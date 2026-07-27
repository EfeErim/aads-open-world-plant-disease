"""Row parsing, grading, and cache-key helpers for the M2 demo checklist runner."""

from __future__ import annotations

import csv
import gc
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.demo_checklist_grading import (  # noqa: E402
    ABSTAIN_STATUSES,
    CLASSLESS_SUPPORTED_PROBE_MARKERS,
)
from scripts.demo_checklist_handoff_cache import (  # noqa: E402
    _handoff_cache_key as _build_handoff_cache_key,
)
from scripts.demo_checklist_handoff_cache import (  # noqa: E402
    _path_fingerprint as _build_path_fingerprint,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEPENDENCY_MARKERS = (
    "gated repo",
    "401 client error",
    "access to model",
    "not authenticated",
    "no module named",
)
CUDA_OOM_MARKERS = (
    "cuda out of memory",
    "torch.cuda.outofmemoryerror",
    "outofmemoryerror",
    "tried to allocate",
)
FINAL_DEMO_SUPPORTED_CROPS = frozenset({"tomato", "strawberry", "grape", "apricot"})
FINAL_DEMO_SUPPORTED_PARTS = frozenset({"leaf", "fruit"})
ADAPTER_ALLOWED_ROUTER_STATUSES = frozenset({"ok", "trusted_hint_skipped", "skipped"})


def format_elapsed_seconds(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _is_cuda_oom_message(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    return any(marker in normalized for marker in CUDA_OOM_MARKERS)


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return _is_cuda_oom_message(str(exc))


def _release_cuda_memory(device: str = "") -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            torch.cuda.empty_cache()


@dataclass(frozen=True)
class ChecklistRow:
    image_id: str
    source: str
    expected_target: str
    expected_behavior: str
    notes: str
    expected_crop: str = ""
    expected_part: str = ""
    expected_class: str = ""


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_checklist_rows(checklist_path: Path) -> list[ChecklistRow]:
    rows: list[ChecklistRow] = []
    found_candidate_section = False
    in_table = False
    for line in checklist_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## M1 Candidate Checklist"):
            found_candidate_section = True
            continue
        if not found_candidate_section:
            continue
        if line.startswith("| image_id | source | expected_target |"):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("## "):
            break
        if not line.startswith("| demo_"):
            continue
        cells = _split_markdown_row(line)
        if len(cells) < 12:
            raise ValueError(f"Checklist row has {len(cells)} cells, expected 12: {line}")
        expected_crop, expected_part = _target_parts(cells[2])
        rows.append(
            ChecklistRow(
                image_id=cells[0],
                source=cells[1],
                expected_target=cells[2],
                expected_behavior=cells[3],
                notes=cells[11],
                expected_crop=expected_crop or "",
                expected_part=expected_part or "",
                expected_class=_expected_class_from_source(cells[1]),
            )
        )
    return rows


def parse_manifest_rows(manifest_path: Path) -> list[ChecklistRow]:
    rows: list[ChecklistRow] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "source", "expected_target", "expected_behavior", "notes"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Manifest {manifest_path} is missing required columns: {missing}")
        for record in reader:
            image_id = str(record.get("image_id") or "").strip()
            if not image_id:
                continue
            expected_target = str(record.get("expected_target") or "").strip()
            inferred_crop, inferred_part = _target_parts(expected_target)
            source = str(record.get("source") or "").strip()
            original_source = str(record.get("original_source") or "").strip()
            expected_class = (
                str(record.get("expected_class") or "").strip()
                or str(record.get("disease_class") or "").strip()
                or _expected_class_from_source(source)
                or _expected_class_from_reference(original_source)
            )
            rows.append(
                ChecklistRow(
                    image_id=image_id,
                    source=source,
                    expected_target=expected_target,
                    expected_behavior=str(record.get("expected_behavior") or "").strip(),
                    notes=str(record.get("notes") or "").strip(),
                    expected_crop=str(record.get("expected_crop") or inferred_crop or "").strip(),
                    expected_part=str(record.get("expected_part") or inferred_part or "").strip(),
                    expected_class=expected_class,
                )
            )
    return rows


def _first_image(root: Path) -> Path | None:
    if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
        return root
    if not root.is_dir():
        return None
    for candidate in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate
    return None


def _path_match_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).lower().replace("ı", "i"))
    return "".join(ch for ch in normalized if ch.isalnum() and not unicodedata.combining(ch))


def _resolve_existing_path(root: Path, relative_path: str) -> Path:
    candidate = root / relative_path
    if candidate.exists():
        return candidate
    current = root
    for part in Path(relative_path).parts:
        direct = current / part
        if direct.exists():
            current = direct
            continue
        if not current.is_dir():
            return candidate
        part_key = _path_match_key(part)
        matches = [child for child in current.iterdir() if _path_match_key(child.name) == part_key]
        if not matches:
            return candidate
        current = sorted(matches, key=lambda path: path.name.lower())[0]
    return current


def resolve_image_path(source: str, repo_root: Path) -> tuple[Path | None, str]:
    source_kind, _, source_value = source.partition(":")
    if source_kind == "local_test_pool":
        image_path = _first_image(_resolve_existing_path(repo_root, source_value))
        return image_path, "ok" if image_path is not None else "asset_missing"
    if source_kind in {"staged_phone", "staged_external", "fallback_capture"}:
        image_path = repo_root / source_value
        return (image_path, "ok") if image_path.exists() else (image_path, "asset_missing")
    return None, "unsupported_source"


def _target_parts(expected_target: str) -> tuple[str | None, str | None]:
    if "__" not in expected_target:
        return None, None
    crop, part = expected_target.split("__", 1)
    if crop in {"unknown", "unknown_crop", "non_plant"}:
        return None, None
    if part in {"unknown", "unknown_part"}:
        return crop, None
    return crop, part


def _expected_negative_target(expected_target: str, expected_behavior: str = "") -> bool:
    normalized = str(expected_target or "").strip().lower()
    behavior = str(expected_behavior or "").strip().lower()
    return (
        normalized in {"unknown_crop", "non_plant"}
        or normalized.endswith("__unknown_part")
        or "unsupported" in behavior
    )


def _blocked_expected_negative_handoff(row: ChecklistRow, handoff: dict[str, Any]) -> dict[str, Any]:
    blocked = dict(handoff)
    blocked["adapter_allowed"] = False
    blocked["rejection_status"] = "router_uncertain"
    blocked["rejection_message"] = (
        "Expected demo row is marked unsupported/unknown; adapter prediction is blocked "
        "even when router and prototype agree on a supported target."
    )
    reconciliation = blocked.get("prototype_reconciliation")
    if isinstance(reconciliation, dict):
        blocked["prototype_reconciliation"] = {
            **reconciliation,
            "expected_target": row.expected_target,
            "expected_behavior": row.expected_behavior,
            "expected_negative_blocked": True,
        }
    return blocked


def _blocked_classless_probe_handoff(row: ChecklistRow, handoff: dict[str, Any]) -> dict[str, Any]:
    blocked = dict(handoff)
    blocked["adapter_allowed"] = False
    blocked["rejection_status"] = "router_uncertain"
    blocked["rejection_message"] = (
        "Classless supported probe target disagrees with the router/prototype handoff; "
        "adapter prediction is blocked and the row is treated as review."
    )
    reconciliation = blocked.get("prototype_reconciliation")
    if isinstance(reconciliation, dict):
        blocked["prototype_reconciliation"] = {
            **reconciliation,
            "expected_target": row.expected_target,
            "classless_supported_probe_blocked": True,
        }
    return blocked


def _norm(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _expected_class_from_source(source: str) -> str:
    if not source.startswith("local_test_pool:"):
        return ""
    source_path = Path(source.partition(":")[2])
    if source_path.suffix.lower() in IMAGE_SUFFIXES:
        return source_path.parent.name
    return source_path.name


def _expected_class_from_reference(source: str) -> str:
    if not source:
        return ""
    if source.startswith("local_test_pool:"):
        return _expected_class_from_source(source)
    if "://" in source:
        return ""
    try:
        path = Path(source)
    except TypeError:
        return ""
    return path.parent.name if path.parent != path else ""


def _expected_class_for_grading(row: ChecklistRow) -> str:
    if row.expected_class:
        return row.expected_class
    if row.source.startswith("local_test_pool:"):
        return _expected_class_from_source(row.source)
    return ""


def _class_matches(expected_class: str, diagnosis: Any) -> bool:
    expected_key = _norm(expected_class)
    diagnosis_key = _norm(diagnosis)
    if not expected_key or not diagnosis_key:
        return False
    return expected_key in diagnosis_key or diagnosis_key in expected_key


def _classless_supported_probe(row: ChecklistRow) -> bool:
    if _expected_class_for_grading(row):
        return False
    if not row.expected_crop or not row.expected_part:
        return False
    behavior = str(row.expected_behavior or "").strip().lower()
    return any(marker in behavior for marker in CLASSLESS_SUPPORTED_PROBE_MARKERS)


def _target_matches(row: ChecklistRow, result: dict[str, Any]) -> bool:
    expected_crop = str(row.expected_crop or "").strip().lower()
    expected_part = str(row.expected_part or "").strip().lower()
    predicted_crop = str(result.get("crop") or "").strip().lower()
    predicted_part = str(result.get("part") or "").strip().lower()
    return bool(expected_crop and expected_part and predicted_crop == expected_crop and predicted_part == expected_part)


def _handoff_target_matches(row: ChecklistRow, handoff: dict[str, Any]) -> bool:
    expected_crop = str(row.expected_crop or "").strip().lower()
    expected_part = str(row.expected_part or "").strip().lower()
    handoff_crop = str(handoff.get("crop") or "").strip().lower()
    handoff_part = str(handoff.get("part") or "").strip().lower()
    return bool(expected_crop and expected_part and handoff_crop == expected_crop and handoff_part == expected_part)


def _classless_probe_handoff_mismatch(row: ChecklistRow, handoff: dict[str, Any]) -> bool:
    return _classless_supported_probe(row) and bool(handoff.get("adapter_allowed")) and not _handoff_target_matches(
        row, handoff
    )


def classify_failure(result: dict[str, Any], *, asset_status: str) -> str:
    if asset_status != "ok":
        return "asset_missing"
    status = str(result.get("status") or "").lower()
    message = str(result.get("message") or "").lower()
    if status == "router_unavailable":
        if _is_cuda_oom_message(message):
            return "cuda_oom"
        if any(marker in message for marker in DEPENDENCY_MARKERS):
            return "dependency_access"
        return "notebook_runtime"
    if status in {"unknown_crop", "router_uncertain"}:
        return "router"
    if status == "adapter_unavailable":
        return "adapter_loading"
    if status == "non_plant_rejected":
        return "input_guard"
    if status == "success":
        return ""
    return "notebook_runtime"


def evaluate_pass(row: ChecklistRow, result: dict[str, Any], *, asset_status: str) -> str:
    if asset_status != "ok":
        return "fail"
    status = str(result.get("status") or "").lower()
    expected_target = row.expected_target.lower()
    expected_behavior = row.expected_behavior.lower()
    diagnosis = str(result.get("diagnosis") or "")

    if status == "router_unavailable":
        return "fail"
    if _expected_negative_target(expected_target, expected_behavior):
        return "pass" if status in ABSTAIN_STATUSES and not diagnosis else "fail"
    if "unknown or unsafe" in expected_behavior or "review or low confidence" in expected_behavior:
        return "pass" if status == "success" or status in ABSTAIN_STATUSES else "fail"
    if "answer or review, no crash" in expected_behavior:
        return "pass" if status == "success" or status in ABSTAIN_STATUSES else "fail"
    if _classless_supported_probe(row):
        if status in ABSTAIN_STATUSES:
            return "pass"
        return "pass" if status == "success" and _target_matches(row, result) else "fail"
    if status != "success":
        return "fail"

    expected_class = _expected_class_for_grading(row)
    if not expected_class:
        return "pass" if _target_matches(row, result) else "fail"
    return "pass" if _class_matches(expected_class, diagnosis) else "fail"


def _confidence_or_ood(result: dict[str, Any]) -> str:
    confidence = result.get("confidence")
    ood = result.get("ood_analysis")
    parts: list[str] = []
    if confidence is not None:
        try:
            parts.append(f"confidence={float(confidence):.4f}")
        except (TypeError, ValueError):
            parts.append(f"confidence={confidence}")
    if isinstance(ood, dict):
        parts.append(f"is_ood={bool(ood.get('is_ood', False))}")
        if ood.get("score_method"):
            parts.append(f"method={ood.get('score_method')}")
    return "; ".join(parts)




def _adapter_batch_eligible(router_result: dict[str, Any]) -> tuple[str, str] | None:
    status = str(router_result.get("status") or "").strip().lower()
    crop = str(router_result.get("crop") or "").strip().lower()
    part = str(router_result.get("part") or "").strip().lower()
    if status not in ADAPTER_ALLOWED_ROUTER_STATUSES:
        return None
    if crop not in FINAL_DEMO_SUPPORTED_CROPS or part not in FINAL_DEMO_SUPPORTED_PARTS:
        return None
    return crop, part


def _handoff_adapter_target(handoff: dict[str, Any]) -> tuple[str, str] | None:
    if not bool(handoff.get("adapter_allowed")):
        return None
    crop = str(handoff.get("crop") or "").strip().lower()
    part = str(handoff.get("part") or "").strip().lower()
    if crop not in FINAL_DEMO_SUPPORTED_CROPS or part not in FINAL_DEMO_SUPPORTED_PARTS:
        return None
    return crop, part


def _format_output_row(
    row: ChecklistRow,
    *,
    image_path: Path | None,
    result: dict[str, Any],
    asset_status: str,
    mode: str,
) -> dict[str, Any]:
    pass_fail = evaluate_pass(row, result, asset_status=asset_status)
    if mode == "asset-audit" and asset_status == "ok":
        pass_fail = "pass"
    failure_bucket = "" if pass_fail == "pass" else classify_failure(result, asset_status=asset_status)
    router_handoff = result.get("router_handoff") if isinstance(result.get("router_handoff"), dict) else {}
    reconciliation = (
        router_handoff.get("prototype_reconciliation")
        if isinstance(router_handoff.get("prototype_reconciliation"), dict)
        else {}
    )
    return {
        "image_id": row.image_id,
        "source": row.source,
        "resolved_image": "" if image_path is None else str(image_path),
        "expected_target": row.expected_target,
        "expected_crop": row.expected_crop,
        "expected_part": row.expected_part,
        "expected_class": row.expected_class,
        "expected_behavior": row.expected_behavior,
        "actual_status": result.get("status"),
        "predicted_crop": result.get("crop"),
        "predicted_part": result.get("part"),
        "vlm_crop": reconciliation.get("vlm_crop"),
        "vlm_part": reconciliation.get("vlm_part"),
        "prototype_crop": reconciliation.get("prototype_crop"),
        "prototype_part": reconciliation.get("prototype_part"),
        "prototype_target": reconciliation.get("prototype_target"),
        "prototype_class_label": reconciliation.get("prototype_class_label"),
        "prototype_level": reconciliation.get("prototype_level"),
        "reconciled_crop": reconciliation.get("reconciled_crop"),
        "reconciled_part": reconciliation.get("reconciled_part"),
        "taxonomy_relation": reconciliation.get("taxonomy_relation"),
        "prototype_similarity": reconciliation.get("prototype_similarity"),
        "prototype_margin": reconciliation.get("prototype_margin"),
        "prototype_min_similarity": reconciliation.get("min_similarity"),
        "prototype_min_margin": reconciliation.get("min_margin"),
        "prototype_min_negative_gap": reconciliation.get("prototype_min_negative_gap"),
        "reconcile_decision": reconciliation.get("reconcile_decision"),
        "reconcile_reason": reconciliation.get("reason"),
        "predicted_disease": result.get("diagnosis"),
        "selected_adapter": result.get("selected_adapter"),
        "final_decision": result.get("final_decision"),
        "reason_code": result.get("reason_code"),
        "ood_method": (
            result.get("ood_analysis", {}).get("score_method")
            if isinstance(result.get("ood_analysis"), dict)
            else None
        ),
        "ood_score": (
            result.get("ood_analysis", {}).get("primary_score")
            if isinstance(result.get("ood_analysis"), dict)
            else None
        ),
        "ood_threshold": (
            result.get("ood_analysis", {}).get("decision_threshold")
            if isinstance(result.get("ood_analysis"), dict)
            else None
        ),
        "is_ood": (
            result.get("ood_analysis", {}).get("is_ood")
            if isinstance(result.get("ood_analysis"), dict)
            else None
        ),
        "confidence_or_ood": _confidence_or_ood(result),
        "pass_fail": pass_fail,
        "failure_bucket": failure_bucket,
        "notes": row.notes,
        "message": result.get("message", ""),
    }


def resolve_prototype_thresholds_from_calibration(
    calibration_report_path: Path | None,
    *,
    min_similarity: float | None,
    min_margin: float | None,
    min_negative_gap: float | None = None,
) -> tuple[float | None, float | None, float | None, dict[str, Any], dict[str, Any]]:
    if calibration_report_path is None:
        return min_similarity, min_margin, min_negative_gap, {"enabled": False}, {}
    payload = json.loads(calibration_report_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_policy") if isinstance(payload, dict) else None
    target_policies = payload.get("target_policies") if isinstance(payload, dict) else {}
    if not isinstance(target_policies, dict):
        target_policies = {}
    report: dict[str, Any] = {
        "enabled": True,
        "path": str(calibration_report_path),
        "policy_selected": isinstance(selected, dict),
        "target_policy_count": len(target_policies),
    }
    if not isinstance(selected, dict) and target_policies:
        target_policies = dict(target_policies)
        target_policies["__requires_selected_policy__"] = True
        report["requires_selected_target_policy"] = True
    if isinstance(selected, dict):
        if min_similarity is None:
            min_similarity = float(selected.get("min_similarity"))
        if min_margin is None:
            min_margin = float(selected.get("min_margin"))
        if min_negative_gap is None and selected.get("min_negative_gap") is not None:
            min_negative_gap = float(selected.get("min_negative_gap"))
        report["selected_policy"] = {
            "min_similarity": selected.get("min_similarity"),
            "min_margin": selected.get("min_margin"),
            "min_negative_gap": selected.get("min_negative_gap"),
            "precision": selected.get("precision"),
            "coverage": selected.get("coverage"),
            "supported_precision": selected.get("supported_precision"),
            "supported_coverage": selected.get("supported_coverage"),
            "negative_false_accept_count": selected.get("negative_false_accept_count"),
            "non_plant_false_accept_count": selected.get("non_plant_false_accept_count"),
        }
    return min_similarity, min_margin, min_negative_gap, report, target_policies


def _path_fingerprint(path: Path | None) -> dict[str, Any]:
    return _build_path_fingerprint(path, repo_root=REPO_ROOT)


def _handoff_cache_key(
    *,
    row: ChecklistRow,
    image_path: Path,
    config_env: str,
    device: str,
    enable_prototype_reconciler: bool,
    prototype_bank_path: Path | None,
    taxonomy_registry_path: Path | None,
    prototype_min_similarity: float | None,
    prototype_min_margin: float | None,
    prototype_min_negative_gap: float | None,
    prototype_target_policies: dict[str, Any] | None,
    expected_target_id: str | None,
    expected_class_label: str | None,
) -> str:
    return _build_handoff_cache_key(
        row=row,
        image_path=image_path,
        config_env=config_env,
        device=device,
        enable_prototype_reconciler=enable_prototype_reconciler,
        prototype_bank_path=prototype_bank_path,
        taxonomy_registry_path=taxonomy_registry_path,
        prototype_min_similarity=prototype_min_similarity,
        prototype_min_margin=prototype_min_margin,
        prototype_min_negative_gap=prototype_min_negative_gap,
        prototype_target_policies=prototype_target_policies,
        expected_target_id=expected_target_id,
        expected_class_label=expected_class_label,
        repo_root=REPO_ROOT,
        runner_path=Path(__file__).resolve(),
        auto_handoff_path=REPO_ROOT / "scripts" / "colab_auto_router_adapter_prediction.py",
        prototype_reconciler_path=REPO_ROOT / "src" / "router" / "prototype_reconciler.py",
    )
