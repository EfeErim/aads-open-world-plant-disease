"""Adapter OOD/OE recovery reporting and promotion-gate helpers."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.ood.behavioral_acceptance import DEFAULT_BEHAVIORAL_THRESHOLDS, behavioral_acceptance_pass

TARGET_ADAPTERS = (
    "apricot__fruit",
    "apricot__leaf",
    "grape__fruit",
    "grape__leaf",
    "strawberry__fruit",
    "strawberry__leaf",
    "tomato__fruit",
    "tomato__leaf",
)

READINESS_THRESHOLDS = {
    "accuracy": (">=", 0.93),
    "balanced_accuracy": (">=", 0.90),
    "macro_f1": (">=", 0.90),
    "ood_auroc": (">=", 0.92),
    "ood_false_positive_rate": ("<=", 0.05),
    "id_test_samples": (">=", 30.0),
    "ood_test_samples": (">=", 30.0),
    "min_ood_type_samples": (">=", 5.0),
}


@dataclass(frozen=True)
class RecoveryBlocker:
    priority: str
    code: str
    message: str


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _nested(payload: Mapping[str, Any], *path: str) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _metric(payload: Mapping[str, Any], name: str) -> float | None:
    aliases = {
        "ood_test_samples": ("ood_samples", "sample_count", "image_count", "n_ood"),
        "id_test_samples": (
            "id_test_samples",
            "test_samples",
            "test_sample_count",
            "in_distribution_samples",
            "samples",
        ),
    }
    names = aliases.get(name, (name,))
    containers = (
        payload,
        _mapping(payload.get("metrics")),
        _mapping(payload.get("classification_metrics")),
        _mapping(_nested(payload, "classification_evidence", "metrics")),
        _mapping(payload.get("ood_metrics")),
        _mapping(payload.get("ood_evidence")),
        _mapping(_nested(payload, "ood_evidence", "metrics")),
        _mapping(payload.get("context")),
    )
    for container in containers:
        value = _number(*(container.get(candidate) for candidate in names))
        if value is not None:
            return value

    checks = _mapping(payload.get("checks"))
    for candidate in names:
        check = _mapping(checks.get(candidate))
        value = _number(check.get("actual"), check.get("value"))
        if value is not None:
            return value
    return None


def _slice_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    candidates = (
        payload.get("ood_type_counts"),
        payload.get("ood_slice_counts"),
        _nested(payload, "ood_evidence", "type_counts"),
        _nested(payload, "ood_evidence", "slice_counts"),
        _nested(payload, "context", "ood_type_counts"),
        payload.get("ood_type_sample_checks"),
    )
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        result: dict[str, int] = {}
        for key, value in candidate.items():
            number = _number(
                value,
                _mapping(value).get("actual"),
                _mapping(value).get("value"),
                _mapping(value).get("sample_count"),
            )
            if number is not None:
                result[str(key)] = int(number)
        if result:
            return result
    return {}


def _passes(value: float | None, operator: str, threshold: float) -> bool:
    if value is None:
        return False
    return value >= threshold if operator == ">=" else value <= threshold


def _infer_target(path: Path, payload: Mapping[str, Any]) -> str:
    target = str(payload.get("dataset_key") or payload.get("adapter_target") or "").strip().lower()
    if target in TARGET_ADAPTERS:
        return target
    crop = str(payload.get("crop_name") or _nested(payload, "context", "crop_name") or "").strip().lower()
    part = str(payload.get("part_name") or _nested(payload, "context", "part_name") or "").strip().lower()
    if crop and part and f"{crop}__{part}" in TARGET_ADAPTERS:
        return f"{crop}__{part}"
    lowered_parts = [part.lower() for part in path.parts]
    for candidate in TARGET_ADAPTERS:
        crop, plant_part = candidate.split("__", 1)
        if candidate in lowered_parts or (crop in lowered_parts and plant_part in lowered_parts):
            return candidate
    return ""


def discover_latest_readiness(runs_root: Path) -> dict[str, Path]:
    """Return the newest readiness artifact for every known crop/part target."""
    selected: dict[str, tuple[str, int, Path]] = {}
    if not runs_root.exists():
        return {}
    for path in runs_root.rglob("production_readiness.json"):
        if "_index" in path.relative_to(runs_root).parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                continue
            target = _infer_target(path, payload)
            run_timestamps = re.findall(r"20\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", path.as_posix())
            run_timestamp = max(run_timestamps) if run_timestamps else ""
            timestamp = path.stat().st_mtime_ns
        except (OSError, json.JSONDecodeError):
            continue
        candidate = (run_timestamp, timestamp, path)
        if target and (target not in selected or candidate[:2] > selected[target][:2]):
            selected[target] = candidate
    return {target: item[2] for target, item in selected.items()}


def discover_latest_behavioral_acceptance(runs_root: Path) -> dict[str, Path]:
    """Return the newest authoritative behavioral artifact for every target."""

    selected: dict[str, tuple[str, int, Path]] = {}
    if not runs_root.exists():
        return {}
    for path in runs_root.rglob("adapter_behavioral_acceptance.json"):
        if "_index" in path.relative_to(runs_root).parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                continue
            target = _infer_target(path, payload)
            run_timestamps = re.findall(r"20\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", path.as_posix())
            run_timestamp = max(run_timestamps) if run_timestamps else ""
            timestamp = path.stat().st_mtime_ns
        except (OSError, json.JSONDecodeError):
            continue
        candidate = (run_timestamp, timestamp, path)
        if target and (target not in selected or candidate[:2] > selected[target][:2]):
            selected[target] = candidate
    return {target: item[2] for target, item in selected.items()}


def evaluate_readiness(target: str, path: Path | None) -> dict[str, Any]:
    """Normalize one readiness artifact and classify recovery blockers."""
    if path is None:
        return {
            "target": target,
            "status": "missing",
            "readiness_path": "",
            "metrics": {key: None for key in READINESS_THRESHOLDS},
            "ood_type_counts": {},
            "worst_ood_slice": {},
            "ood_method": "",
            "ood_threshold": None,
            "training": {},
            "provenance": {},
            "blockers": [
                asdict(
                    RecoveryBlocker(
                        "P2",
                        "readiness_artifact_missing",
                        "No production_readiness.json artifact was found for this required target.",
                    )
                )
            ],
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "target": target,
            "status": "invalid",
            "readiness_path": str(path),
            "metrics": {key: None for key in READINESS_THRESHOLDS},
            "ood_type_counts": {},
            "blockers": [asdict(RecoveryBlocker("P0", "readiness_artifact_invalid", str(exc)))],
        }
    if not isinstance(payload, Mapping):
        return evaluate_readiness(target, None)

    metrics = {key: _metric(payload, key) for key in READINESS_THRESHOLDS}
    type_counts = _slice_counts(payload)
    metrics["min_ood_type_samples"] = float(min(type_counts.values())) if type_counts else None
    blockers: list[RecoveryBlocker] = []
    for name, (operator, threshold) in READINESS_THRESHOLDS.items():
        value = metrics[name]
        if value is None:
            priority = "P2" if name.endswith("_samples") else "P1"
            blockers.append(RecoveryBlocker(priority, f"{name}_missing", f"{name} is not reported."))
        elif not _passes(value, operator, threshold):
            blockers.append(
                RecoveryBlocker(
                    "P1",
                    f"{name}_failed",
                    f"{name}={value:.4f} does not satisfy {operator} {threshold:.4f}.",
                )
            )

    overlap_count = int(_number(payload.get("evidence_overlap_count"), 0) or 0)
    if overlap_count:
        blockers.append(
            RecoveryBlocker("P0", "evidence_hash_overlap", f"Readiness reports {overlap_count} evidence overlaps.")
        )
    status = str(payload.get("status") or payload.get("readiness_status") or "unknown").lower()
    if status != "ready":
        blockers.append(RecoveryBlocker("P1", "readiness_not_ready", f"Artifact status is {status!r}, not 'ready'."))

    method = str(
        payload.get("ood_primary_score_method")
        or _nested(payload, "context", "ood_primary_score_method")
        or _nested(payload, "ood_calibration", "primary_score_method")
        or ""
    )
    threshold = _number(
        payload.get("ood_threshold"),
        _nested(payload, "context", "ood_primary_score_selection", "selected_threshold"),
        _nested(payload, "ood_calibration", "selected_threshold"),
    )
    return {
        "target": target,
        "status": status,
        "readiness_path": str(path),
        "metrics": metrics,
        "ood_type_counts": type_counts,
        "worst_ood_slice": _mapping(payload.get("worst_ood_slice")),
        "ood_method": method,
        "ood_threshold": threshold,
        "training": {
            "loss_name": _nested(payload, "context", "optimization", "loss_name")
            or _nested(payload, "optimization", "loss_name"),
            "logitnorm_tau": _nested(payload, "context", "optimization", "logitnorm_tau")
            or _nested(payload, "optimization", "logitnorm_tau"),
            "oe": _mapping(_nested(payload, "context", "oe") or payload.get("oe")),
        },
        "provenance": {
            "run_id": payload.get("run_id") or _nested(payload, "context", "run_id"),
            "dataset": payload.get("dataset_key") or _nested(payload, "context", "dataset_key"),
            "ood_evidence_source": payload.get("ood_evidence_source"),
        },
        "blockers": [asdict(blocker) for blocker in blockers],
    }


def evaluate_behavioral_acceptance(target: str, path: Path | None) -> dict[str, Any]:
    """Normalize the authoritative adapter decision for campaign reporting."""

    if path is None:
        return {
            "target": target,
            "status": "missing",
            "pass": False,
            "behavioral_acceptance_path": "",
            "metrics": {},
            "checks": {},
            "blockers": [
                asdict(
                    RecoveryBlocker(
                        "P2",
                        "behavioral_acceptance_artifact_missing",
                        "No adapter_behavioral_acceptance.json artifact was found for this required target.",
                    )
                )
            ],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "target": target,
            "status": "invalid",
            "pass": False,
            "behavioral_acceptance_path": str(path),
            "metrics": {},
            "checks": {},
            "blockers": [asdict(RecoveryBlocker("P0", "behavioral_acceptance_artifact_invalid", str(exc)))],
        }
    if not isinstance(payload, Mapping):
        return evaluate_behavioral_acceptance(target, None)
    checks = dict(payload.get("checks") or {})
    blockers = [
        asdict(
            RecoveryBlocker(
                "P2" if name.endswith("samples") or name.endswith("complete") else "P1",
                f"{name}_failed",
                f"Behavioral acceptance check {name!r} did not pass.",
            )
        )
        for name, check in checks.items()
        if not isinstance(check, Mapping) or check.get("passed") is not True
    ]
    passed = behavioral_acceptance_pass(payload)
    if not passed and not blockers:
        blockers.append(
            asdict(RecoveryBlocker("P1", "behavioral_acceptance_failed", "Behavioral contract did not pass."))
        )
    return {
        "target": target,
        "status": "pass" if passed else str(payload.get("status") or "fail"),
        "pass": passed,
        "behavioral_acceptance_path": str(path),
        "metrics": dict(payload.get("metrics") or {}),
        "checks": checks,
        "same_crop_ood_per_disease": dict(payload.get("same_crop_ood_per_disease") or {}),
        "context": dict(payload.get("context") or {}),
        "blockers": blockers,
    }


def build_recovery_report(runs_root: Path, readiness_paths: Mapping[str, Path] | None = None) -> dict[str, Any]:
    selected = dict(readiness_paths or discover_latest_behavioral_acceptance(runs_root))
    adapters = [evaluate_behavioral_acceptance(target, selected.get(target)) for target in TARGET_ADAPTERS]
    priorities = {"P0": 0, "P1": 0, "P2": 0}
    for adapter in adapters:
        for blocker in adapter["blockers"]:
            priorities[blocker["priority"]] += 1
    passed_targets = [adapter["target"] for adapter in adapters if adapter["pass"] and not adapter["blockers"]]
    return {
        "schema_version": "v2_adapter_behavioral_recovery_report",
        "decision_authority": "adapter_behavioral_acceptance",
        "runs_root": str(runs_root),
        "overall_status": "pass" if len(passed_targets) == len(TARGET_ADAPTERS) else "recovery_required",
        "required_target_count": len(TARGET_ADAPTERS),
        "passed_target_count": len(passed_targets),
        "passed_targets": passed_targets,
        "ready_target_count": len(passed_targets),
        "ready_targets": passed_targets,
        "blocker_counts": priorities,
        "thresholds": {
            key: {"operator": operator, "value": value}
            for key, (operator, value) in DEFAULT_BEHAVIORAL_THRESHOLDS.items()
        },
        "adapters": adapters,
    }


def render_recovery_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Adapter OOD/OE Recovery Baseline",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- Behavioral pass: `{report['passed_target_count']}/{report['required_target_count']}`",
        f"- Blockers: `{report['blocker_counts']}`",
        "",
        "| Target | Status | Accuracy | Balanced accuracy | Macro F1 | ID false reject | Same-crop reject | Forced answers | Blockers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    def value(adapter: Mapping[str, Any], key: str) -> str:
        item = _mapping(adapter.get("metrics")).get(key)
        return "-" if item is None else f"{float(item):.4f}"

    for adapter in report["adapters"]:
        blocker_codes = ", ".join(item["code"] for item in adapter["blockers"]) or "-"
        lines.append(
            f"| `{adapter['target']}` | `{adapter['status']}` | {value(adapter, 'accuracy')} | "
            f"{value(adapter, 'balanced_accuracy')} | {value(adapter, 'macro_f1')} | "
            f"{value(adapter, 'id_false_rejection_rate')} | {value(adapter, 'same_crop_ood_rejection_rate')} | "
            f"{value(adapter, 'forced_supported_answer_count')} | {blocker_codes} |"
        )
    lines.extend(["", "This report is a baseline inventory. Missing evidence is a blocker, never a pass.", ""])
    return "\n".join(lines)


def paths_from_arguments(paths: Iterable[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} must contain a JSON object")
        target = _infer_target(path, payload)
        if not target:
            raise ValueError(f"Cannot infer required crop/part target from {path}")
        result[target] = path
    return result
