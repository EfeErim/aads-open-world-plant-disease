"""Gate decisions and summaries for the single-notebook OOD/OE recovery flow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.ood.behavioral_acceptance import behavioral_acceptance_pass, behavioral_dev_report_pass
from src.ood.recovery import TARGET_ADAPTERS

CLASSIFICATION_CHECKS = ("accuracy", "balanced_accuracy", "macro_f1", "id_test_samples")
ID_BEHAVIOR_CHECKS = (*CLASSIFICATION_CHECKS, "id_false_rejection_rate", "id_decisions_complete")
SAME_CROP_OOD_CHECKS = (
    "same_crop_ood_test_samples",
    "forced_supported_answer_count",
    "same_crop_ood_rejection_rate",
    "same_crop_ood_decisions_complete",
)


def read_behavioral_acceptance(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def read_behavioral_dev_report(path: Path | None) -> dict[str, Any]:
    return read_behavioral_acceptance(path)


def read_readiness(path: Path | None) -> dict[str, Any]:
    """Deprecated compatibility alias; campaign decisions expect behavioral artifacts."""

    return read_behavioral_acceptance(path)


def _required_checks_pass(checks: Mapping[str, Any], required: tuple[str, ...]) -> bool:
    for key in required:
        check = checks.get(key)
        if not isinstance(check, Mapping) or check.get("passed") is not True:
            return False
    return True


def classification_gates_pass(payload: Mapping[str, Any]) -> bool:
    return _required_checks_pass(dict(payload.get("checks") or {}), CLASSIFICATION_CHECKS)


def id_behavior_gates_pass(payload: Mapping[str, Any]) -> bool:
    return _required_checks_pass(dict(payload.get("checks") or {}), ID_BEHAVIOR_CHECKS)


def same_crop_evidence_floor_pass(payload: Mapping[str, Any]) -> bool:
    return _required_checks_pass(dict(payload.get("checks") or {}), ("same_crop_ood_test_samples",))


def same_crop_rejection_pass(payload: Mapping[str, Any]) -> bool:
    return _required_checks_pass(dict(payload.get("checks") or {}), SAME_CROP_OOD_CHECKS)


def readiness_ready(payload: Mapping[str, Any]) -> bool:
    """Deprecated name retained for callers; the decision is behavioral acceptance."""

    return behavioral_acceptance_pass(payload)


def experiment_gate(stage: str, current: Mapping[str, Any]) -> tuple[bool, str]:
    resolved_stage = str(stage or "").upper()
    if behavioral_dev_report_pass(current):
        return False, "already_passed_dev_selection"
    if resolved_stage in {"TRAIN", "A"}:
        return True, "candidate_a_required"
    if resolved_stage == "B":
        return (not id_behavior_gates_pass(current), "id_dev_or_false_rejection_recovery_required")
    if resolved_stage == "C":
        if not id_behavior_gates_pass(current):
            return False, "id_behavior_gate_failed"
        if not same_crop_evidence_floor_pass(current):
            return False, "same_crop_ood_sample_floor_failed"
        if same_crop_rejection_pass(current):
            return False, "ood_dev_already_passed"
        return True, "ood_dev_recovery_required"
    if resolved_stage == "D":
        if not id_behavior_gates_pass(current):
            return False, "id_behavior_gate_failed"
        if not same_crop_evidence_floor_pass(current):
            return False, "same_crop_ood_sample_floor_failed"
        if same_crop_rejection_pass(current):
            return False, "same_crop_rejection_already_passed"
        return True, "bounded_capacity_recovery_required"
    return False, "unknown_stage"


def json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evidence_preflight_blocked_targets(
    report: Mapping[str, Any],
    *,
    required_targets: Iterable[str],
) -> set[str]:
    """Map strict manifest errors to targets without hiding global integrity failures."""
    targets = {str(target) for target in required_targets if str(target)}
    errors = [
        issue
        for issue in list(report.get("issues") or [])
        if isinstance(issue, Mapping) and str(issue.get("severity") or "") == "error"
    ]
    if any(not str(issue.get("target") or "").strip() for issue in errors):
        return targets
    return {str(issue.get("target")) for issue in errors if str(issue.get("target") or "") in targets}


def build_campaign_lineage(
    *,
    input_revision: str,
    campaign_digest: str,
    evidence_manifest_digest: str,
) -> dict[str, str]:
    lineage = {
        "input_revision": str(input_revision),
        "campaign_digest": str(campaign_digest),
        "evidence_manifest_digest": str(evidence_manifest_digest),
    }
    lineage["campaign_id"] = hashlib.sha256(
        "|".join(lineage.values()).encode("utf-8")
    ).hexdigest()[:20]
    return lineage


def load_campaign_ledger(path: Path, *, lineage: Mapping[str, str]) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": "v1_adapter_ood_oe_campaign_ledger",
            "lineage": dict(lineage),
            "experiments": {},
            "targets": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Campaign ledger must be a JSON object: {path}")
    if payload.get("schema_version") != "v1_adapter_ood_oe_campaign_ledger":
        raise ValueError(f"Unsupported campaign ledger schema: {path}")
    if dict(payload.get("lineage") or {}) != dict(lineage):
        raise ValueError("Campaign ledger lineage does not match the pinned recovery inputs.")
    return payload


def write_campaign_ledger(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resumable_experiment(
    ledger: Mapping[str, Any],
    *,
    experiment_id: str,
    resolved_config_digest: str,
) -> tuple[Path | None, dict[str, Any]]:
    entry = dict((ledger.get("experiments") or {}).get(experiment_id) or {})
    if entry.get("status") != "completed" or entry.get("resolved_config_digest") != resolved_config_digest:
        return None, {}
    report_path_text = str(entry.get("dev_report_path") or "")
    adapter_artifact_path_text = str(entry.get("adapter_artifact_path") or "")
    if not report_path_text or not adapter_artifact_path_text:
        return None, {}
    report_path = Path(report_path_text)
    adapter_artifact_path = Path(adapter_artifact_path_text)
    if not report_path.is_file() or not adapter_artifact_path.exists():
        return None, {}
    try:
        report = read_behavioral_dev_report(report_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None, {}
    if not report or report.get("authoritative") is not False:
        return None, {}
    return report_path, report


def build_notebook_completion_report(
    *,
    target_results: Mapping[str, Any],
    preflight_ok: bool,
) -> dict[str, Any]:
    passed_targets = []
    failed_targets = []
    blocked_targets = []
    for target in TARGET_ADAPTERS:
        item = target_results.get(target)
        payload = item if isinstance(item, Mapping) else {}
        if bool(payload.get("pass")):
            passed_targets.append(target)
        elif str(payload.get("status") or "") == "blocked":
            blocked_targets.append(target)
        else:
            failed_targets.append(target)
    passed = bool(preflight_ok and len(passed_targets) == len(TARGET_ADAPTERS))
    return {
        "schema_version": "v2_adapter_ood_oe_behavioral_recovery_notebook_report",
        "decision_authority": "adapter_behavioral_acceptance",
        "pass": passed,
        "ready": passed,
        "preflight_ok": bool(preflight_ok),
        "required_target_count": len(TARGET_ADAPTERS),
        "passed_target_count": len(passed_targets),
        "passed_targets": passed_targets,
        "ready_target_count": len(passed_targets),
        "ready_targets": passed_targets,
        "blocked_targets": blocked_targets,
        "failed_targets": failed_targets,
        "target_results": dict(target_results),
    }
