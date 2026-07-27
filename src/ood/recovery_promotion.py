"""Safety-first promotion rules for staged adapter OOD/OE recovery candidates."""

from __future__ import annotations

from typing import Any, Mapping

from src.ood.behavioral_acceptance import behavioral_acceptance_pass
from src.ood.recovery import TARGET_ADAPTERS

PROMOTION_SCHEMA = "v2_adapter_behavioral_promotion_report"


def _adapter_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("target") or ""): item
        for item in report.get("adapters", [])
        if isinstance(item, Mapping)
    }


def evaluate_candidate_promotion(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    integrity_report: Mapping[str, Any],
    reload_parity: Mapping[str, bool],
    max_worst_slice_fpr_regression: float = 0.0,
) -> dict[str, Any]:
    """Promote only candidates that pass the default behavioral contract."""
    del baseline, max_worst_slice_fpr_regression
    candidate_adapters = _adapter_map(candidate)
    results: list[dict[str, Any]] = []
    for target in TARGET_ADAPTERS:
        current = candidate_adapters.get(target, {})
        acceptance = current.get("behavioral_acceptance", current)
        acceptance = acceptance if isinstance(acceptance, Mapping) else {}
        blockers: list[str] = []
        if not behavioral_acceptance_pass(acceptance):
            failed_checks = [
                str(name)
                for name, check in dict(acceptance.get("checks") or {}).items()
                if not isinstance(check, Mapping) or check.get("passed") is not True
            ]
            blockers.extend(f"behavioral_{name}" for name in failed_checks)
            if not failed_checks:
                blockers.append("behavioral_acceptance")
        if not bool(reload_parity.get(target, False)):
            blockers.append("serialization_reload_parity")
        results.append(
            {
                "target": target,
                "promote": not blockers,
                "blockers": blockers,
                "behavioral_acceptance_status": acceptance.get("status", "missing"),
            }
        )

    integrity_ok = bool(integrity_report.get("ok", False))
    if not integrity_ok:
        for result in results:
            result["promote"] = False
            result["blockers"].append("evidence_integrity")
    promoted = [result["target"] for result in results if result["promote"]]
    return {
        "schema_version": PROMOTION_SCHEMA,
        "decision_authority": "adapter_behavioral_acceptance",
        "overall_promote": len(promoted) == len(TARGET_ADAPTERS),
        "integrity_ok": integrity_ok,
        "promoted_target_count": len(promoted),
        "promoted_targets": promoted,
        "targets": results,
    }
