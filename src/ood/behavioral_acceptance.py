"""Behavioral acceptance contract for crop/part adapter training runs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

BEHAVIORAL_ACCEPTANCE_SCHEMA = "v1_adapter_behavioral_acceptance"
BEHAVIORAL_DEV_REPORT_SCHEMA = "v1_adapter_behavioral_dev_report"
SAME_CROP_UNSUPPORTED_DISEASE = "same_crop_unsupported_disease"

DEFAULT_BEHAVIORAL_THRESHOLDS: dict[str, tuple[str, float]] = {
    "accuracy": (">=", 0.93),
    "balanced_accuracy": (">=", 0.90),
    "macro_f1": (">=", 0.90),
    "id_test_samples": (">=", 30.0),
    "id_false_rejection_rate": ("<=", 0.05),
    "same_crop_ood_test_samples": (">=", 30.0),
    "forced_supported_answer_count": ("<=", 0.0),
    "same_crop_ood_rejection_rate": (">=", 1.0),
}
REQUIRED_BEHAVIORAL_CHECKS = frozenset(
    {
        *DEFAULT_BEHAVIORAL_THRESHOLDS,
        "id_decisions_complete",
        "same_crop_ood_decisions_complete",
    }
)


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _passes(value: float | None, operator: str, threshold: float) -> bool:
    if value is None:
        return False
    return value >= threshold if operator == ">=" else value <= threshold


def _disease_id(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("disease_id") or "").strip()
    if explicit:
        return explicit
    image_path = str(row.get("image_path") or "").strip()
    return Path(image_path).parent.name if image_path else "unlabeled"


def _family_id(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("evidence_family_id") or "").strip()
    if explicit:
        return explicit
    image_path = str(row.get("image_path") or "").strip()
    return image_path or f"missing-family:{id(row)}"


def build_behavioral_acceptance(
    *,
    classification_metrics: Mapping[str, Any] | None,
    prediction_rows: Sequence[Mapping[str, Any]] | None,
    context: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, tuple[str, float]] | None = None,
) -> dict[str, Any]:
    """Evaluate the two default adapter behaviors from held-out per-sample decisions."""

    rows = [dict(row) for row in list(prediction_rows or [])]
    id_rows = [row for row in rows if str(row.get("sample_origin") or "") == "in_distribution"]
    same_crop_ood_rows = [
        row
        for row in rows
        if str(row.get("sample_origin") or "") == "ood"
        and str(row.get("ood_type") or "") == SAME_CROP_UNSUPPORTED_DISEASE
    ]

    id_rows_with_decision = [row for row in id_rows if row.get("ood_predicted") is not None]
    id_false_rejections = sum(bool(row.get("ood_predicted")) for row in id_rows_with_decision)
    id_accepted_correct = sum(
        not bool(row.get("ood_predicted")) and bool(row.get("is_correct")) for row in id_rows_with_decision
    )
    ood_rows_with_decision = [row for row in same_crop_ood_rows if row.get("ood_predicted") is not None]
    ood_rejections = sum(bool(row.get("ood_predicted")) for row in ood_rows_with_decision)
    forced_answers = sum(not bool(row.get("ood_predicted")) for row in ood_rows_with_decision)

    id_family_count = len({_family_id(row) for row in id_rows})
    same_crop_ood_family_count = len({_family_id(row) for row in same_crop_ood_rows})
    metrics = {
        "accuracy": _number(dict(classification_metrics or {}).get("accuracy")),
        "balanced_accuracy": _number(dict(classification_metrics or {}).get("balanced_accuracy")),
        "macro_f1": _number(dict(classification_metrics or {}).get("macro_f1")),
        "id_test_samples": float(id_family_count),
        "id_accepted_correct_count": int(id_accepted_correct),
        "id_false_rejection_count": int(id_false_rejections),
        "id_missing_decision_count": int(len(id_rows) - len(id_rows_with_decision)),
        "id_false_rejection_rate": (
            None if not id_rows_with_decision else float(id_false_rejections) / float(len(id_rows_with_decision))
        ),
        "same_crop_ood_test_samples": float(same_crop_ood_family_count),
        "same_crop_ood_rejected_count": int(ood_rejections),
        "forced_supported_answer_count": int(forced_answers),
        "same_crop_ood_missing_decision_count": int(len(same_crop_ood_rows) - len(ood_rows_with_decision)),
        "same_crop_ood_rejection_rate": (
            None if not ood_rows_with_decision else float(ood_rejections) / float(len(ood_rows_with_decision))
        ),
    }
    resolved_thresholds = dict(thresholds or DEFAULT_BEHAVIORAL_THRESHOLDS)
    checks = {
        name: {
            "value": metrics.get(name),
            "operator": operator,
            "threshold": threshold,
            "passed": _passes(_number(metrics.get(name)), operator, threshold),
        }
        for name, (operator, threshold) in resolved_thresholds.items()
    }
    checks["id_decisions_complete"] = {
        "value": metrics["id_missing_decision_count"],
        "operator": "==",
        "threshold": 0,
        "passed": metrics["id_missing_decision_count"] == 0,
    }
    checks["same_crop_ood_decisions_complete"] = {
        "value": metrics["same_crop_ood_missing_decision_count"],
        "operator": "==",
        "threshold": 0,
        "passed": metrics["same_crop_ood_missing_decision_count"] == 0,
    }
    failure_reasons = [name for name, check in checks.items() if not bool(check["passed"])]
    disease_counts: dict[str, Counter[str]] = {}
    for row in same_crop_ood_rows:
        disease = _disease_id(row)
        counts = disease_counts.setdefault(disease, Counter())
        counts["samples"] += 1
        if row.get("ood_predicted") is None:
            counts["missing_decision"] += 1
        elif bool(row.get("ood_predicted")):
            counts["rejected"] += 1
        else:
            counts["forced_supported_answer"] += 1

    passed = not failure_reasons
    return {
        "schema_version": BEHAVIORAL_ACCEPTANCE_SCHEMA,
        "decision_authority": "adapter_behavioral_acceptance",
        "status": "pass" if passed else "fail",
        "pass": passed,
        "metrics": metrics,
        "checks": checks,
        "failure_reasons": failure_reasons,
        "same_crop_ood_per_disease": {
            disease: dict(sorted(counts.items())) for disease, counts in sorted(disease_counts.items())
        },
        "context": dict(context or {}),
    }


def behavioral_acceptance_pass(payload: Mapping[str, Any] | None) -> bool:
    """Return the canonical campaign decision without consulting readiness status."""

    resolved = dict(payload or {})
    checks = dict(resolved.get("checks") or {})
    return (
        resolved.get("schema_version") == BEHAVIORAL_ACCEPTANCE_SCHEMA
        and resolved.get("decision_authority") == "adapter_behavioral_acceptance"
        and resolved.get("status") == "pass"
        and resolved.get("pass") is True
        and REQUIRED_BEHAVIORAL_CHECKS.issubset(checks)
        and all(isinstance(checks[name], Mapping) and checks[name].get("passed") is True for name in REQUIRED_BEHAVIORAL_CHECKS)
    )


def build_behavioral_dev_report(
    *,
    classification_metrics: Mapping[str, Any] | None,
    prediction_rows: Sequence[Mapping[str, Any]] | None,
    context: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, tuple[str, float]] | None = None,
) -> dict[str, Any]:
    """Build a non-authoritative selection report from ID validation and OOD-dev only."""

    payload = build_behavioral_acceptance(
        classification_metrics=classification_metrics,
        prediction_rows=prediction_rows,
        context={
            **dict(context or {}),
            "classification_split": "validation",
            "ood_split": "ood_dev",
            "locked_test_evidence_loaded": False,
        },
        thresholds=thresholds,
    )
    payload.update(
        {
            "schema_version": BEHAVIORAL_DEV_REPORT_SCHEMA,
            "decision_authority": "adapter_behavioral_dev_selection",
            "authoritative": False,
            "evaluation_scope": "id_validation_and_ood_dev",
        }
    )
    payload["metrics"] = {
        **dict(payload["metrics"]),
        "id_validation_families": payload["metrics"]["id_test_samples"],
        "same_crop_ood_dev_families": payload["metrics"]["same_crop_ood_test_samples"],
    }
    return payload


def behavioral_dev_report_pass(payload: Mapping[str, Any] | None) -> bool:
    resolved = dict(payload or {})
    checks = dict(resolved.get("checks") or {})
    return (
        resolved.get("schema_version") == BEHAVIORAL_DEV_REPORT_SCHEMA
        and resolved.get("decision_authority") == "adapter_behavioral_dev_selection"
        and resolved.get("authoritative") is False
        and resolved.get("evaluation_scope") == "id_validation_and_ood_dev"
        and resolved.get("pass") is True
        and REQUIRED_BEHAVIORAL_CHECKS.issubset(checks)
        and all(checks[name].get("passed") is True for name in REQUIRED_BEHAVIORAL_CHECKS)
    )
