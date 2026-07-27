"""Deterministic metric helpers for training workflows and notebook artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve

from src.shared.json_utils import read_json, write_json

logger = logging.getLogger(__name__)

DEFAULT_PLAN_TARGETS = {
    "accuracy": 0.93,
    "balanced_accuracy": 0.90,
    "macro_f1": 0.90,
    "ood_auroc": 0.92,
    "ood_false_positive_rate": 0.05,
    "ood_samples": 30.0,
    "in_distribution_samples": 30.0,
    "ood_samples_per_type": 5.0,
    "sure_ds_f1": 0.90,
    "conformal_empirical_coverage": 0.95,
    "conformal_avg_set_size": 2.0,
}

HARD_OOD_METRIC_NAMES = (
    "ood_auroc",
    "ood_false_positive_rate",
    "ood_samples",
    "in_distribution_samples",
)

AUXILIARY_OOD_METRIC_NAMES = (
    "sure_ds_f1",
    "conformal_empirical_coverage",
    "conformal_avg_set_size",
)

OOD_METRIC_NAMES = (
    *HARD_OOD_METRIC_NAMES,
    *AUXILIARY_OOD_METRIC_NAMES,
)

_TARGET_FALLBACK_KEYS = {
    "accuracy": ("accuracy", "continual_accuracy"),
    "balanced_accuracy": ("balanced_accuracy",),
    "macro_f1": ("macro_f1",),
    "ood_auroc": ("ood_auroc",),
    "ood_false_positive_rate": ("ood_false_positive_rate",),
    "ood_samples": ("ood_samples", "ood_min_samples", "min_ood_samples"),
    "ood_samples_per_type": ("ood_samples_per_type", "min_ood_samples_per_type"),
    "in_distribution_samples": (
        "in_distribution_samples",
        "in_distribution_min_samples",
        "min_in_distribution_samples",
    ),
    "sure_ds_f1": ("sure_ds_f1",),
    "conformal_empirical_coverage": ("conformal_empirical_coverage",),
    "conformal_avg_set_size": ("conformal_avg_set_size", "conformal_max_avg_set_size"),
}


def load_plan_targets(spec_path: Optional[Path] = None) -> Dict[str, float]:
    if spec_path is None:
        return dict(DEFAULT_PLAN_TARGETS)

    resolved = Path(spec_path)
    if not resolved.exists():
        return dict(DEFAULT_PLAN_TARGETS)

    payload = read_json(resolved, default={}, expect_type=dict)
    targets = payload.get("targets", {}) if isinstance(payload, dict) else {}
    return _resolve_target_values(targets)


def _resolve_target_value(targets: Dict[str, Any], metric_name: str) -> float:
    for key in _TARGET_FALLBACK_KEYS.get(metric_name, (metric_name,)):
        if key in targets:
            return float(targets[key])
    return float(DEFAULT_PLAN_TARGETS[metric_name])


def _resolve_target_values(targets: Optional[Dict[str, Any]]) -> Dict[str, float]:
    raw_targets = dict(targets or {})
    return {
        metric_name: float(_resolve_target_value(raw_targets, metric_name))
        for metric_name in DEFAULT_PLAN_TARGETS
    }


def _build_threshold_checks(
    metrics: Dict[str, Optional[float]],
    target_values: Dict[str, float],
    definitions: Sequence[tuple[str, str]],
) -> Dict[str, Dict[str, Any]]:
    return {
        metric_name: _build_check(
            metrics.get(metric_name),
            float(target_values[metric_name]),
            operator=operator,
        )
        for metric_name, operator in definitions
    }


def _resolve_ood_metric_payload(ood_metrics: Optional[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    resolved_ood_metrics: Dict[str, Optional[float]] = {name: None for name in OOD_METRIC_NAMES}
    for name, value in dict(ood_metrics or {}).items():
        if name in resolved_ood_metrics:
            resolved_ood_metrics[name] = None if value is None else float(value)
    return resolved_ood_metrics


def _collect_missing_requirements(
    classification_checks: Dict[str, Dict[str, Any]],
    ood_checks: Dict[str, Dict[str, Any]],
    ood_type_sample_checks: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    require_ood: bool,
) -> list[str]:
    missing_requirements: list[str] = []
    for metric_name, detail in classification_checks.items():
        if not detail.get("asserted", False) or not detail.get("passed", False):
            missing_requirements.append(metric_name)
    for metric_name, detail in ood_checks.items():
        asserted = bool(detail.get("asserted", False))
        passed = bool(detail.get("passed", False))
        if require_ood:
            if not asserted or not passed:
                missing_requirements.append(metric_name)
            continue
        if asserted and not passed:
            missing_requirements.append(metric_name)
    for ood_type, detail in dict(ood_type_sample_checks or {}).items():
        asserted = bool(detail.get("asserted", False))
        passed = bool(detail.get("passed", False))
        if require_ood:
            if not asserted or not passed:
                missing_requirements.append(f"ood_samples_per_type:{ood_type}")
            continue
        if asserted and not passed:
            missing_requirements.append(f"ood_samples_per_type:{ood_type}")
    return missing_requirements


def _coerce_ood_type_breakdown(context: Dict[str, Any]) -> Dict[str, Any]:
    direct = context.get("ood_type_breakdown")
    if isinstance(direct, dict):
        return dict(direct)
    comparison = context.get("ood_method_comparison")
    if isinstance(comparison, dict) and isinstance(comparison.get("ood_type_breakdown"), dict):
        return dict(comparison.get("ood_type_breakdown", {}))
    return {}


def _build_ood_type_sample_checks(
    *,
    ood_type_breakdown: Dict[str, Any],
    target: float,
    require_real_ood_types: bool,
) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}
    for ood_type, payload in sorted(dict(ood_type_breakdown or {}).items()):
        row = dict(payload or {}) if isinstance(payload, dict) else {}
        sample_count = row.get("sample_count", row.get("ood_samples"))
        try:
            value = None if sample_count is None else float(sample_count)
        except (TypeError, ValueError):
            value = None
        checks[str(ood_type)] = _build_check(value, float(target), operator=">=")
    if not checks and require_real_ood_types:
        checks["untyped"] = _build_check(None, float(target), operator=">=")
    return checks


def compute_ood_detection_metrics(
    *,
    ood_labels: Optional[Sequence[int]] = None,
    ood_scores: Optional[Sequence[float]] = None,
) -> Dict[str, Optional[float | int]]:
    ood_auroc: Optional[float] = None
    ood_fpr: Optional[float] = None
    ood_total: Optional[int] = None
    in_dist_total: Optional[int] = None
    if ood_labels is not None and ood_scores is not None:
        if len(ood_labels) != len(ood_scores):
            raise ValueError("ood_labels and ood_scores must have same length")
        if len(ood_labels) > 0:
            labels_t = torch.tensor(list(ood_labels), dtype=torch.long)
            scores_t = torch.tensor(list(ood_scores), dtype=torch.float32)
            ood_total = int((labels_t == 1).sum().item())
            in_dist_total = int((labels_t == 0).sum().item())
            if ood_total > 0 and in_dist_total > 0:
                try:
                    labels_np = labels_t.cpu().numpy()
                    scores_np = scores_t.cpu().numpy()
                    ood_auroc = float(roc_auc_score(labels_np, scores_np))
                    fpr, tpr, _ = roc_curve(labels_np, scores_np, pos_label=1)
                    valid_tpr = tpr >= 0.95
                    if valid_tpr.any():
                        ood_fpr = float(fpr[valid_tpr].min())
                except Exception as exc:
                    logger.warning(
                        "Failed to compute OOD detection metrics with ood_total=%s in_dist_total=%s: %s",
                        ood_total,
                        in_dist_total,
                        exc,
                    )
                    ood_auroc = None
                    ood_fpr = None
    return {
        "ood_auroc": ood_auroc,
        "ood_false_positive_rate": ood_fpr,
        "ood_samples": ood_total,
        "in_distribution_samples": in_dist_total,
    }


def compute_plan_metrics(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    ood_labels: Optional[Sequence[int]] = None,
    ood_scores: Optional[Sequence[float]] = None,
    sure_ds_f1: Optional[float] = None,
    conformal_empirical_coverage: Optional[float] = None,
    conformal_avg_set_size: Optional[float] = None,
) -> Dict[str, Optional[float | int]]:
    if len(y_true) == 0:
        raise ValueError("y_true must not be empty")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    y_true_t = torch.tensor(list(y_true), dtype=torch.long)
    y_pred_t = torch.tensor(list(y_pred), dtype=torch.long)
    y_true_list = y_true_t.cpu().tolist()
    y_pred_list = y_pred_t.cpu().tolist()
    accuracy = float((y_true_t == y_pred_t).float().mean().item())
    balanced_accuracy = float(balanced_accuracy_score(y_true_list, y_pred_list))
    macro_f1 = float(f1_score(y_true_list, y_pred_list, average="macro", zero_division=0))

    ood_metrics = compute_ood_detection_metrics(ood_labels=ood_labels, ood_scores=ood_scores)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "ood_auroc": ood_metrics["ood_auroc"],
        "ood_false_positive_rate": ood_metrics["ood_false_positive_rate"],
        "classification_samples": int(y_true_t.numel()),
        "ood_samples": (
            None if ood_metrics["ood_samples"] is None else int(ood_metrics["ood_samples"])
        ),
        "in_distribution_samples": (
            None
            if ood_metrics["in_distribution_samples"] is None
            else int(ood_metrics["in_distribution_samples"])
        ),
        "sure_ds_f1": sure_ds_f1,
        "conformal_empirical_coverage": conformal_empirical_coverage,
        "conformal_avg_set_size": conformal_avg_set_size,
    }


def _build_check(
    value: Optional[float],
    target: float,
    *,
    operator: str,
) -> Dict[str, Any]:
    if operator == ">=":
        passed = bool(value is not None and float(value) >= float(target))
    elif operator == "<=":
        passed = bool(value is not None and float(value) <= float(target))
    else:  # pragma: no cover - internal misuse guard
        raise ValueError(f"Unsupported operator: {operator}")
    return {
        "value": value,
        "target": float(target),
        "operator": operator,
        "asserted": value is not None,
        "passed": passed,
    }


def _finalize_validation(
    checks: Dict[str, Dict[str, Any]],
    *,
    require_metrics: bool,
) -> Dict[str, Any]:
    missing_checks = [name for name, detail in checks.items() if not detail["asserted"]]
    if require_metrics:
        gating_status = "failed" if missing_checks else "ready"
        gating_reason = "missing_required_metrics" if missing_checks else "all_required_metrics_present"
    else:
        gating_status = "soft" if missing_checks else "ready"
        gating_reason = "missing_optional_metrics" if missing_checks else "all_metrics_present"

    all_asserted_passed = all(detail["passed"] for detail in checks.values() if detail["asserted"])
    hard_fail = require_metrics and bool(missing_checks)
    passed = bool(all_asserted_passed and not hard_fail)

    return {
        "passed": passed,
        "checks": checks,
        "gating": {
            "status": gating_status,
            "reason": gating_reason,
            "missing_metrics": missing_checks,
        },
    }


def validate_ood_metrics(
    metrics: Dict[str, Optional[float]],
    targets: Optional[Dict[str, float]] = None,
    *,
    require_ood: bool = False,
    gate_auxiliary_ood_diagnostics: bool = False,
) -> Dict[str, Any]:
    target_values = _resolve_target_values(targets)
    hard_checks = _build_threshold_checks(
        metrics,
        target_values,
        (
            ("ood_auroc", ">="),
            ("ood_false_positive_rate", "<="),
            ("ood_samples", ">="),
            ("in_distribution_samples", ">="),
        ),
    )
    auxiliary_checks = _build_threshold_checks(
        metrics,
        target_values,
        (
            ("sure_ds_f1", ">="),
            ("conformal_empirical_coverage", ">="),
            ("conformal_avg_set_size", "<="),
        ),
    )
    checks = {**hard_checks, **auxiliary_checks}
    gating_checks = checks if bool(gate_auxiliary_ood_diagnostics) else hard_checks
    finalized = _finalize_validation(gating_checks, require_metrics=require_ood)
    return {
        "passed": finalized["passed"],
        "require_ood": bool(require_ood),
        "gate_auxiliary_ood_diagnostics": bool(gate_auxiliary_ood_diagnostics),
        "targets": target_values,
        "checks": checks,
        "hard_checks": hard_checks,
        "auxiliary_checks": auxiliary_checks,
        "gating": finalized["gating"],
    }


def validate_plan_metrics(
    metrics: Dict[str, Optional[float]],
    targets: Optional[Dict[str, float]] = None,
    *,
    require_ood: bool = False,
    gate_auxiliary_ood_diagnostics: bool = False,
) -> Dict[str, Any]:
    target_values = _resolve_target_values(targets)
    checks: Dict[str, Dict[str, Any]] = _build_threshold_checks(
        metrics,
        target_values,
        (
            ("accuracy", ">="),
            ("balanced_accuracy", ">="),
            ("macro_f1", ">="),
        ),
    )
    ood_validation = validate_ood_metrics(
        metrics,
        target_values,
        require_ood=require_ood,
        gate_auxiliary_ood_diagnostics=gate_auxiliary_ood_diagnostics,
    )
    checks.update(ood_validation["checks"])
    gating_checks = {
        key: value
        for key, value in checks.items()
        if key in {"accuracy", "balanced_accuracy", "macro_f1"}
    }
    gating_checks.update(ood_validation["hard_checks"])
    if bool(gate_auxiliary_ood_diagnostics):
        gating_checks.update(ood_validation["auxiliary_checks"])
    finalized = _finalize_validation(gating_checks, require_metrics=require_ood)
    return {
        "passed": finalized["passed"],
        "require_ood": bool(require_ood),
        "targets": target_values,
        "checks": checks,
        "gating": finalized["gating"],
    }


def build_production_readiness(
    *,
    classification_metric_gate: Optional[Dict[str, Any]],
    classification_split: str,
    ood_evidence_source: Optional[str],
    ood_metrics: Optional[Dict[str, Optional[float]]],
    targets: Optional[Dict[str, float]] = None,
    context: Optional[Dict[str, Any]] = None,
    require_ood: bool = True,
) -> Dict[str, Any]:
    target_values = _resolve_target_values(targets)
    classification_gate = dict(classification_metric_gate or {})
    metric_gate_context = (
        dict(classification_gate.get("context", {}))
        if isinstance(classification_gate.get("context"), dict)
        else {}
    )
    readiness_context = {**metric_gate_context, **dict(context or {})}
    dataset_lineage = (
        dict(readiness_context.get("dataset_lineage", {}))
        if isinstance(readiness_context.get("dataset_lineage"), dict)
        else {}
    )
    classification_metrics = dict(classification_gate.get("metrics", {}))
    classification_eval = dict(classification_gate.get("evaluation", {}))
    classification_checks = dict(classification_eval.get("checks", {}))
    resolved_classification_checks = {
        "accuracy": classification_checks.get(
            "accuracy",
            _build_check(classification_metrics.get("accuracy"), target_values["accuracy"], operator=">="),
        ),
        "balanced_accuracy": classification_checks.get(
            "balanced_accuracy",
            _build_check(
                classification_metrics.get("balanced_accuracy"),
                target_values["balanced_accuracy"],
                operator=">=",
            ),
        ),
        "macro_f1": classification_checks.get(
            "macro_f1",
            _build_check(classification_metrics.get("macro_f1"), target_values["macro_f1"], operator=">="),
        ),
    }

    resolved_ood_metrics = _resolve_ood_metric_payload(ood_metrics)
    gate_auxiliary = bool(
        readiness_context.get(
            "gate_auxiliary_ood_diagnostics",
            target_values.get("gate_auxiliary_ood_diagnostics", False),
        )
    )
    ood_validation = validate_ood_metrics(
        resolved_ood_metrics,
        target_values,
        require_ood=require_ood,
        gate_auxiliary_ood_diagnostics=gate_auxiliary,
    )
    ood_type_breakdown = _coerce_ood_type_breakdown(readiness_context)
    ood_type_sample_checks = _build_ood_type_sample_checks(
        ood_type_breakdown=ood_type_breakdown,
        target=target_values["ood_samples_per_type"],
        require_real_ood_types=bool(require_ood and str(ood_evidence_source or "") == "real_ood_split"),
    )

    missing_requirements = _collect_missing_requirements(
        resolved_classification_checks,
        {
            **ood_validation["hard_checks"],
            **(ood_validation["auxiliary_checks"] if gate_auxiliary else {}),
        },
        ood_type_sample_checks,
        require_ood=require_ood,
    )
    policy_passed = not missing_requirements
    real_ood_evidence = str(ood_evidence_source or "unavailable") == "real_ood_split"
    deployable = bool(policy_passed and real_ood_evidence)
    missing_deployment_requirements: list[str] = []
    if policy_passed and not real_ood_evidence:
        missing_deployment_requirements.append("real_ood_evidence")
    if bool(readiness_context.get("require_immutable_dataset_lineage", False)):
        from src.data.dataset_lineage import immutable_lineage_blockers

        missing_deployment_requirements.extend(immutable_lineage_blockers(dataset_lineage))
        deployable = bool(deployable and not missing_deployment_requirements)
    if deployable:
        status = "ready"
    elif policy_passed:
        status = "provisional"
    else:
        status = "failed"
    return {
        "status": status,
        "readiness_tier": status,
        "passed": bool(deployable),
        "policy_passed": bool(policy_passed),
        "deployable": bool(deployable),
        "ood_evidence_source": str(ood_evidence_source or "unavailable"),
        "classification_evidence": {
            "split_name": str(classification_split),
            "metrics": classification_metrics,
            "evaluation": {
                "checks": resolved_classification_checks,
            },
        },
        "ood_evidence": {
            "source": str(ood_evidence_source or "unavailable"),
            "metrics": resolved_ood_metrics,
            "evaluation": ood_validation,
        },
        "auxiliary_checks": ood_validation["auxiliary_checks"],
        "ood_type_sample_checks": ood_type_sample_checks,
        "missing_requirements": missing_requirements,
        "missing_deployment_requirements": missing_deployment_requirements,
        "dataset_lineage": dataset_lineage,
        "targets": target_values,
        "context": readiness_context,
    }


def write_plan_metric_artifact(
    *,
    output_path: Path,
    metrics: Dict[str, Optional[float]],
    targets: Optional[Dict[str, float]] = None,
    require_ood: bool = False,
    gate_auxiliary_ood_diagnostics: bool = False,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    artifact = {
        "schema_version": "v6_plan_metric_gate",
        "metrics": metrics,
        "evaluation": validate_plan_metrics(
            metrics,
            targets,
            require_ood=require_ood,
            gate_auxiliary_ood_diagnostics=gate_auxiliary_ood_diagnostics,
        ),
        "context": dict(context or {}),
    }
    write_json(output_path, artifact)
    return artifact
