"""Helpers for requested-vs-selected OOD score methods."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Sequence

from src.training.services.metrics import compute_plan_metrics, load_plan_targets, validate_ood_metrics
from src.training.types import EvaluationArtifactsPayload

AUTO_PRIMARY_SCORE_METHOD = "auto"
SUPPORTED_CONCRETE_OOD_SCORE_METHODS = ("ensemble", "energy", "knn")
SUPPORTED_REQUESTED_OOD_SCORE_METHODS = (AUTO_PRIMARY_SCORE_METHOD, *SUPPORTED_CONCRETE_OOD_SCORE_METHODS)
_METHOD_PREFERENCE = {"ensemble": 0, "energy": 1, "knn": 2}


def normalize_requested_primary_score_method(value: Any) -> str:
    resolved = str(value or AUTO_PRIMARY_SCORE_METHOD).strip().lower()
    if resolved not in SUPPORTED_REQUESTED_OOD_SCORE_METHODS:
        raise ValueError(
            "ood.primary_score_method must be one of: "
            + ", ".join(SUPPORTED_REQUESTED_OOD_SCORE_METHODS)
            + "."
        )
    return resolved


def is_auto_primary_score_method(value: Any) -> bool:
    return normalize_requested_primary_score_method(value) == AUTO_PRIMARY_SCORE_METHOD


def resolve_runtime_primary_score_method(value: Any) -> str:
    requested = normalize_requested_primary_score_method(value)
    if requested == AUTO_PRIMARY_SCORE_METHOD:
        return "ensemble"
    return requested


def compute_method_metrics_from_evaluation(
    evaluation: Optional[EvaluationArtifactsPayload],
) -> Dict[str, Dict[str, Optional[float]]]:
    if evaluation is None or not evaluation.ood_labels:
        return {}
    method_metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for method_name, scores in dict(evaluation.ood_scores_by_method or {}).items():
        score_values = list(scores)
        if not score_values:
            continue
        method_metrics[str(method_name)] = compute_plan_metrics(
            y_true=evaluation.y_true,
            y_pred=evaluation.y_pred,
            ood_labels=evaluation.ood_labels,
            ood_scores=score_values,
            sure_ds_f1=evaluation.sure_ds_f1,
            conformal_empirical_coverage=evaluation.conformal_empirical_coverage,
            conformal_avg_set_size=evaluation.conformal_avg_set_size,
        )
    return method_metrics


def select_threshold_at_target_fpr(
    *,
    ood_labels: Sequence[int],
    ood_scores: Sequence[float],
    target_fpr: float,
) -> Dict[str, Any]:
    labels = [int(label) for label in list(ood_labels)]
    scores = [float(score) for score in list(ood_scores)]
    if len(labels) != len(scores):
        raise ValueError("ood_labels and ood_scores must have same length")
    id_scores = sorted(score for label, score in zip(labels, scores) if int(label) == 0)
    ood_only_scores = [score for label, score in zip(labels, scores) if int(label) == 1]
    if not id_scores or not ood_only_scores:
        return {
            "threshold": None,
            "target_fpr": float(target_fpr),
            "false_positive_rate": None,
            "true_positive_rate": None,
            "in_distribution_samples": len(id_scores),
            "ood_samples": len(ood_only_scores),
        }
    clipped_target = max(0.0, min(1.0, float(target_fpr)))
    rank = int(math.ceil((1.0 - clipped_target) * len(id_scores)) - 1)
    rank = max(0, min(len(id_scores) - 1, rank))
    threshold = float(id_scores[rank])
    false_positive_count = sum(1 for score in id_scores if score > threshold)
    true_positive_count = sum(1 for score in ood_only_scores if score > threshold)
    return {
        "threshold": threshold,
        "target_fpr": clipped_target,
        "false_positive_rate": float(false_positive_count) / float(len(id_scores)),
        "true_positive_rate": float(true_positive_count) / float(len(ood_only_scores)),
        "in_distribution_samples": len(id_scores),
        "ood_samples": len(ood_only_scores),
    }


def build_real_ood_dev_selection(
    evaluation: Optional[EvaluationArtifactsPayload],
    *,
    fallback: str = "ensemble",
    target_fpr: float = 0.05,
) -> Dict[str, Any]:
    if evaluation is None or not evaluation.ood_labels or not evaluation.ood_scores_by_method:
        return {}
    method_metrics = compute_method_metrics_from_evaluation(evaluation)
    selected_method = select_best_ood_score_method(method_metrics, fallback=fallback)
    method_scores = list(dict(evaluation.ood_scores_by_method or {}).get(selected_method, []) or [])
    threshold_payload = select_threshold_at_target_fpr(
        ood_labels=evaluation.ood_labels,
        ood_scores=method_scores,
        target_fpr=float(target_fpr),
    )
    return {
        "selection_source": "real_ood_dev",
        "selected_primary_score_method": selected_method,
        "selected_threshold": threshold_payload.get("threshold"),
        "target_fpr": float(threshold_payload.get("target_fpr", target_fpr)),
        "dev_metrics": dict(method_metrics.get(selected_method, {})),
        "threshold_selection": threshold_payload,
        "method_metrics": method_metrics,
    }


def _coerce_metric_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_method_details(method_metrics: Mapping[str, Any], method_name: str) -> Dict[str, Any]:
    methods = _coerce_metric_mapping(method_metrics.get("methods"))
    if method_name in methods:
        details = _coerce_metric_mapping(methods.get(method_name))
        pooled_metrics = _coerce_metric_mapping(details.get("pooled_metrics"))
        if not pooled_metrics:
            pooled_metrics = {
                key: value
                for key, value in details.items()
                if key not in {"pooled_gate_eligible", "worst_slice", "worst_fold", "metric_std"}
            }
        return {
            "pooled_metrics": pooled_metrics,
            "pooled_gate_eligible": details.get("pooled_gate_eligible"),
            "worst_slice": _coerce_metric_mapping(details.get("worst_slice")),
            "worst_fold": _coerce_metric_mapping(details.get("worst_fold")),
        }
    details = _coerce_metric_mapping(method_metrics.get(method_name))
    return {
        "pooled_metrics": details,
        "pooled_gate_eligible": details.get("pooled_gate_eligible"),
        "worst_slice": _coerce_metric_mapping(details.get("worst_slice")),
        "worst_fold": _coerce_metric_mapping(details.get("worst_fold")),
    }


def _resolve_gate_eligibility(pooled_metrics: Dict[str, Any], explicit_value: Any) -> bool:
    if explicit_value is not None:
        return bool(explicit_value)
    if not pooled_metrics:
        return False
    return bool(validate_ood_metrics(pooled_metrics, load_plan_targets(), require_ood=True).get("passed", False))


def _resolve_robust_fpr(details: Dict[str, Any], pooled_metrics: Dict[str, Any]) -> Optional[float]:
    for key in ("worst_slice", "worst_fold"):
        robust_payload = _coerce_metric_mapping(details.get(key))
        metrics = _coerce_metric_mapping(robust_payload.get("metrics")) if robust_payload else {}
        candidate = _coerce_float(robust_payload.get("ood_false_positive_rate"))
        if candidate is None:
            candidate = _coerce_float(metrics.get("ood_false_positive_rate"))
        if candidate is not None:
            return candidate
    return _coerce_float(pooled_metrics.get("ood_false_positive_rate"))


def select_best_ood_score_method(
    method_metrics: Mapping[str, Mapping[str, Any]],
    *,
    fallback: str = "ensemble",
) -> str:
    resolved_fallback = resolve_runtime_primary_score_method(fallback)
    candidates = []
    for method_name in SUPPORTED_CONCRETE_OOD_SCORE_METHODS:
        details = _resolve_method_details(method_metrics, method_name)
        pooled_metrics = _coerce_metric_mapping(details.get("pooled_metrics"))
        pooled_auroc = _coerce_float(pooled_metrics.get("ood_auroc"))
        robust_fpr = _resolve_robust_fpr(details, pooled_metrics)
        if pooled_auroc is None and robust_fpr is None:
            continue
        gate_eligible = _resolve_gate_eligibility(pooled_metrics, details.get("pooled_gate_eligible"))
        candidates.append(
            (
                -int(bool(gate_eligible)),
                float("inf") if robust_fpr is None else float(robust_fpr),
                float("inf") if pooled_auroc is None else -float(pooled_auroc),
                _METHOD_PREFERENCE[method_name],
                method_name,
            )
        )
    if not candidates:
        return resolved_fallback
    candidates.sort()
    return str(candidates[0][4])


def apply_primary_score_method_to_evaluation(
    evaluation: Optional[EvaluationArtifactsPayload],
    primary_score_method: str,
    *,
    requested_primary_score_method: Optional[str] = None,
    selection_source: str = "",
) -> Optional[EvaluationArtifactsPayload]:
    if evaluation is None:
        return None
    resolved_method = resolve_runtime_primary_score_method(primary_score_method)
    selected_scores = list(
        dict(evaluation.ood_scores_by_method or {}).get(
            resolved_method,
            evaluation.ood_scores or [],
        )
    )
    updated_context = dict(evaluation.context)
    updated_context["ood_primary_score_method"] = resolved_method
    if requested_primary_score_method is not None:
        updated_context["ood_requested_primary_score_method"] = normalize_requested_primary_score_method(
            requested_primary_score_method
        )
    if selection_source:
        updated_context["ood_primary_score_selection_source"] = str(selection_source)
    return replace(
        evaluation,
        ood_scores=selected_scores if selected_scores else evaluation.ood_scores,
        ood_primary_score_method=resolved_method,
        context=updated_context,
    )
