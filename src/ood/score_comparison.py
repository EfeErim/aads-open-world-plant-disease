"""Per-adapter OOD-dev score comparison and threshold-freeze reporting."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from sklearn.metrics import roc_auc_score

from src.ood.recovery import TARGET_ADAPTERS
from src.training.services.ood_score_selection import (
    SUPPORTED_CONCRETE_OOD_SCORE_METHODS,
    select_best_ood_score_method,
    select_threshold_at_target_fpr,
)

SCORE_COMPARISON_SCHEMA = "v1_adapter_ood_score_comparison"


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _method_report(rows: list[dict[str, Any]], method: str, *, target_fpr: float) -> dict[str, Any]:
    labels = [int(row["ood_label"]) for row in rows]
    scores = [float(row["scores"][method]) for row in rows]
    threshold_payload = select_threshold_at_target_fpr(
        ood_labels=labels,
        ood_scores=scores,
        target_fpr=target_fpr,
    )
    threshold = threshold_payload["threshold"]
    auroc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
    per_type: dict[str, dict[str, Any]] = {}
    if threshold is not None:
        for ood_type in sorted({str(row.get("ood_type") or "unlabeled") for row in rows if int(row["ood_label"]) == 1}):
            selected = [
                (float(row["scores"][method]), int(row["ood_label"]))
                for row in rows
                if int(row["ood_label"]) == 0 or str(row.get("ood_type") or "unlabeled") == ood_type
            ]
            id_scores = [score for score, label in selected if label == 0]
            ood_scores = [score for score, label in selected if label == 1]
            per_type[ood_type] = {
                "id_samples": len(id_scores),
                "ood_samples": len(ood_scores),
                "false_positive_rate": _rate(sum(score > threshold for score in id_scores), len(id_scores)),
                "true_positive_rate": _rate(sum(score > threshold for score in ood_scores), len(ood_scores)),
            }
    worst_slice_name = ""
    worst_slice: dict[str, Any] = {}
    if per_type:
        worst_slice_name, worst_slice = min(
            per_type.items(),
            key=lambda item: (
                float("inf") if item[1]["true_positive_rate"] is None else float(item[1]["true_positive_rate"]),
                item[0],
            ),
        )

    accepted = [
        row
        for row in rows
        if int(row["ood_label"]) == 0 and threshold is not None and float(row["scores"][method]) <= threshold
    ]
    correctness = [bool(row["classification_correct"]) for row in accepted if row.get("classification_correct") is not None]
    id_count = sum(int(row["ood_label"]) == 0 for row in rows)
    return {
        "score_direction": "higher_is_more_ood",
        "threshold_semantics": "is_ood = score > threshold",
        "selected_threshold": threshold,
        "target_fpr": float(target_fpr),
        "pooled_metrics": {
            "ood_auroc": auroc,
            "ood_false_positive_rate": threshold_payload["false_positive_rate"],
            "ood_true_positive_rate": threshold_payload["true_positive_rate"],
            "in_distribution_samples": threshold_payload["in_distribution_samples"],
            "ood_samples": threshold_payload["ood_samples"],
        },
        "per_ood_type": per_type,
        "worst_slice": {"name": worst_slice_name, "metrics": worst_slice},
        "risk_coverage": {
            "accepted_id_samples": len(accepted),
            "id_coverage": _rate(len(accepted), id_count),
            "accepted_classification_risk": (
                _rate(sum(not value for value in correctness), len(correctness)) if correctness else None
            ),
            "classification_correctness_samples": len(correctness),
        },
    }


def build_score_comparison(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_fpr: float = 0.05,
) -> dict[str, Any]:
    """Compare supported scores using OOD-dev only and freeze one method per target."""
    normalized = [dict(row) for row in rows]
    issues: list[dict[str, Any]] = []
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(normalized, start=1):
        target = str(row.get("target") or "").strip().lower()
        split_role = str(row.get("split_role") or "").strip().lower()
        scores = row.get("scores")
        if target not in TARGET_ADAPTERS:
            issues.append({"code": "invalid_target", "row": index, "target": target})
            continue
        if split_role != "ood_dev":
            issues.append(
                {
                    "code": "non_dev_selection_evidence",
                    "row": index,
                    "target": target,
                    "split_role": split_role,
                }
            )
            continue
        try:
            label = int(row.get("ood_label"))
        except (TypeError, ValueError):
            issues.append({"code": "invalid_ood_label", "row": index, "target": target})
            continue
        if label not in {0, 1}:
            issues.append({"code": "invalid_ood_label", "row": index, "target": target})
            continue
        if not isinstance(scores, Mapping):
            issues.append({"code": "scores_missing", "row": index, "target": target})
            continue
        missing_methods = [method for method in SUPPORTED_CONCRETE_OOD_SCORE_METHODS if scores.get(method) is None]
        if missing_methods:
            issues.append(
                {"code": "score_method_missing", "row": index, "target": target, "methods": missing_methods}
            )
            continue
        try:
            row["scores"] = {method: float(scores[method]) for method in SUPPORTED_CONCRETE_OOD_SCORE_METHODS}
        except (TypeError, ValueError):
            issues.append({"code": "invalid_score", "row": index, "target": target})
            continue
        row["ood_label"] = label
        row["target"] = target
        by_target[target].append(row)

    adapters: list[dict[str, Any]] = []
    for target in TARGET_ADAPTERS:
        target_rows = by_target[target]
        label_counts = Counter(int(row["ood_label"]) for row in target_rows)
        if not target_rows:
            issues.append({"code": "target_dev_evidence_missing", "target": target})
            adapters.append({"target": target, "status": "missing", "selected_method": "", "methods": {}})
            continue
        if not label_counts[0] or not label_counts[1]:
            issues.append(
                {
                    "code": "target_dev_label_class_missing",
                    "target": target,
                    "id_samples": label_counts[0],
                    "ood_samples": label_counts[1],
                }
            )
            adapters.append({"target": target, "status": "invalid", "selected_method": "", "methods": {}})
            continue
        methods = {
            method: _method_report(target_rows, method, target_fpr=target_fpr)
            for method in SUPPORTED_CONCRETE_OOD_SCORE_METHODS
        }
        selector_payload = {
            "methods": {
                method: {
                    "pooled_metrics": details["pooled_metrics"],
                    "worst_slice": {
                        "metrics": {
                            "ood_false_positive_rate": details["pooled_metrics"]["ood_false_positive_rate"],
                            "ood_auroc": details["pooled_metrics"]["ood_auroc"],
                            "ood_true_positive_rate": details["worst_slice"]["metrics"].get("true_positive_rate"),
                        }
                    },
                }
                for method, details in methods.items()
            }
        }
        selected = select_best_ood_score_method(selector_payload)
        adapters.append(
            {
                "target": target,
                "status": "selected",
                "selection_source": "ood_dev",
                "selected_method": selected,
                "selected_threshold": methods[selected]["selected_threshold"],
                "methods": methods,
            }
        )

    return {
        "schema_version": SCORE_COMPARISON_SCHEMA,
        "ok": not issues,
        "selection_evidence_role": "ood_dev",
        "final_ood_test_used_for_selection": False,
        "target_fpr": float(target_fpr),
        "selected_target_count": sum(adapter["status"] == "selected" for adapter in adapters),
        "adapters": adapters,
        "issues": issues,
    }
