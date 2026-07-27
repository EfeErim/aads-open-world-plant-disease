from src.training.services.ood_score_selection import (
    apply_primary_score_method_to_evaluation,
    build_real_ood_dev_selection,
    normalize_requested_primary_score_method,
    select_best_ood_score_method,
    select_threshold_at_target_fpr,
)
from src.training.types import EvaluationArtifactsPayload, ValidationReport


def _evaluation_payload() -> EvaluationArtifactsPayload:
    return EvaluationArtifactsPayload(
        report=ValidationReport.from_dict({}),
        y_true=[0, 1],
        y_pred=[0, 1],
        ood_labels=[0, 1],
        ood_scores=[0.1, 0.9],
        ood_primary_score_method="ensemble",
        ood_scores_by_method={
            "ensemble": [0.4, 0.6],
            "energy": [0.1, 0.9],
            "knn": [0.2, 0.8],
        },
        context={},
    )


def test_normalize_requested_primary_score_method_accepts_auto():
    assert normalize_requested_primary_score_method("AUTO") == "auto"


def test_select_best_ood_score_method_prefers_gate_eligible_method():
    selected = select_best_ood_score_method(
        {
            "methods": {
                "ensemble": {
                    "pooled_metrics": {"ood_auroc": 0.98, "ood_false_positive_rate": 0.08},
                    "pooled_gate_eligible": False,
                    "worst_slice": {"metrics": {"ood_false_positive_rate": 0.08, "ood_auroc": 0.98}},
                },
                "energy": {
                    "pooled_metrics": {"ood_auroc": 0.93, "ood_false_positive_rate": 0.05},
                    "pooled_gate_eligible": True,
                    "worst_slice": {"metrics": {"ood_false_positive_rate": 0.05, "ood_auroc": 0.93}},
                },
            }
        }
    )

    assert selected == "energy"


def test_select_best_ood_score_method_prefers_lower_worst_slice_fpr_before_pooled_auroc():
    selected = select_best_ood_score_method(
        {
            "methods": {
                "ensemble": {
                    "pooled_metrics": {"ood_auroc": 0.98, "ood_false_positive_rate": 0.03},
                    "pooled_gate_eligible": True,
                    "worst_slice": {"metrics": {"ood_false_positive_rate": 0.22, "ood_auroc": 0.92}},
                },
                "energy": {
                    "pooled_metrics": {"ood_auroc": 0.95, "ood_false_positive_rate": 0.05},
                    "pooled_gate_eligible": True,
                    "worst_slice": {"metrics": {"ood_false_positive_rate": 0.07, "ood_auroc": 0.89}},
                },
            }
        }
    )

    assert selected == "energy"


def test_select_best_ood_score_method_prefers_lower_worst_fold_fpr_before_pooled_auroc():
    selected = select_best_ood_score_method(
        {
            "methods": {
                "ensemble": {
                    "pooled_metrics": {"ood_auroc": 0.99, "ood_false_positive_rate": 0.02},
                    "pooled_gate_eligible": True,
                    "worst_fold": {"metrics": {"ood_false_positive_rate": 0.18, "ood_auroc": 0.91}},
                },
                "knn": {
                    "pooled_metrics": {"ood_auroc": 0.94, "ood_false_positive_rate": 0.04},
                    "pooled_gate_eligible": True,
                    "worst_fold": {"metrics": {"ood_false_positive_rate": 0.06, "ood_auroc": 0.88}},
                },
            }
        }
    )

    assert selected == "knn"


def test_apply_primary_score_method_to_evaluation_rewrites_selected_scores():
    rewritten = apply_primary_score_method_to_evaluation(
        _evaluation_payload(),
        "energy",
        requested_primary_score_method="auto",
        selection_source="real_ood_split",
    )

    assert rewritten is not None
    assert rewritten.ood_primary_score_method == "energy"
    assert rewritten.ood_scores == [0.1, 0.9]
    assert rewritten.context["ood_requested_primary_score_method"] == "auto"
    assert rewritten.context["ood_primary_score_selection_source"] == "real_ood_split"


def test_select_threshold_at_target_fpr_uses_id_score_quantile():
    selected = select_threshold_at_target_fpr(
        ood_labels=[0, 0, 0, 0, 1, 1],
        ood_scores=[0.1, 0.2, 0.3, 0.4, 0.35, 0.9],
        target_fpr=0.25,
    )

    assert selected["threshold"] == 0.3
    assert selected["false_positive_rate"] == 0.25
    assert selected["true_positive_rate"] == 1.0


def test_build_real_ood_dev_selection_returns_method_and_threshold():
    selection = build_real_ood_dev_selection(
        _evaluation_payload(),
        fallback="ensemble",
        target_fpr=0.0,
    )

    assert selection["selection_source"] == "real_ood_dev"
    assert selection["selected_primary_score_method"] == "ensemble"
    assert selection["selected_threshold"] == 0.4
    assert selection["target_fpr"] == 0.0
    assert "energy" in selection["method_metrics"]
