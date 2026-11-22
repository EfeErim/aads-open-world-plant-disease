import numpy as np
import pytest

from scripts.posthoc_linear_probe import (
    evaluate_adapter_rows,
    evaluate_probe,
    fit_linear_probe,
    knn_distance,
    ood_auroc,
    ood_scores_from_logits,
    selective_risk_metrics,
)


def test_ood_auroc_counts_ties_as_half():
    assert ood_auroc([0.0, 1.0], [2.0, 3.0]) == 1.0
    assert ood_auroc([2.0, 3.0], [0.0, 1.0]) == 0.0
    assert ood_auroc([1.0], [1.0]) == 0.5
    assert ood_auroc([], [1.0]) is None


def test_selective_risk_metrics_match_hand_computation():
    # Accept order by confidence: correct, wrong, correct, wrong.
    result = selective_risk_metrics([0.9, 0.8, 0.7, 0.6], [True, False, True, False])
    # Selective risk at coverage 1..4: 0, 1/2, 1/3, 2/4. Generalized risk: 0, 1/4, 1/4, 2/4.
    assert result["aurc"] == pytest.approx((0 + 1 / 2 + 1 / 3 + 2 / 4) / 4)
    assert result["augrc"] == pytest.approx((0 + 1 / 4 + 1 / 4 + 2 / 4) / 4)
    assert result["full_coverage_risk"] == pytest.approx(0.5)


def test_confident_errors_cost_more_augrc_than_unconfident_errors():
    correct = [True, True, False, False]
    good_ranking = selective_risk_metrics([0.9, 0.8, 0.2, 0.1], correct)
    bad_ranking = selective_risk_metrics([0.1, 0.2, 0.8, 0.9], correct)
    assert good_ranking["augrc"] < bad_ranking["augrc"]


def test_logit_scores_rank_flat_logits_as_more_ood():
    scores = ood_scores_from_logits(np.array([[8.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    for values in scores.values():
        assert values[1] > values[0]


def test_knn_distance_is_larger_away_from_the_training_bank():
    bank = np.array([[1.0, 0.0], [0.9, 0.1], [0.95, 0.05]])
    distances = knn_distance(np.array([[1.0, 0.02], [0.0, 1.0]]), bank, k=2)
    assert distances[1] > distances[0]


def test_probe_separates_clusters_and_flags_far_samples_as_ood():
    rng = np.random.default_rng(0)
    centers = np.array([[6.0, 0.0, 0.0, 0.0], [0.0, 6.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0]])
    train_y = np.repeat(np.arange(3), 30)
    train_x = centers[train_y] + rng.normal(size=(90, 4))
    test_y = np.repeat(np.arange(3), 10)
    test_x = centers[test_y] + rng.normal(size=(30, 4))
    ood_x = np.array([0.0, 0.0, 0.0, 12.0]) + rng.normal(size=(20, 4))

    result = evaluate_probe(
        fit_linear_probe(train_x, train_y),
        train_features=train_x,
        test_features=test_x,
        test_labels=test_y,
        ood_features=ood_x,
        knn_k=5,
    )

    assert result["classification"]["accuracy"] == 1.0
    assert result["ood_auroc"]["knn"] == 1.0
    assert len(result["rows"]["test"]) == 30
    assert len(result["rows"]["ood"]) == 20


def test_adapter_rows_use_primary_score_for_ood():
    test_rows = [
        {"label": 0, "prediction": 0, "confidence": 0.9, "primary_score": 0.1},
        {"label": 1, "prediction": 0, "confidence": 0.6, "primary_score": 0.4},
    ]
    ood_rows = [{"label": -1, "prediction": 0, "confidence": 0.5, "primary_score": 0.9}]

    result = evaluate_adapter_rows(test_rows, ood_rows)

    assert result["classification"]["accuracy"] == 0.5
    assert result["ood_auroc"]["primary_score"] == 1.0
    assert result["selective"]["augrc"] == pytest.approx((0 + 1 / 2) / 2)
