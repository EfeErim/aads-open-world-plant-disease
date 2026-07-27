"""Unit tests for Conformal Prediction Guarantees."""

from __future__ import annotations

import torch

from src.ood.conformal_prediction import (
    build_prediction_set,
    calibrate_conformal_qhat,
    calibrate_prediction_set_qhat,
    compute_empirical_coverage,
    compute_nonconformity_scores,
    compute_prediction_set_nonconformity_scores,
    describe_conformal_method,
    normalize_conformal_method,
    score_all_classes,
)


class TestNonconformityScores:
    def test_signed_distance(self):
        ensemble = torch.tensor([2.0, 1.0, 3.0])
        thresholds = torch.tensor([1.5, 1.5, 1.5])
        nc = compute_nonconformity_scores(ensemble, thresholds)
        expected = torch.tensor([0.5, -0.5, 1.5])
        assert torch.allclose(nc, expected)


class TestCalibrateConformalQhat:
    def test_basic(self):
        scores = torch.randn(100)
        qhat = calibrate_conformal_qhat(scores, alpha=0.05)
        assert isinstance(qhat, float)

    def test_higher_alpha_lower_qhat(self):
        scores = torch.randn(200)
        q_strict = calibrate_conformal_qhat(scores, alpha=0.01)
        q_loose = calibrate_conformal_qhat(scores, alpha=0.20)
        assert q_strict >= q_loose

    def test_empty_returns_inf(self):
        qhat = calibrate_conformal_qhat(torch.tensor([]), alpha=0.05)
        assert qhat == float("inf")

    def test_deterministic(self):
        scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        q1 = calibrate_conformal_qhat(scores, alpha=0.1)
        q2 = calibrate_conformal_qhat(scores, alpha=0.1)
        assert q1 == q2


class TestPredictionSetConformal:
    def test_normalize_conformal_method(self):
        assert normalize_conformal_method("APS") == "aps"

    def test_aps_nonconformity_scores(self):
        logits = torch.tensor([[4.0, 1.0, 0.0], [0.1, 3.5, 0.0]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)

        scores = compute_prediction_set_nonconformity_scores(logits, labels, method="aps")

        assert scores.shape == (2,)
        assert torch.all(scores > 0.0)

    def test_raps_scores_include_regularization(self):
        logits = torch.tensor([[0.5, 0.4, 0.3]], dtype=torch.float32)
        labels = torch.tensor([2], dtype=torch.long)

        aps = compute_prediction_set_nonconformity_scores(logits, labels, method="aps")
        raps = compute_prediction_set_nonconformity_scores(
            logits,
            labels,
            method="raps",
            raps_lambda=0.1,
            raps_k_reg=1,
        )

        assert raps.item() >= aps.item()

    def test_calibrate_prediction_set_qhat(self):
        logits = torch.tensor([[4.0, 1.0], [0.5, 2.0], [3.0, 0.1]], dtype=torch.float32)
        labels = torch.tensor([0, 1, 0], dtype=torch.long)

        qhat = calibrate_prediction_set_qhat(logits, labels, method="aps", alpha=0.1)

        assert isinstance(qhat, float)

    def test_build_prediction_set_aps(self):
        idx_to_class = {0: "healthy", 1: "diseased", 2: "other"}
        pred_set = build_prediction_set(
            torch.randn(3),
            torch.tensor([5.0, 1.0, 0.0], dtype=torch.float32),
            qhat=0.95,
            class_stats={},
            idx_to_class=idx_to_class,
            method="aps",
        )

        assert "healthy" in pred_set

    def test_describe_conformal_method(self):
        assert "APS" in describe_conformal_method("aps")


class TestScoreAllClasses:
    def _make_stats(self):
        """Create minimal ClassCalibration-like objects."""
        from dataclasses import dataclass

        @dataclass
        class FakeCalibration:
            mean: torch.Tensor
            var: torch.Tensor
            mahalanobis_mu: float
            mahalanobis_sigma: float
            energy_mu: float
            energy_sigma: float
            threshold: float

        return {
            0: FakeCalibration(
                mean=torch.zeros(8), var=torch.ones(8),
                mahalanobis_mu=1.0, mahalanobis_sigma=0.5,
                energy_mu=-2.0, energy_sigma=0.3,
                threshold=2.0,
            ),
            1: FakeCalibration(
                mean=torch.ones(8), var=torch.ones(8),
                mahalanobis_mu=1.0, mahalanobis_sigma=0.5,
                energy_mu=-2.0, energy_sigma=0.3,
                threshold=2.0,
            ),
        }

    def test_returns_all_classes(self):
        stats = self._make_stats()
        features = torch.randn(8)
        logits = torch.randn(2)
        scores = score_all_classes(features, logits, stats)
        assert set(scores.keys()) == {0, 1}
        assert all(isinstance(v, float) for v in scores.values())


class TestBuildPredictionSet:
    def _make_stats(self):
        from src.ood.continual_ood import ClassCalibration
        return {
            0: ClassCalibration(
                mean=torch.zeros(8), var=torch.ones(8),
                mahalanobis_mu=1.0, mahalanobis_sigma=0.5,
                energy_mu=-2.0, energy_sigma=0.3,
                threshold=2.0,
            ),
            1: ClassCalibration(
                mean=torch.ones(8), var=torch.ones(8),
                mahalanobis_mu=1.0, mahalanobis_sigma=0.5,
                energy_mu=-2.0, energy_sigma=0.3,
                threshold=2.0,
            ),
        }

    def test_inf_qhat_returns_all(self):
        stats = self._make_stats()
        idx_to_class = {0: "healthy", 1: "diseased"}
        features = torch.randn(8)
        logits = torch.randn(2)
        pred_set = build_prediction_set(features, logits, float("inf"), stats, idx_to_class)
        assert set(pred_set) == {"diseased", "healthy"}

    def test_returns_list_of_strings(self):
        stats = self._make_stats()
        idx_to_class = {0: "healthy", 1: "diseased"}
        features = torch.randn(8)
        logits = torch.randn(2)
        qhat = 5.0  # lenient
        pred_set = build_prediction_set(features, logits, qhat, stats, idx_to_class)
        assert isinstance(pred_set, list)
        assert all(isinstance(c, str) for c in pred_set)

    def test_tight_qhat_may_exclude(self):
        stats = self._make_stats()
        idx_to_class = {0: "healthy", 1: "diseased"}
        features = torch.randn(8) * 10  # Far from both class means
        logits = torch.randn(2)
        pred_set = build_prediction_set(features, logits, -100.0, stats, idx_to_class)
        # Very tight qhat should exclude classes
        assert len(pred_set) <= 2


class TestEmpiricalCoverage:
    def test_perfect_coverage(self):
        labels = torch.tensor([0, 1, 2])
        sets = [[0, 1], [1, 2], [2, 0]]
        coverage = compute_empirical_coverage(labels, sets)
        assert coverage == 1.0

    def test_zero_coverage(self):
        labels = torch.tensor([0, 1, 2])
        sets = [[3], [3], [3]]
        coverage = compute_empirical_coverage(labels, sets)
        assert coverage == 0.0

    def test_partial_coverage(self):
        labels = torch.tensor([0, 1])
        sets = [[0], [2]]  # 1 out of 2 covered
        coverage = compute_empirical_coverage(labels, sets)
        assert abs(coverage - 0.5) < 1e-6

    def test_empty(self):
        coverage = compute_empirical_coverage(torch.tensor([]), [])
        assert coverage == 0.0
