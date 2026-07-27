"""Unit tests for extended OOD calibration with all four techniques."""

from __future__ import annotations

import pytest
import torch

from src.ood.continual_ood import ClassCalibration, ContinualOODDetector


class TestExtendedCalibration:
    """End-to-end tests of the OOD detector with all techniques enabled."""

    @pytest.fixture
    def detector(self):
        return ContinualOODDetector(
            threshold_factor=2.0,
            radial_l2_enabled=True,
            radial_beta_range=(0.5, 2.0),
            radial_beta_steps=4,
            sure_enabled=True,
            sure_semantic_percentile=95.0,
            sure_confidence_percentile=90.0,
            conformal_enabled=True,
            conformal_alpha=0.10,
        )

    @pytest.fixture
    def synthetic_data(self):
        torch.manual_seed(42)
        n_per_class = 50
        features = torch.cat([
            torch.randn(n_per_class, 32) + 2.0,
            torch.randn(n_per_class, 32) - 2.0,
        ])
        labels = torch.cat([
            torch.zeros(n_per_class, dtype=torch.long),
            torch.ones(n_per_class, dtype=torch.long),
        ])
        logits = torch.randn(n_per_class * 2, 2)
        # Make logits somewhat correlated with labels
        logits[labels == 0, 0] += 2.0
        logits[labels == 1, 1] += 2.0
        return features, logits, labels

    def test_calibrate_populates_all_fields(self, detector, synthetic_data):
        features, logits, labels = synthetic_data
        result = detector.calibrate(features, logits, labels)

        assert "num_classes" in result
        assert float(result["num_classes"]) == 2.0
        assert result["primary_score_method"] == "ensemble"
        assert "radial_beta" in result
        assert "conformal_qhat" in result
        assert detector.radial_beta is not None
        assert detector.conformal_qhat is not None

        for class_id, stats in detector.class_stats.items():
            assert isinstance(stats, ClassCalibration)
            assert stats.sure_semantic_threshold != 0.0 or stats.sure_confidence_threshold != 0.0
            assert stats.threshold != 0.0
            assert stats.knn_bank is not None
            assert int(stats.knn_bank.shape[0]) <= detector.knn_bank_cap

    def test_score_returns_sure_fields(self, detector, synthetic_data):
        features, logits, labels = synthetic_data
        detector.calibrate(features, logits, labels)

        ood = detector.score(features[:2], logits[:2])
        assert "sure_semantic_score" in ood
        assert "sure_confidence_score" in ood
        assert "sure_semantic_ood" in ood
        assert "sure_confidence_reject" in ood
        assert "radial_beta" in ood
        assert "primary_score" in ood
        assert "knn_distance" in ood

    def test_conformal_set_built(self, detector, synthetic_data):
        features, logits, labels = synthetic_data
        detector.calibrate(features, logits, labels)

        idx_to_class = {0: "class_a", 1: "class_b"}
        pred_set = detector.build_conformal_set(features[0], logits[0], idx_to_class)
        assert isinstance(pred_set, list)
        # With α=0.10 and well-separated data, set should be small
        assert len(pred_set) <= 2

    def test_disabled_techniques_no_crash(self):
        detector = ContinualOODDetector(
            threshold_factor=2.0,
            radial_l2_enabled=False,
            sure_enabled=False,
            conformal_enabled=False,
        )
        torch.manual_seed(42)
        features = torch.randn(20, 16)
        logits = torch.randn(20, 2)
        labels = torch.cat([torch.zeros(10), torch.ones(10)]).long()

        result = detector.calibrate(features, logits, labels)
        assert "num_classes" in result
        assert detector.radial_beta is None
        assert detector.conformal_qhat is None

        ood = detector.score(features[:1], logits[:1])
        assert "primary_score" in ood
        # SURE+ fields should not be in output when disabled
        assert "sure_semantic_score" not in ood

    def test_score_class_with_sure_disabled(self):
        detector = ContinualOODDetector(threshold_factor=2.0, sure_enabled=False)
        features = torch.randn(20, 8)
        logits = torch.randn(20, 2)
        labels = torch.cat([torch.zeros(10), torch.ones(10)]).long()
        detector.calibrate(features, logits, labels)

        result = detector._score_class(0, features[0], logits[0])
        assert result["sure_semantic_score"] is None
        assert result["sure_confidence_score"] is None

    def test_conformal_set_empty_when_disabled(self):
        detector = ContinualOODDetector(threshold_factor=2.0, conformal_enabled=False)
        features = torch.randn(20, 8)
        logits = torch.randn(20, 2)
        labels = torch.cat([torch.zeros(10), torch.ones(10)]).long()
        detector.calibrate(features, logits, labels)

        pred_set = detector.build_conformal_set(features[0], logits[0], {0: "a", 1: "b"})
        assert pred_set == []

    def test_knn_bank_is_deterministically_capped(self):
        detector = ContinualOODDetector(knn_bank_cap=8)
        torch.manual_seed(7)
        features = torch.randn(60, 6)
        logits = torch.randn(60, 2)
        labels = torch.cat([torch.zeros(30), torch.ones(30)]).long()

        detector.calibrate(features, logits, labels)
        first_banks = {class_id: stats.knn_bank.clone() for class_id, stats in detector.class_stats.items()}

        detector.calibrate(features, logits, labels)
        second_banks = {class_id: stats.knn_bank.clone() for class_id, stats in detector.class_stats.items()}

        for class_id in first_banks:
            assert first_banks[class_id].shape[0] == 8
            assert torch.equal(first_banks[class_id], second_banks[class_id])
