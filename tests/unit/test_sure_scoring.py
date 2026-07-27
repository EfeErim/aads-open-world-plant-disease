"""Unit tests for SURE+ Double Scoring Functions."""

from __future__ import annotations

import torch

from src.ood.sure_scoring import (
    apply_sure_decision,
    calibrate_sure_thresholds,
    compute_confidence_score,
    compute_ds_f1,
    compute_semantic_score,
)


class TestComputeSemanticScore:
    def test_identity(self):
        z = torch.tensor([1.0, -0.5, 2.3])
        result = compute_semantic_score(z)
        assert torch.equal(result, z)


class TestComputeConfidenceScore:
    def test_range(self):
        logits = torch.randn(10, 5)
        scores = compute_confidence_score(logits)
        assert (scores >= 0.0).all()
        assert (scores < 1.0).all()

    def test_high_confidence_low_score(self):
        # One class has very high logit → confidence should be low (close to 0)
        logits = torch.tensor([[10.0, -10.0, -10.0]])
        score = compute_confidence_score(logits)
        assert score.item() < 0.01

    def test_uniform_logits_high_score(self):
        # Uniform logits → max prob = 1/C → confidence score = 1 - 1/C
        logits = torch.zeros(1, 5)
        score = compute_confidence_score(logits)
        assert abs(score.item() - 0.8) < 0.01  # 1 - 1/5 = 0.8


class TestCalibrateSureThresholds:
    def test_basic(self):
        semantic = torch.randn(100)
        confidence = torch.rand(100)
        sem_t, conf_t = calibrate_sure_thresholds(semantic, confidence)
        assert isinstance(sem_t, float)
        assert isinstance(conf_t, float)

    def test_percentile_ordering(self):
        scores = torch.arange(100, dtype=torch.float32)
        t_90, _ = calibrate_sure_thresholds(scores, scores, semantic_percentile=90.0)
        t_50, _ = calibrate_sure_thresholds(scores, scores, semantic_percentile=50.0)
        assert t_90 > t_50

    def test_empty_input(self):
        sem_t, conf_t = calibrate_sure_thresholds(torch.tensor([]), torch.tensor([]))
        assert sem_t == 0.0
        assert conf_t == 0.0


class TestApplySureDecision:
    def test_below_both_thresholds(self):
        result = apply_sure_decision(0.5, 0.3, 1.0, 0.5)
        assert not result["semantic_ood"]
        assert not result["confidence_reject"]
        assert not result["combined_reject"]

    def test_above_semantic_only(self):
        result = apply_sure_decision(1.5, 0.3, 1.0, 0.5)
        assert result["semantic_ood"]
        assert not result["confidence_reject"]
        assert result["combined_reject"]

    def test_above_confidence_only(self):
        result = apply_sure_decision(0.5, 0.8, 1.0, 0.5)
        assert not result["semantic_ood"]
        assert result["confidence_reject"]
        assert result["combined_reject"]

    def test_above_both(self):
        result = apply_sure_decision(1.5, 0.8, 1.0, 0.5)
        assert result["semantic_ood"]
        assert result["confidence_reject"]
        assert result["combined_reject"]

    def test_at_threshold_not_ood(self):
        result = apply_sure_decision(1.0, 0.5, 1.0, 0.5)
        assert not result["semantic_ood"]
        assert not result["confidence_reject"]


class TestDSF1:
    def test_perfect_scores(self):
        labels = torch.tensor([1, 1, 0, 0])
        preds = torch.tensor([1, 1, 0, 0])
        result = compute_ds_f1(labels, labels, preds, preds)
        assert result["semantic_f1"] == 1.0
        assert result["confidence_f1"] == 1.0
        assert result["ds_f1"] == 1.0

    def test_zero_scores(self):
        labels = torch.tensor([1, 1, 0, 0])
        preds = torch.tensor([0, 0, 1, 1])  # All wrong
        result = compute_ds_f1(labels, labels, preds, preds)
        assert result["ds_f1"] == 0.0
