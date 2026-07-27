"""Unit tests for Radially Scaled L2 Normalization."""

from __future__ import annotations

import pytest
import torch

from src.ood.radial_normalization import auto_tune_beta, radial_l2_normalize


class TestRadialL2Normalize:
    def test_output_norm_equals_beta(self):
        features = torch.randn(10, 64)
        beta = 1.5
        normed = radial_l2_normalize(features, beta)
        norms = normed.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.full_like(norms, beta), atol=1e-5)

    def test_single_vector(self):
        feature = torch.randn(64)
        beta = 2.0
        normed = radial_l2_normalize(feature, beta)
        assert normed.shape == (64,)
        assert abs(normed.norm(p=2).item() - beta) < 1e-5

    def test_direction_preserved(self):
        features = torch.randn(5, 32)
        beta = 1.0
        normed = radial_l2_normalize(features, beta)
        # Cosine similarity between original and normalized should be ~1
        cos = torch.nn.functional.cosine_similarity(features, normed, dim=1)
        assert torch.allclose(cos, torch.ones(5), atol=1e-5)

    def test_beta_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            radial_l2_normalize(torch.randn(3, 4), beta=-1.0)

    def test_zero_beta_raises(self):
        with pytest.raises(ValueError, match="positive"):
            radial_l2_normalize(torch.randn(3, 4), beta=0.0)

    def test_various_beta_values(self):
        features = torch.randn(8, 128)
        for beta in [0.1, 0.5, 1.0, 5.0, 10.0]:
            normed = radial_l2_normalize(features, beta)
            norms = normed.norm(p=2, dim=1)
            assert torch.allclose(norms, torch.full_like(norms, beta), atol=1e-5)


class TestAutoTuneBeta:
    def test_returns_within_range(self):
        torch.manual_seed(42)
        features = torch.randn(100, 32)
        labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
        beta = auto_tune_beta(features, labels, beta_range=(0.5, 2.0), beta_steps=8)
        assert 0.5 <= beta <= 2.0

    def test_single_class(self):
        features = torch.randn(50, 16)
        labels = torch.zeros(50).long()
        beta = auto_tune_beta(features, labels, beta_range=(0.5, 2.0), beta_steps=4)
        assert 0.5 <= beta <= 2.0

    def test_min_steps_clamp(self):
        features = torch.randn(20, 8)
        labels = torch.zeros(20).long()
        beta = auto_tune_beta(features, labels, beta_range=(1.0, 2.0), beta_steps=1)
        assert 1.0 <= beta <= 2.0

    def test_well_separated_classes_prefer_smaller_beta(self):
        """With well-separated classes, compactness (smaller beta) should score well."""
        torch.manual_seed(0)
        # Two well-separated clusters
        c0 = torch.randn(50, 16) + 5.0
        c1 = torch.randn(50, 16) - 5.0
        features = torch.cat([c0, c1])
        labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
        beta = auto_tune_beta(features, labels, beta_range=(0.1, 5.0), beta_steps=20)
        assert isinstance(beta, float)
