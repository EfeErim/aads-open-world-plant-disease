"""Known in-distribution inputs are accepted and far-away, low-confidence inputs are rejected."""

import math

import pytest
import torch

from src.ood._scoring_utils import energy_from_logits
from src.ood.continual_ood import ContinualOODDetector


def test_energy_is_lower_for_confident_logits():
    flat = torch.zeros(1, 3)
    confident = torch.tensor([[10.0, 0.0, 0.0]])

    assert energy_from_logits(flat).item() == pytest.approx(-math.log(3.0))
    assert energy_from_logits(confident).item() < energy_from_logits(flat).item()


def _calibrated(method: str) -> ContinualOODDetector:
    generator = torch.Generator().manual_seed(0)
    centers = torch.eye(3, 8) * 10.0
    labels = torch.arange(3).repeat_interleave(20)
    features = centers[labels] + 0.1 * torch.randn(60, 8, generator=generator)
    logits = torch.nn.functional.one_hot(labels, 3).float() * 10.0
    detector = ContinualOODDetector(threshold_factor=2.0, primary_score_method=method)
    detector.calibrate(features, logits, labels)
    return detector


@pytest.mark.parametrize("method", ["ensemble", "energy", "knn"])
def test_in_distribution_is_accepted_and_far_input_is_rejected(method):
    detector = _calibrated(method)
    labels = torch.tensor([0, 1, 2])
    known = torch.eye(3, 8) * 10.0
    known_logits = torch.nn.functional.one_hot(labels, 3).float() * 10.0
    far = torch.full((3, 8), -50.0)
    flat_logits = torch.zeros(3, 3)

    accepted = detector.score(known, known_logits, predicted_labels=labels)
    rejected = detector.score(far, flat_logits, predicted_labels=labels)

    assert not accepted["is_ood"].any()
    assert rejected["is_ood"].all()
    assert torch.all(rejected["primary_score"] > accepted["primary_score"])
