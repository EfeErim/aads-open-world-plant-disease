import numpy as np
import pytest

from aads_public.training import LossWeights, ReplayBuffer, continual_loss


def test_continual_loss_exposes_all_components() -> None:
    result = continual_loss(
        logits=np.array([[3.0, 0.2], [0.1, 2.5]]),
        labels=np.array([0, 1]),
        current_features=np.array([[1.0, 0.0], [0.0, 1.0]]),
        teacher_features=np.array([[0.9, 0.1], [0.1, 0.9]]),
        adapter_parameters=np.array([0.2, -0.2]),
        weights=LossWeights(classification=1.0, distillation=0.5, adapter_l2=0.1),
    )
    assert result.total > 0
    assert result.classification > 0
    assert result.distillation == pytest.approx(0.01)
    assert result.adapter_l2 == pytest.approx(0.04)


def test_feature_shapes_must_match() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        continual_loss(
            logits=np.array([[1.0, 0.0]]),
            labels=np.array([0]),
            current_features=np.zeros((1, 2)),
            teacher_features=np.zeros((1, 3)),
            adapter_parameters=np.zeros(1),
        )


def test_replay_buffer_is_bounded_and_deterministic() -> None:
    first = ReplayBuffer(3, seed=7)
    second = ReplayBuffer(3, seed=7)
    for value in range(20):
        first.add(value)
        second.add(value)
    assert len(first) == 3
    assert first.sample(3) == second.sample(3)


def test_replay_buffer_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="positive"):
        ReplayBuffer(0)
