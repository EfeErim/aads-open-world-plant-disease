"""Small, executable version of the continual-adapter objective."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


@dataclass(frozen=True)
class LossWeights:
    classification: float = 1.0
    distillation: float = 0.5
    adapter_l2: float = 1e-4


@dataclass(frozen=True)
class ContinualLoss:
    total: float
    classification: float
    distillation: float
    adapter_l2: float


def _cross_entropy(logits: FloatArray, labels: IntArray) -> float:
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits must be [batch, classes] and labels must be [batch]")
    if logits.shape[0] == 0:
        raise ValueError("empty batches are not valid")
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probabilities[np.arange(labels.size), labels].mean())


def continual_loss(
    *,
    logits: FloatArray,
    labels: IntArray,
    current_features: FloatArray,
    teacher_features: FloatArray,
    adapter_parameters: FloatArray,
    weights: LossWeights = LossWeights(),
) -> ContinualLoss:
    """Combine task learning, feature retention, and bounded adapter movement."""
    if current_features.shape != teacher_features.shape:
        raise ValueError("current and teacher feature tensors must have identical shapes")
    classification = _cross_entropy(logits, labels)
    distillation = float(np.mean(np.square(current_features - teacher_features)))
    adapter_l2 = float(np.mean(np.square(adapter_parameters)))
    total = (
        weights.classification * classification
        + weights.distillation * distillation
        + weights.adapter_l2 * adapter_l2
    )
    return ContinualLoss(total, classification, distillation, adapter_l2)


class ReplayBuffer:
    """Deterministic reservoir sampler for bounded continual-learning replay."""

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._seen = 0
        self._items: list[object] = []

    def add(self, item: object) -> None:
        self._seen += 1
        if len(self._items) < self.capacity:
            self._items.append(item)
            return
        replacement = int(self._rng.integers(0, self._seen))
        if replacement < self.capacity:
            self._items[replacement] = item

    def sample(self, size: int) -> tuple[object, ...]:
        if size < 0 or size > len(self._items):
            raise ValueError("sample size must fit the current replay buffer")
        if size == 0:
            return ()
        indices = self._rng.choice(len(self._items), size=size, replace=False)
        return tuple(self._items[int(index)] for index in indices)

    def __len__(self) -> int:
        return len(self._items)
