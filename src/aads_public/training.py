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

    def __post_init__(self) -> None:
        values = (self.classification, self.distillation, self.adapter_l2)
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError("loss weights must be finite and non-negative")


@dataclass(frozen=True)
class ContinualLoss:
    total: float
    classification: float
    distillation: float
    adapter_l2: float


def _cross_entropy(logits: FloatArray, labels: IntArray) -> float:
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits must be [batch, classes] and labels must be [batch]")
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("empty batch or class dimensions are not valid")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must use an integer dtype")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("labels must be within the logits class range")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must contain only finite values")
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
    if current_features.ndim == 0 or current_features.shape[0] != logits.shape[0] or current_features.size == 0:
        raise ValueError("feature tensors must contain one non-empty row per logit row")
    if adapter_parameters.size == 0:
        raise ValueError("adapter_parameters must not be empty")
    if not all(
        np.all(np.isfinite(values))
        for values in (current_features, teacher_features, adapter_parameters)
    ):
        raise ValueError("features and adapter_parameters must contain only finite values")
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
