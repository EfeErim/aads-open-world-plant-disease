#!/usr/bin/env python3
"""Multi-scale feature fusion for v6 continual learning."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


def _to_2d(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 2:
        return features
    if features.ndim == 3:
        # [B, T, D] -> mean token pooling
        return features.mean(dim=1)
    if features.ndim == 4:
        # [B, C, H, W] -> global avg pooling
        return features.mean(dim=(2, 3))
    return features.reshape(features.shape[0], -1)


def select_multiscale_features(hidden_states: Sequence[torch.Tensor], layers: Iterable[int]) -> List[torch.Tensor]:
    """Select hidden states by layer index with safe clamping."""
    states = list(hidden_states or [])
    if not states:
        return []
    selected: List[torch.Tensor] = []
    max_idx = len(states) - 1
    for layer in layers:
        idx = max(0, min(int(layer), max_idx))
        selected.append(states[idx])
    return selected


class MultiScaleFeatureFusion(nn.Module):
    """
    Lightweight gated fusion over selected transformer stages.

    Default use-case is 4-stage fusion for DINOv3 hidden states.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 768,
        num_scales: int = 4,
        dropout: float = 0.1,
        gating: str = "softmax",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_scales = int(max(1, num_scales))
        self.gating = str(gating)

        self.projections = nn.ModuleList(
            [nn.Linear(self.input_dim, self.output_dim) for _ in range(self.num_scales)]
        )
        self.scale_logits = nn.Parameter(torch.zeros(self.num_scales))
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.output_dim)

    def _projection(self, index: int) -> nn.Linear:
        projection = self.projections[index]
        if not isinstance(projection, nn.Linear):
            raise TypeError(f"Expected Linear projection at index {index}, got {type(projection)!r}")
        return projection

    def _weights(self, active_scales: int) -> torch.Tensor:
        logits = self.scale_logits[:active_scales]
        if self.gating == "softmax":
            return torch.softmax(logits, dim=0)
        return torch.ones_like(logits) / max(1, logits.numel())

    def forward(self, feature_list: Sequence[torch.Tensor]) -> torch.Tensor:
        if not feature_list:
            raise ValueError("feature_list must contain at least one tensor")

        first_projection = self._projection(0)
        projection_weight = first_projection.weight
        normalized = [
            _to_2d(feat).to(device=projection_weight.device, dtype=projection_weight.dtype)
            for feat in feature_list
        ]
        while len(normalized) < self.num_scales:
            normalized.append(normalized[-1])
        active = normalized[: self.num_scales]

        projected = [self._projection(idx)(feat) for idx, feat in enumerate(active)]
        weights = self._weights(len(projected))

        fused = torch.zeros_like(projected[0])
        for idx, feat in enumerate(projected):
            fused = fused + feat * weights[idx]

        return self.norm(self.dropout(fused))
