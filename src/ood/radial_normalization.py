"""Radially Scaled L2 Normalization for OOD feature geometry control.

Implements β-scaled L2 normalization that projects feature vectors onto a
hypersphere of radius β, and provides auto-tuning of β via grid search
over Mahalanobis-based OOD separability.

Reference: Mahalanobis++ dissection — tunable radial geometry for OOD detection.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def radial_l2_normalize(features: torch.Tensor, beta: float) -> torch.Tensor:
    """Project feature vectors onto a hypersphere of radius β.

    Computes  f̂ = β · f / ‖f‖₂  for each row vector.

    Args:
        features: Tensor of shape ``[N, D]`` or ``[D]``.
        beta: Radius of the target hypersphere.  Must be positive.

    Returns:
        Normalized tensor with the same shape, where each row has L2 norm ≈ β.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")

    norms = features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
    return beta * (features / norms)


def _mahalanobis_separability(
    features: torch.Tensor,
    labels: torch.Tensor,
    var: Dict[int, torch.Tensor],
    means: Dict[int, torch.Tensor],
) -> float:
    """Compute OOD separability score as ratio of inter-class to intra-class Mahalanobis spread.

    Higher is better — indicates cleaner separation between class manifolds.
    """
    unique_classes = torch.unique(labels).tolist()
    if len(unique_classes) < 2:
        # With one class, measure intra-class compactness (lower distance = better).
        # Return inverse of mean distance so higher = more compact.
        class_id = unique_classes[0]
        mask = labels == class_id
        class_features = features[mask]
        m = means[class_id]
        v = var[class_id]
        distances = ((class_features - m) ** 2 / v).sum(dim=1).sqrt()
        mean_dist = float(distances.mean().item())
        return 1.0 / max(mean_dist, 1e-6)

    # Multi-class: measure ratio of between-class distance to within-class spread
    intra_distances = []
    for class_id in unique_classes:
        class_id = int(class_id)
        mask = labels == class_id
        class_features = features[mask]
        if class_features.shape[0] == 0:
            continue
        m = means[class_id]
        v = var[class_id]
        distances = ((class_features - m) ** 2 / v).sum(dim=1).sqrt()
        intra_distances.append(float(distances.mean().item()))

    mean_intra = sum(intra_distances) / max(len(intra_distances), 1)

    # Compute inter-class distances (pairwise between means)
    class_ids = sorted(means.keys())
    inter_distances = []
    for i, c1 in enumerate(class_ids):
        for c2 in class_ids[i + 1:]:
            # Use pooled variance for inter-class distance
            pooled_var = (var[c1] + var[c2]) / 2.0
            diff = means[c1] - means[c2]
            dist = float(((diff ** 2) / pooled_var).sum().sqrt().item())
            inter_distances.append(dist)

    mean_inter = sum(inter_distances) / max(len(inter_distances), 1)
    return mean_inter / max(mean_intra, 1e-6)


def auto_tune_beta(
    features: torch.Tensor,
    labels: torch.Tensor,
    beta_range: Tuple[float, float] = (0.5, 2.0),
    beta_steps: int = 16,
) -> float:
    """Grid-search for the optimal β that maximizes Mahalanobis OOD separability.

    For each candidate β, features are radially normalized, per-class statistics
    are recomputed, and a separability score is measured.  The β yielding the
    highest score is returned.

    Args:
        features: Raw feature tensor ``[N, D]``.
        labels: Class labels ``[N]``.
        beta_range: ``(min_beta, max_beta)`` inclusive search range.
        beta_steps: Number of grid points (≥ 2).

    Returns:
        The optimal β value.
    """
    if beta_steps < 2:
        beta_steps = 2

    beta_min, beta_max = float(beta_range[0]), float(beta_range[1])
    if beta_min >= beta_max:
        return beta_min

    candidates = torch.linspace(beta_min, beta_max, beta_steps).tolist()
    best_beta = candidates[0]
    best_score = -float("inf")

    unique_classes = torch.unique(labels).tolist()

    for beta in candidates:
        normed = radial_l2_normalize(features, beta)

        # Compute per-class mean and variance on normalized features
        means: Dict[int, torch.Tensor] = {}
        variances: Dict[int, torch.Tensor] = {}
        for class_id in unique_classes:
            class_id = int(class_id)
            mask = labels == class_id
            class_features = normed[mask]
            if class_features.shape[0] == 0:
                continue
            means[class_id] = class_features.mean(dim=0)
            variances[class_id] = class_features.var(dim=0, unbiased=False).clamp_min(1e-6)

        if not means:
            continue

        score = _mahalanobis_separability(normed, labels, variances, means)
        if score > best_score:
            best_score = score
            best_beta = beta

    return float(best_beta)
