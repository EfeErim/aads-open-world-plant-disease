"""Conformal helpers for thresholded OOD scores and set-valued classification.

This module supports three split-conformal modes:

- ``threshold``: conformalized OOD-threshold residuals based on the detector's
  ensemble score. This calibrates a score threshold, not a standard
  label-prediction set method.
- ``aps``: adaptive prediction sets on class probabilities.
- ``raps``: regularized adaptive prediction sets.

The usual finite-sample coverage guarantee for APS/RAPS requires exchangeable
calibration and evaluation samples drawn from the same label space.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import torch

from src.ood._scoring_utils import energy_from_logits, ensemble_z_score, mahalanobis_distance

SUPPORTED_CONFORMAL_METHODS = ("threshold", "aps", "raps")


def normalize_conformal_method(value: Any) -> str:
    resolved = str(value or "threshold").strip().lower()
    if resolved not in SUPPORTED_CONFORMAL_METHODS:
        raise ValueError(
            "conformal_method must be one of: " + ", ".join(SUPPORTED_CONFORMAL_METHODS) + "."
        )
    return resolved


def compute_nonconformity_scores(
    ensemble_scores: torch.Tensor,
    class_thresholds: torch.Tensor,
) -> torch.Tensor:
    """Convert OOD ensemble scores to threshold-mode non-conformity scores."""
    return ensemble_scores - class_thresholds


def _sorted_probability_payload(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    return torch.sort(probabilities, dim=-1, descending=True)


def _label_ranks(sorted_indices: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.reshape(-1, 1).to(device=sorted_indices.device, dtype=torch.long)
    return torch.argmax((sorted_indices == labels).to(dtype=torch.long), dim=1)


def compute_prediction_set_nonconformity_scores(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    method: str = "aps",
    raps_lambda: float = 0.0,
    raps_k_reg: int = 1,
) -> torch.Tensor:
    resolved_method = normalize_conformal_method(method)
    if resolved_method == "threshold":
        raise ValueError("threshold mode uses compute_nonconformity_scores(...), not APS/RAPS logits scoring.")
    if logits.ndim != 2:
        raise ValueError("logits must be [N, C].")
    if logits.shape[0] <= 0:
        return torch.empty(0, dtype=torch.float32, device=logits.device)

    sorted_probs, sorted_indices = _sorted_probability_payload(logits)
    ranks = _label_ranks(sorted_indices, labels)
    cumulative = torch.cumsum(sorted_probs, dim=1)
    row_indices = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    scores = cumulative[row_indices, ranks]
    if resolved_method == "raps":
        penalties = torch.clamp(ranks.to(dtype=torch.float32) + 1.0 - float(max(int(raps_k_reg), 1)), min=0.0)
        scores = scores + (float(raps_lambda) * penalties)
    return scores.to(dtype=torch.float32)


def calibrate_conformal_qhat(
    nonconformity_scores: torch.Tensor,
    alpha: float = 0.05,
) -> float:
    """Compute the split-conformal quantile q-hat."""
    if nonconformity_scores.numel() == 0:
        return float("inf")

    n = nonconformity_scores.numel()
    adjusted_quantile = min(1.0, math.ceil((1 - alpha) * (n + 1)) / n)

    return float(
        torch.quantile(
            nonconformity_scores.float(),
            min(max(adjusted_quantile, 0.0), 1.0),
        ).item()
    )


def calibrate_prediction_set_qhat(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = 0.05,
    method: str = "aps",
    raps_lambda: float = 0.0,
    raps_k_reg: int = 1,
) -> float:
    scores = compute_prediction_set_nonconformity_scores(
        logits,
        labels,
        method=method,
        raps_lambda=raps_lambda,
        raps_k_reg=raps_k_reg,
    )
    return calibrate_conformal_qhat(scores, alpha=alpha)


def score_all_classes(
    features: torch.Tensor,
    logits: torch.Tensor,
    class_stats: Dict[int, Any],
    *,
    energy_temperature: float = 1.0,
) -> Dict[int, float]:
    """Score a single sample against every calibrated class for threshold mode."""
    scores: Dict[int, float] = {}
    energy = energy_from_logits(logits.unsqueeze(0), temperature=energy_temperature)[0]

    for class_id, stats in class_stats.items():
        mean = stats.mean.to(device=features.device, dtype=features.dtype)
        var = stats.var.to(device=features.device, dtype=features.dtype)
        distance = mahalanobis_distance(features, mean, var)
        ensemble = ensemble_z_score(
            distance,
            energy,
            mahalanobis_mu=stats.mahalanobis_mu,
            mahalanobis_sigma=stats.mahalanobis_sigma,
            energy_mu=stats.energy_mu,
            energy_sigma=stats.energy_sigma,
        )
        scores[int(class_id)] = float(ensemble.item())

    return scores


def build_prediction_set(
    features: torch.Tensor,
    logits: torch.Tensor,
    qhat: float,
    class_stats: Dict[int, Any],
    idx_to_class: Dict[int, str],
    *,
    method: str = "threshold",
    raps_lambda: float = 0.0,
    raps_k_reg: int = 1,
    energy_temperature: float = 1.0,
) -> List[str]:
    """Build the conformal prediction set for a single sample."""
    resolved_method = normalize_conformal_method(method)
    if qhat == float("inf"):
        return [idx_to_class[class_id] for class_id in sorted(idx_to_class)]

    if resolved_method in {"aps", "raps"}:
        if logits.ndim != 1:
            raise ValueError("APS/RAPS prediction-set building expects logits [C].")
        sorted_probs, sorted_indices = _sorted_probability_payload(logits.unsqueeze(0))
        cumulative = torch.cumsum(sorted_probs[0], dim=0)
        included: List[str] = []
        for rank, class_id in enumerate(sorted_indices[0].tolist(), start=1):
            score = float(cumulative[rank - 1].item())
            if resolved_method == "raps":
                score += float(raps_lambda) * max(rank - max(int(raps_k_reg), 1), 0)
            if score <= float(qhat):
                included.append(idx_to_class.get(int(class_id), str(int(class_id))))
        if not included and sorted_indices.shape[1] > 0:
            top_class = int(sorted_indices[0, 0].item())
            return [idx_to_class.get(top_class, str(top_class))]
        return included

    all_scores = score_all_classes(features, logits, class_stats, energy_temperature=energy_temperature)

    included_pairs: List[tuple[float, str]] = []
    for class_id, ensemble_score in all_scores.items():
        threshold = class_stats[class_id].threshold
        nonconformity = ensemble_score - threshold
        if nonconformity <= qhat:
            class_name = idx_to_class.get(class_id, str(class_id))
            included_pairs.append((ensemble_score, class_name))

    included_pairs.sort(key=lambda x: x[0])
    return [name for _, name in included_pairs]


def describe_conformal_method(
    method: str,
    *,
    raps_lambda: float = 0.0,
    raps_k_reg: int = 1,
) -> str:
    resolved_method = normalize_conformal_method(method)
    if resolved_method == "threshold":
        return "threshold-conformalized OOD residual sets"
    if resolved_method == "aps":
        return "APS set-valued classification"
    return f"RAPS set-valued classification (lambda={float(raps_lambda):.3g}, k_reg={int(raps_k_reg)})"


def compute_empirical_coverage(
    true_labels: torch.Tensor,
    prediction_sets: List[List[int]],
) -> float:
    """Compute empirical coverage: fraction of samples where true label is in the prediction set."""
    if len(prediction_sets) == 0:
        return 0.0

    covered = 0
    for label, pred_set in zip(true_labels.tolist(), prediction_sets):
        if int(label) in pred_set:
            covered += 1

    return float(covered) / float(len(prediction_sets))
