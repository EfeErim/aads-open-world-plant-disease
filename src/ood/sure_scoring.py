"""SURE+/DS-F1-inspired double scoring for OOD and failure rejection.

Implements independent semantic shift and predictive confidence scoring with
separate calibrated thresholds, enabling joint evaluation via the DS-F1 metric.

The repo uses "SURE+" as internal shorthand for this two-threshold path. Treat
it as an inspiration-level implementation rather than a claim of exact
reproduction of a standalone upstream paper.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def compute_semantic_score(ensemble_z: torch.Tensor) -> torch.Tensor:
    """Reuse the Mahalanobis+energy ensemble z-score as semantic shift score."""
    return ensemble_z


def compute_confidence_score(logits: torch.Tensor) -> torch.Tensor:
    """Compute inverse max-softmax confidence; higher means less confident."""
    probs = torch.softmax(logits, dim=-1)
    max_probs, _ = probs.max(dim=-1)
    return 1.0 - max_probs


def calibrate_sure_thresholds(
    semantic_scores: torch.Tensor,
    confidence_scores: torch.Tensor,
    semantic_percentile: float = 95.0,
    confidence_percentile: float = 90.0,
) -> Tuple[float, float]:
    """Calibrate independent double-scoring thresholds from ID calibration data."""
    if semantic_scores.numel() == 0 or confidence_scores.numel() == 0:
        return 0.0, 0.0

    semantic_q = float(semantic_percentile) / 100.0
    confidence_q = float(confidence_percentile) / 100.0

    semantic_threshold = float(
        torch.quantile(semantic_scores.float(), min(max(semantic_q, 0.0), 1.0)).item()
    )
    confidence_threshold = float(
        torch.quantile(confidence_scores.float(), min(max(confidence_q, 0.0), 1.0)).item()
    )
    return semantic_threshold, confidence_threshold


def apply_sure_decision(
    semantic_score: float,
    confidence_score: float,
    semantic_threshold: float,
    confidence_threshold: float,
) -> Dict[str, Any]:
    """Apply the repo's double-scoring rejection logic."""
    semantic_ood = bool(semantic_score > semantic_threshold)
    confidence_reject = bool(confidence_score > confidence_threshold)
    return {
        "semantic_ood": semantic_ood,
        "confidence_reject": confidence_reject,
        "combined_reject": semantic_ood or confidence_reject,
    }


def compute_ds_f1(
    semantic_ood_labels: torch.Tensor,
    confidence_reject_labels: torch.Tensor,
    semantic_predictions: torch.Tensor,
    confidence_predictions: torch.Tensor,
) -> Dict[str, float]:
    """Compute DS-F1 from semantic OOD and confidence rejection F1 scores."""

    def _binary_f1(labels: torch.Tensor, preds: torch.Tensor) -> float:
        labels = labels.bool()
        preds = preds.bool()
        tp = float((labels & preds).sum().item())
        fp = float((~labels & preds).sum().item())
        fn = float((labels & ~preds).sum().item())
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        if precision + recall < 1e-8:
            return 0.0
        return float(2.0 * precision * recall / (precision + recall))

    semantic_f1 = _binary_f1(semantic_ood_labels, semantic_predictions)
    confidence_f1 = _binary_f1(confidence_reject_labels, confidence_predictions)

    if semantic_f1 + confidence_f1 < 1e-8:
        ds_f1 = 0.0
    else:
        ds_f1 = float(2.0 * semantic_f1 * confidence_f1 / (semantic_f1 + confidence_f1))

    return {
        "semantic_f1": semantic_f1,
        "confidence_f1": confidence_f1,
        "ds_f1": ds_f1,
    }
