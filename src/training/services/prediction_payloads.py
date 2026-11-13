"""Build the public per-image prediction payload from classifier outputs and OOD scores."""

from __future__ import annotations

from typing import Any, Dict

import torch


def _tensor_item(values: Any, index: int, default: float = 0.0) -> float:
    if torch.is_tensor(values) and values.numel() > index:
        return float(values[index].item())
    return float(default)


def _bool_tensor_item(values: Any, index: int, default: bool = False) -> bool:
    if torch.is_tensor(values) and values.numel() > index:
        return bool(values[index].item())
    return bool(default)


def _int_tensor_item(values: Any, index: int, default: int = 0) -> int:
    if torch.is_tensor(values) and values.numel() > index:
        return int(values[index].item())
    return int(default)


def _score_dict_for_index(values: Any, index: int) -> Dict[str, float]:
    return {
        name: _tensor_item(tensor, index)
        for name, tensor in dict(values or {}).items()
        if torch.is_tensor(tensor) and tensor.numel() > index
    }


def build_ood_prediction_payloads(
    *,
    confidence: torch.Tensor,
    indices: torch.Tensor,
    ood: Dict[str, Any],
    score_features: torch.Tensor,
    logits: torch.Tensor,
    ood_detector: Any,
    idx_to_class: Dict[int, str],
) -> list[Dict[str, Any]]:
    """Turn one batch of classifier outputs and OOD scores into the public per-image prediction payloads."""
    primary_score_method = str(
        ood.get("primary_score_method", getattr(ood_detector, "primary_score_method", "ensemble")) or "ensemble"
    )
    results: list[Dict[str, Any]] = []
    for index in range(int(indices.numel())):
        predicted_idx = int(indices[index].item())
        candidate_scores = _score_dict_for_index(ood.get("candidate_scores"), index)
        candidate_thresholds = _score_dict_for_index(ood.get("candidate_thresholds"), index)
        primary_score_tensor = ood.get("primary_score")
        if torch.is_tensor(primary_score_tensor):
            primary_score = _tensor_item(primary_score_tensor, index)
        elif primary_score_method in candidate_scores:
            primary_score = float(candidate_scores[primary_score_method])
        else:
            primary_score = float(candidate_scores.get("ensemble", 0.0))
        decision_threshold_tensor = ood.get("decision_threshold")
        if torch.is_tensor(decision_threshold_tensor):
            decision_threshold = _tensor_item(decision_threshold_tensor, index)
        elif primary_score_method in candidate_thresholds:
            decision_threshold = float(candidate_thresholds[primary_score_method])
        else:
            decision_threshold = float(candidate_thresholds.get("ensemble", 0.0))
        ood_analysis: Dict[str, Any] = {
            "score_method": primary_score_method,
            "primary_score": primary_score,
            "decision_threshold": decision_threshold,
            "is_ood": _bool_tensor_item(ood.get("is_ood"), index),
            "candidate_scores": candidate_scores,
            "candidate_thresholds": candidate_thresholds,
            "calibration_version": _int_tensor_item(ood.get("calibration_version"), index),
        }
        for optional_key in ("mahalanobis_z", "energy_z", "knn_distance"):
            if torch.is_tensor(ood.get(optional_key)) and ood[optional_key].numel() > index:
                ood_analysis[optional_key] = _tensor_item(ood[optional_key], index)

        if ood_detector.radial_beta is not None:
            ood_analysis["radial_beta"] = float(ood_detector.radial_beta)

        if ood_detector.sure_enabled and "sure_semantic_score" in ood:
            ood_analysis["sure_semantic_score"] = _tensor_item(ood.get("sure_semantic_score"), index)
            ood_analysis["sure_confidence_score"] = _tensor_item(ood.get("sure_confidence_score"), index)
            ood_analysis["sure_semantic_ood"] = _bool_tensor_item(ood.get("sure_semantic_ood"), index)
            ood_analysis["sure_confidence_reject"] = _bool_tensor_item(ood.get("sure_confidence_reject"), index)

        if ood_detector.conformal_enabled:
            with torch.inference_mode():
                conformal_set = ood_detector.build_conformal_set(
                    score_features[index],
                    logits[index],
                    idx_to_class,
                )
            ood_analysis["conformal_set"] = conformal_set
            ood_analysis["conformal_set_size"] = len(conformal_set)
            ood_analysis["conformal_coverage"] = 1.0 - ood_detector.conformal_alpha

        results.append(
            {
                "status": "success",
                "disease": {
                    "class_index": predicted_idx,
                    "name": idx_to_class.get(predicted_idx, str(predicted_idx)),
                    "confidence": _tensor_item(confidence, index),
                },
                "ood_analysis": ood_analysis,
            }
        )
    return results

