"""Materialized OOD calibration helpers for continual SD-LoRA training.

The multi-score OOD stack needs one shared feature/logit materialization so it
can calibrate ensemble, energy, and kNN statistics together.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch

from src.ood.continual_ood import ContinualOODDetector


def _collect_materialized_tensors(
    trainer: Any,
    loader: Iterable[Dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features_list: List[torch.Tensor] = []
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(trainer.device, non_blocking=True)
            labels = batch["labels"].to(trainer.device, non_blocking=True).reshape(-1)
            features = trainer.encode(images)
            logits = trainer.classifier(features)
            features_list.append(features.detach().cpu())
            logits_list.append(logits.detach().cpu())
            labels_list.append(labels.detach().cpu())
    if not features_list:
        raise ValueError("Cannot calibrate OOD with an empty loader.")
    return (
        torch.cat(features_list, dim=0),
        torch.cat(logits_list, dim=0),
        torch.cat(labels_list, dim=0),
    )


def _move_detector_stats_to_device(detector: ContinualOODDetector, device: torch.device) -> None:
    for stats in detector.class_stats.values():
        stats.mean = stats.mean.to(device=device, dtype=torch.float32)
        stats.var = stats.var.to(device=device, dtype=torch.float32)
        if stats.knn_bank is not None:
            stats.knn_bank = stats.knn_bank.to(device=device, dtype=torch.float32)


def _build_calibration_summary(detector: ContinualOODDetector) -> Dict[str, float | str]:
    summary: Dict[str, float | str] = {
        "num_classes": float(len(detector.class_stats)),
        "calibration_version": float(detector.calibration_version),
        "primary_score_method": str(detector.primary_score_method),
        "knn_k": float(detector.knn_k),
        "knn_bank_cap": float(detector.knn_bank_cap),
        "knn_backend": str(getattr(detector, "knn_backend", "auto")),
        "knn_chunk_size": float(getattr(detector, "knn_chunk_size", 2048)),
        "conformal_method": str(getattr(detector, "conformal_method", "threshold")),
        "energy_temperature": float(getattr(detector, "energy_temperature", 1.0)),
        "react_enabled": bool(getattr(detector, "react_enabled", False)),
        "react_percentile": float(getattr(detector, "react_percentile", 0.99)),
    }
    if getattr(detector, "react_threshold", None) is not None:
        summary["react_threshold"] = float(getattr(detector, "react_threshold"))
    if detector.radial_beta is not None:
        summary["radial_beta"] = float(detector.radial_beta)
    if detector.conformal_qhat is not None:
        summary["conformal_qhat"] = float(detector.conformal_qhat)
    return summary


def calibrate_trainer_ood(
    trainer: Any,
    loader: Iterable[Dict[str, torch.Tensor]],
) -> Dict[str, float | str]:
    """Calibrate trainer-owned OOD statistics from one shared materialization."""
    if trainer.adapter_model is None or trainer.classifier is None or trainer.fusion is None:
        raise RuntimeError("Cannot calibrate OOD before adapter, classifier, and fusion are initialized.")

    trainer.adapter_model.eval()
    trainer.classifier.eval()
    trainer.fusion.eval()

    features, logits, labels = _collect_materialized_tensors(trainer, loader)
    calibration_features = features
    calibration_logits = logits
    if getattr(trainer.ood_detector, "react_enabled", False):
        trainer.ood_detector.fit_react_threshold(features)
        if getattr(trainer.ood_detector, "react_apply_during_calibration", True):
            calibration_features = trainer.ood_detector._maybe_reactify(features, apply=True)
            with torch.inference_mode():
                calibration_logits = trainer.classifier(
                    calibration_features.to(device=trainer.device, dtype=torch.float32)
                ).detach().cpu()
    calibration_result = trainer.ood_detector.calibrate(
        features=calibration_features,
        logits=calibration_logits,
        labels=labels,
    )
    _move_detector_stats_to_device(trainer.ood_detector, trainer.device)
    if isinstance(calibration_result, dict):
        return calibration_result
    return _build_calibration_summary(trainer.ood_detector)
