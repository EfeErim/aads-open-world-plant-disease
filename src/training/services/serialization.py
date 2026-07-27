"""Serialization helpers for adapter metadata, checkpoints, and OOD state."""

from __future__ import annotations

from typing import Any, Dict

import torch


def _read_mapping_or_attr(source: Any, field_name: str) -> Any:
    if isinstance(source, dict):
        return source.get(field_name)
    return getattr(source, field_name, None)


def _tensor_to_list(value: Any, *, strict: bool, class_id: str, field_name: str) -> list[Any]:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if not strict:
        return []
    raise ValueError(
        f"OOD class_stats[{class_id}] is missing exportable '{field_name}' tensor/list payload."
    )


def _float_field(value: Any, *, default: float) -> float:
    fallback = float(default)
    try:
        return fallback if value is None else float(value)
    except (TypeError, ValueError, OverflowError):
        return fallback


def serialize_ood_state(ood_detector: Any, *, strict: bool = True) -> Dict[str, Any]:
    class_stats_payload: Dict[str, Any] = {}
    raw_class_stats = getattr(ood_detector, "class_stats", {}) or {}
    if not isinstance(raw_class_stats, dict):
        if strict:
            raise ValueError("OOD detector class_stats must be a dictionary for export.")
        raw_class_stats = {}

    for class_id, stats in raw_class_stats.items():
        class_key = str(class_id)
        mean = _read_mapping_or_attr(stats, "mean")
        var = _read_mapping_or_attr(stats, "var")
        if mean is None or var is None:
            if strict:
                raise ValueError(
                    f"OOD class_stats[{class_key}] is missing required 'mean'/'var' calibration tensors."
                )
            continue
        class_stats_payload[class_key] = {
            "mean": _tensor_to_list(mean, strict=strict, class_id=class_key, field_name="mean"),
            "var": _tensor_to_list(var, strict=strict, class_id=class_key, field_name="var"),
            "mahalanobis_mu": _float_field(_read_mapping_or_attr(stats, "mahalanobis_mu"), default=0.0),
            "mahalanobis_sigma": _float_field(_read_mapping_or_attr(stats, "mahalanobis_sigma"), default=1.0),
            "energy_mu": _float_field(_read_mapping_or_attr(stats, "energy_mu"), default=0.0),
            "energy_sigma": _float_field(_read_mapping_or_attr(stats, "energy_sigma"), default=1.0),
            "threshold": _float_field(_read_mapping_or_attr(stats, "threshold"), default=0.0),
            "energy_threshold": _float_field(_read_mapping_or_attr(stats, "energy_threshold"), default=0.0),
            "knn_distance_mu": _float_field(_read_mapping_or_attr(stats, "knn_distance_mu"), default=0.0),
            "knn_distance_sigma": _float_field(_read_mapping_or_attr(stats, "knn_distance_sigma"), default=1.0),
            "knn_threshold": _float_field(_read_mapping_or_attr(stats, "knn_threshold"), default=0.0),
            "knn_bank": _tensor_to_list(
                _read_mapping_or_attr(stats, "knn_bank"),
                strict=False,
                class_id=class_key,
                field_name="knn_bank",
            ),
            "knn_k": int(_read_mapping_or_attr(stats, "knn_k") or 10),
            "sure_semantic_threshold": _float_field(
                _read_mapping_or_attr(stats, "sure_semantic_threshold"),
                default=0.0,
            ),
            "sure_confidence_threshold": _float_field(
                _read_mapping_or_attr(stats, "sure_confidence_threshold"),
                default=0.0,
            ),
        }

    return {
        "threshold_factor": _float_field(getattr(ood_detector, "threshold_factor", 2.0), default=2.0),
        "primary_score_method": str(getattr(ood_detector, "primary_score_method", "ensemble") or "ensemble"),
        "score_threshold_overrides": {
            str(method_name): _float_field(threshold, default=0.0)
            for method_name, threshold in dict(getattr(ood_detector, "score_threshold_overrides", {}) or {}).items()
        },
        "calibration_version": int(getattr(ood_detector, "calibration_version", 0)),
        "class_stats": class_stats_payload,
        "knn_k": int(getattr(ood_detector, "knn_k", 10)),
        "knn_bank_cap": int(getattr(ood_detector, "knn_bank_cap", 256)),
        "knn_backend": str(getattr(ood_detector, "knn_backend", "auto") or "auto"),
        "knn_chunk_size": int(getattr(ood_detector, "knn_chunk_size", 2048)),
        "radial_l2_enabled": bool(getattr(ood_detector, "radial_l2_enabled", False)),
        "radial_beta": (
            None
            if getattr(ood_detector, "radial_beta", None) is None
            else _float_field(getattr(ood_detector, "radial_beta"), default=0.0)
        ),
        "radial_beta_range": list(getattr(ood_detector, "radial_beta_range", (0.5, 2.0))),
        "radial_beta_steps": int(getattr(ood_detector, "radial_beta_steps", 16)),
        "sure_enabled": bool(getattr(ood_detector, "sure_enabled", False)),
        "sure_semantic_percentile": _float_field(
            getattr(ood_detector, "sure_semantic_percentile", 95.0),
            default=95.0,
        ),
        "sure_confidence_percentile": _float_field(
            getattr(ood_detector, "sure_confidence_percentile", 90.0),
            default=90.0,
        ),
        "conformal_enabled": bool(getattr(ood_detector, "conformal_enabled", False)),
        "conformal_alpha": _float_field(getattr(ood_detector, "conformal_alpha", 0.05), default=0.05),
        "conformal_method": str(getattr(ood_detector, "conformal_method", "threshold") or "threshold"),
        "conformal_raps_lambda": _float_field(getattr(ood_detector, "conformal_raps_lambda", 0.0), default=0.0),
        "conformal_raps_k_reg": int(getattr(ood_detector, "conformal_raps_k_reg", 1)),
        "conformal_qhat": (
            None
            if getattr(ood_detector, "conformal_qhat", None) is None
            else _float_field(getattr(ood_detector, "conformal_qhat"), default=0.0)
        ),
        "energy_temperature": _float_field(getattr(ood_detector, "energy_temperature", 1.0), default=1.0),
        "energy_temperature_mode": str(getattr(ood_detector, "energy_temperature_mode", "fixed") or "fixed"),
        "energy_temperature_range": list(getattr(ood_detector, "energy_temperature_range", (0.5, 3.0))),
        "energy_temperature_steps": int(getattr(ood_detector, "energy_temperature_steps", 16)),
        "react_enabled": bool(getattr(ood_detector, "react_enabled", False)),
        "react_percentile": _float_field(getattr(ood_detector, "react_percentile", 0.99), default=0.99),
        "react_apply_during_calibration": bool(getattr(ood_detector, "react_apply_during_calibration", True)),
        "react_apply_during_inference": bool(getattr(ood_detector, "react_apply_during_inference", True)),
        "react_threshold": (
            None
            if getattr(ood_detector, "react_threshold", None) is None
            else _float_field(getattr(ood_detector, "react_threshold"), default=0.0)
        ),
    }
