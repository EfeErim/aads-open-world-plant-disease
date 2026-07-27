"""Composable helpers for the continual training surface."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "amp_scaler_enabled",
    "autocast_context",
    "build_adapter_metadata",
    "build_grad_scaler",
    "capture_rng_state",
    "compute_config_hash",
    "compute_plan_metrics",
    "configure_runtime_reproducibility",
    "load_plan_targets",
    "resolve_amp_dtype",
    "resolve_session_num_epochs",
    "restore_ood_state",
    "restore_rng_state",
    "serialize_ood_state",
    "validate_plan_metrics",
    "write_plan_metric_artifact",
]

_EXPORTS = {
    "amp_scaler_enabled": ("src.training.services.runtime", "amp_scaler_enabled"),
    "autocast_context": ("src.training.services.runtime", "autocast_context"),
    "build_adapter_metadata": ("src.training.services.persistence", "build_adapter_metadata"),
    "build_grad_scaler": ("src.training.services.runtime", "build_grad_scaler"),
    "capture_rng_state": ("src.training.services.persistence", "capture_rng_state"),
    "compute_config_hash": ("src.training.services.persistence", "compute_config_hash"),
    "compute_plan_metrics": ("src.training.services.metrics", "compute_plan_metrics"),
    "configure_runtime_reproducibility": ("src.training.services.runtime", "configure_runtime_reproducibility"),
    "load_plan_targets": ("src.training.services.metrics", "load_plan_targets"),
    "resolve_amp_dtype": ("src.training.services.runtime", "resolve_amp_dtype"),
    "resolve_session_num_epochs": ("src.training.services.runtime", "resolve_session_num_epochs"),
    "restore_ood_state": ("src.training.services.persistence", "restore_ood_state"),
    "restore_rng_state": ("src.training.services.persistence", "restore_rng_state"),
    "serialize_ood_state": ("src.training.services.persistence", "serialize_ood_state"),
    "validate_plan_metrics": ("src.training.services.metrics", "validate_plan_metrics"),
    "write_plan_metric_artifact": ("src.training.services.metrics", "write_plan_metric_artifact"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name), attribute_name)
