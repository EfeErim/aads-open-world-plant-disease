"""Training package with lazy exports to avoid eager trainer imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ContinualSDLoRAConfig",
    "ContinualSDLoRATrainer",
    "ContinualTrainingSession",
    "TrainBatchStats",
    "TrainingCheckpointPayload",
    "TrainingHistory",
    "TrainingProgressState",
    "ValidationReport",
    "evaluate_model",
]

_EXPORTS = {
    "ContinualSDLoRAConfig": ("src.training.continual_sd_lora", "ContinualSDLoRAConfig"),
    "ContinualSDLoRATrainer": ("src.training.continual_sd_lora", "ContinualSDLoRATrainer"),
    "ContinualTrainingSession": ("src.training.session", "ContinualTrainingSession"),
    "TrainBatchStats": ("src.training.types", "TrainBatchStats"),
    "TrainingCheckpointPayload": ("src.training.types", "TrainingCheckpointPayload"),
    "TrainingHistory": ("src.training.types", "TrainingHistory"),
    "TrainingProgressState": ("src.training.types", "TrainingProgressState"),
    "ValidationReport": ("src.training.types", "ValidationReport"),
    "evaluate_model": ("src.training.validation", "evaluate_model"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name), attribute_name)
