"""Workflow package with lazy exports to avoid cross-surface eager imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["InferenceWorkflow", "TrainingWorkflow", "TrainingWorkflowResult"]


def __getattr__(name: str) -> Any:
    if name == "InferenceWorkflow":
        return import_module("src.workflows.inference").InferenceWorkflow
    if name in {"TrainingWorkflow", "TrainingWorkflowResult"}:
        module = import_module("src.workflows.training")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
