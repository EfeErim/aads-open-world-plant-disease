"""Pipeline package with lazy exports to avoid eager runtime imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["RouterAdapterRuntime"]


def __getattr__(name: str) -> Any:
    if name == "RouterAdapterRuntime":
        return import_module("src.pipeline.router_adapter_runtime").RouterAdapterRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
