#!/usr/bin/env python3
"""Compatibility CLI wrapper for Colab router-adapter inference."""

from __future__ import annotations

from PIL import Image

from src.core.config_manager import get_config
from src.pipeline import colab_router_adapter_inference as _runtime
from src.pipeline.colab_router_adapter_inference import *  # noqa: F403
from src.router.router_pipeline import RouterPipeline

_runtime_run_inference = _runtime.run_inference
_runtime_main = _runtime.main


def _sync_runtime_dependencies() -> None:
    """Forward compatibility-module monkeypatches to the canonical runtime."""
    _runtime.get_config = get_config
    _runtime.RouterPipeline = RouterPipeline
    _runtime.Image = Image


def run_inference(*args, **kwargs):
    _sync_runtime_dependencies()
    return _runtime_run_inference(*args, **kwargs)


def main() -> int:
    _sync_runtime_dependencies()
    previous = _runtime.run_inference
    _runtime.run_inference = run_inference
    try:
        return _runtime_main()
    finally:
        _runtime.run_inference = previous


def __getattr__(name: str) -> object:
    return getattr(_runtime, name)


if __name__ == "__main__":
    raise SystemExit(main())
