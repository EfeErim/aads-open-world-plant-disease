#!/usr/bin/env python3
"""Compatibility CLI wrapper for Colab ROI ablation helpers."""

from __future__ import annotations

from src.pipeline import colab_roi_ablation_runtime as _runtime
from src.pipeline.colab_roi_ablation_runtime import *  # noqa: F403
from src.pipeline.colab_roi_ablation_runtime import main


def __getattr__(name: str) -> object:
    return getattr(_runtime, name)


if __name__ == "__main__":
    raise SystemExit(main())
