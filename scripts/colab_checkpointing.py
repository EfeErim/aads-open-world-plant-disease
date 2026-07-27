#!/usr/bin/env python3
"""Compatibility alias for Colab checkpointing helpers."""

from __future__ import annotations

import sys

from src.training import colab_checkpointing as _runtime

sys.modules[__name__] = _runtime
