#!/usr/bin/env python3
"""Compatibility wrapper for Notebook 2 training/runtime helpers."""

from __future__ import annotations

import sys

from src.training import notebook_runtime_helpers as _runtime

sys.modules[__name__] = _runtime
