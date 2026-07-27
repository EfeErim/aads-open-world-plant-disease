#!/usr/bin/env python3
"""Compatibility alias for Colab repo bootstrap helpers."""

from __future__ import annotations

import sys

from src.notebook import repo_bootstrap as _runtime

sys.modules[__name__] = _runtime
