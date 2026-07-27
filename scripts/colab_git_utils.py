"""Compatibility alias for Colab Git utilities."""

from __future__ import annotations

import sys

from src.notebook import git_utils as _runtime

sys.modules[__name__] = _runtime
