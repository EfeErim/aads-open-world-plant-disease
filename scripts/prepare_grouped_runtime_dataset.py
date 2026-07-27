#!/usr/bin/env python3
"""Compatibility CLI wrapper for grouped runtime dataset preparation."""

from __future__ import annotations

import sys

from src.data import grouped_runtime_dataset_preparation as _runtime

sys.modules[__name__] = _runtime


if __name__ == "__main__":
    raise SystemExit(_runtime.main())
