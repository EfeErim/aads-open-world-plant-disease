#!/usr/bin/env python3
"""Build a local training-run registry for Pareto preparation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.services.run_registry import (  # noqa: E402
    build_pareto_inputs,
    build_run_registry,
    collect_trial_records,
    discover_artifact_roots,
    main,
    write_registry,
)

__all__ = [
    "build_pareto_inputs",
    "build_run_registry",
    "collect_trial_records",
    "discover_artifact_roots",
    "main",
    "write_registry",
]


if __name__ == "__main__":
    raise SystemExit(main())
