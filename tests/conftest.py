"""Minimal pytest configuration for the slimmed repo."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

RUNTIME_TMP_ROOT = (Path(__file__).resolve().parents[1] / ".runtime_tmp").resolve()
RUNTIME_TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DIR", str(RUNTIME_TMP_ROOT / "wandb"))

def pytest_configure(config):
    if not getattr(config.option, "basetemp", None):
        config.option.basetemp = str(RUNTIME_TMP_ROOT / f"p{uuid.uuid4().hex[:6]}")
    config.addinivalue_line("markers", "integration: integration tests")


def pytest_addoption(parser):
    parser.addoption(
        "--runintegration",
        action="store_true",
        default=False,
        help="run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runintegration"):
        return
    skip_integration = pytest.mark.skip(reason="need --runintegration option to run")
    for item in items:
        normalized = str(item.fspath).replace("\\", "/")
        if "/tests/integration/" in normalized:
            item.add_marker(skip_integration)

