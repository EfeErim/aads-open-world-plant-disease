"""Keep pytest temp files under .runtime_tmp and gate integration tests behind --runintegration."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

RUNTIME_TMP_ROOT = (Path(__file__).resolve().parents[1] / ".runtime_tmp").resolve()
RUNTIME_TMP_ROOT.mkdir(parents=True, exist_ok=True)

def pytest_configure(config):
    if not getattr(config.option, "basetemp", None):
        config.option.basetemp = str(RUNTIME_TMP_ROOT / f"p{uuid.uuid4().hex[:6]}")


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

