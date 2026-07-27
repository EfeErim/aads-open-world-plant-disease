"""Minimal pytest configuration for the slimmed repo."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

RUNTIME_TMP_ROOT = (Path(__file__).resolve().parents[1] / ".runtime_tmp").resolve()
RUNTIME_TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DIR", str(RUNTIME_TMP_ROOT / "wandb"))

_ORIG_PATH_MKDIR = Path.mkdir
_ORIG_RMTREE = shutil.rmtree
_ORIG_TEMPORARY_DIRECTORY = tempfile.TemporaryDirectory
_PYTEST_PATHLIB = None
_PYTEST_TMPDIR = None
_ORIG_PYTEST_PATHLIB_CLEANUP = None
_ORIG_PYTEST_TMPDIR_CLEANUP = None


def _safe_path_mkdir(self, mode=0o777, parents=False, exist_ok=False):
    if os.name == "nt" and mode == 0o700:
        mode = 0o777
    return _ORIG_PATH_MKDIR(self, mode=mode, parents=parents, exist_ok=exist_ok)


def _safe_rmtree(path, *args, **kwargs):
    try:
        return _ORIG_RMTREE(path, *args, **kwargs)
    except PermissionError:
        return None
    except OSError:
        return None


class _SafeTemporaryDirectory(tempfile.TemporaryDirectory):
    def cleanup(self):
        try:
            super().cleanup()
        except PermissionError:
            return None
        except OSError:
            return None


Path.mkdir = _safe_path_mkdir
shutil.rmtree = _safe_rmtree
tempfile.tempdir = str(RUNTIME_TMP_ROOT)
tempfile.TemporaryDirectory = _SafeTemporaryDirectory

try:
    import _pytest.pathlib as _pytest_pathlib

    _PYTEST_PATHLIB = _pytest_pathlib
    _ORIG_PYTEST_PATHLIB_CLEANUP = _pytest_pathlib.cleanup_dead_symlinks

    def _patched_cleanup_dead_symlinks(root):
        try:
            return _ORIG_PYTEST_PATHLIB_CLEANUP(root)
        except PermissionError:
            return None

    _pytest_pathlib.cleanup_dead_symlinks = _patched_cleanup_dead_symlinks

    import _pytest.tmpdir as _pytest_tmpdir

    _PYTEST_TMPDIR = _pytest_tmpdir
    _ORIG_PYTEST_TMPDIR_CLEANUP = getattr(_pytest_tmpdir, "cleanup_dead_symlinks", None)
    _pytest_tmpdir.cleanup_dead_symlinks = _patched_cleanup_dead_symlinks
except Exception:
    pass


@pytest.fixture(scope="session")
def test_device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def test_seed():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory(dir=str(RUNTIME_TMP_ROOT)) as tmpdir:
        yield Path(tmpdir)


def pytest_configure(config):
    if not getattr(config.option, "basetemp", None):
        config.option.basetemp = str(RUNTIME_TMP_ROOT / f"pytest_{uuid.uuid4().hex}")
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


def pytest_sessionfinish(session, exitstatus):
    Path.mkdir = _ORIG_PATH_MKDIR
    shutil.rmtree = _ORIG_RMTREE
    tempfile.TemporaryDirectory = _ORIG_TEMPORARY_DIRECTORY
    if _PYTEST_PATHLIB is not None and _ORIG_PYTEST_PATHLIB_CLEANUP is not None:
        _PYTEST_PATHLIB.cleanup_dead_symlinks = _ORIG_PYTEST_PATHLIB_CLEANUP
    if _PYTEST_TMPDIR is not None and _ORIG_PYTEST_TMPDIR_CLEANUP is not None:
        _PYTEST_TMPDIR.cleanup_dead_symlinks = _ORIG_PYTEST_TMPDIR_CLEANUP
