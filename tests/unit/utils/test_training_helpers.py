import types
from pathlib import Path

from src.utils import training_helpers as th


def test_loader_size_with_sequence():
    seq = [1, 2, 3]
    assert th.loader_size(seq) == 3


def test_loader_size_with_none():
    assert th.loader_size(None) == 0


def test_stringify_paths_converts_path():
    p = Path("/tmp/example")
    assert th.stringify_paths(p) == str(p)


def test_build_loader_sizes_with_mapping():
    fake_loader = types.SimpleNamespace(dataset=[1, 2, 3])
    sizes = th.build_loader_sizes({"train": fake_loader, "val": None})
    assert sizes["train"] == 3
    assert sizes["val"] == 0
