import json
from pathlib import Path

import pytest

from src.training.services.runtime_dataset import resolve_runtime_dataset


def _write_manifest(root: Path, *, crop_name: str, part_name: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "split_manifest.json").write_text(
        json.dumps(
            {
                "crop_name": crop_name,
                "part_name": part_name or "unspecified",
                "dataset_key": root.name,
            }
        ),
        encoding="utf-8",
    )


def test_resolve_runtime_dataset_accepts_exact_dataset_root(tmp_path: Path):
    dataset_root = tmp_path / "tomato__fruit"
    for split_name in ("continual", "val", "test"):
        (dataset_root / split_name / "healthy").mkdir(parents=True, exist_ok=True)
    _write_manifest(dataset_root, crop_name="tomato", part_name="fruit")

    resolved = resolve_runtime_dataset(data_dir=dataset_root, crop_name="tomato")

    assert resolved.dataset_key == "tomato__fruit"
    assert resolved.crop_root == dataset_root
    assert resolved.resolution_source == "exact_root"


def test_resolve_runtime_dataset_rejects_ambiguous_parent_root(tmp_path: Path):
    (tmp_path / "tomato" / "continual" / "healthy").mkdir(parents=True, exist_ok=True)
    dataset_root = tmp_path / "tomato__fruit"
    for split_name in ("continual", "val", "test"):
        (dataset_root / split_name / "healthy").mkdir(parents=True, exist_ok=True)
    _write_manifest(dataset_root, crop_name="tomato", part_name="fruit")

    with pytest.raises(ValueError, match="Multiple runtime datasets matched crop 'tomato'"):
        resolve_runtime_dataset(data_dir=tmp_path, crop_name="tomato")
