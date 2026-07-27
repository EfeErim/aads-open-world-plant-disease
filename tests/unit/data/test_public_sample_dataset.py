from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.public_sample_dataset import PUBLIC_SAMPLE_SCHEMA, materialize_public_sample_dataset


def test_public_sample_dataset_materializes_complete_runtime_layout(tmp_path: Path) -> None:
    result = materialize_public_sample_dataset(tmp_path, target="tomato__leaf")
    target_root = Path(result["target_root"])

    assert result["created"] is True
    assert result["production_eligible"] is False
    assert result["sample_count"] == 56
    for split in ("continual", "val", "test", "ood", "oe"):
        assert (target_root / split).is_dir()
    manifest = json.loads((target_root / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == PUBLIC_SAMPLE_SCHEMA
    assert manifest["synthetic"] is True
    assert manifest["production_eligible"] is False


def test_public_sample_dataset_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    first = materialize_public_sample_dataset(tmp_path, target="grape__fruit")
    manifest_path = Path(first["manifest_path"])
    first_bytes = manifest_path.read_bytes()

    second = materialize_public_sample_dataset(tmp_path, target="grape__fruit")

    assert second["created"] is False
    assert manifest_path.read_bytes() == first_bytes


def test_public_sample_dataset_rejects_invalid_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="<crop>__<part>"):
        materialize_public_sample_dataset(tmp_path, target="../tomato")
