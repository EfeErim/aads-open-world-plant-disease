from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.ood_splits import ensure_ood_split_manifest
from src.shared.hash_utils import sha256_file


def _write_frozen_manifest(ood_root: Path, image: Path) -> Path:
    manifest_path = ood_root / "ood_split_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "v2_family_frozen_ood_split_manifest",
                "assignment_policy": "reviewed_explicit_family_assignment",
                "entries": {
                    image.relative_to(ood_root).as_posix(): {
                        "sha256": sha256_file(image),
                        "slice": "same_crop_unsupported_disease",
                        "ood_type": "same_crop_unsupported_disease",
                        "evidence_family_id": "family-one",
                        "split": "test",
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_frozen_family_manifest_is_reused_without_mutating_release_materialization(tmp_path: Path) -> None:
    ood_root = tmp_path / "ood"
    image = ood_root / "same_crop_unsupported_disease" / "sample.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"reviewed-image")
    manifest_path = _write_frozen_manifest(ood_root, image)
    original = manifest_path.read_bytes()

    result = ensure_ood_split_manifest(ood_root, seed=999, dev_fraction=0.25)

    assert result["schema_version"] == "v2_family_frozen_ood_split_manifest"
    assert manifest_path.read_bytes() == original


def test_frozen_family_manifest_rebinds_relocated_member_by_hash_without_writing(tmp_path: Path) -> None:
    ood_root = tmp_path / "ood"
    original_image = ood_root / "old_slice" / "sample.jpg"
    original_image.parent.mkdir(parents=True)
    original_image.write_bytes(b"reviewed-image")
    manifest_path = _write_frozen_manifest(ood_root, original_image)
    original_manifest = manifest_path.read_bytes()
    relocated_image = ood_root / "reviewed_slice" / "sample.jpg"
    relocated_image.parent.mkdir(parents=True)
    original_image.replace(relocated_image)

    result = ensure_ood_split_manifest(ood_root)

    assert set(result["entries"]) == {"reviewed_slice/sample.jpg"}
    assert result["entries"]["reviewed_slice/sample.jpg"]["split"] == "test"
    assert result["entries"]["reviewed_slice/sample.jpg"]["slice"] == "reviewed_slice"
    assert result["runtime_path_reconciled_by_sha256"] is True
    assert manifest_path.read_bytes() == original_manifest


def test_frozen_family_manifest_fails_closed_when_image_inventory_drifts(tmp_path: Path) -> None:
    ood_root = tmp_path / "ood"
    image = ood_root / "same_crop_unsupported_disease" / "sample.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"reviewed-image")
    manifest_path = _write_frozen_manifest(ood_root, image)
    original = manifest_path.read_bytes()
    (image.parent / "unexpected.jpg").write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="Frozen OOD split manifest does not match"):
        ensure_ood_split_manifest(ood_root)

    assert manifest_path.read_bytes() == original
