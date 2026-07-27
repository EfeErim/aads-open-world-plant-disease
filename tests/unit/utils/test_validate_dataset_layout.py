import json
from pathlib import Path

from scripts.validate_dataset_layout import validate_runtime_dataset_layout


def _touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def _write_runtime_dataset(root: Path, rows: list[dict]) -> None:
    for split in ("continual", "val", "test"):
        _touch_image(root / split / "healthy" / f"{split}.jpg")
    (root / "split_manifest.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")


def test_validate_runtime_dataset_layout_accepts_family_safe_manifest(tmp_path: Path):
    root = tmp_path / "runtime"
    _write_runtime_dataset(
        root,
        [
            {
                "runtime_relative_path": "continual/healthy/continual.jpg",
                "normalized_class_name": "healthy",
                "split": "continual",
                "family_id": "healthy__a",
            },
            {
                "runtime_relative_path": "val/healthy/val.jpg",
                "normalized_class_name": "healthy",
                "split": "val",
                "family_id": "healthy__b",
            },
            {
                "runtime_relative_path": "test/healthy/test.jpg",
                "normalized_class_name": "healthy",
                "split": "test",
                "family_id": "healthy__c",
            },
        ],
    )

    result = validate_runtime_dataset_layout(root)

    assert result["ok"] is True
    assert result["summary"]["families_checked"] == 3
    assert result["summary"]["leakage_family_count"] == 0


def test_validate_runtime_dataset_layout_rejects_family_cross_split_leakage(tmp_path: Path):
    root = tmp_path / "runtime"
    _write_runtime_dataset(
        root,
        [
            {
                "runtime_relative_path": "continual/healthy/continual.jpg",
                "normalized_class_name": "healthy",
                "split": "continual",
                "family_id": "healthy__same",
            },
            {
                "runtime_relative_path": "val/healthy/val.jpg",
                "normalized_class_name": "healthy",
                "split": "val",
                "family_id": "healthy__same",
            },
            {
                "runtime_relative_path": "test/healthy/test.jpg",
                "normalized_class_name": "healthy",
                "split": "test",
                "family_id": "healthy__other",
            },
        ],
    )

    result = validate_runtime_dataset_layout(root)

    assert result["ok"] is False
    assert result["summary"]["leakage_family_count"] == 1
    assert any("Family leakage detected" in item for item in result["errors"])


def test_validate_runtime_dataset_layout_prefers_precise_family_bundle_key(tmp_path: Path):
    root = tmp_path / "runtime"
    _write_runtime_dataset(
        root,
        [
            {
                "runtime_relative_path": "continual/healthy/continual.jpg",
                "normalized_class_name": "healthy",
                "split": "continual",
                "family_id": "healthy__coarse",
                "family_bundle_key": "family:healthy/continual.jpg",
            },
            {
                "runtime_relative_path": "val/healthy/val.jpg",
                "normalized_class_name": "healthy",
                "split": "val",
                "family_id": "healthy__coarse",
                "family_bundle_key": "family:healthy/val.jpg",
            },
            {
                "runtime_relative_path": "test/healthy/test.jpg",
                "normalized_class_name": "healthy",
                "split": "test",
                "family_id": "healthy__coarse",
                "family_bundle_key": "family:healthy/test.jpg",
            },
        ],
    )

    result = validate_runtime_dataset_layout(root)

    assert result["ok"] is True
    assert result["summary"]["families_checked"] == 3
    assert result["summary"]["leakage_family_count"] == 0


def test_validate_runtime_dataset_layout_rejects_bundle_key_cross_split_leakage(tmp_path: Path):
    root = tmp_path / "runtime"
    _write_runtime_dataset(
        root,
        [
            {
                "runtime_relative_path": "continual/healthy/continual.jpg",
                "normalized_class_name": "healthy",
                "split": "continual",
                "family_id": "healthy__coarse",
                "family_bundle_key": "family:healthy/shared-source.jpg",
            },
            {
                "runtime_relative_path": "val/healthy/val.jpg",
                "normalized_class_name": "healthy",
                "split": "val",
                "family_id": "healthy__coarse",
                "family_bundle_key": "family:healthy/shared-source.jpg",
            },
            {
                "runtime_relative_path": "test/healthy/test.jpg",
                "normalized_class_name": "healthy",
                "split": "test",
                "family_id": "healthy__other",
                "family_bundle_key": "family:healthy/test.jpg",
            },
        ],
    )

    result = validate_runtime_dataset_layout(root)

    assert result["ok"] is False
    assert result["summary"]["leakage_family_count"] == 1
    assert any("Family leakage detected" in item for item in result["errors"])


def test_validate_runtime_dataset_layout_rejects_generated_eval_rows_and_files(tmp_path: Path):
    root = tmp_path / "runtime"
    _write_runtime_dataset(
        root,
        [
            {
                "runtime_relative_path": "continual/healthy/continual.jpg",
                "normalized_class_name": "healthy",
                "split": "continual",
                "family_id": "healthy__a",
            },
            {
                "runtime_relative_path": "val/healthy/_offline_aug/val_aug.jpg",
                "normalized_class_name": "healthy",
                "split": "val",
                "family_id": "healthy__b",
                "generated_offline_augmentation": True,
            },
            {
                "runtime_relative_path": "test/healthy/test.jpg",
                "normalized_class_name": "healthy",
                "split": "test",
                "family_id": "healthy__c",
            },
        ],
    )
    _touch_image(root / "val" / "healthy" / "_offline_aug" / "val_aug.jpg")

    result = validate_runtime_dataset_layout(root)

    assert result["ok"] is False
    assert result["summary"]["generated_rows_outside_train"] == 1
    assert any("Generated offline augmentation rows" in item for item in result["errors"])
    assert any("generated augmentation files under val" in item for item in result["errors"])
