import json
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.augment_runtime_train_split import augment_runtime_train_split


def _write_image(path: Path, *, offset: int) -> None:
    image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((offset, offset, offset + 6, offset + 6), fill=(200, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def test_augment_runtime_train_split_only_adds_continual_variants(tmp_path: Path):
    source_root = tmp_path / "runtime" / "grape__fruit"
    output_root = tmp_path / "runtime" / "grape__fruit_train_aug"

    for split_name in ("continual", "val", "test"):
        _write_image(source_root / split_name / "healthy" / f"{split_name}_healthy.jpg", offset=3)
        _write_image(source_root / split_name / "powdery_mildew" / f"{split_name}_powdery.jpg", offset=8)
    _write_image(source_root / "ood" / "unknown" / "ood.jpg", offset=12)
    (source_root / "split_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "v1_grouped_runtime_layout",
                "dataset_key": "grape__fruit",
                "rows": [
                    {
                        "runtime_relative_path": "continual/healthy/continual_healthy.jpg",
                        "normalized_class_name": "healthy",
                        "split": "continual",
                    },
                    {"runtime_relative_path": "val/healthy/val_healthy.jpg", "normalized_class_name": "healthy", "split": "val"},
                    {"runtime_relative_path": "test/healthy/test_healthy.jpg", "normalized_class_name": "healthy", "split": "test"},
                    {
                        "runtime_relative_path": "continual/powdery_mildew/continual_powdery.jpg",
                        "normalized_class_name": "powdery_mildew",
                        "split": "continual",
                    },
                    {
                        "runtime_relative_path": "val/powdery_mildew/val_powdery.jpg",
                        "normalized_class_name": "powdery_mildew",
                        "split": "val",
                    },
                    {
                        "runtime_relative_path": "test/powdery_mildew/test_powdery.jpg",
                        "normalized_class_name": "powdery_mildew",
                        "split": "test",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = augment_runtime_train_split(
        source_root=source_root,
        output_root=output_root,
        variants_per_image=2,
        seed=123,
    )

    assert report["generated_image_count"] == 4
    assert len(list((output_root / "continual").rglob("*__aug_*.jpg"))) == 4
    assert not list((output_root / "val").rglob("*__aug_*.jpg"))
    assert not list((output_root / "test").rglob("*__aug_*.jpg"))
    assert (output_root / "ood" / "unknown" / "ood.jpg").exists()

    manifest = json.loads((output_root / "split_manifest.json").read_text(encoding="utf-8"))
    by_class = {row["class_name"]: row for row in manifest["classes"]}

    assert by_class["healthy"]["reference_image_count"] == 3
    assert by_class["powdery_mildew"]["reference_image_count"] == 3
    assert by_class["healthy"]["split_counts"] == {"continual": 3, "val": 1, "test": 1}
    assert by_class["healthy"]["non_augmented_split_counts"] == {"continual": 1, "val": 1, "test": 1}
    assert by_class["healthy"]["offline_augmented_count"] == 2
    assert manifest["offline_train_augmentation"]["leakage_policy"].startswith("val/test/ood are copied unchanged")

    generated_rows = [row for row in manifest["rows"] if row.get("generated_offline_augmentation")]
    assert len(generated_rows) == 4
    assert {row["split"] for row in generated_rows} == {"continual"}
    assert all(row["synthetic_hint"] for row in generated_rows)
    assert all(row["train_only_routed"] for row in generated_rows)
