from pathlib import Path

from src.data.datasets import CropDataset, infer_crop_classes_from_layout
from tests.utils.test_helpers import make_image


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    make_image(path, size=(4, 4), color=(255, 0, 0))


def test_infer_crop_classes_from_layout_reads_split_dirs(tmp_path: Path):
    root = tmp_path / "runtime"
    crop = "tomato leaf"

    for split in ("continual", "val", "test"):
        for cls in ("bacterial_spot", "healthy", "late_blight"):
            target = root / crop / split / cls
            target.mkdir(parents=True, exist_ok=True)

    classes = infer_crop_classes_from_layout(str(root), crop)

    assert classes == ["bacterial_spot", "healthy", "late_blight"]


def test_crop_dataset_uses_inferred_classes_for_custom_crop_name(tmp_path: Path):
    root = tmp_path / "runtime"
    crop = "tomato leaf"

    class_to_count = {
        "bacterial_spot": 2,
        "healthy": 2,
        "late_blight": 2,
    }

    for cls, count in class_to_count.items():
        cls_dir = root / crop / "val" / cls
        for i in range(count):
            _write_image(cls_dir / f"img_{i}.jpg")

    dataset = CropDataset(
        data_dir=str(root),
        crop=crop,
        split="val",
        transform=False,
        use_cache=False,
    )

    assert dataset.classes == ["bacterial_spot", "healthy", "late_blight"]
    assert set(dataset.labels) == {0, 1, 2}
    assert len(dataset) == sum(class_to_count.values())
