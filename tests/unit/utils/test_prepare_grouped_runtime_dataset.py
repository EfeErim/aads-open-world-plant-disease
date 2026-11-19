import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.data.grouped_runtime_dataset_preparation import (
    _refresh_record_availability,
    build_grouped_dataset_plan,
    materialize_grouped_runtime_dataset,
    scan_class_root_dataset,
)


def _write_pattern(path: Path, *, offset: int) -> None:
    image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((offset, offset, offset + 7, offset + 7), fill=(255, 0, 0))
    draw.line((0, 31 - offset, 31, offset), fill=(0, 128, 0), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _fake_embeddings(paths, *, model_id: str, batch_size: int, device: str):
    size = len(paths)
    if size == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.eye(size, dtype=np.float32)


class _FakeModel:
    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self


def test_build_grouped_dataset_plan_blocks_cross_class_exact_duplicate(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    image_path = source_root / "Healthy" / "shared.jpg"
    _write_pattern(image_path, offset=2)
    duplicate_target = source_root / "Late Blight" / "shared.jpg"
    duplicate_target.parent.mkdir(parents=True, exist_ok=True)
    duplicate_target.write_bytes(image_path.read_bytes())

    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["runtime_ready"] is False
    assert summary["summary"]["cross_class_conflicts"] == 1
    assert (artifact_root / "cross_class_conflicts.csv").exists()


def test_materialize_grouped_runtime_dataset_writes_runtime_layout(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    runtime_root = tmp_path / "runtime"

    for index, offset in enumerate((2, 10, 18)):
        _write_pattern(source_root / "Healthy" / f"healthy_{index}.jpg", offset=offset)
        _write_pattern(source_root / "Early Blight" / f"disease_{index}.jpg", offset=offset + 1)

    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["runtime_ready"] is True

    result_root = materialize_grouped_runtime_dataset(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        runtime_root=runtime_root,
    )

    crop_root = result_root / "tomato"
    assert (crop_root / "continual").exists()
    assert (crop_root / "val").exists()
    assert (crop_root / "test").exists()
    assert (crop_root / "split_manifest.json").exists()
    manifest = json.loads((crop_root / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_policy"] == "grouped_family_canonical_eval_60_20_20"


def test_materialize_grouped_runtime_dataset_includes_optional_ood_tree(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    runtime_root = tmp_path / "runtime"
    ood_root = tmp_path / "ood_pool"

    for index, offset in enumerate((2, 10, 18)):
        _write_pattern(source_root / "Healthy" / f"healthy_{index}.jpg", offset=offset)
        _write_pattern(source_root / "Early Blight" / f"disease_{index}.jpg", offset=offset + 1)

    _write_pattern(ood_root / "unsupported_tomato" / "ood_a.jpg", offset=5)
    _write_pattern(ood_root / "background" / "nested" / "ood_b.jpg", offset=7)

    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["runtime_ready"] is True

    result_root = materialize_grouped_runtime_dataset(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        runtime_root=runtime_root,
        ood_root=ood_root,
    )

    crop_root = result_root / "tomato"
    manifest = json.loads((crop_root / "split_manifest.json").read_text(encoding="utf-8"))
    handoff = json.loads((artifact_root / "ood_handoff_checklist.json").read_text(encoding="utf-8"))

    assert (crop_root / "ood" / "unsupported_tomato" / "ood_a.jpg").exists()
    assert (crop_root / "ood" / "background" / "nested" / "ood_b.jpg").exists()
    assert manifest["ood"]["source_root"] == str(ood_root.resolve())
    assert manifest["ood"]["image_count"] == 2
    assert handoff["status"] == "materialized"
    assert handoff["image_count"] == 2


def test_source_like_group_bundles_eval_families_into_one_split(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"

    for class_name, offsets in {
        "Healthy": (2, 10, 18),
        "Early Blight": (3, 11, 19),
    }.items():
        _write_pattern(source_root / class_name / "source_a" / f"{class_name.lower().replace(' ', '_')}_a.jpg", offset=offsets[0])
        _write_pattern(source_root / class_name / "source_a" / f"{class_name.lower().replace(' ', '_')}_b.jpg", offset=offsets[1])
        _write_pattern(source_root / class_name / "source_b" / f"{class_name.lower().replace(' ', '_')}_c.jpg", offset=offsets[2])
        _write_pattern(source_root / class_name / "source_c" / f"{class_name.lower().replace(' ', '_')}_d.jpg", offset=offsets[2] + 5)

    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._compute_neighbor_pairs",
        lambda embeddings, *, paths, neighbors: {},
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["runtime_ready"] is True
    manifest_rows = json.loads((artifact_root / "proposed_split_manifest.json").read_text(encoding="utf-8"))["rows"]
    source_a_eval_rows = [
        row for row in manifest_rows
        if row["source_like_group"].startswith("hint:source_a") and row["is_family_canonical"]
    ]
    assert source_a_eval_rows
    assert len({row["split"] for row in source_a_eval_rows}) == 1


def test_eval_risk_families_are_continual_only(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    runtime_root = tmp_path / "runtime"

    for index, offset in enumerate((2, 10, 18)):
        _write_pattern(source_root / "Healthy" / f"healthy_clean_{index}.jpg", offset=offset)
        _write_pattern(source_root / "Early Blight" / f"disease_clean_{index}.jpg", offset=offset + 1)

    _write_pattern(source_root / "Healthy" / "Ekran görüntüsü 2026-03-04 001701.jpg", offset=23)
    _write_pattern(source_root / "Early Blight" / "pngtree-preview-image.jpg", offset=24)

    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._compute_neighbor_pairs",
        lambda embeddings, *, paths, neighbors: {},
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )
    assert summary["runtime_ready"] is True
    assert summary["summary"]["eval_risk_images"] == 2

    manifest_rows = json.loads((artifact_root / "proposed_split_manifest.json").read_text(encoding="utf-8"))["rows"]
    risky_rows = [row for row in manifest_rows if row["eval_quality_risk"]]
    assert risky_rows
    assert all(row["split"] == "continual" for row in risky_rows)

    result_root = materialize_grouped_runtime_dataset(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        runtime_root=runtime_root,
    )
    crop_root = result_root / "tomato"
    eval_files = {
        path.name
        for split_name in ("val", "test")
        for path in (crop_root / split_name).rglob("*.*")
        if path.is_file()
    }
    assert "Ekran görüntüsü 2026-03-04 001701.jpg" not in eval_files
    assert "pngtree-preview-image.jpg" not in eval_files


def test_materialized_eval_splits_only_contain_canonical_clean_family_members(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    runtime_root = tmp_path / "runtime"

    for index, offset in enumerate((2, 10, 18)):
        _write_pattern(source_root / "Healthy" / f"healthy_clean_{index}.jpg", offset=offset)
        _write_pattern(source_root / "Early Blight" / f"disease_clean_{index}.jpg", offset=offset + 1)

    _write_pattern(source_root / "Healthy" / "healthy_real.jpg", offset=22)
    _write_pattern(source_root / "Healthy" / "healthy_aug_flip.jpg", offset=23)
    _write_pattern(source_root / "Early Blight" / "disease_real.jpg", offset=24)
    _write_pattern(source_root / "Early Blight" / "disease_aug_flip.jpg", offset=25)

    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._compute_neighbor_pairs",
        lambda embeddings, *, paths, neighbors: {
            tuple(sorted((real_path, aug_path))): 0.99
            for real_path in paths
            for aug_path in paths
            if "real" in real_path and "aug" in aug_path
        },
    )
    monkeypatch.setattr(
        "src.data.grouped_runtime_dataset_preparation._phash_distance",
        lambda a, b: 6,
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["runtime_ready"] is True

    result_root = materialize_grouped_runtime_dataset(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        runtime_root=runtime_root,
    )

    crop_root = result_root / "tomato"
    eval_files = {
        path.relative_to(crop_root).as_posix()
        for split_name in ("val", "test")
        for path in (crop_root / split_name).rglob("*.jpg")
    }
    assert all("aug" not in path for path in eval_files)

    manifest_rows = json.loads((artifact_root / "proposed_split_manifest.json").read_text(encoding="utf-8"))["rows"]
    synthetic_rows = [row for row in manifest_rows if row["family_role"] == "canonical_with_synthetic_derivatives"]
    assert synthetic_rows
    assert all(row["split"] == "continual" for row in synthetic_rows if not row["is_family_canonical"])


def test_scan_marks_unreadable_images_as_excluded(tmp_path: Path):
    source_root = tmp_path / "source"
    _write_pattern(source_root / "Healthy" / "good.png", offset=2)
    broken = source_root / "Healthy" / "broken.jpg"
    broken.write_bytes(b"not an image")

    records, _ = scan_class_root_dataset(class_root=source_root, crop_name="tomato")

    by_name = {Path(record.relative_path).name: record for record in records}
    assert by_name["broken.jpg"].readable is False
    assert by_name["broken.jpg"].excluded_reason == "unreadable"
    assert by_name["good.png"].readable is True
    assert by_name["good.png"].excluded_reason == ""


def test_refresh_record_availability_excludes_images_corrupted_after_scan(tmp_path: Path):
    source_root = tmp_path / "source"
    image_path = source_root / "Healthy" / "good.png"
    _write_pattern(image_path, offset=2)
    records, _ = scan_class_root_dataset(class_root=source_root, crop_name="tomato")
    image_path.write_bytes(b"corrupted later")

    counts = _refresh_record_availability(records)

    assert counts == {"missing_after_scan": 0, "unreadable_after_scan": 1}
    assert records[0].readable is False
    assert records[0].excluded_reason == "unreadable_after_scan"
