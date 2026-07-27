import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.prepare_grouped_runtime_dataset import (
    ImageRecord,
    _compute_neighbor_pairs,
    _encode_with_oom_retries,
    _estimate_grouped_split_counts,
    _has_synthetic_hint,
    _infer_source_like_group,
    _resolve_embedding_device,
    build_grouped_dataset_plan,
    build_human_review_packet,
    build_label_review_summary,
    build_prepared_dataset_key,
    format_human_review_packet,
    materialize_grouped_runtime_dataset,
    normalize_prepared_class_name,
    resolve_safe_embedding_batch_size,
    scan_class_root_dataset,
)


def _write_pattern(path: Path, *, offset: int) -> None:
    image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((offset, offset, offset + 7, offset + 7), fill=(255, 0, 0))
    draw.line((0, 31 - offset, 31, offset), fill=(0, 128, 0), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _fake_embeddings(paths, *, model_id: str, batch_size: int, device: str):  # noqa: ARG001
    size = len(paths)
    if size == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.eye(size, dtype=np.float32)


def test_encode_with_oom_retries_forwards_batch_progress_and_reduces_batch_size(monkeypatch):
    attempts = []
    batch_updates = []
    status_updates = []

    def encode(paths, *, batch_size, progress_fn):
        attempts.append(batch_size)
        if batch_size == 4:
            raise RuntimeError("CUDA out of memory")
        progress_fn(len(paths), len(paths))
        return np.ones((len(paths), 1), dtype=np.float32)

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._is_cuda_oom",
        lambda exc: "out of memory" in str(exc).lower(),
    )
    result = _encode_with_oom_retries(
        encode,
        paths=[Path("a.jpg"), Path("b.jpg")],
        batch_size=4,
        progress_fn=status_updates.append,
        batch_progress_fn=lambda done, total: batch_updates.append((done, total)),
        label="test encoder",
    )

    assert result.shape == (2, 1)
    assert attempts == [4, 2]
    assert batch_updates == [(2, 2)]
    assert status_updates == ["test encoder CUDA OOM at batch_size=4; retrying with batch_size=2."]


class _FakeModel:
    def eval(self):
        return self

    def to(self, *args, **kwargs):  # noqa: ARG002
        return self


def test_normalize_prepared_class_name_uses_taxonomy_aliases():
    normalized = normalize_prepared_class_name(
        "Tomato Healthy Leaf",
        crop_name="tomato",
        expected_classes={"healthy", "early_blight"},
    )

    assert normalized == "healthy"


def test_build_prepared_dataset_key_includes_part_only_when_specified():
    assert build_prepared_dataset_key("tomato", "unspecified") == "tomato"
    assert build_prepared_dataset_key("Tomato", "Fruit") == "tomato__fruit"


def test_synthetic_hint_ignores_class_name_rot_token():
    assert not _has_synthetic_hint("botrytis_bunch_rot/Botrytis-cinerea.jpg")
    assert not _has_synthetic_hint("botrytis_bunch_rot/0893d858-botrite_grappolo42.jpg")
    assert _has_synthetic_hint("healthy/image_rot_90.jpg")
    assert _has_synthetic_hint("healthy/image_rot90.jpg")
    assert _has_synthetic_hint("healthy/image_aug_flip.jpg")


def test_grouped_split_counts_use_60_20_20_targets():
    assert _estimate_grouped_split_counts(0) == (0, 0, 0)
    assert _estimate_grouped_split_counts(2) == (2, 0, 0)
    assert _estimate_grouped_split_counts(3) == (1, 1, 1)
    assert _estimate_grouped_split_counts(10) == (6, 2, 2)
    assert _estimate_grouped_split_counts(39) == (23, 8, 8)


def test_infer_source_like_group_prefers_capture_group_and_web_signals():
    captured = ImageRecord(
        relative_path="Healthy/source_a/img001.jpg",
        absolute_path="C:/tmp/img001.jpg",
        raw_class_name="Healthy",
        normalized_class_name="healthy",
        source_hint="source_a",
        source_like_group="unknown",
        synthetic_hint=False,
        eval_quality_risk=False,
        readable=True,
        width=256,
        height=256,
        blur_score=1.0,
        brightness_mean=0.5,
        exact_hash="a",
        phash_hex="0" * 16,
        class_order_index=0,
    )
    web = ImageRecord(
        relative_path="Healthy/istockphoto-1320751459-612x612.jpg",
        absolute_path="C:/tmp/istock.jpg",
        raw_class_name="Healthy",
        normalized_class_name="healthy",
        source_hint="unknown",
        source_like_group="unknown",
        synthetic_hint=False,
        eval_quality_risk=False,
        readable=True,
        width=256,
        height=256,
        blur_score=1.0,
        brightness_mean=0.5,
        exact_hash="b",
        phash_hex="1" * 16,
        class_order_index=1,
    )

    assert _infer_source_like_group(captured) == "hint:source_a"
    assert _infer_source_like_group(web) == "web:istockphoto:1320751459"


def test_compute_neighbor_pairs_skips_non_finite_embedding_rows():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [np.nan, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    paths = ["a.jpg", "b.jpg", "c.jpg"]

    pairs = _compute_neighbor_pairs(embeddings, paths=paths, neighbors=2)

    # Non-finite row (b.jpg) should be ignored instead of crashing NearestNeighbors.
    assert all("b.jpg" not in pair for pair in pairs)
    assert ("a.jpg", "c.jpg") in pairs


def test_resolve_embedding_device_falls_back_when_cuda_unavailable(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert _resolve_embedding_device("cuda") == "cpu"
    assert _resolve_embedding_device("cuda:0") == "cpu"
    assert _resolve_embedding_device("cpu") == "cpu"


def test_resolve_embedding_device_falls_back_when_cuda_probe_fails(monkeypatch):
    import torch

    def _raise_cuda_probe_error():
        raise RuntimeError("CUDA driver probe failed")

    monkeypatch.setattr(torch.cuda, "is_available", _raise_cuda_probe_error)

    assert _resolve_embedding_device("cuda") == "cpu"
    assert _resolve_embedding_device("cuda:0") == "cpu"
    assert _resolve_embedding_device("cpu") == "cpu"


def test_resolve_safe_embedding_batch_size_uses_t4_safe_default(monkeypatch):
    import torch

    class _Props:
        name = "Tesla T4"
        total_memory = 15 * 1024**3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: _Props())

    assert resolve_safe_embedding_batch_size("cuda") == 4


def test_resolve_safe_embedding_batch_size_respects_manual_override(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_safe_embedding_batch_size("cuda", requested_batch_size=2) == 2


def test_build_grouped_dataset_plan_blocks_cross_class_exact_duplicate(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    image_path = source_root / "Healthy" / "shared.jpg"
    _write_pattern(image_path, offset=2)
    duplicate_target = source_root / "Late Blight" / "shared.jpg"
    duplicate_target.parent.mkdir(parents=True, exist_ok=True)
    duplicate_target.write_bytes(image_path.read_bytes())

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
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
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
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


def test_materialize_grouped_runtime_dataset_uses_part_aware_dataset_key(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    runtime_root = tmp_path / "runtime"

    for index, offset in enumerate((2, 10, 18)):
        _write_pattern(source_root / "Healthy" / f"healthy_{index}.jpg", offset=offset)
        _write_pattern(source_root / "Early Blight" / f"disease_{index}.jpg", offset=offset + 1)

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        part_name="fruit",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["runtime_ready"] is True
    assert summary["part_name"] == "fruit"
    assert Path(summary["prepared_runtime_root"]).name == "tomato__fruit"
    proposed = json.loads((artifact_root / "proposed_split_manifest.json").read_text(encoding="utf-8"))
    assert proposed["part_name"] == "fruit"
    assert proposed["dataset_key"] == "tomato__fruit"

    result_root = materialize_grouped_runtime_dataset(
        class_root=source_root,
        crop_name="tomato",
        part_name="fruit",
        artifact_root=artifact_root,
        runtime_root=runtime_root,
    )

    crop_root = result_root / "tomato__fruit"
    manifest = json.loads((crop_root / "split_manifest.json").read_text(encoding="utf-8"))

    assert (crop_root / "continual").exists()
    assert manifest["crop_name"] == "tomato"
    assert manifest["part_name"] == "fruit"
    assert manifest["dataset_key"] == "tomato__fruit"


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
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
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


def test_review_candidates_include_adjacency_ranking_fields(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    _write_pattern(source_root / "Healthy" / "source_a" / "healthy_0.jpg", offset=2)
    _write_pattern(source_root / "Healthy" / "source_b" / "healthy_1.jpg", offset=4)
    _write_pattern(source_root / "Healthy" / "source_c" / "healthy_2.jpg", offset=20)

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._compute_neighbor_pairs",
        lambda embeddings, *, paths, neighbors: {
            tuple(sorted((paths[0], paths[1]))): 0.97,
            tuple(sorted((paths[0], paths[2]))): 0.966,
        },
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["summary"]["adjacency_used_for_review_ranking_only"] is True
    review_csv = artifact_root / "same_class_review_candidates.csv"
    with review_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "adjacency_distance" in rows[0]
    assert "review_rank" in rows[0]
    assert int(rows[0]["adjacency_distance"]) <= int(rows[-1]["adjacency_distance"])
    assert rows[0]["triage_resolution"] == "manual_review"

    label_summary = json.loads((artifact_root / "label_risk_summary.json").read_text(encoding="utf-8"))
    with (artifact_root / "label_review_candidates.csv").open("r", encoding="utf-8", newline="") as handle:
        label_rows = list(csv.DictReader(handle))
    assert label_summary["review_candidate_count"] == len(label_rows)
    assert label_rows
    assert {row["label_risk_level"] for row in label_rows} == {"review_candidate"}

    packet = json.loads((artifact_root / "human_review_packet.json").read_text(encoding="utf-8"))
    label_review_summary = json.loads((artifact_root / "label_review_summary.json").read_text(encoding="utf-8"))
    assert packet["pause_recommended"] is True
    assert packet["recommended_action"] in {
        "prepare_clean_working_copy_or_stop",
        "confirm_train_only_routing_before_materialization",
    }
    assert any(point["id"] == "label_or_family_review_queue" for point in packet["decision_points"])
    assert "label_review_candidates.csv" in packet["review_artifacts"]
    assert "label_review_summary.json" in packet["review_artifacts"]
    assert "dino_auto_min" in packet["threshold_policy"]
    rebuilt = build_human_review_packet(summary, artifact_root=artifact_root, max_review_items=1)
    assert rebuilt["counts"]["label_review_candidates"] == len(label_rows)
    rebuilt_label_review_summary = build_label_review_summary(
        summary,
        label_risk_summary=label_summary,
        human_review_packet=rebuilt,
        label_review_candidates=label_rows,
        max_preview_items=1,
    )
    assert label_review_summary["human_in_the_loop"]["enabled"] is True
    assert label_review_summary["review_queue"]["path"] == "label_review_candidates.csv"
    assert label_review_summary["counts"]["label_review_candidates"] == len(label_rows)
    assert rebuilt_label_review_summary["review_queue"]["preview"] == [label_rows[0]]
    rendered = format_human_review_packet(rebuilt)
    assert "Notebook 0 audit gate" in rendered
    assert "label_review_candidates" in rendered


def test_low_risk_same_class_review_clusters_are_auto_resolved(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    _write_pattern(source_root / "Healthy" / "healthy_aug_0.jpg", offset=2)
    _write_pattern(source_root / "Healthy" / "healthy_aug_1.jpg", offset=3)
    _write_pattern(source_root / "Healthy" / "healthy_aug_2.jpg", offset=4)

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._compute_neighbor_pairs",
        lambda embeddings, *, paths, neighbors: {
            tuple(sorted((paths[0], paths[1]))): 0.97,
            tuple(sorted((paths[1], paths[2]))): 0.969,
        },
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._phash_distance",
        lambda a, b: 6,
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["summary"]["same_class_review_pairs_total"] == 2
    assert summary["summary"]["same_class_auto_resolved_clusters"] == 1
    assert summary["summary"]["same_class_review_pairs"] == 0
    auto_clusters_csv = artifact_root / "same_class_auto_resolved_clusters.csv"
    with auto_clusters_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["resolution"] == "auto_resolve"


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
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._compute_neighbor_pairs",
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
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._compute_neighbor_pairs",
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


def test_source_style_risk_is_train_only_and_excluded_from_eval(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"

    for index, offset in enumerate((2, 10, 18)):
        _write_pattern(source_root / "Healthy" / f"healthy_clean_{index}.jpg", offset=offset)
        _write_pattern(source_root / "Early Blight" / f"disease_clean_{index}.jpg", offset=offset + 1)

    _write_pattern(source_root / "Healthy" / "download_612x612_leaf_a.jpg", offset=23)
    _write_pattern(source_root / "Early Blight" / "download_612x612_leaf_b.jpg", offset=24)

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._compute_neighbor_pairs",
        lambda embeddings, *, paths, neighbors: {},
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["runtime_ready"] is True
    assert summary["summary"]["source_style_risk_images"] == 2
    assert summary["summary"]["train_only_routed_images"] == 2

    manifest_rows = json.loads((artifact_root / "proposed_split_manifest.json").read_text(encoding="utf-8"))["rows"]
    style_risk_rows = [row for row in manifest_rows if row["source_style_risk"]]
    assert style_risk_rows
    assert all(row["split"] == "continual" for row in style_risk_rows)
    assert all(row["train_only_routed"] for row in style_risk_rows)


def test_source_style_filter_blocks_when_safe_eval_families_are_insufficient(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"

    for class_name, base_offset in {"Healthy": 2, "Early Blight": 5}.items():
        _write_pattern(source_root / class_name / "clean_0.jpg", offset=base_offset)
        _write_pattern(source_root / class_name / "clean_1.jpg", offset=base_offset + 8)
        _write_pattern(source_root / class_name / "download_612x612_leaf.jpg", offset=base_offset + 16)

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._compute_neighbor_pairs",
        lambda embeddings, *, paths, neighbors: {},
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["runtime_ready"] is False
    assert summary["summary"]["source_style_risk_images"] == 2
    assert any("only 2 evaluation-eligible" in item for item in summary["blocking_issues"])


def test_under_min_eval_policy_skip_skips_classes_with_too_few_eval_families(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"

    for class_name, base_offset in {"Healthy": 2, "Early Blight": 5}.items():
        _write_pattern(source_root / class_name / "clean_0.jpg", offset=base_offset)
        _write_pattern(source_root / class_name / "clean_1.jpg", offset=base_offset + 8)
        _write_pattern(source_root / class_name / "download_612x612_leaf.jpg", offset=base_offset + 16)

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._compute_neighbor_pairs",
        lambda embeddings, *, paths, neighbors: {},
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
        under_min_eval_policy="skip",
    )

    assert summary["runtime_ready"] is False
    assert summary["under_min_eval_policy"] == "skip"
    assert summary["summary"]["skipped_classes"] == 2
    assert "No classes remain" in summary["blocking_issues"][0]
    assert summary["class_health"]["healthy"]["runtime_action"] == "skipped"

    manifest_rows = json.loads((artifact_root / "proposed_split_manifest.json").read_text(encoding="utf-8"))["rows"]
    assert manifest_rows
    assert all(row["split"] == "skipped" for row in manifest_rows)


def test_label_train_only_risk_is_excluded_from_canonical_eval(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"

    for index, offset in enumerate((2, 10, 18)):
        _write_pattern(source_root / "Healthy" / f"healthy_clean_{index}.jpg", offset=offset)
        _write_pattern(source_root / "Early Blight" / f"disease_clean_{index}.jpg", offset=offset + 1)

    _write_pattern(source_root / "Healthy" / "leaf_real.jpg", offset=24)
    _write_pattern(source_root / "Healthy" / "leaf_aug_flip.jpg", offset=25)

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._compute_neighbor_pairs",
        lambda embeddings, *, paths, neighbors: {
            tuple(sorted((real_path, aug_path))): 0.97
            for real_path in paths
            for aug_path in paths
            if "real" in real_path and "aug" in aug_path
        },
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._phash_distance",
        lambda a, b: 6,
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["runtime_ready"] is True
    assert summary["summary"]["label_train_only_risk_images"] == 2

    manifest_rows = json.loads((artifact_root / "proposed_split_manifest.json").read_text(encoding="utf-8"))["rows"]
    label_risk_rows = [row for row in manifest_rows if row["label_risk_level"] == "train_only_risk"]
    assert len(label_risk_rows) == 2
    assert all(row["split"] == "continual" for row in label_risk_rows)
    assert all(row["train_only_routed"] for row in label_risk_rows)


def test_zero_eval_eligible_class_is_skipped_during_materialization(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    runtime_root = tmp_path / "runtime"

    for index, offset in enumerate((2, 10, 18)):
        _write_pattern(source_root / "Healthy" / f"healthy_clean_{index}.jpg", offset=offset)
        _write_pattern(source_root / "Early Blight" / f"disease_clean_{index}.jpg", offset=offset + 1)
        _write_pattern(source_root / "Skip Me" / f"gan_leaf_{index}.jpg", offset=offset + 2)

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._compute_neighbor_pairs",
        lambda embeddings, *, paths, neighbors: {},
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["runtime_ready"] is True
    assert summary["summary"]["skipped_classes"] == 1
    assert summary["skipped_classes"][0]["class_name"] == "skip_me"
    assert summary["class_health"]["skip_me"]["runtime_action"] == "skipped"

    manifest_rows = json.loads((artifact_root / "proposed_split_manifest.json").read_text(encoding="utf-8"))["rows"]
    skipped_rows = [row for row in manifest_rows if row["normalized_class_name"] == "skip_me"]
    assert skipped_rows
    assert all(row["split"] == "skipped" for row in skipped_rows)
    assert all(row["runtime_skipped"] for row in skipped_rows)

    result_root = materialize_grouped_runtime_dataset(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        runtime_root=runtime_root,
    )

    crop_root = result_root / "tomato"
    assert not any((crop_root / split_name / "skip_me").exists() for split_name in ("continual", "val", "test"))
    assert (crop_root / "continual" / "healthy").exists()
    assert (crop_root / "continual" / "early_blight").exists()


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
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._compute_neighbor_pairs",
        lambda embeddings, *, paths, neighbors: {
            tuple(sorted((real_path, aug_path))): 0.99
            for real_path in paths
            for aug_path in paths
            if "real" in real_path and "aug" in aug_path
        },
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._phash_distance",
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


def test_build_grouped_dataset_plan_excludes_images_missing_after_scan(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"

    existing_paths = [
        source_root / "Healthy" / "healthy_0.jpg",
        source_root / "Healthy" / "healthy_1.jpg",
        source_root / "Healthy" / "healthy_2.jpg",
    ]
    for index, path in enumerate(existing_paths):
        _write_pattern(path, offset=index + 2)

    missing_path = source_root / "Healthy" / "vanished.jpg"

    real_records, normalization_report = scan_class_root_dataset(
        class_root=source_root,
        crop_name="tomato",
        taxonomy_path=None,
    )
    missing_record = next(record for record in real_records if record.relative_path.endswith("healthy_0.jpg"))
    missing_record.relative_path = "Healthy/vanished.jpg"
    missing_record.absolute_path = str(missing_path.resolve())

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset.scan_class_root_dataset",
        lambda **kwargs: (real_records, normalization_report),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["summary"]["excluded_images"] == 1
    assert summary["summary"]["readable_images"] == 2
    assert summary["summary"]["excluded_reason_breakdown"]["missing_after_scan"] == 1

def test_grouped_dataset_plan_writes_guided_catalog(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    runtime_root = tmp_path / "runtime"

    for index, offset in enumerate((2, 10, 18)):
        _write_pattern(source_root / "Healthy" / f"healthy_{index}.jpg", offset=offset)
        _write_pattern(source_root / "Early Blight" / f"disease_{index}.jpg", offset=offset + 1)

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )

    build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )
    materialize_grouped_runtime_dataset(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        runtime_root=runtime_root,
    )

    guided_dir = artifact_root / "guided"
    catalog = json.loads((guided_dir / "02_file_catalog.json").read_text(encoding="utf-8"))

    assert (guided_dir / "00_start_here.md").exists()
    assert (guided_dir / "01_prep_overview.json").exists()
    assert any(entry["relative_path"] == "prep_summary.json" for entry in catalog["entries"])
    assert any(entry["relative_path"] == "human_review_packet.json" for entry in catalog["entries"])
    assert any(entry["relative_path"] == "label_review_summary.json" for entry in catalog["entries"])
    assert any(entry["relative_path"] == "label_risk_summary.json" for entry in catalog["entries"])
    assert any(entry["relative_path"] == "label_review_candidates.csv" for entry in catalog["entries"])
    assert any(entry["relative_path"] == "source_style_groups.csv" for entry in catalog["entries"])
    assert any(entry["title_tr"] == "Materyalize edilmis runtime split manifesti" for entry in catalog["entries"])
