import json
from pathlib import Path

import pytest

from src.training.services.class_balance import (
    build_class_balance_runtime,
    compute_effective_number_weights,
    resolve_reference_class_counts,
)


def _write_manifest(crop_root: Path, rows: list[dict], *, filename: str = "split_manifest.json") -> None:
    crop_root.mkdir(parents=True, exist_ok=True)
    (crop_root / filename).write_text(
        json.dumps({"classes": rows}, indent=2),
        encoding="utf-8",
    )


def test_resolve_reference_class_counts_prefers_manifest_and_normalizes_names(tmp_path: Path):
    crop_root = tmp_path / "tomato"
    _write_manifest(
        crop_root,
        [
            {"class_name": "Healthy", "image_count": 240},
            {"class_name": "Disease A", "image_count": 120},
        ],
    )

    resolved = resolve_reference_class_counts(
        crop_name="tomato",
        data_dir=tmp_path,
        detected_classes=["healthy", "disease_a"],
        split_class_counts={"train": {"healthy": 10, "disease_a": 10}},
    )

    assert resolved["count_source"] == "manifest:split_manifest.json"
    assert resolved["resolved_class_counts"] == {"healthy": 240, "disease_a": 120}
    assert resolved["missing_classes"] == []


def test_resolve_reference_class_counts_prefers_reference_image_count(tmp_path: Path):
    crop_root = tmp_path / "grape__fruit"
    _write_manifest(
        crop_root,
        [
            {"class_name": "healthy", "image_count": 300, "reference_image_count": 100},
            {"class_name": "powdery_mildew", "image_count": 138, "reference_image_count": 46},
        ],
    )

    resolved = resolve_reference_class_counts(
        crop_name="grape",
        data_dir=tmp_path,
        dataset_key="grape__fruit",
        detected_classes=["healthy", "powdery_mildew"],
        split_class_counts={"train": {"healthy": 260, "powdery_mildew": 92}},
    )

    assert resolved["count_source"] == "manifest:split_manifest.json"
    assert resolved["resolved_class_counts"] == {"healthy": 100, "powdery_mildew": 46}


def test_resolve_reference_class_counts_uses_grouped_manifest_rows(tmp_path: Path):
    crop_root = tmp_path / "grape__fruit"
    crop_root.mkdir(parents=True)
    (crop_root / "split_manifest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"normalized_class_name": "healthy", "split": "continual"},
                    {"normalized_class_name": "healthy", "split": "val"},
                    {"normalized_class_name": "healthy", "split": "test"},
                    {"normalized_class_name": "healthy", "split": "continual", "generated_offline_augmentation": True},
                    {"normalized_class_name": "healthy", "split": "continual", "synthetic_hint": True},
                    {"normalized_class_name": "powdery_mildew", "split": "continual"},
                    {"normalized_class_name": "powdery_mildew", "split": "skipped", "runtime_skipped": True},
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_reference_class_counts(
        crop_name="grape",
        data_dir=tmp_path,
        dataset_key="grape__fruit",
        detected_classes=["healthy", "powdery_mildew"],
        split_class_counts={"train": {"healthy": 30, "powdery_mildew": 30}},
    )

    assert resolved["count_source"] == "manifest:split_manifest.json"
    assert resolved["resolved_class_counts"] == {"healthy": 3, "powdery_mildew": 1}


def test_resolve_reference_class_counts_falls_back_to_train_split_without_manifest(tmp_path: Path):
    resolved = resolve_reference_class_counts(
        crop_name="tomato",
        data_dir=tmp_path,
        detected_classes=["healthy", "disease_a"],
        split_class_counts={"train": {"healthy": 160, "disease_a": 110}},
    )

    assert resolved["count_source"] == "train_split"
    assert resolved["resolved_class_counts"] == {"healthy": 160, "disease_a": 110}
    assert resolved["missing_classes"] == []


def test_build_class_balance_runtime_activates_and_normalizes_weights(tmp_path: Path):
    crop_root = tmp_path / "tomato"
    _write_manifest(
        crop_root,
        [
            {"class_name": "healthy", "image_count": 260},
            {"class_name": "disease_a", "image_count": 120},
        ],
    )

    runtime = build_class_balance_runtime(
        crop_name="tomato",
        data_dir=tmp_path,
        detected_classes=["healthy", "disease_a"],
        split_class_counts={"train": {"healthy": 200, "disease_a": 90}},
    )

    assert runtime["active"] is True
    assert runtime["eligible_classes"] == ["disease_a"]
    assert set(runtime["weights_by_class"].keys()) == {"healthy", "disease_a"}
    assert pytest.approx(sum(runtime["weights_by_class"].values()) / 2.0) == 1.0
    assert runtime["weights_by_class"]["disease_a"] > runtime["weights_by_class"]["healthy"]


def test_build_class_balance_runtime_disables_loss_weights_when_sampler_is_weighted(tmp_path: Path):
    crop_root = tmp_path / "tomato"
    _write_manifest(
        crop_root,
        [
            {"class_name": "healthy", "image_count": 260},
            {"class_name": "disease_a", "image_count": 120},
        ],
    )

    runtime = build_class_balance_runtime(
        crop_name="tomato",
        data_dir=tmp_path,
        detected_classes=["healthy", "disease_a"],
        split_class_counts={"train": {"healthy": 200, "disease_a": 90}},
        resolved_train_sampler="weighted",
    )

    assert runtime["active"] is False
    assert runtime["loss_weighting_disabled_reason"] == "weighted_sampler_active"
    assert runtime["weights_by_class"] == {}


def test_build_class_balance_runtime_allows_explicit_sampler_and_loss_rebalance(tmp_path: Path):
    crop_root = tmp_path / "tomato"
    _write_manifest(
        crop_root,
        [
            {"class_name": "healthy", "image_count": 260},
            {"class_name": "disease_a", "image_count": 120},
        ],
    )

    runtime = build_class_balance_runtime(
        crop_name="tomato",
        data_dir=tmp_path,
        detected_classes=["healthy", "disease_a"],
        split_class_counts={"train": {"healthy": 200, "disease_a": 90}},
        resolved_train_sampler="weighted",
        allow_sampler_and_loss_rebalance=True,
    )

    assert runtime["active"] is True
    assert runtime["loss_weighting_disabled_reason"] == ""
    assert set(runtime["weights_by_class"].keys()) == {"healthy", "disease_a"}


def test_build_class_balance_runtime_reports_under_min_classes(tmp_path: Path):
    crop_root = tmp_path / "tomato"
    _write_manifest(
        crop_root,
        [
            {"class_name": "healthy", "image_count": 240},
            {"class_name": "disease_a", "image_count": 54},
        ],
    )

    runtime = build_class_balance_runtime(
        crop_name="tomato",
        data_dir=tmp_path,
        detected_classes=["healthy", "disease_a"],
        split_class_counts={"train": {"healthy": 180, "disease_a": 40}},
    )

    assert runtime["active"] is False
    assert runtime["under_min_classes"] == ["disease_a"]
    assert runtime["production_under_min_classes"] == ["disease_a"]
    assert runtime["production_guardrail_bypassed"] is False
    assert runtime["weights_by_class"] == {}


def test_build_class_balance_runtime_records_under_min_training_bypass(tmp_path: Path):
    crop_root = tmp_path / "tomato"
    _write_manifest(
        crop_root,
        [
            {"class_name": "healthy", "image_count": 12},
            {"class_name": "disease_a", "image_count": 8},
        ],
    )

    runtime = build_class_balance_runtime(
        crop_name="tomato",
        data_dir=tmp_path,
        detected_classes=["healthy", "disease_a"],
        split_class_counts={"train": {"healthy": 9, "disease_a": 6}},
        allow_under_min_training=True,
    )

    assert runtime["allow_under_min_training"] is True
    assert runtime["under_min_classes"] == ["healthy", "disease_a"]
    assert runtime["production_under_min_classes"] == ["healthy", "disease_a"]
    assert runtime["production_guardrail_bypassed"] is True
    assert runtime["active"] is False


def test_compute_effective_number_weights_normalizes_to_mean_one():
    weights = compute_effective_number_weights([260, 120, 180])

    assert len(weights) == 3
    assert pytest.approx(sum(weights) / 3.0) == 1.0
