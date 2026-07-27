from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_adapter_class_index_alignment import build_report


def _write_adapter_meta(adapter_dir: Path, *, class_to_idx: dict[str, int]) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_meta.json").write_text(
        json.dumps(
            {
                "schema_version": "v6",
                "engine": "continual_sd_lora",
                "crop_name": "tomato",
                "part_name": "leaf",
                "class_to_idx": class_to_idx,
                "backbone": {"model_name": "fake/backbone"},
                "ood_calibration": {"version": 1},
            }
        ),
        encoding="utf-8",
    )


def test_class_index_alignment_passes_when_adapter_uses_runtime_class_order(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf"
    for class_name in ("Disease A", "Healthy Leaf"):
        (dataset_root / "continual" / class_name).mkdir(parents=True)

    adapter_dir = tmp_path / "runs" / "tomato" / "leaf" / "run_1" / "outputs" / "colab_notebook_training" / "tomato" / "leaf" / "continual_sd_lora_adapter"
    _write_adapter_meta(adapter_dir, class_to_idx={"Disease A": 0, "Healthy Leaf": 1})

    report = build_report(
        adapter_roots=[tmp_path / "runs"],
        dataset_root=tmp_path / "data" / "prepared_runtime_datasets",
    )

    assert report["status"] == "pass"
    assert report["fail_count"] == 0


def test_class_index_alignment_fails_when_adapter_uses_normalized_class_names(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf"
    for class_name in ("Disease A", "Healthy Leaf"):
        (dataset_root / "continual" / class_name).mkdir(parents=True)

    adapter_dir = tmp_path / "runs" / "tomato" / "leaf" / "run_1" / "outputs" / "colab_notebook_training" / "tomato" / "leaf" / "continual_sd_lora_adapter"
    _write_adapter_meta(adapter_dir, class_to_idx={"healthy_leaf": 0, "disease_a": 1})

    report = build_report(
        adapter_roots=[tmp_path / "runs"],
        dataset_root=tmp_path / "data" / "prepared_runtime_datasets",
    )

    assert report["status"] == "fail"
    assert report["fail_count"] == 1
    assert "adapter class order does not match" in report["adapters"][0]["errors"][0]


def test_class_index_alignment_warns_when_only_raw_names_differ(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf"
    for class_name in ("Disease A", "Healthy Leaf"):
        (dataset_root / "continual" / class_name).mkdir(parents=True)

    adapter_dir = tmp_path / "runs" / "tomato" / "leaf" / "run_1" / "outputs" / "colab_notebook_training" / "tomato" / "leaf" / "continual_sd_lora_adapter"
    _write_adapter_meta(adapter_dir, class_to_idx={"disease_a": 0, "healthy_leaf": 1})

    report = build_report(
        adapter_roots=[tmp_path / "runs"],
        dataset_root=tmp_path / "data" / "prepared_runtime_datasets",
    )

    assert report["status"] == "warn"
    assert report["fail_count"] == 0
    assert report["warn_count"] == 1
    assert report["adapters"][0]["mismatch_type"] == "raw_name_mismatch"
