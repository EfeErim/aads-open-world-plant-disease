from __future__ import annotations

import json
from pathlib import Path

from src.data.adapter_ood_oe_split_rebalancing import rebalance_runtime_id_splits


def _row(name: str, family: str, split: str) -> dict[str, object]:
    return {
        "relative_path": f"source/{name}.jpg",
        "normalized_class_name": "disease",
        "family_bundle_key": family,
        "family_eval_eligible": True,
        "canonical_eval_safe": True,
        "family_assignment": split,
        "runtime_skipped": False,
        "split": split,
        "runtime_relative_path": f"{split}/disease/{name}.jpg",
    }


def test_rebalance_moves_whole_families_and_repairs_overlap(tmp_path: Path) -> None:
    rows = [
        _row("overlap_train", "family-overlap", "continual"),
        _row("overlap_val", "FAMILY-OVERLAP", "val"),
        _row("train_a", "family-train-a", "continual"),
        _row("train_b", "family-train-b", "continual"),
        _row("val", "family-val", "val"),
        _row("test", "family-test", "test"),
    ]
    for row in rows:
        path = tmp_path / str(row["runtime_relative_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(row["relative_path"]).encode())
    (tmp_path / "split_manifest.json").write_text(
        json.dumps({"schema_version": "v1_grouped_runtime_layout", "rows": rows}), encoding="utf-8"
    )

    report = rebalance_runtime_id_splits(
        tmp_path,
        dry_run=False,
        min_train_families=2,
        min_val_families=1,
        min_test_families=2,
    )

    assert report["ok"] is True
    written = json.loads((tmp_path / "split_manifest.json").read_text(encoding="utf-8"))["rows"]
    overlap_splits = {
        row["split"] for row in written if str(row["family_bundle_key"]).casefold() == "family-overlap"
    }
    assert len(overlap_splits) == 1
    assert sum(row["split"] == "test" for row in written) == 2
    assert all((tmp_path / row["runtime_relative_path"]).is_file() for row in written)


def test_rebalance_reports_unmet_floor_without_inventing_families(tmp_path: Path) -> None:
    rows = [_row("train", "family-train", "continual"), _row("val", "family-val", "val")]
    for row in rows:
        path = tmp_path / str(row["runtime_relative_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    (tmp_path / "split_manifest.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")

    report = rebalance_runtime_id_splits(
        tmp_path,
        min_train_families=1,
        min_val_families=1,
        min_test_families=1,
    )

    assert report["ok"] is False
    assert report["unmet"]["disease"]["test"] == {"actual": 0, "required": 1}
