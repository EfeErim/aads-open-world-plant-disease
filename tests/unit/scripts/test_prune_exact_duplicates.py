import csv
import zipfile
from pathlib import Path

from scripts.prune_exact_duplicates import (
    apply_cleanup_plan,
    build_cleanup_plan,
    build_combined_cleanup_plan,
    build_review_cleanup_plan,
    write_cleanup_report,
)


def _write_file(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_build_cleanup_plan_selects_n_minus_one_and_variants(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    csv_path = tmp_path / "exact_duplicates.csv"

    _write_file(dataset_root / "Healthy" / "dup_a.jpg", b"a")
    _write_file(dataset_root / "Healthy" / "dup_b.jpg", b"b")
    _write_file(dataset_root / "Healthy" / "dup_b_mirror.jpg", b"bm")
    _write_file(dataset_root / "Healthy" / "dup_b-change.jpg", b"bc")
    _write_file(dataset_root / "Healthy" / "dup_b_flip.jpg", b"bf")
    _write_file(dataset_root / "Healthy" / "dup_b_180deg.jpg", b"b180")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["count", "exact_hash", "normalized_class_name", "relative_paths"])
        writer.writeheader()
        writer.writerow(
            {
                "count": "2",
                "exact_hash": "hash1",
                "normalized_class_name": "healthy",
                "relative_paths": "Healthy/dup_a.jpg|Healthy/dup_b.jpg",
            }
        )

    actions = build_cleanup_plan(dataset_root=dataset_root, exact_duplicates_source=csv_path, seed=7)
    deleted = {action.deleted_relative_path for action in actions}
    assert len({action.selected_relative_path for action in actions}) == 1
    assert deleted & {"Healthy/dup_a.jpg", "Healthy/dup_b.jpg"}
    assert "Healthy/dup_b_mirror.jpg" in deleted or "Healthy/dup_a_mirror.jpg" in deleted
    assert "Healthy/dup_b-change.jpg" in deleted or "Healthy/dup_a-change.jpg" in deleted
    assert "Healthy/dup_b_flip.jpg" in deleted or "Healthy/dup_a_flip.jpg" in deleted
    assert "Healthy/dup_b_180deg.jpg" in deleted or "Healthy/dup_a_180deg.jpg" in deleted


def test_build_cleanup_plan_supports_zip_source(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    zip_path = tmp_path / "run.zip"

    _write_file(dataset_root / "Healthy" / "dup_a.jpg", b"a")
    _write_file(dataset_root / "Healthy" / "dup_b.jpg", b"b")
    payload = (
        "count,exact_hash,normalized_class_name,relative_paths\n"
        "2,hash1,healthy,Healthy/dup_a.jpg|Healthy/dup_b.jpg\n"
    )
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("run/artifacts/data_prep_artifacts/exact_duplicates.csv", payload)

    actions = build_cleanup_plan(dataset_root=dataset_root, exact_duplicates_source=zip_path, seed=11)
    assert len(actions) == 1


def test_apply_cleanup_plan_deletes_only_planned_files(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    csv_path = tmp_path / "exact_duplicates.csv"
    report_path = tmp_path / "plan.csv"

    _write_file(dataset_root / "Healthy" / "dup_a.jpg", b"a")
    _write_file(dataset_root / "Healthy" / "dup_b.jpg", b"b")
    _write_file(dataset_root / "Healthy" / "dup_b_lower.jpg", b"bl")
    _write_file(dataset_root / "Healthy" / "keep.jpg", b"k")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["count", "exact_hash", "normalized_class_name", "relative_paths"])
        writer.writeheader()
        writer.writerow(
            {
                "count": "2",
                "exact_hash": "hash1",
                "normalized_class_name": "healthy",
                "relative_paths": "Healthy/dup_a.jpg|Healthy/dup_b.jpg",
            }
        )

    actions = build_cleanup_plan(dataset_root=dataset_root, exact_duplicates_source=csv_path, seed=0)
    write_cleanup_report(report_path, actions)
    assert report_path.exists()

    deleted_count = apply_cleanup_plan(dataset_root=dataset_root, actions=actions)
    deleted_paths = {action.deleted_relative_path for action in actions}
    assert deleted_count == len(deleted_paths)
    assert (dataset_root / "Healthy" / "keep.jpg").exists()


def test_build_review_cleanup_plan_only_deletes_auto_resolved_clusters(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    review_csv = tmp_path / "same_class_review_candidates.csv"
    manifest_csv = tmp_path / "dataset_manifest.csv"

    _write_file(dataset_root / "Healthy" / "aug_a.jpg", b"a")
    _write_file(dataset_root / "Healthy" / "aug_b.jpg", b"b")
    _write_file(dataset_root / "Healthy" / "aug_a_mirror.jpg", b"am")
    _write_file(dataset_root / "Healthy" / "manual_a.jpg", b"c")
    _write_file(dataset_root / "Healthy" / "manual_b.jpg", b"d")

    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "relative_path",
            "absolute_path",
            "raw_class_name",
            "normalized_class_name",
            "source_hint",
            "synthetic_hint",
            "width",
            "height",
            "blur_score",
            "brightness_mean",
            "exact_hash",
            "phash_hex",
            "class_order_index",
            "excluded_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, relpath in enumerate(
            [
                "Healthy/aug_a.jpg",
                "Healthy/aug_b.jpg",
                "Healthy/manual_a.jpg",
                "Healthy/manual_b.jpg",
            ]
        ):
            writer.writerow(
                {
                    "relative_path": relpath,
                    "absolute_path": str((dataset_root / relpath).resolve()),
                    "raw_class_name": "Healthy",
                    "normalized_class_name": "healthy",
                    "source_hint": "unknown",
                    "synthetic_hint": "true" if "aug_" in relpath else "false",
                    "width": "32",
                    "height": "32",
                    "blur_score": "1.0",
                    "brightness_mean": "0.5",
                    "exact_hash": "",
                    "phash_hex": "0",
                    "class_order_index": str(index),
                    "excluded_reason": "",
                }
            )

    with review_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "pair_type",
            "class_a",
            "class_b",
            "path_a",
            "path_b",
            "exact_match",
            "phash_distance",
            "dino_cosine",
            "bioclip_cosine",
            "adjacency_distance",
            "review_rank",
            "decision",
            "reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "pair_type": "same_class_review",
                "class_a": "healthy",
                "class_b": "healthy",
                "path_a": "Healthy/aug_a.jpg",
                "path_b": "Healthy/aug_b.jpg",
                "exact_match": "false",
                "phash_distance": "20",
                "dino_cosine": "0.97",
                "bioclip_cosine": "-1.0",
                "adjacency_distance": "1",
                "review_rank": "1",
                "decision": "review",
                "reason": "synthetic pair",
            }
        )
        writer.writerow(
            {
                "pair_type": "same_class_review",
                "class_a": "healthy",
                "class_b": "healthy",
                "path_a": "Healthy/manual_a.jpg",
                "path_b": "Healthy/manual_b.jpg",
                "exact_match": "false",
                "phash_distance": "20",
                "dino_cosine": "0.97",
                "bioclip_cosine": "-1.0",
                "adjacency_distance": "10",
                "review_rank": "10",
                "decision": "review",
                "reason": "manual pair",
            }
        )

    actions = build_review_cleanup_plan(
        dataset_root=dataset_root,
        review_source=review_csv,
        dataset_manifest_source=manifest_csv,
        seed=1,
    )
    deleted = {action.deleted_relative_path for action in actions}
    assert deleted & {"Healthy/aug_a.jpg", "Healthy/aug_b.jpg"}
    assert "Healthy/aug_a_mirror.jpg" in deleted
    assert "Healthy/manual_a.jpg" not in deleted
    assert "Healthy/manual_b.jpg" not in deleted


def test_build_combined_cleanup_plan_merges_exact_and_review_actions(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    exact_csv = tmp_path / "exact_duplicates.csv"
    review_csv = tmp_path / "same_class_review_candidates.csv"
    manifest_csv = tmp_path / "dataset_manifest.csv"

    _write_file(dataset_root / "Healthy" / "dup_a.jpg", b"a")
    _write_file(dataset_root / "Healthy" / "dup_b.jpg", b"b")
    _write_file(dataset_root / "Healthy" / "aug_a.jpg", b"c")
    _write_file(dataset_root / "Healthy" / "aug_b.jpg", b"d")

    with exact_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["count", "exact_hash", "normalized_class_name", "relative_paths"])
        writer.writeheader()
        writer.writerow(
            {
                "count": "2",
                "exact_hash": "hash1",
                "normalized_class_name": "healthy",
                "relative_paths": "Healthy/dup_a.jpg|Healthy/dup_b.jpg",
            }
        )

    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "relative_path",
            "absolute_path",
            "raw_class_name",
            "normalized_class_name",
            "source_hint",
            "synthetic_hint",
            "width",
            "height",
            "blur_score",
            "brightness_mean",
            "exact_hash",
            "phash_hex",
            "class_order_index",
            "excluded_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, relpath in enumerate(["Healthy/aug_a.jpg", "Healthy/aug_b.jpg"]):
            writer.writerow(
                {
                    "relative_path": relpath,
                    "absolute_path": str((dataset_root / relpath).resolve()),
                    "raw_class_name": "Healthy",
                    "normalized_class_name": "healthy",
                    "source_hint": "unknown",
                    "synthetic_hint": "true",
                    "width": "32",
                    "height": "32",
                    "blur_score": "1.0",
                    "brightness_mean": "0.5",
                    "exact_hash": "",
                    "phash_hex": "0",
                    "class_order_index": str(index),
                    "excluded_reason": "",
                }
            )

    with review_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_type",
                "class_a",
                "class_b",
                "path_a",
                "path_b",
                "exact_match",
                "phash_distance",
                "dino_cosine",
                "bioclip_cosine",
                "adjacency_distance",
                "review_rank",
                "decision",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_type": "same_class_review",
                "class_a": "healthy",
                "class_b": "healthy",
                "path_a": "Healthy/aug_a.jpg",
                "path_b": "Healthy/aug_b.jpg",
                "exact_match": "false",
                "phash_distance": "20",
                "dino_cosine": "0.97",
                "bioclip_cosine": "-1.0",
                "adjacency_distance": "1",
                "review_rank": "1",
                "decision": "review",
                "reason": "synthetic pair",
            }
        )

    actions = build_combined_cleanup_plan(
        dataset_root=dataset_root,
        exact_duplicates_source=exact_csv,
        review_source=review_csv,
        dataset_manifest_source=manifest_csv,
        seed=3,
    )
    deleted = {action.deleted_relative_path for action in actions}
    assert deleted & {"Healthy/dup_a.jpg", "Healthy/dup_b.jpg"}
    assert deleted & {"Healthy/aug_a.jpg", "Healthy/aug_b.jpg"}
