import csv
import json
from pathlib import Path

from scripts.audit_ood_oe_quality import (
    WINDOWS_SAFE_QUARANTINE_PATH_LENGTH,
    _safe_quarantine_path,
    apply_review_decisions,
    discover_prepared_datasets,
    run_audit,
    run_batch_audit,
)
from tests.utils.test_helpers import make_image


def _write_image(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    make_image(path, size=(12, 12), color=color)


def test_audit_flags_exact_ood_oe_overlap_and_writes_review_csv(tmp_path: Path):
    dataset_root = tmp_path / "tomato__fruit"
    _write_image(dataset_root / "ood" / "unsupported_same_crop" / "a.jpg", color=(255, 0, 0))
    _write_image(dataset_root / "oe" / "aux_unknown" / "a_copy.jpg", color=(255, 0, 0))

    result = run_audit(
        dataset_root=dataset_root,
        dataset_key="tomato__fruit",
        output_dir=tmp_path / "audit",
        near_duplicate_hamming=6,
    )

    issues = result["issues"]
    assert any(item["issue_type"] == "exact_hash_overlap" for item in issues)
    assert any(item["severity"] == "blocker" for item in issues)
    assert (tmp_path / "audit" / "review_decisions.csv").is_file()
    summary = json.loads((tmp_path / "audit" / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts_by_role"] == {"ood": 1, "oe": 1}


def test_audit_flags_semantic_slice_suspicion(tmp_path: Path):
    dataset_root = tmp_path / "strawberry__fruit"
    _write_image(
        dataset_root / "ood" / "unsupported_same_crop" / "strawberry_leaf_like_case.jpg",
        color=(0, 255, 0),
    )

    result = run_audit(
        dataset_root=dataset_root,
        dataset_key="strawberry__fruit",
        output_dir=tmp_path / "audit",
        near_duplicate_hamming=6,
    )

    assert any(item["issue_type"] == "semantic_slice_suspicion" for item in result["issues"])


def test_audit_does_not_flag_intentional_opposite_part_or_slice_name_tokens(tmp_path: Path):
    dataset_root = tmp_path / "strawberry__fruit"
    _write_image(
        dataset_root / "oe" / "same_crop_leaf_negatives" / "strawberry_leaf_case.jpg",
        color=(0, 255, 0),
    )
    _write_image(
        dataset_root / "oe" / "same_crop_disease_negatives" / "strawberry_leaf_disease.jpg",
        color=(0, 200, 0),
    )
    _write_image(dataset_root / "ood" / "off_crop_secondary" / "grape_leaf_case.jpg", color=(0, 0, 255))
    _write_image(dataset_root / "ood" / "non_plant_misc" / "non_plant_misc_car.jpg", color=(255, 0, 0))

    result = run_audit(
        dataset_root=dataset_root,
        dataset_key="strawberry__fruit",
        output_dir=tmp_path / "audit",
        near_duplicate_hamming=6,
    )

    assert not any(item["issue_type"] == "semantic_slice_suspicion" for item in result["issues"])


def test_audit_rejects_dhash_only_false_positive(tmp_path: Path):
    dataset_root = tmp_path / "tomato__leaf"
    _write_image(dataset_root / "test" / "healthy" / "green.jpg", color=(0, 255, 0))
    _write_image(dataset_root / "oe" / "unsupported" / "red.jpg", color=(255, 0, 0))

    result = run_audit(
        dataset_root=dataset_root,
        dataset_key="tomato__leaf",
        output_dir=tmp_path / "audit",
        near_duplicate_hamming=6,
    )

    assert not any(item["issue_type"] == "near_duplicate_perceptual_hash" for item in result["issues"])


def test_apply_review_decisions_quarantines_only_marked_rows(tmp_path: Path):
    dataset_root = tmp_path / "tomato__leaf"
    target = dataset_root / "oe" / "aux_unknown" / "dup.jpg"
    _write_image(target, color=(0, 0, 255))
    (dataset_root / "split_manifest.json").write_text(
        json.dumps({"rows": [{"runtime_relative_path": "oe/aux_unknown/dup.jpg"}]}),
        encoding="utf-8",
    )
    evidence_manifest = dataset_root.parent / "adapter_ood_oe_evidence_manifest.csv"
    with evidence_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "relative_path"])
        writer.writeheader()
        writer.writerow(
            {
                "target": "tomato__leaf",
                "relative_path": "data/prepared_runtime_datasets/tomato__leaf/oe/aux_unknown/dup.jpg",
            }
        )
    decisions_csv = tmp_path / "review_decisions.csv"
    with decisions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["issue_id", "target_path", "decision"])
        writer.writeheader()
        writer.writerow({"issue_id": "exact_overlap_00001", "target_path": "oe/aux_unknown/dup.jpg", "decision": "quarantine"})

    summary = apply_review_decisions(
        dataset_root=dataset_root,
        decisions_csv=decisions_csv,
        quarantine_root=tmp_path / "quarantine",
    )

    assert summary["applied_count"] == 1
    assert not target.exists()
    assert (tmp_path / "quarantine" / "oe" / "aux_unknown" / "dup.jpg").is_file()
    assert summary["manifest_sync"] == {
        "removed_split_manifest_rows": 1,
        "removed_evidence_manifest_rows": 1,
    }
    assert json.loads((dataset_root / "split_manifest.json").read_text(encoding="utf-8"))["rows"] == []
    with evidence_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        assert list(csv.DictReader(handle)) == []


def test_safe_quarantine_path_shortens_long_windows_destination(tmp_path: Path):
    dataset_root = tmp_path / "tomato__fruit"
    quarantine_root = tmp_path / "quarantine"
    target_relative = f"oe/{'nested_' * 12}/{'very_long_name_' * 12}.jpg"

    destination = _safe_quarantine_path(dataset_root, quarantine_root, target_relative)

    assert len(str(destination)) <= WINDOWS_SAFE_QUARANTINE_PATH_LENGTH
    assert destination.parent.name == "oe"
    assert destination.suffix == ".jpg"
    assert destination == _safe_quarantine_path(dataset_root, quarantine_root, target_relative)


def test_batch_audit_runs_all_prepared_datasets_with_ood_or_oe(tmp_path: Path):
    prepared_root = tmp_path / "prepared_runtime_datasets"
    _write_image(prepared_root / "tomato__leaf" / "ood" / "unsupported" / "a.jpg", color=(255, 0, 0))
    _write_image(prepared_root / "grape__fruit" / "oe" / "aux" / "b.jpg", color=(0, 255, 0))
    _write_image(prepared_root / "empty__leaf" / "continual" / "healthy" / "c.jpg", color=(0, 0, 255))

    discovered = discover_prepared_datasets(prepared_root)
    assert [path.name for path in discovered] == ["grape__fruit", "tomato__leaf"]

    summary = run_batch_audit(
        prepared_root=prepared_root,
        output_dir=tmp_path / "batch_audit",
        near_duplicate_hamming=6,
    )

    assert summary["dataset_count"] == 2
    assert (tmp_path / "batch_audit" / "batch_summary.json").is_file()
    assert (tmp_path / "batch_audit" / "batch_summary.csv").is_file()
    assert (tmp_path / "batch_audit" / "tomato__leaf" / "review_decisions.csv").is_file()
    assert (tmp_path / "batch_audit" / "grape__fruit" / "review_decisions.csv").is_file()
