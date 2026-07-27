import csv
import json
from pathlib import Path

from scripts.apply_m2_hard_example_audit_decisions import apply_hard_example_decisions, load_decision_rows, main

HEADERS = [
    "rank",
    "image_id",
    "source",
    "resolved_image",
    "expected_target",
    "expected_crop",
    "expected_part",
    "expected_class",
    "predicted_disease",
    "review_decision",
    "corrected_class",
    "review_notes",
]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in HEADERS})


def test_apply_hard_example_decisions_writes_separate_manifests(tmp_path: Path):
    decisions = tmp_path / "docs" / "demo_results" / "m2" / "run_a" / "hard_example_audit.csv"
    output_dir = tmp_path / "curation"
    _write_rows(
        decisions,
        [
            {
                "rank": "1",
                "image_id": "demo_001",
                "expected_target": "tomato__leaf",
                "expected_crop": "tomato",
                "expected_part": "leaf",
                "expected_class": "healthy",
                "review_decision": "add_prototype_positive",
            },
            {
                "rank": "2",
                "image_id": "demo_002",
                "expected_target": "tomato__fruit",
                "expected_crop": "tomato",
                "expected_part": "fruit",
                "expected_class": "old",
                "review_decision": "add_adapter_train",
            },
            {
                "rank": "3",
                "image_id": "demo_003",
                "expected_target": "apricot__fruit",
                "expected_crop": "apricot",
                "expected_part": "fruit",
                "expected_class": "old",
                "review_decision": "exclude_ambiguous",
            },
            {
                "rank": "4",
                "image_id": "demo_004",
                "expected_target": "strawberry__fruit",
                "expected_crop": "strawberry",
                "expected_part": "fruit",
                "expected_class": "old",
                "review_decision": "relabel:new_class",
            },
            {
                "rank": "5",
                "image_id": "demo_005",
                "expected_target": "tomato__leaf",
                "review_decision": "add_prototype_hard_negative",
            },
        ],
    )

    summary = apply_hard_example_decisions(decisions_csv=decisions, output_dir=output_dir)

    assert summary["prototype_positive_count"] == 1
    assert summary["prototype_hard_negative_count"] == 1
    assert summary["adapter_hard_example_count"] == 1
    assert summary["excluded_ambiguous_count"] == 1
    assert summary["relabeled_count"] == 1
    assert (output_dir / "prototype_positive_manifest.csv").is_file()
    relabeled = (output_dir / "relabeled_rows.csv").read_text(encoding="utf-8-sig")
    assert "new_class" in relabeled
    assert json.loads((output_dir / "curation_summary.json").read_text(encoding="utf-8"))["relabeled_count"] == 1


def test_packet_dir_overlays_review_decisions(tmp_path: Path):
    full_csv = tmp_path / "hard_example_audit.csv"
    packet_csv = tmp_path / "packets" / "01_tomato__leaf" / "review_rows.csv"
    _write_rows(full_csv, [{"rank": "1", "image_id": "demo_001", "review_decision": ""}])
    _write_rows(
        packet_csv,
        [
            {
                "rank": "1",
                "image_id": "demo_001",
                "review_decision": "add_prototype_positive",
                "review_notes": "clear",
            }
        ],
    )

    rows, merged_count = load_decision_rows(decisions_csv=full_csv, packet_dir=tmp_path / "packets")

    assert merged_count == 1
    assert rows[0]["review_decision"] == "add_prototype_positive"
    assert rows[0]["review_notes"] == "clear"


def test_cli_require_reviewed_returns_two_for_pending_rows(tmp_path: Path):
    decisions = tmp_path / "hard_example_audit.csv"
    output_dir = tmp_path / "out"
    _write_rows(decisions, [{"rank": "1", "image_id": "demo_001", "review_decision": ""}])

    status = main(["--decisions-csv", str(decisions), "--output-dir", str(output_dir), "--require-reviewed"])

    summary = json.loads((output_dir / "curation_summary.json").read_text(encoding="utf-8"))
    assert status == 2
    assert summary["pending_review_count"] == 1


def test_unsupported_decisions_are_reported_without_outputs(tmp_path: Path):
    decisions = tmp_path / "hard_example_audit.csv"
    output_dir = tmp_path / "out"
    _write_rows(decisions, [{"rank": "1", "image_id": "demo_001", "review_decision": "delete"}])

    summary = apply_hard_example_decisions(decisions_csv=decisions, output_dir=output_dir)

    assert summary["skipped_count"] == 1
    assert summary["skipped"][0]["reason"] == "unsupported_decision:delete"
    assert summary["prototype_positive_count"] == 0
