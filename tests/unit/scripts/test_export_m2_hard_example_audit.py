import csv
import json
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from scripts.export_m2_hard_example_audit import build_hard_example_rows, load_m2_rows, main


def test_build_hard_example_rows_prioritizes_answered_wrong_and_filters_targets():
    rows = [
        {
            "image_id": "demo_001",
            "expected_target": "tomato__leaf",
            "expected_crop": "tomato",
            "expected_part": "leaf",
            "expected_class": "healthy",
            "actual_status": "success",
            "predicted_disease": "blight",
            "pass_fail": "fail",
            "prototype_margin": 0.08,
        },
        {
            "image_id": "demo_002",
            "expected_target": "tomato__leaf",
            "expected_crop": "tomato",
            "expected_part": "leaf",
            "actual_status": "unknown_crop",
            "pass_fail": "fail",
            "failure_bucket": "router",
            "reconcile_reason": "prototype_evidence_weak",
            "prototype_margin": "",
        },
        {
            "image_id": "demo_003",
            "expected_target": "grape__leaf",
            "expected_crop": "grape",
            "expected_part": "leaf",
            "pass_fail": "fail",
            "failure_bucket": "router",
        },
    ]

    audit_rows = build_hard_example_rows(rows, targets=("tomato__leaf",))

    assert [row["image_id"] for row in audit_rows] == ["demo_001", "demo_002"]
    assert "answered_wrong" in audit_rows[0]["priority_reasons"]
    assert "prototype_evidence_weak" in audit_rows[1]["priority_reasons"]


def test_build_hard_example_rows_excludes_unsupported_by_default():
    rows = [
        {
            "image_id": "demo_044",
            "expected_target": "grape__unknown_part",
            "expected_crop": "grape",
            "expected_part": "unknown_part",
            "pass_fail": "fail",
            "failure_bucket": "router",
        }
    ]

    assert build_hard_example_rows(rows, targets=("grape__unknown_part",)) == []
    assert build_hard_example_rows(rows, targets=("grape__unknown_part",), include_unsupported=True)


def test_build_hard_example_rows_router_only_filters_adapter_wrong_and_reasons():
    rows = [
        {
            "image_id": "demo_001",
            "expected_target": "tomato__leaf",
            "expected_crop": "tomato",
            "expected_part": "leaf",
            "actual_status": "success",
            "predicted_disease": "blight",
            "pass_fail": "fail",
            "reconcile_reason": "prototype_evidence_weak",
        },
        {
            "image_id": "demo_002",
            "expected_target": "tomato__leaf",
            "expected_crop": "tomato",
            "expected_part": "leaf",
            "actual_status": "router_uncertain",
            "pass_fail": "fail",
            "failure_bucket": "router",
            "reconcile_reason": "prototype_evidence_weak",
        },
        {
            "image_id": "demo_003",
            "expected_target": "tomato__leaf",
            "expected_crop": "tomato",
            "expected_part": "leaf",
            "actual_status": "router_uncertain",
            "pass_fail": "fail",
            "failure_bucket": "router",
            "reconcile_reason": "unsupported_expected_target",
        },
        {
            "image_id": "demo_004",
            "expected_target": "tomato__leaf",
            "expected_crop": "tomato",
            "expected_part": "leaf",
            "actual_status": "success",
            "pass_fail": "pass",
            "failure_bucket": "",
            "reconcile_reason": "prototype_evidence_weak",
        },
    ]

    audit_rows = build_hard_example_rows(
        rows,
        targets=("tomato__leaf",),
        reconcile_reasons=("prototype_evidence_weak",),
        exclude_answered_wrong=True,
        failed_only=True,
    )

    assert [row["image_id"] for row in audit_rows] == ["demo_002"]
    assert "router_failure" in audit_rows[0]["priority_reasons"]


def test_build_hard_example_rows_answered_wrong_only_filters_and_ranks_class_pairs():
    rows = [
        {
            "image_id": "demo_001",
            "expected_target": "grape__leaf",
            "expected_crop": "grape",
            "expected_part": "leaf",
            "expected_class": "mildew",
            "actual_status": "success",
            "predicted_crop": "grape",
            "predicted_part": "leaf",
            "predicted_disease": "anthracnose",
            "pass_fail": "fail",
        },
        {
            "image_id": "demo_002",
            "expected_target": "grape__leaf",
            "expected_crop": "grape",
            "expected_part": "leaf",
            "expected_class": "mildew",
            "actual_status": "success",
            "predicted_crop": "grape",
            "predicted_part": "leaf",
            "predicted_disease": "anthracnose",
            "pass_fail": "fail",
        },
        {
            "image_id": "demo_003",
            "expected_target": "grape__leaf",
            "expected_crop": "grape",
            "expected_part": "leaf",
            "expected_class": "",
            "actual_status": "success",
            "predicted_disease": "anthracnose",
            "pass_fail": "fail",
        },
        {
            "image_id": "demo_004",
            "expected_target": "grape__leaf",
            "expected_crop": "grape",
            "expected_part": "leaf",
            "expected_class": "mildew",
            "actual_status": "success",
            "predicted_disease": "anthracnose",
            "pass_fail": "fail",
            "failure_bucket": "router",
        },
    ]

    audit_rows = build_hard_example_rows(rows, targets=(), answered_wrong_only=True)

    assert [row["image_id"] for row in audit_rows] == ["demo_001", "demo_002"]
    assert "repeated_class_pair" in audit_rows[0]["priority_reasons"]
    assert "crop_part_correct_disease_wrong" in audit_rows[0]["priority_reasons"]


def test_load_m2_rows_falls_back_to_git_object_when_sparse(tmp_path: Path):
    missing = tmp_path / "docs" / "demo_results" / "m2" / "run" / "m2_demo_checklist_run.json"
    completed = Mock(stdout=json.dumps({"rows": [{"image_id": "demo_001"}]}))

    with patch("scripts.export_m2_hard_example_audit.subprocess.run", return_value=completed) as run:
        rows = load_m2_rows(missing)

    assert rows == [{"image_id": "demo_001"}]
    run.assert_called_once()


def test_cli_exports_csv_packets_and_index(tmp_path: Path):
    repo_root = tmp_path
    image = (
        repo_root
        / "docs"
        / "demo_assets"
        / "m2_full_image_set"
        / "images"
        / "demo_001.jpg"
    )
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "green").save(image)
    run = tmp_path / "m2_demo_checklist_run.json"
    run.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "image_id": "demo_001",
                        "source": "staged_external:docs/demo_assets/m2_full_image_set/images/demo_001.jpg",
                        "resolved_image": (
                            "/content/bitirmeprojesi/"
                            "docs/demo_assets/m2_full_image_set/images/demo_001.jpg"
                        ),
                        "expected_target": "tomato__leaf",
                        "expected_crop": "tomato",
                        "expected_part": "leaf",
                        "expected_class": "healthy",
                        "actual_status": "success",
                        "predicted_disease": "blight",
                        "prototype_target": "tomato__leaf",
                        "prototype_class_label": "healthy",
                        "prototype_similarity": 0.66,
                        "prototype_margin": 0.03,
                        "pass_fail": "fail",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    csv_output = tmp_path / "hard_example_audit.csv"
    packet_dir = tmp_path / "packets"
    index = packet_dir / "index.html"

    status = main(
        [
            "--input",
            str(run),
            "--run-id",
            "test_run",
            "--repo-root",
            str(repo_root),
            "--csv-output",
            str(csv_output),
            "--packet-output-dir",
            str(packet_dir),
            "--review-index-output",
            str(index),
        ]
    )

    assert status == 0
    with csv_output.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["review_decision"] == ""
    assert rows[0]["image_id"] == "demo_001"
    assert (packet_dir / "packet_summary.json").is_file()
    assert (packet_dir / "01_tomato__leaf" / "review_rows.csv").is_file()
    assert (packet_dir / "01_tomato__leaf" / "contact_sheet.jpg").is_file()
    assert "tomato__leaf" in index.read_text(encoding="utf-8")


def test_cli_answered_wrong_mode_uses_answered_wrong_defaults(tmp_path: Path):
    run = tmp_path / "m2_demo_checklist_run.json"
    run.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "image_id": "demo_001",
                        "expected_target": "grape__leaf",
                        "expected_crop": "grape",
                        "expected_part": "leaf",
                        "expected_class": "mildew",
                        "actual_status": "success",
                        "predicted_crop": "grape",
                        "predicted_part": "leaf",
                        "predicted_disease": "anthracnose",
                        "pass_fail": "fail",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    csv_output = tmp_path / "answered_wrong_audit.csv"
    packet_dir = tmp_path / "packets"

    status = main(
        [
            "--input",
            str(run),
            "--run-id",
            "test_run",
            "--answered-wrong-only",
            "--csv-output",
            str(csv_output),
            "--packet-output-dir",
            str(packet_dir),
        ]
    )

    assert status == 0
    with csv_output.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["image_id"] == "demo_001"
    assert (packet_dir / "01_grape__leaf__mildew__predicted_anthracnose" / "review_rows.csv").is_file()
