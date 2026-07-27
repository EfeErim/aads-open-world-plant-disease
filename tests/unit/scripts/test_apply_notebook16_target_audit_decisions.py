import csv
import json
from pathlib import Path

from scripts.apply_notebook16_target_audit_decisions import apply_audit_decisions, load_decision_rows, main


def _write_decisions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "local_path", "review_decision"])
        writer.writeheader()
        writer.writerows(rows)


def test_apply_audit_decisions_dry_run_does_not_move_files(tmp_path: Path):
    dataset_root = tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf"
    image = dataset_root / "test" / "old_class" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    decisions = tmp_path / "decisions.csv"
    _write_decisions(
        decisions,
        [
            {
                "rank": "1",
                "local_path": "data/prepared_runtime_datasets/tomato__leaf/test/old_class/a.jpg",
                "review_decision": "relabel:new_class",
            }
        ],
    )

    summary = apply_audit_decisions(
        decisions_csv=decisions,
        dataset_root=dataset_root,
        quarantine_root=tmp_path / "quarantine",
        apply=False,
    )

    assert summary["mode"] == "dry_run"
    assert summary["action_count"] == 1
    assert summary["actions"][0]["action"] == "relabel"
    assert image.is_file()
    assert not (dataset_root / "test" / "new_class" / "a.jpg").exists()


def test_apply_audit_decisions_moves_marked_rows_only(tmp_path: Path):
    dataset_root = tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf"
    remove_image = dataset_root / "test" / "old_class" / "remove.jpg"
    relabel_image = dataset_root / "test" / "old_class" / "relabel.jpg"
    keep_image = dataset_root / "test" / "old_class" / "keep.jpg"
    remove_image.parent.mkdir(parents=True)
    for image in (remove_image, relabel_image, keep_image):
        image.write_bytes(b"image")
    decisions = tmp_path / "decisions.csv"
    _write_decisions(
        decisions,
        [
            {"rank": "1", "local_path": "test/old_class/remove.jpg", "review_decision": "remove_from_test"},
            {"rank": "2", "local_path": "test/old_class/relabel.jpg", "review_decision": "relabel:new_class"},
            {"rank": "3", "local_path": "test/old_class/keep.jpg", "review_decision": "keep"},
        ],
    )

    summary = apply_audit_decisions(
        decisions_csv=decisions,
        dataset_root=dataset_root,
        quarantine_root=tmp_path / "quarantine",
        apply=True,
    )

    assert summary["mode"] == "apply"
    assert summary["action_count"] == 2
    assert not remove_image.exists()
    assert not relabel_image.exists()
    assert keep_image.is_file()
    assert (tmp_path / "quarantine" / "test" / "old_class" / "remove.jpg").is_file()
    assert (dataset_root / "test" / "new_class" / "relabel.jpg").is_file()


def test_cli_writes_dry_run_summary(tmp_path: Path):
    dataset_root = tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf"
    image = dataset_root / "test" / "old_class" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    decisions = tmp_path / "decisions.csv"
    output = tmp_path / "summary.json"
    _write_decisions(
        decisions,
        [{"rank": "1", "local_path": "test/old_class/a.jpg", "review_decision": "remove_from_test"}],
    )

    status = main(
        [
            "--decisions-csv",
            str(decisions),
            "--dataset-root",
            str(dataset_root),
            "--quarantine-root",
            str(tmp_path / "quarantine"),
            "--output",
            str(output),
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["mode"] == "dry_run"
    assert summary["action_count"] == 1
    assert summary["pending_review_count"] == 0
    assert image.is_file()


def test_cli_require_reviewed_fails_on_empty_decisions(tmp_path: Path):
    dataset_root = tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf"
    image = dataset_root / "test" / "old_class" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    decisions = tmp_path / "decisions.csv"
    output = tmp_path / "summary.json"
    _write_decisions(
        decisions,
        [{"rank": "1", "local_path": "test/old_class/a.jpg", "review_decision": ""}],
    )

    status = main(
        [
            "--decisions-csv",
            str(decisions),
            "--dataset-root",
            str(dataset_root),
            "--quarantine-root",
            str(tmp_path / "quarantine"),
            "--output",
            str(output),
            "--require-reviewed",
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert status == 2
    assert summary["pending_review_count"] == 1
    assert summary["skipped"][0]["reason"] == "pending_review_decision"
    assert image.is_file()


def test_packet_dir_decisions_overlay_full_audit_rows(tmp_path: Path):
    full_csv = tmp_path / "full.csv"
    packet_csv = tmp_path / "packets" / "01_pair" / "review_rows.csv"
    _write_decisions(
        full_csv,
        [{"rank": "1", "local_path": "test/old_class/a.jpg", "review_decision": ""}],
    )
    _write_decisions(
        packet_csv,
        [{"rank": "1", "local_path": "test/old_class/a.jpg", "review_decision": "remove_from_test"}],
    )

    rows, merged_count = load_decision_rows(decisions_csv=full_csv, packet_dir=tmp_path / "packets")

    assert merged_count == 1
    assert rows[0]["review_decision"] == "remove_from_test"


def test_cli_uses_packet_dir_decisions(tmp_path: Path):
    dataset_root = tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf"
    image = dataset_root / "test" / "old_class" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    full_csv = tmp_path / "full.csv"
    packet_csv = tmp_path / "packets" / "01_pair" / "review_rows.csv"
    output = tmp_path / "summary.json"
    _write_decisions(
        full_csv,
        [{"rank": "1", "local_path": "test/old_class/a.jpg", "review_decision": ""}],
    )
    _write_decisions(
        packet_csv,
        [{"rank": "1", "local_path": "test/old_class/a.jpg", "review_decision": "remove_from_test"}],
    )

    status = main(
        [
            "--decisions-csv",
            str(full_csv),
            "--dataset-root",
            str(dataset_root),
            "--quarantine-root",
            str(tmp_path / "quarantine"),
            "--packet-dir",
            str(tmp_path / "packets"),
            "--output",
            str(output),
            "--require-reviewed",
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["packet_decision_count"] == 1
    assert summary["pending_review_count"] == 0
    assert summary["action_count"] == 1
    assert image.is_file()
