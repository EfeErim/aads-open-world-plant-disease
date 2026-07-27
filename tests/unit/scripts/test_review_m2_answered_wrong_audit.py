import csv
import json
from pathlib import Path

from scripts.review_m2_answered_wrong_audit import main, review_answered_wrong_audit


HEADERS = [
    "rank",
    "image_id",
    "expected_target",
    "expected_crop",
    "expected_part",
    "expected_class",
    "predicted_crop",
    "predicted_part",
    "predicted_disease",
    "review_decision",
    "prototype_quality",
    "adapter_training_quality",
    "review_notes",
]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in HEADERS})


def test_review_answered_wrong_audit_marks_crop_part_correct_rows_for_adapter_training(tmp_path: Path):
    csv_path = tmp_path / "answered_wrong_audit.csv"
    _write_rows(
        csv_path,
        [
            {
                "image_id": "demo_001",
                "expected_target": "grape__leaf",
                "expected_crop": "grape",
                "expected_part": "leaf",
                "expected_class": "mildew",
                "predicted_crop": "grape",
                "predicted_part": "leaf",
                "predicted_disease": "anthracnose",
            },
            {
                "image_id": "demo_002",
                "expected_target": "grape__leaf",
                "expected_crop": "grape",
                "expected_part": "leaf",
                "expected_class": "mildew",
                "predicted_crop": "tomato",
                "predicted_part": "leaf",
                "predicted_disease": "blight",
            },
        ],
    )

    summary = review_answered_wrong_audit(csv_path)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["decisions"] == {"add_adapter_train": 1, "exclude_ambiguous": 1}
    assert rows[0]["review_decision"] == "add_adapter_train"
    assert rows[0]["adapter_training_quality"] == "crop_part_correct_hard_example"
    assert rows[1]["review_decision"] == "exclude_ambiguous"


def test_cli_updates_answered_wrong_audit_in_place(tmp_path: Path, capsys):
    csv_path = tmp_path / "answered_wrong_audit.csv"
    packet_csv = tmp_path / "packets" / "01_grape" / "review_rows.csv"
    _write_rows(
        csv_path,
        [
            {
                "rank": "1",
                "image_id": "demo_001",
                "expected_target": "grape__leaf",
                "expected_crop": "grape",
                "expected_part": "leaf",
                "expected_class": "mildew",
                "predicted_crop": "grape",
                "predicted_part": "leaf",
                "predicted_disease": "anthracnose",
            }
        ],
    )
    _write_rows(
        packet_csv,
        [
            {
                "rank": "1",
                "image_id": "demo_001",
                "expected_target": "grape__leaf",
                "expected_crop": "grape",
                "expected_part": "leaf",
                "expected_class": "mildew",
                "predicted_crop": "grape",
                "predicted_part": "leaf",
                "predicted_disease": "anthracnose",
            }
        ],
    )

    status = main(["--csv", str(csv_path), "--packet-dir", str(tmp_path / "packets")])

    assert status == 0
    output = json.loads(capsys.readouterr().out)
    assert output["row_count"] == 1
    assert output["packet_row_count"] == 1
    assert output["decisions"] == {"add_adapter_train": 1}
    with packet_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        packet_rows = list(csv.DictReader(handle))
    assert packet_rows[0]["review_decision"] == "add_adapter_train"
