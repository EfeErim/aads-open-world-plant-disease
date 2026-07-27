import json
from pathlib import Path

from PIL import Image

from scripts.export_notebook16_target_audit import main


def test_cli_exports_target_audit_and_grouped_packets(tmp_path: Path):
    image_path = tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf" / "test" / "healthy" / "a.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "green").save(image_path)
    report = tmp_path / "multi_target_report.json"
    report.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "image_path": "/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/healthy/a.jpg",
                        "target_id": "tomato__leaf",
                        "expected_label": "healthy",
                        "diagnosis": "blight",
                        "full_confidence": 0.99,
                        "requires_review": False,
                        "roi_evidence_status": "supports_full",
                        "roi_quality_status": "roi_ok",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    json_output = tmp_path / "audit.json"
    csv_output = tmp_path / "audit.csv"
    markdown_output = tmp_path / "audit.md"
    contact_sheet = tmp_path / "contact.jpg"
    packet_dir = tmp_path / "packets"
    review_index = packet_dir / "index.html"

    status = main(
        [
            "--input",
            str(report),
            "--repo-root",
            str(tmp_path),
            "--json-output",
            str(json_output),
            "--csv-output",
            str(csv_output),
            "--markdown-output",
            str(markdown_output),
            "--contact-sheet-output",
            str(contact_sheet),
            "--packet-output-dir",
            str(packet_dir),
            "--review-index-output",
            str(review_index),
        ]
    )

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    packet_summary = json.loads((packet_dir / "packet_summary.json").read_text(encoding="utf-8"))
    assert status == 0
    assert payload["missed_wrong_count"] == 1
    assert payload["local_available_count"] == 1
    assert csv_output.is_file()
    assert markdown_output.is_file()
    assert contact_sheet.is_file()
    assert packet_summary["packet_count"] == 1
    assert packet_summary["rows_written"] == 1
    assert review_index.is_file()
    assert "healthy -&gt; blight" in review_index.read_text(encoding="utf-8")
    assert (packet_dir / "README.md").is_file()
    readme = (packet_dir / "README.md").read_text(encoding="utf-8")
    assert "--packet-dir" in readme
    assert "--require-reviewed" in readme
