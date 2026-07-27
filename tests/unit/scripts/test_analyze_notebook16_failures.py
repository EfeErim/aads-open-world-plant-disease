import json

from scripts.analyze_notebook16_failures import main


def _row(image_path, target_id="tomato__leaf", expected="healthy", diagnosis="healthy", **overrides):
    row = {
        "image_path": image_path,
        "target_id": target_id,
        "expected_label": expected,
        "diagnosis": diagnosis,
        "full_confidence": 0.90,
        "roi_evidence_status": "supports_full",
        "roi_quality_status": "roi_ok",
        "requires_review": False,
        "review_reasons": [],
    }
    row.update(overrides)
    return row


def test_cli_writes_json_and_markdown_outputs(tmp_path):
    report = tmp_path / "multi_target_report.json"
    calibration = tmp_path / "evidence_gate_calibration.json"
    json_output = tmp_path / "analysis.json"
    markdown_output = tmp_path / "analysis.md"
    report.write_text(
        json.dumps(
            {
                "rows": [
                    _row("missed.jpg", diagnosis="blight", full_confidence=0.96),
                    _row("audit.jpg", "strawberry__fruit", diagnosis="ripe", full_confidence=0.97),
                ]
            }
        ),
        encoding="utf-8",
    )
    calibration.write_text(
        json.dumps({"target_policies": {"tomato__leaf": {"status": "target_specific"}}}),
        encoding="utf-8",
    )

    status = main(
        [
            "--input",
            str(report),
            "--calibration-input",
            str(calibration),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--focus-target",
            "tomato__leaf",
            "--top-examples",
            "1",
        ]
    )

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")
    assert status == 0
    assert payload["focus_target"] == "tomato__leaf"
    assert payload["targets"]["tomato__leaf"]["calibration_status"] == "target_specific"
    assert len(payload["targets"]["tomato__leaf"]["missed_wrong_examples"]) == 1
    assert "## Focus Target: `tomato__leaf`" in markdown
    assert "## All Targets" in markdown
