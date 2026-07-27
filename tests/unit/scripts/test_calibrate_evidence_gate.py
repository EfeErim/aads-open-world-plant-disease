import json

from scripts.calibrate_evidence_gate import main


def _row(image_path, expected="healthy", diagnosis="healthy", **overrides):
    row = {
        "image_path": image_path,
        "target_id": "tomato__leaf",
        "expected_label": expected,
        "diagnosis": diagnosis,
        "full_confidence": 0.90,
        "roi_evidence_status": "supports_full",
        "roi_quality_status": "roi_ok",
        "full_ood_is_ood": False,
        "roi_ood_is_ood": False,
    }
    row.update(overrides)
    return row


def test_cli_writes_calibration_json_without_samples_by_default(tmp_path):
    report = tmp_path / "multi_target_report.json"
    output = tmp_path / "evidence_gate_calibration.json"
    rows = [
        _row("wrong.jpg", diagnosis="blight", full_confidence=0.55),
        _row("correct.jpg", full_confidence=0.95),
    ]
    report.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    status = main(
        [
            "--input",
            str(report),
            "--output",
            str(output),
            "--min-capture",
            "1.0",
            "--max-false-positive-rate",
            "0.0",
            "--min-target-errors",
            "1",
            "--seed",
            "7",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["schema_version"] == "v1_evidence_gate_calibration"
    assert payload["constraints"]["min_capture"] == 1.0
    assert payload["global_policy"]["status"] in {"eligible", "no_eligible_policy"}
    assert "samples" not in payload


def test_cli_includes_samples_when_requested(tmp_path):
    report = tmp_path / "multi_target_report.json"
    output = tmp_path / "evidence_gate_calibration.json"
    report.write_text(json.dumps({"rows": [_row("sample.jpg")]}), encoding="utf-8")

    status = main(["--input", str(report), "--output", str(output), "--include-samples"])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["samples"][0]["image_path"] == "sample.jpg"


def test_cli_writes_v2_schema_when_requested(tmp_path):
    report = tmp_path / "multi_target_report.json"
    output = tmp_path / "evidence_gate_calibration.json"
    rows = [
        _row("wrong.jpg", diagnosis="blight", full_confidence=0.55),
        _row("correct.jpg", full_confidence=0.95),
    ]
    report.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    status = main(
        [
            "--input",
            str(report),
            "--output",
            str(output),
            "--schema-version",
            "v2",
            "--min-calibration-errors",
            "1",
            "--min-holdout-errors",
            "1",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["schema_version"] == "v2_evidence_gate_calibration"
    assert payload["runtime_recommendation"]["status"] == "advisory_only"
    assert "audit_queue" in payload
