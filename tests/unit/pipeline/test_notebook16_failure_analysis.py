from src.pipeline.notebook16_failure_analysis import (
    build_notebook16_failure_analysis,
    build_target_missed_wrong_audit,
    render_notebook16_failure_markdown,
    write_target_missed_wrong_audit_outputs,
)


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


def test_analysis_computes_review_metrics_and_confusions():
    payload = build_notebook16_failure_analysis(
        [
            _row("missed.jpg", diagnosis="blight", full_confidence=0.97),
            _row("captured.jpg", diagnosis="rust", full_confidence=0.52, requires_review=True),
            _row("false-positive.jpg", requires_review=True, review_reasons="low_full_confidence;roi_conflict"),
            _row("correct.jpg"),
        ],
        calibration_statuses={"tomato__leaf": {"status": "target_specific"}},
    )

    target = payload["targets"]["tomato__leaf"]
    assert target["sample_count"] == 4
    assert target["wrong_count"] == 2
    assert target["accuracy"] == 0.5
    assert target["review_capture_rate"] == 0.5
    assert target["missed_wrong_count"] == 1
    assert target["false_positive_review_rate"] == 0.5
    assert target["high_confidence_wrong_count"] == 1
    assert target["confusion_pairs"] == [
        {"key": "healthy -> blight", "count": 1},
        {"key": "healthy -> rust", "count": 1},
    ]
    assert target["missed_wrong_confusion_pairs"] == [{"key": "healthy -> blight", "count": 1}]
    assert target["review_reason_counts"] == [
        {"key": "low_full_confidence", "count": 1},
        {"key": "roi_conflict", "count": 1},
    ]
    assert target["calibration_status"] == "target_specific"


def test_analysis_adds_missed_wrong_drilldown_and_threshold_sweep():
    payload = build_notebook16_failure_analysis(
        [
            _row("missed-very-high.jpg", diagnosis="blight", full_confidence=0.995),
            _row("missed-high.jpg", diagnosis="blight", full_confidence=0.965, roi_evidence_status="supports_full"),
            _row("missed-mid.jpg", diagnosis="rust", full_confidence=0.91, roi_quality_status="roi_too_large"),
            _row("captured.jpg", diagnosis="rust", full_confidence=0.80, requires_review=True),
            _row("correct-low.jpg", full_confidence=0.94),
            _row("correct-high.jpg", full_confidence=0.99),
        ]
    )

    target = payload["targets"]["tomato__leaf"]
    assert target["missed_wrong_confidence_distribution"] == [
        {"key": "0.90-0.95", "count": 1},
        {"key": "0.95-0.98", "count": 1},
        {"key": ">=0.99", "count": 1},
    ]
    assert target["missed_wrong_roi_evidence_status_counts"] == [{"key": "supports_full", "count": 3}]
    assert target["top_missed_confusion_examples"][0]["confusion_pair"] == "healthy -> blight"
    assert target["top_missed_confusion_examples"][0]["count"] == 2
    threshold_095 = target["confidence_threshold_sweep"][0]
    threshold_098 = target["confidence_threshold_sweep"][1]
    threshold_099 = target["confidence_threshold_sweep"][2]
    assert threshold_095["threshold"] == 0.95
    assert threshold_095["newly_captured_wrong_count"] == 1
    assert threshold_095["added_false_positive_count"] == 1
    assert threshold_098["newly_captured_wrong_count"] == 2
    assert threshold_099["newly_captured_wrong_count"] == 2


def test_analysis_orders_targets_by_missed_wrong_then_sample_count_and_false_positive_rate():
    rows = [
        _row("tomato-missed-1.jpg", "tomato__leaf", diagnosis="blight"),
        _row("tomato-missed-2.jpg", "tomato__leaf", diagnosis="rust"),
        _row("tomato-correct.jpg", "tomato__leaf"),
        _row("strawberry-missed.jpg", "strawberry__fruit", diagnosis="ripe"),
        _row("strawberry-correct-reviewed.jpg", "strawberry__fruit", requires_review=True),
        _row("apricot-missed.jpg", "apricot__leaf", diagnosis="rust"),
        _row("apricot-correct.jpg", "apricot__leaf"),
    ]

    payload = build_notebook16_failure_analysis(rows)

    assert payload["ordered_targets"] == ["tomato__leaf", "apricot__leaf", "strawberry__fruit"]


def test_analysis_handles_missing_optional_fields_and_caps_examples_deterministically():
    rows = [
        _row("b.jpg", diagnosis="blight", full_confidence=0.91),
        _row("a.jpg", diagnosis="blight", full_confidence=0.99),
        {
            "image_path": "c.jpg",
            "target_id": "tomato__leaf",
            "expected_label": "healthy",
            "diagnosis": "blight",
            "full_confidence": 0.99,
            "requires_review": False,
        },
    ]

    payload = build_notebook16_failure_analysis(rows, top_examples=2)
    target = payload["targets"]["tomato__leaf"]

    assert target["roi_quality_status_counts"] == [{"key": "roi_ok", "count": 2}, {"key": "missing", "count": 1}]
    assert [row["image_path"] for row in target["missed_wrong_examples"]] == ["a.jpg", "c.jpg"]
    assert len(target["missed_wrong_examples"]) == 2


def test_render_markdown_includes_focus_and_all_target_sections():
    payload = build_notebook16_failure_analysis(
        [
            _row("missed.jpg", "tomato__leaf", diagnosis="blight"),
            _row("audit.jpg", "strawberry__fruit", diagnosis="ripe"),
        ]
    )

    markdown = render_notebook16_failure_markdown(payload)

    assert "## Focus Target: `tomato__leaf`" in markdown
    assert "## All Targets" in markdown
    assert "## Data/Label Audit Target: `strawberry__fruit`" in markdown
    assert "### Missed-Wrong Drilldown" in markdown
    assert "Confidence threshold sweep over existing review decisions" in markdown
    assert "Review-gate focus decision:" in markdown
    assert "`tomato__leaf` stays report-only; it is not a runtime promotion." in markdown
    assert "`0.95` confidence-threshold simulation: review capture `1.0000`" in markdown


def test_target_missed_wrong_audit_exports_all_rows_and_local_status(tmp_path):
    local_image = tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf" / "test" / "healthy" / "a.jpg"
    local_image.parent.mkdir(parents=True)
    local_image.write_bytes(b"not-an-image")
    rows = [
        _row(
            "/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/healthy/a.jpg",
            diagnosis="blight",
            full_confidence=0.99,
        ),
        _row("missing.jpg", diagnosis="rust", full_confidence=0.95),
        _row("captured.jpg", diagnosis="rust", requires_review=True),
        _row("correct.jpg"),
        _row("other-target.jpg", target_id="strawberry__fruit", diagnosis="ripe"),
    ]

    payload = build_target_missed_wrong_audit(rows, repo_root=tmp_path)

    assert payload["wrong_count"] == 3
    assert payload["missed_wrong_count"] == 2
    assert payload["local_available_count"] == 1
    assert [row["diagnosis"] for row in payload["rows"]] == ["blight", "rust"]
    assert payload["rows"][0]["local_exists"] is True
    assert payload["confusion_pairs"] == [
        {"key": "healthy -> blight", "count": 1},
        {"key": "healthy -> rust", "count": 1},
    ]

    json_output = tmp_path / "audit.json"
    csv_output = tmp_path / "audit.csv"
    markdown_output = tmp_path / "audit.md"
    write_target_missed_wrong_audit_outputs(
        payload,
        json_output=json_output,
        csv_output=csv_output,
        markdown_output=markdown_output,
    )

    assert "v1_notebook16_target_missed_wrong_audit" in json_output.read_text(encoding="utf-8")
    assert "confusion_pair" in csv_output.read_text(encoding="utf-8-sig")
    assert "Missed-Wrong Audit" in markdown_output.read_text(encoding="utf-8")
