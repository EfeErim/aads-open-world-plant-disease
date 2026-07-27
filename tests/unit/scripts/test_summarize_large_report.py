from __future__ import annotations

import json

from scripts.summarize_large_report import main, summarize_report


def test_summarize_json_extracts_metrics_statuses_and_rows(tmp_path):
    report = tmp_path / "multi_target_report.json"
    report.write_text(
        json.dumps(
            {
                "accuracy": 0.88,
                "macro_f1": 0.81,
                "rows": [
                    {
                        "target_id": "tomato__leaf",
                        "expected_label": "healthy",
                        "diagnosis": "blight",
                        "full_confidence": 0.91,
                        "roi_evidence_status": "conflicts_full",
                    },
                    {
                        "target_id": "grape__leaf",
                        "expected_label": "healthy",
                        "diagnosis": "healthy",
                        "full_confidence": 0.97,
                        "roi_evidence_status": "supports_full",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_report(report, max_items=4)

    metric_keys = {metric["key"] for metric in summary["numeric_metrics"]}
    assert {"accuracy", "macro_f1", "full_confidence"} <= metric_keys
    assert summary["representative_rows"]["row_count"] == 2
    assert summary["representative_rows"]["sample_rows"][0]["target_id"] == "tomato__leaf"
    status = {item["key"]: item["top_values"] for item in summary["status_counts"]}
    assert status["roi_evidence_status"]["conflicts_full"] == 1


def test_summarize_csv_extracts_counts_samples_and_numeric_columns(tmp_path):
    report = tmp_path / "rows.csv"
    report.write_text(
        "target_id,accuracy,status\n"
        "tomato__leaf,0.8,ok\n"
        "grape__leaf,0.9,ok\n"
        "strawberry__fruit,0.4,fail\n",
        encoding="utf-8",
    )

    summary = summarize_report(report, max_items=2)

    assert summary["row_count"] == 3
    assert summary["sample_rows"][0]["target_id"] == "tomato__leaf"
    numeric = {item["key"]: item for item in summary["numeric_columns"]}
    assert numeric["accuracy"]["mean"] == 0.7000000000000001
    categorical = {item["key"]: item["top_values"] for item in summary["categorical_columns"]}
    assert categorical["status"]["ok"] == 2


def test_cli_prints_bounded_json_summary(tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"status": "ok", "items": [{"score": 1.0}]}), encoding="utf-8")

    exit_code = main([str(report), "--max-items", "2"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["format"] == "json"
    assert output["status_counts"][0]["top_values"]["ok"] == 1
