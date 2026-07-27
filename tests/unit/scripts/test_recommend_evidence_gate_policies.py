import json

from scripts.recommend_evidence_gate_policies import main


def test_cli_writes_policy_recommendation_outputs(tmp_path):
    calibration = tmp_path / "evidence_gate_calibration.json"
    failure = tmp_path / "notebook16_failure_analysis.json"
    json_output = tmp_path / "recommendations.json"
    markdown_output = tmp_path / "recommendations.md"
    calibration.write_text(
        json.dumps(
            {
                "global_policy": {"status": "no_eligible_policy"},
                "target_policies": {
                    "tomato__leaf": {
                        "status": "target_specific",
                        "policy": {"full_confidence_threshold": 0.95},
                        "calibration_metrics": {"wrong_capture_rate": 0.75},
                        "holdout_metrics": {"wrong_capture_rate": 0.70, "false_positive_review_rate": 0.12},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    failure.write_text(
        json.dumps({"targets": {"tomato__leaf": {"wrong_count": 121, "missed_wrong_count": 84}}}),
        encoding="utf-8",
    )

    status = main(
        [
            "--calibration-input",
            str(calibration),
            "--failure-analysis-input",
            str(failure),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")
    assert status == 0
    assert payload["summary"]["target_specific_count"] == 1
    assert payload["targets"]["tomato__leaf"]["missed_wrong_count"] == 84
    assert "Evidence Gate Policy Recommendations" in markdown
