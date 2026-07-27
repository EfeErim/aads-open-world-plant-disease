from src.pipeline.evidence_gate_policy_recommendations import (
    build_policy_recommendations,
    render_policy_recommendations_markdown,
)


def _policy(status, *, threshold=0.95, group_key="", fallback_reason=""):
    return {
        "status": status,
        "group_key": group_key,
        "fallback_reason": fallback_reason,
        "policy": {"full_confidence_threshold": threshold} if status != "no_eligible_policy" else {},
        "calibration_metrics": {
            "wrong_capture_rate": 0.75,
            "false_positive_review_rate": 0.10,
        },
        "holdout_metrics": {
            "wrong_capture_rate": 0.70,
            "false_positive_review_rate": 0.12,
            "review_rate": 0.20,
        },
        "evidence_status": {"total_wrong": 30},
    }


def test_build_policy_recommendations_groups_all_targets():
    payload = build_policy_recommendations(
        {
            "schema_version": "v2_evidence_gate_calibration",
            "global_policy": {"status": "no_eligible_policy"},
            "target_policies": {
                "tomato__leaf": _policy("target_specific", threshold=0.95),
                "apricot__leaf": _policy("group_fallback", threshold=0.90, group_key="leaf_targets"),
                "strawberry__fruit": _policy("no_eligible_policy", fallback_reason="target_selection_no_eligible_policy"),
            },
        },
        failure_analysis_payload={
            "targets": {
                "tomato__leaf": {"wrong_count": 121, "missed_wrong_count": 84},
                "strawberry__fruit": {"wrong_count": 112, "missed_wrong_count": 84},
            }
        },
    )

    assert payload["schema_version"] == "v1_evidence_gate_policy_recommendations"
    assert payload["summary"]["target_specific_count"] == 1
    assert payload["summary"]["group_fallback_count"] == 1
    assert payload["summary"]["no_eligible_policy_count"] == 1
    assert payload["categories"]["target_specific"] == ["tomato__leaf"]
    assert payload["categories"]["group_fallback"] == ["apricot__leaf"]
    assert payload["targets"]["tomato__leaf"]["recommendation"] == "report_only_candidate"
    assert payload["targets"]["apricot__leaf"]["source"] == "group:leaf_targets"
    assert payload["targets"]["strawberry__fruit"]["recommendation"] == "audit_required"
    assert payload["targets"]["strawberry__fruit"]["missed_wrong_count"] == 84


def test_render_policy_recommendations_markdown_separates_candidates_and_audits():
    payload = build_policy_recommendations(
        {
            "global_policy": {"status": "no_eligible_policy"},
            "target_policies": {
                "tomato__leaf": _policy("target_specific", threshold=0.95),
                "strawberry__fruit": _policy("no_eligible_policy"),
            },
        }
    )

    markdown = render_policy_recommendations_markdown(payload)

    assert "## Recommended Report-Only Candidates" in markdown
    assert "| `tomato__leaf` | `report_only_candidate`" in markdown
    assert "## `tomato__leaf` Pilot Decision" in markdown
    assert "Keep `tomato__leaf` as a report-only candidate" in markdown
    assert "## Audit Required" in markdown
    assert "| `strawberry__fruit` | `no_safe_target_group_or_global_policy`" in markdown
