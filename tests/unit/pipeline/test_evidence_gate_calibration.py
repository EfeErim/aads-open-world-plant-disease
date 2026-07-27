from src.pipeline.evidence_gate_calibration import (
    calibrate_evidence_gate_report,
    candidate_policies,
    evaluate_policy,
    select_policy,
    split_rows,
)


def _row(image_path, target_id, expected="healthy", diagnosis="healthy", **overrides):
    row = {
        "image_path": image_path,
        "target_id": target_id,
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


def test_evaluate_policy_computes_review_metrics():
    policy = {
        "full_confidence_threshold": 0.70,
        "review_on_roi_conflict": True,
        "review_on_roi_quality_bad": False,
        "review_on_full_ood": False,
        "review_on_roi_ood": False,
    }
    rows = [
        _row("wrong.jpg", "tomato__leaf", diagnosis="blight", roi_evidence_status="conflicts_with_full"),
        _row("correct.jpg", "tomato__leaf"),
        _row("reviewed-correct.jpg", "tomato__leaf", roi_evidence_status="conflicts_with_full"),
    ]

    metrics = evaluate_policy(rows, policy)["metrics"]

    assert metrics["sample_count"] == 3
    assert metrics["wrong_count"] == 1
    assert metrics["correct_count"] == 2
    assert metrics["review_count"] == 2
    assert metrics["wrong_capture_rate"] == 1.0
    assert metrics["false_positive_review_rate"] == 0.5
    assert metrics["wrong_missed_count"] == 0


def test_select_policy_chooses_highest_coverage_eligible_policy():
    rows = [
        _row("wrong-low.jpg", "tomato__leaf", diagnosis="blight", full_confidence=0.55),
        _row("wrong-conflict.jpg", "tomato__leaf", diagnosis="blight", roi_evidence_status="conflicts_with_full"),
        _row("correct-high.jpg", "tomato__leaf"),
        _row("correct-low.jpg", "tomato__leaf", full_confidence=0.95),
    ]

    selected = select_policy(rows, min_capture=1.0, max_false_positive_rate=0.0)

    assert selected["status"] == "eligible"
    assert selected["selected"]["policy"]["full_confidence_threshold"] == 0.60
    assert selected["selected"]["policy"]["review_on_roi_conflict"] is True
    assert selected["selected"]["metrics"]["coverage"] == 0.5


def test_select_policy_emits_best_rejected_when_no_policy_is_eligible():
    rows = [
        _row("wrong.jpg", "tomato__leaf", diagnosis="blight", full_confidence=0.95),
        _row("correct.jpg", "tomato__leaf", full_confidence=0.95),
    ]

    selected = select_policy(rows, min_capture=1.0, max_false_positive_rate=0.0)

    assert selected["status"] == "no_eligible_policy"
    assert selected["best_rejected"]["metrics"]["wrong_capture_rate"] == 0.0


def test_calibration_uses_target_policy_when_evidence_is_sufficient():
    rows = []
    for idx in range(1, 31):
        rows.append(
            _row(
                f"target-wrong-{idx}.jpg",
                "tomato__leaf",
                diagnosis="blight",
                full_confidence=0.55,
            )
        )
        rows.append(_row(f"target-correct-{idx}.jpg", "tomato__leaf", full_confidence=0.95))

    payload = calibrate_evidence_gate_report(
        rows,
        min_capture=1.0,
        max_false_positive_rate=0.0,
        min_target_errors=5,
        holdout_ratio=0.30,
        seed=7,
    )

    assert payload["global_policy"]["status"] == "eligible"
    assert payload["target_policies"]["tomato__leaf"]["status"] == "target_specific"


def test_calibration_uses_global_fallback_when_target_errors_are_insufficient():
    rows = []
    for idx in range(1, 31):
        rows.append(_row(f"global-wrong-{idx}.jpg", "tomato__leaf", diagnosis="blight", full_confidence=0.55))
        rows.append(_row(f"global-correct-{idx}.jpg", "tomato__leaf", full_confidence=0.95))
    rows.extend(
        [
            _row("small-wrong.jpg", "strawberry__fruit", diagnosis="ripe", full_confidence=0.55),
            _row("small-correct.jpg", "strawberry__fruit", full_confidence=0.95),
        ]
    )

    payload = calibrate_evidence_gate_report(
        rows,
        min_capture=1.0,
        max_false_positive_rate=0.0,
        min_target_errors=5,
        holdout_ratio=0.30,
        seed=7,
    )

    small_target = payload["target_policies"]["strawberry__fruit"]
    assert small_target["status"] == "global_fallback"
    assert small_target["fallback_reason"].startswith("target_wrong_count_below_min_target_errors")


def test_v2_candidate_grid_includes_high_confidence_thresholds():
    thresholds = {
        policy["full_confidence_threshold"]
        for policy in candidate_policies(confidence_thresholds=[0.50, 0.95, 0.98, 0.99])
    }

    assert {0.95, 0.98, 0.99}.issubset(thresholds)


def test_v2_uses_group_fallback_when_target_evidence_is_sparse():
    rows = []
    for idx in range(1, 41):
        target_id = "tomato__leaf" if idx <= 20 else "grape__leaf"
        rows.append(_row(f"{target_id}-wrong-{idx}.jpg", target_id, diagnosis="blight", full_confidence=0.55))
        rows.append(_row(f"{target_id}-correct-{idx}.jpg", target_id, full_confidence=0.95))

    payload = calibrate_evidence_gate_report(
        rows,
        schema_version="v2",
        min_capture=1.0,
        max_false_positive_rate=0.0,
        min_target_errors=25,
        min_calibration_errors=1,
        min_holdout_errors=1,
        max_review_rate=0.60,
        holdout_ratio=0.30,
        seed=7,
    )

    target_policy = payload["target_policies"]["tomato__leaf"]
    assert payload["schema_version"] == "v2_evidence_gate_calibration"
    assert payload["group_policies"]["leaf_targets"]["status"] == "eligible"
    assert target_policy["status"] == "group_fallback"
    assert target_policy["group_key"] == "leaf_targets"


def test_v2_keeps_no_eligible_policy_explicit_and_adds_audit_queue():
    rows = []
    for idx in range(1, 21):
        rows.append(_row(f"wrong-{idx}.jpg", "strawberry__fruit", diagnosis="ripe", full_confidence=0.97))
        rows.append(_row(f"correct-{idx}.jpg", "strawberry__fruit", full_confidence=0.97))

    payload = calibrate_evidence_gate_report(
        rows,
        schema_version="v2",
        min_capture=1.0,
        max_false_positive_rate=0.0,
        min_target_errors=5,
        min_calibration_errors=1,
        min_holdout_errors=1,
        max_review_rate=0.50,
        holdout_ratio=0.30,
        seed=7,
    )

    assert payload["global_policy"]["status"] == "no_eligible_policy"
    assert payload["target_policies"]["strawberry__fruit"]["status"] == "no_eligible_policy"
    assert payload["audit_queue"][0]["target_id"] == "strawberry__fruit"
    assert "no_calibration_policy_meets_constraints" in payload["audit_queue"][0]["reasons"]


def test_v2_risk_coverage_curves_are_deterministic():
    rows = [
        _row("wrong-low.jpg", "tomato__leaf", diagnosis="blight", full_confidence=0.55),
        _row("wrong-high.jpg", "tomato__leaf", diagnosis="blight", full_confidence=0.97),
        _row("correct.jpg", "tomato__leaf", full_confidence=0.95),
    ]

    first = calibrate_evidence_gate_report(rows, schema_version="v2")
    second = calibrate_evidence_gate_report(rows, schema_version="v2")

    assert first["risk_coverage_curves"] == second["risk_coverage_curves"]
    assert [row["threshold"] for row in first["risk_coverage_curves"]["global"]] == [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        0.98,
        0.99,
    ]


def test_split_rows_is_stable_across_repeated_runs():
    rows = [_row(f"sample-{idx}.jpg", "tomato__leaf") for idx in range(20)]

    first = split_rows(rows, holdout_ratio=0.30, seed=20260613)
    second = split_rows(rows, holdout_ratio=0.30, seed=20260613)

    assert first == second
