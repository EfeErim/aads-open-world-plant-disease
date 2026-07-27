from src.ood.recovery import TARGET_ADAPTERS
from src.ood.recovery_campaign import build_recovery_campaign


def test_campaign_builds_bounded_dev_selected_sequence() -> None:
    baseline = {
        "adapters": [
            {
                "target": target,
                "status": "failed",
                "blockers": (
                    [{"code": "accuracy_failed"}] if target != "strawberry__leaf" else [{"code": "ood_false_positive_rate_failed"}]
                ),
            }
            for target in TARGET_ADAPTERS
        ]
    }
    stage_a = {
        "adapters": [
            {
                "target": target,
                "selected_method": "knn",
                "sample_floors_passed": target != "strawberry__leaf",
                "stage_a_promotable": False,
            }
            for target in TARGET_ADAPTERS
        ]
    }

    report = build_recovery_campaign(
        baseline,
        stage_a,
        target_defaults={target: {"OE_LOSS_WEIGHT": 0.24, "LORA_R": 24} for target in TARGET_ADAPTERS},
    )

    assert report["target_count"] == 8
    assert report["schema_version"] == "v4_bounded_adapter_recovery_campaign"
    assert report["experiment_count"] == 32
    assert report["training_run_count"] == 32
    assert report["max_attempts_per_target"] == 4
    assert report["final_evaluation_once_per_target"] is True
    assert report["selection_test_reuse_forbidden"] is True
    apricot = next(item for item in report["targets"] if item["target"] == "apricot__fruit")
    strawberry = next(item for item in report["targets"] if item["target"] == "strawberry__leaf")
    assert [item["stage"] for item in apricot["experiments"]] == ["A", "B", "C", "D"]
    assert [item["stage"] for item in strawberry["experiments"]] == ["A", "B", "C", "D"]
    assert apricot["experiments"][0]["resolved_config"]["OE_ENABLED"] is True
    assert apricot["experiments"][1]["resolved_config"]["OE_LOSS_WEIGHT"] == 0.14
    assert apricot["experiments"][2]["resolved_config"]["OE_LOSS_WEIGHT"] == 0.34
    assert apricot["experiments"][3]["resolved_config"]["LORA_R"] == 32
    assert strawberry["evidence_repair_required"] is True


def test_campaign_uses_current_manifest_readiness_over_historical_stage_a_floor() -> None:
    baseline = {"adapters": [{"target": target, "status": "failed"} for target in TARGET_ADAPTERS]}
    stage_a = {
        "adapters": [
            {"target": target, "sample_floors_passed": True, "stage_a_promotable": False}
            for target in TARGET_ADAPTERS
        ]
    }

    report = build_recovery_campaign(
        baseline,
        stage_a,
        evidence_manifest_digest="abc123",
        evidence_ready_targets={"strawberry__leaf"},
    )

    assert report["evidence_manifest_digest"] == "abc123"
    assert report["evidence_ready_target_count"] == 1
    ready = next(item for item in report["targets"] if item["target"] == "strawberry__leaf")
    blocked = next(item for item in report["targets"] if item["target"] == "apricot__fruit")
    assert ready["evidence_repair_required"] is False
    assert blocked["evidence_repair_required"] is True
