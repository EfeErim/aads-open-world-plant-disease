from pathlib import Path

from scripts import validate_router_calibration_stability as validator


def _variant(**metrics):
    return {"metrics": metrics}


def test_count_router_eval_images_ignores_non_images(tmp_path: Path):
    (tmp_path / "id" / "tomato" / "leaf").mkdir(parents=True)
    (tmp_path / "id" / "tomato" / "leaf" / "a.jpg").write_bytes(b"image")
    (tmp_path / "id" / "tomato" / "leaf" / "notes.txt").write_text("skip", encoding="utf-8")

    assert validator.count_router_eval_images(tmp_path) == 1


def test_validate_calibration_payload_accepts_stable_recommendation():
    payload = {
        "sample_count": 4,
        "baseline": _variant(
            crop_accuracy=0.90,
            part_non_unknown_precision=0.92,
            negative_false_accept_rate=0.10,
        ),
        "recommended": _variant(
            crop_accuracy=0.89,
            part_non_unknown_precision=0.91,
            negative_false_accept_rate=0.02,
            abstention_rate=0.20,
        ),
    }

    issues = validator.validate_calibration_payload(
        payload,
        target_negative_far=0.05,
        max_crop_accuracy_drop=0.02,
        max_part_precision_drop=0.02,
        max_abstention_rate=0.30,
        min_samples=1,
    )

    assert issues == []


def test_validate_calibration_payload_flags_unstable_recommendation():
    payload = {
        "sample_count": 1,
        "baseline": _variant(crop_accuracy=0.95, part_non_unknown_precision=0.95),
        "recommended": {
            "eligible": False,
            "eligibility_reasons": ["negative_false_accept_rate_above_target"],
            "metrics": {
                "crop_accuracy": 0.80,
                "part_non_unknown_precision": 0.75,
                "negative_false_accept_rate": 0.20,
                "abstention_rate": 0.40,
            },
        },
    }

    issues = validator.validate_calibration_payload(
        payload,
        target_negative_far=0.05,
        max_crop_accuracy_drop=0.02,
        max_part_precision_drop=0.02,
        max_abstention_rate=0.30,
        min_samples=2,
    )

    assert {issue.code for issue in issues} == {
        "sample_count_too_low",
        "negative_false_accept_rate_above_target",
        "abstention_rate_above_target",
        "crop_accuracy_drop",
        "part_precision_drop",
        "recommended_variant_ineligible",
    }


def test_validate_calibration_payload_reports_no_eligible_recommendation():
    payload = {
        "sample_count": 3,
        "baseline": _variant(crop_accuracy=0.95, part_non_unknown_precision=0.95),
        "best_rejected": _variant(
            crop_accuracy=0.90,
            part_non_unknown_precision=0.88,
            negative_false_accept_rate=0.14,
            abstention_rate=0.25,
        ),
    }

    issues = validator.validate_calibration_payload(
        payload,
        target_negative_far=0.05,
        max_crop_accuracy_drop=0.02,
        max_part_precision_drop=0.02,
        max_abstention_rate=0.30,
        min_samples=1,
    )

    assert [issue.code for issue in issues] == ["no_eligible_recommendation"]
