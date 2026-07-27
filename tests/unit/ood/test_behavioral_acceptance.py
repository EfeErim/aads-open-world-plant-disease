from src.ood.behavioral_acceptance import (
    behavioral_acceptance_pass,
    behavioral_dev_report_pass,
    build_behavioral_acceptance,
    build_behavioral_dev_report,
)


def _rows(*, forced_answers: int = 0, id_rejections: int = 0) -> list[dict]:
    rows = [
        {
            "sample_origin": "in_distribution",
            "ood_predicted": index < id_rejections,
            "is_correct": True,
            "image_path": f"test/healthy/{index}.jpg",
        }
        for index in range(30)
    ]
    rows.extend(
        {
            "sample_origin": "ood",
            "ood_type": "same_crop_unsupported_disease",
            "ood_predicted": index >= forced_answers,
            "is_correct": False,
            "image_path": f"ood/test/disease_{index % 3}/{index}.jpg",
        }
        for index in range(30)
    )
    return rows


def _metrics() -> dict:
    return {"accuracy": 0.94, "balanced_accuracy": 0.91, "macro_f1": 0.91}


def test_contract_passes_only_when_id_is_good_and_all_same_crop_ood_is_rejected() -> None:
    payload = build_behavioral_acceptance(classification_metrics=_metrics(), prediction_rows=_rows())

    assert behavioral_acceptance_pass(payload) is True
    assert payload["metrics"]["forced_supported_answer_count"] == 0


def test_one_forced_supported_answer_fails_closed() -> None:
    payload = build_behavioral_acceptance(classification_metrics=_metrics(), prediction_rows=_rows(forced_answers=1))

    assert behavioral_acceptance_pass(payload) is False
    assert "forced_supported_answer_count" in payload["failure_reasons"]


def test_id_false_rejection_rate_is_part_of_default_acceptance() -> None:
    payload = build_behavioral_acceptance(classification_metrics=_metrics(), prediction_rows=_rows(id_rejections=2))

    assert behavioral_acceptance_pass(payload) is False
    assert "id_false_rejection_rate" in payload["failure_reasons"]


def test_far_ood_does_not_satisfy_same_crop_adapter_evidence_floor() -> None:
    rows = _rows()[:30]
    rows.extend(
        {
            "sample_origin": "ood",
            "ood_type": "non_plant",
            "ood_predicted": True,
            "image_path": f"ood/non_plant/{index}.jpg",
        }
        for index in range(30)
    )
    payload = build_behavioral_acceptance(classification_metrics=_metrics(), prediction_rows=rows)

    assert behavioral_acceptance_pass(payload) is False
    assert payload["metrics"]["same_crop_ood_test_samples"] == 0


def test_spoofed_empty_checks_cannot_pass() -> None:
    assert behavioral_acceptance_pass(
        {
            "schema_version": "v1_adapter_behavioral_acceptance",
            "decision_authority": "adapter_behavioral_acceptance",
            "status": "pass",
            "pass": True,
            "checks": {},
        }
    ) is False


def test_dev_report_is_selection_only_and_cannot_spoof_authoritative_acceptance() -> None:
    payload = build_behavioral_dev_report(classification_metrics=_metrics(), prediction_rows=_rows())

    assert behavioral_dev_report_pass(payload) is True
    assert behavioral_acceptance_pass(payload) is False
    assert payload["authoritative"] is False
    assert payload["context"]["locked_test_evidence_loaded"] is False


def test_derived_rows_count_once_for_behavioral_evidence_floor() -> None:
    rows = _rows()
    for index, row in enumerate(rows[30:]):
        row["evidence_family_id"] = f"family-{index // 2}"

    payload = build_behavioral_acceptance(classification_metrics=_metrics(), prediction_rows=rows)

    assert payload["metrics"]["same_crop_ood_test_samples"] == 15
    assert behavioral_acceptance_pass(payload) is False
