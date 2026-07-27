from __future__ import annotations

from src.ood.behavioral_acceptance import build_behavioral_acceptance
from src.ood.recovery import TARGET_ADAPTERS
from src.ood.recovery_promotion import evaluate_candidate_promotion


def _acceptance(*, forced_answers: int = 0) -> dict:
    rows = [
        {"sample_origin": "in_distribution", "ood_predicted": False, "is_correct": True}
        for _ in range(30)
    ]
    rows.extend(
        {
            "sample_origin": "ood",
            "ood_type": "same_crop_unsupported_disease",
            "ood_predicted": index >= forced_answers,
        }
        for index in range(30)
    )
    return build_behavioral_acceptance(
        classification_metrics={"accuracy": 0.94, "balanced_accuracy": 0.91, "macro_f1": 0.91},
        prediction_rows=rows,
    )


def _report(*, forced_answers: int = 0) -> dict:
    return {
        "adapters": [
            {"target": target, "behavioral_acceptance": _acceptance(forced_answers=forced_answers)}
            for target in TARGET_ADAPTERS
        ]
    }


def test_all_behavioral_gates_promote_all_targets() -> None:
    report = evaluate_candidate_promotion(
        {},
        _report(),
        integrity_report={"ok": True},
        reload_parity={target: True for target in TARGET_ADAPTERS},
    )

    assert report["overall_promote"] is True
    assert report["decision_authority"] == "adapter_behavioral_acceptance"


def test_integrity_or_reload_failure_rejects_candidate() -> None:
    parity = {target: True for target in TARGET_ADAPTERS}
    parity["tomato__leaf"] = False
    report = evaluate_candidate_promotion(
        {},
        _report(),
        integrity_report={"ok": False},
        reload_parity=parity,
    )

    tomato = next(item for item in report["targets"] if item["target"] == "tomato__leaf")
    assert "serialization_reload_parity" in tomato["blockers"]
    assert "evidence_integrity" in tomato["blockers"]


def test_one_forced_ood_answer_rejects_candidate() -> None:
    report = evaluate_candidate_promotion(
        {},
        _report(forced_answers=1),
        integrity_report={"ok": True},
        reload_parity={target: True for target in TARGET_ADAPTERS},
    )

    assert report["overall_promote"] is False
    assert all("behavioral_forced_supported_answer_count" in item["blockers"] for item in report["targets"])
