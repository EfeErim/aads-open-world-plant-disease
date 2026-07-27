from __future__ import annotations

import json
from pathlib import Path

from src.ood.behavioral_acceptance import build_behavioral_acceptance
from src.ood.recovery import TARGET_ADAPTERS
from src.ood.stage_a import build_stage_a_report


def _comparison(split: str, *, knn_fpr: float) -> dict:
    return {
        "split_name": split,
        "methods": {
            "ensemble": {
                "pooled_metrics": {"ood_auroc": 0.93, "ood_false_positive_rate": 0.2},
                "pooled_gate_eligible": False,
                "worst_slice": {"ood_false_positive_rate": 0.3},
            },
            "energy": {
                "pooled_metrics": {"ood_auroc": 0.92, "ood_false_positive_rate": 0.15},
                "pooled_gate_eligible": False,
                "worst_slice": {"ood_false_positive_rate": 0.25},
            },
            "knn": {
                "pooled_metrics": {"ood_auroc": 0.95, "ood_false_positive_rate": knn_fpr},
                "pooled_gate_eligible": knn_fpr <= 0.05,
                "worst_slice": {"slice_name": "near_ood", "ood_false_positive_rate": knn_fpr},
            },
        },
    }


def test_stage_a_selects_on_validation_and_reports_test(tmp_path: Path) -> None:
    paths = {}
    for target in TARGET_ADAPTERS:
        root = tmp_path / target / "artifacts"
        (root / "validation").mkdir(parents=True)
        (root / "test").mkdir()
        readiness = root / "production_readiness.json"
        readiness.write_text(
            json.dumps({"ood_type_sample_checks": {"near_ood": {"asserted": True, "passed": True}}}),
            encoding="utf-8",
        )
        rows = [
            {"sample_origin": "in_distribution", "ood_predicted": False, "is_correct": True}
            for _ in range(30)
        ]
        rows.extend(
            {
                "sample_origin": "ood",
                "ood_type": "same_crop_unsupported_disease",
                "ood_predicted": True,
            }
            for _ in range(30)
        )
        (root / "adapter_behavioral_acceptance.json").write_text(
            json.dumps(
                build_behavioral_acceptance(
                    classification_metrics={"accuracy": 0.94, "balanced_accuracy": 0.91, "macro_f1": 0.91},
                    prediction_rows=rows,
                )
            ),
            encoding="utf-8",
        )
        (root / "validation" / "ood_method_comparison.json").write_text(
            json.dumps(_comparison("validation", knn_fpr=0.04)),
            encoding="utf-8",
        )
        (root / "test" / "ood_method_comparison.json").write_text(
            json.dumps(_comparison("test", knn_fpr=0.03)),
            encoding="utf-8",
        )
        paths[target] = readiness

    report = build_stage_a_report(tmp_path, readiness_paths=paths)

    assert report["ok"] is True
    assert report["final_test_used_for_selection"] is False
    assert report["selected_target_count"] == 8
    assert report["pooled_test_gate_pass_target_count"] == 8
    assert report["stage_a_promotable_target_count"] == 8
    assert all(item["selected_method"] == "knn" for item in report["adapters"])
