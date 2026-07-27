from __future__ import annotations

import json
from pathlib import Path

from src.ood.behavioral_acceptance import build_behavioral_acceptance
from src.ood.recovery import TARGET_ADAPTERS, build_recovery_report, evaluate_readiness


def _write_ready(path: Path, *, target: str, status: str = "ready") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset_key": target,
                "status": status,
                "metrics": {
                    "accuracy": 0.94,
                    "balanced_accuracy": 0.91,
                    "macro_f1": 0.91,
                    "id_test_samples": 35,
                },
                "ood_metrics": {
                    "ood_auroc": 0.93,
                    "ood_false_positive_rate": 0.04,
                    "ood_samples": 40,
                },
                "ood_type_counts": {"same_crop_unknown": 10, "wrong_part": 8, "non_plant": 7},
                "context": {
                    "ood_primary_score_method": "energy",
                    "ood_primary_score_selection": {"selected_threshold": -2.5},
                },
            }
        ),
        encoding="utf-8",
    )


def test_missing_artifacts_are_reported_for_all_eight_targets(tmp_path: Path) -> None:
    report = build_recovery_report(tmp_path / "runs")

    assert report["overall_status"] == "recovery_required"
    assert report["required_target_count"] == 8
    assert [item["target"] for item in report["adapters"]] == list(TARGET_ADAPTERS)
    assert report["blocker_counts"] == {"P0": 0, "P1": 0, "P2": 8}


def test_recovery_report_defaults_to_behavioral_acceptance_artifacts(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    for target in TARGET_ADAPTERS:
        path = runs_root / target / "adapter_behavioral_acceptance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
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
        payload = build_behavioral_acceptance(
            classification_metrics={"accuracy": 0.94, "balanced_accuracy": 0.91, "macro_f1": 0.91},
            prediction_rows=rows,
            context={"crop_name": target.split("__")[0], "part_name": target.split("__")[1]},
        )
        path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_recovery_report(runs_root)

    assert report["overall_status"] == "pass"
    assert report["passed_target_count"] == 8
    assert report["decision_authority"] == "adapter_behavioral_acceptance"


def test_ready_artifact_normalizes_metrics_and_score_metadata(tmp_path: Path) -> None:
    path = tmp_path / "tomato" / "leaf" / "production_readiness.json"
    _write_ready(path, target="tomato__leaf")

    result = evaluate_readiness("tomato__leaf", path)

    assert result["blockers"] == []
    assert result["metrics"]["ood_test_samples"] == 40
    assert result["metrics"]["min_ood_type_samples"] == 7
    assert result["ood_method"] == "energy"
    assert result["ood_threshold"] == -2.5


def test_metric_and_evidence_floor_failures_are_classified(tmp_path: Path) -> None:
    path = tmp_path / "grape" / "fruit" / "production_readiness.json"
    _write_ready(path, target="grape__fruit", status="failed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metrics"]["accuracy"] = 0.80
    payload["ood_type_counts"]["non_plant"] = 2
    payload["evidence_overlap_count"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate_readiness("grape__fruit", path)
    blockers = {(item["priority"], item["code"]) for item in result["blockers"]}

    assert ("P0", "evidence_hash_overlap") in blockers
    assert ("P1", "accuracy_failed") in blockers
    assert ("P1", "min_ood_type_samples_failed") in blockers
    assert ("P1", "readiness_not_ready") in blockers


def test_canonical_readiness_schema_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "apricot" / "leaf" / "production_readiness.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "status": "ready",
                "classification_evidence": {
                    "metrics": {
                        "accuracy": 0.95,
                        "balanced_accuracy": 0.92,
                        "macro_f1": 0.91,
                        "samples": 34,
                    }
                },
                "ood_evidence": {
                    "source": "real_ood_split",
                    "metrics": {
                        "ood_auroc": 0.94,
                        "ood_false_positive_rate": 0.03,
                        "ood_samples": 32,
                    },
                },
                "ood_type_sample_checks": {
                    "same_crop_unknown": {"actual": 7, "passed": True},
                    "wrong_part": {"actual": 5, "passed": True},
                },
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_readiness("apricot__leaf", path)

    assert result["blockers"] == []
    assert result["metrics"]["id_test_samples"] == 34
    assert result["metrics"]["min_ood_type_samples"] == 5
