import json
from pathlib import Path

from src.training.services.reporting import (
    persist_batch_metrics_artifacts,
    persist_behavioral_acceptance_artifact,
    persist_behavioral_dev_report_artifact,
    persist_training_history_artifacts,
    persist_training_results_figure,
    persist_training_summary_artifact,
    persist_validation_artifacts,
)


def test_persist_behavioral_acceptance_is_the_default_decision_artifact(tmp_path: Path):
    rows = [
        {"sample_origin": "in_distribution", "ood_predicted": False, "is_correct": True}
        for _ in range(30)
    ]
    rows.extend(
        {
            "sample_origin": "ood",
            "ood_type": "same_crop_unsupported_disease",
            "ood_predicted": True,
            "image_path": f"ood/test/unsupported/{index}.jpg",
        }
        for index in range(30)
    )

    result = persist_behavioral_acceptance_artifact(
        artifact_root=tmp_path,
        classification_metrics={"accuracy": 0.94, "balanced_accuracy": 0.91, "macro_f1": 0.91},
        prediction_rows=rows,
        context={"crop_name": "tomato", "part_name": "leaf"},
    )

    assert result["payload"]["pass"] is True
    assert result["acceptance_json"].name == "adapter_behavioral_acceptance.json"
    assert json.loads(result["acceptance_json"].read_text(encoding="utf-8"))["decision_authority"] == (
        "adapter_behavioral_acceptance"
    )


def test_persist_behavioral_dev_report_is_non_authoritative(tmp_path: Path):
    rows = [
        {
            "sample_origin": "in_distribution",
            "ood_predicted": False,
            "is_correct": True,
            "image_path": f"validation/healthy/{index}.jpg",
        }
        for index in range(30)
    ]
    rows.extend(
        {
            "sample_origin": "ood",
            "ood_type": "same_crop_unsupported_disease",
            "ood_predicted": True,
            "image_path": f"ood/dev/unsupported/{index}.jpg",
        }
        for index in range(30)
    )

    result = persist_behavioral_dev_report_artifact(
        artifact_root=tmp_path,
        classification_metrics={"accuracy": 0.94, "balanced_accuracy": 0.91, "macro_f1": 0.91},
        prediction_rows=rows,
    )

    payload = json.loads(result["report_json"].read_text(encoding="utf-8"))
    assert result["report_json"].name == "adapter_behavioral_dev_report.json"
    assert payload["pass"] is True
    assert payload["authoritative"] is False
    assert payload["context"]["locked_test_evidence_loaded"] is False


def test_reporting_writes_training_and_validation_artifacts(tmp_path: Path):
    artifact_root = tmp_path / "training_metrics"
    history_snapshot = {
        "train_loss": [0.8, 0.4],
        "val_loss": [0.9, 0.5],
        "val_accuracy": [0.4, 0.75],
        "macro_precision": [0.3, 0.7],
        "macro_recall": [0.35, 0.72],
        "macro_f1": [0.32, 0.71],
        "weighted_f1": [0.34, 0.74],
        "balanced_accuracy": [0.35, 0.72],
        "generalization_gap": [0.1, 0.1],
        "best_metric_name": "val_loss",
        "best_metric_value": 0.5,
        "best_epoch": 2,
    }
    batch_history = [
        {"epoch": 1, "batch": 1, "global_step": 1, "optimizer_steps": 1, "loss": 0.8, "lr": 1e-3},
        {"epoch": 1, "batch": 2, "global_step": 2, "optimizer_steps": 2, "loss": 0.6, "lr": 8e-4},
    ]

    training_paths = persist_training_history_artifacts(
        artifact_root=artifact_root,
        history_snapshot=history_snapshot,
    )
    batch_paths = persist_batch_metrics_artifacts(
        artifact_root=artifact_root,
        batch_history=batch_history,
    )
    figure_paths = persist_training_results_figure(
        artifact_root=artifact_root,
        history_snapshot=history_snapshot,
        batch_history=batch_history,
    )
    summary_paths = persist_training_summary_artifact(
        artifact_root=artifact_root,
        summary_payload={"run_id": "run_1", "crop_name": "tomato"},
    )
    validation_paths = persist_validation_artifacts(
        artifact_root=artifact_root,
        y_true=[0, 0],
        y_pred=[0, 0],
        classes=["healthy", "disease_a"],
        context={"crop_name": "tomato"},
    )

    assert training_paths["history_json"].exists()
    assert training_paths["results_csv"].exists()
    assert batch_paths["batch_metrics_csv"].exists()
    assert figure_paths["results_png"].exists()
    assert summary_paths["summary_json"].exists()
    assert validation_paths["paths"]["cm_png"].exists()
    assert validation_paths["paths"]["cm_norm_png"].exists()
    assert validation_paths["paths"]["metric_gate_json"].exists()


def test_reporting_writes_ood_type_breakdown_artifact(tmp_path: Path):
    result = persist_validation_artifacts(
        artifact_root=tmp_path / "training_metrics",
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        classes=["healthy", "disease_a"],
        ood_type_breakdown={
            "blur": {"sample_count": 2, "metrics": {"ood_auroc": 0.8}},
        },
    )

    assert result["paths"]["ood_type_breakdown_json"].exists()


def test_reporting_writes_ood_evidence_summary_artifact(tmp_path: Path):
    result = persist_validation_artifacts(
        artifact_root=tmp_path / "training_metrics",
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        classes=["healthy", "disease_a"],
        ood_labels=[0, 0, 0, 0, 1, 1],
        ood_scores=[0.1, 0.2, 0.15, 0.18, 0.8, 0.9],
        sure_ds_f1=0.95,
        conformal_empirical_coverage=0.97,
        conformal_avg_set_size=1.0,
        ood_type_breakdown={
            "blur": {"sample_count": 2, "metrics": {"ood_auroc": 0.8}},
        },
        context={"ood_primary_score_method": "ensemble", "ood_score_methods": ["ensemble", "energy"]},
    )

    summary_path = result["paths"]["ood_evidence_summary_json"]
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["primary_score_method"] == "ensemble"
    assert summary["score_methods"] == ["ensemble", "energy"]
    assert summary["sample_counts"]["ood_samples"] == 2
    assert summary["sample_counts"]["in_distribution_samples"] == 4
    assert summary["ood_types"] == ["blur"]
    assert summary["ood_type_count"] == 1


def test_reporting_writes_ood_method_comparison_artifact(tmp_path: Path):
    result = persist_validation_artifacts(
        artifact_root=tmp_path / "training_metrics",
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        classes=["healthy", "disease_a"],
        ood_labels=[0, 0, 0, 0, 1, 1],
        ood_scores=[0.1, 0.2, 0.15, 0.18, 0.8, 0.9],
        ood_scores_by_method={
            "ensemble": [0.1, 0.2, 0.15, 0.18, 0.8, 0.9],
            "energy": [0.2, 0.3, 0.22, 0.25, 0.85, 0.95],
            "knn": [0.12, 0.21, 0.19, 0.2, 0.76, 0.88],
        },
        sure_ds_f1=0.95,
        conformal_empirical_coverage=0.97,
        conformal_avg_set_size=1.0,
        ood_type_breakdown={
            "blur": {
                "sample_count": 2,
                "method_metrics": {
                    "ensemble": {"ood_auroc": 0.81, "ood_false_positive_rate": 0.12, "in_distribution_samples": 4},
                    "energy": {"ood_auroc": 0.84, "ood_false_positive_rate": 0.08, "in_distribution_samples": 4},
                    "knn": {"ood_auroc": 0.79, "ood_false_positive_rate": 0.15, "in_distribution_samples": 4},
                },
            }
        },
        context={
            "ood_requested_primary_score_method": "auto",
            "ood_primary_score_method": "ensemble",
            "ood_primary_score_selection_source": "real_ood_guardrail_no_dev",
            "ood_score_methods": ["ensemble", "energy", "knn"],
        },
    )

    comparison_path = result["paths"]["ood_method_comparison_json"]
    summary_path = result["paths"]["ood_evidence_summary_json"]
    assert comparison_path.exists()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["selected_primary_score_method"] == "ensemble"
    assert comparison["selection_source"] == "real_ood_guardrail_no_dev"
    assert comparison["methods"]["energy"]["worst_slice"]["slice_name"] == "blur"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["method_comparison"]["selected_primary_score_method"] == "ensemble"
    assert "energy" in summary["method_comparison"]["methods"]


def test_reporting_writes_prediction_rows_csv(tmp_path: Path):
    result = persist_validation_artifacts(
        artifact_root=tmp_path / "training_metrics",
        y_true=[0, 1],
        y_pred=[0, 1],
        classes=["healthy", "disease_a"],
        prediction_rows=[
            {
                "sample_origin": "in_distribution",
                "split_name": "val",
                "image_path": "runtime/tomato/val/healthy/img1.jpg",
                "true_label": "healthy",
                "pred_label": "healthy",
                "is_correct": True,
                "class_confidence": 0.99,
            }
        ],
    )

    predictions_csv = result["paths"]["predictions_csv"]
    assert predictions_csv.exists()
    body = predictions_csv.read_text(encoding="utf-8")
    assert "image_path" in body
    assert "runtime/tomato/val/healthy/img1.jpg" in body
