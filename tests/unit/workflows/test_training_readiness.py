from src.workflows.training_readiness import (
    build_production_readiness_context,
    record_adapter_export_metadata,
    select_authoritative_artifacts,
    select_authoritative_evaluation,
)


class _FakeEvaluation:
    def __init__(self, y_true, *, prediction_rows=None):
        self.y_true = list(y_true)
        self.prediction_rows = list(prediction_rows or [])


class _RecordingAdapter:
    def __init__(self):
        self.calls = []

    def set_export_metadata(self, *, ood_calibration=None, adapter_runtime=None):
        self.calls.append(
            {
                "ood_calibration": dict(ood_calibration or {}),
                "adapter_runtime": dict(adapter_runtime or {}),
            }
        )


def test_select_authoritative_artifacts_prefers_test_split():
    split_name, artifacts = select_authoritative_artifacts(
        {"metric_gate": {"passed": True}},
        {"metric_gate": {"passed": False}},
        calibration_split_name="val",
    )

    assert split_name == "test"
    assert artifacts["metric_gate"]["passed"] is False


def test_select_authoritative_evaluation_skips_val_when_used_for_calibration():
    split_name, evaluation = select_authoritative_evaluation(
        _FakeEvaluation([0, 1]),
        None,
        calibration_split_name="val",
    )

    assert split_name == ""
    assert evaluation is None


def test_record_adapter_export_metadata_passes_runtime_metadata():
    adapter = _RecordingAdapter()

    record_adapter_export_metadata(
        adapter,
        ood_calibration={"ood_calibration": {"version": 3}},
        calibration_split_name="val",
        calibration_loader_size=12,
        authoritative_split="test",
        ood_evidence_source="real_ood_split",
        crop_name="tomato",
        part_name="fruit",
        requested_primary_score_method="auto",
        selected_primary_score_method="energy",
        selection_source="real_ood_split",
        best_state_restored=True,
    )

    assert adapter.calls == [
        {
            "ood_calibration": {
                "version": 3,
                "source_split": "val",
                "source_loader_size": 12,
                "authoritative_classification_split": "test",
                "ood_evidence_source": "real_ood_split",
                "requested_primary_score_method": "auto",
                "primary_score_method": "energy",
                "selection_source": "real_ood_split",
            },
            "adapter_runtime": {
                "best_state_restored": True,
                "crop_name": "tomato",
                "part_name": "fruit",
            },
        }
    ]


def test_build_production_readiness_context_tracks_selection_fields():
    payload = build_production_readiness_context(
        run_id="run_1",
        crop_name="tomato",
        loader_sizes={"train": 10},
        loader_batch_counts={"train": 2},
        split_class_counts={"train": {"healthy": 10}},
        calibration_split_name="val",
        best_state_restored=True,
        classification_split="test",
        requested_primary_score_method="auto",
        selected_primary_score_method="knn",
        selection_source="held_out_benchmark",
        ood_benchmark={"status": "completed", "passed": True},
        ood_method_comparison={"selected_primary_score_method": "knn"},
    )

    assert payload["classification_split"] == "test"
    assert payload["ood_requested_primary_score_method"] == "auto"
    assert payload["ood_primary_score_method"] == "knn"
    assert payload["ood_primary_score_selection_source"] == "held_out_benchmark"
    assert payload["ood_method_comparison"]["selected_primary_score_method"] == "knn"
