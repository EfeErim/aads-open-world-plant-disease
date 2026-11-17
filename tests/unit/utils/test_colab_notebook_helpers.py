import json
from datetime import datetime
from pathlib import Path

from src.colab.checkpointing import TrainingCheckpointManager
from src.colab.training_helpers import (
    apply_notebook_optimization_proposal,
    ensure_notebook_checkpoint_manager,
    maybe_auto_disconnect_colab_runtime,
    resolve_notebook_optimization_campaign,
    summarize_notebook_optimization_campaign,
)
from src.colab.training_session import (
    build_notebook_completion_report,
    build_notebook_run_dir,
    build_notebook_run_id,
    cleanup_notebook_training_state,
    merge_training_summary_fields,
    persist_production_readiness_artifact,
    persist_validation_artifacts,
)


def test_cleanup_notebook_training_state_releases_loaders_and_globals(monkeypatch):
    class Iterator:
        def __init__(self):
            self.shutdown = False

        def _shutdown_workers(self):
            self.shutdown = True

    class Loader:
        def __init__(self):
            self._iterator = Iterator()

    loader = Loader()
    iterator = loader._iterator
    state = {"adapter": object(), "loaders": {"train": loader}, "history": {"loss": [1.0]}}
    shared_globals = {"STATE": state, "adapter": object(), "trainer": object(), "session": object()}
    messages = []
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    cleanup_notebook_training_state(shared_globals, label="tomato__leaf", print_fn=messages.append)

    assert iterator.shutdown is True
    assert loader._iterator is None
    assert "adapter" not in state
    assert "loaders" not in state
    assert "history" not in state
    assert "adapter" not in shared_globals
    assert "trainer" not in shared_globals
    assert "session" not in shared_globals
    assert messages[-1] == "[CLEANUP] Released runtime state after tomato__leaf."


def test_persist_validation_artifacts_writes_metric_gate(tmp_path: Path):
    result = persist_validation_artifacts(
        root=tmp_path,
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 1, 1],
        classes=["healthy", "disease_a"],
        telemetry=None,
        require_ood=False,
        context={"crop": "tomato"},
    )

    metric_gate_path = result["paths"]["metric_gate_json"]
    assert metric_gate_path.exists()
    assert result["metric_gate"]["evaluation"]["gating"]["status"] in {"soft", "ready"}


def test_persist_validation_artifacts_supports_custom_artifact_subdir(tmp_path: Path):
    result = persist_validation_artifacts(
        root=tmp_path,
        y_true=[0, 1],
        y_pred=[0, 1],
        classes=["healthy", "disease_a"],
        artifact_subdir="test",
        context={"crop": "tomato", "split_name": "test"},
    )

    assert result["paths"]["report_txt"].exists()
    assert result["paths"]["report_txt"].parent.name == "test"


def test_persist_validation_artifacts_forwards_extended_ood_artifacts(tmp_path: Path):
    result = persist_validation_artifacts(
        root=tmp_path,
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        classes=["healthy", "disease_a"],
        require_ood=True,
        emit_metric_gate=False,
        ood_labels=[0, 0, 0, 0, 1, 1],
        ood_scores=[0.1, 0.2, 0.15, 0.18, 0.8, 0.9],
        ood_scores_by_method={
            "ensemble": [0.1, 0.2, 0.15, 0.18, 0.8, 0.9],
            "energy": [0.2, 0.25, 0.22, 0.24, 0.86, 0.95],
        },
        ood_type_breakdown={
            "blur": {
                "sample_count": 2,
                "method_metrics": {
                    "ensemble": {"ood_auroc": 0.81, "ood_false_positive_rate": 0.12, "in_distribution_samples": 4},
                    "energy": {"ood_auroc": 0.84, "ood_false_positive_rate": 0.08, "in_distribution_samples": 4},
                },
            }
        },
        prediction_rows=[
            {
                "sample_origin": "in_distribution",
                "split_name": "test",
                "image_path": "runtime/tomato/test/healthy/img1.jpg",
                "true_label": "healthy",
                "pred_label": "healthy",
                "is_correct": True,
                "class_confidence": 0.99,
            }
        ],
        context={
            "crop_name": "tomato",
            "split_name": "test",
            "ood_requested_primary_score_method": "auto",
            "ood_primary_score_method": "ensemble",
            "ood_primary_score_selection_source": "real_ood_guardrail_no_dev",
        },
    )

    assert "metric_gate_json" not in result["paths"]
    assert result["paths"]["ood_type_breakdown_json"].exists()
    assert result["paths"]["ood_method_comparison_json"].exists()
    assert result["paths"]["predictions_csv"].exists()


def test_persist_production_readiness_artifact_forwards_require_ood_flag(tmp_path: Path):
    readiness = persist_production_readiness_artifact(
        root=tmp_path,
        classification_metric_gate={
            "metrics": {
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "weighted_f1": 1.0,
                "balanced_accuracy": 1.0,
            }
        },
        classification_split="test",
        ood_evidence_source="unavailable",
        ood_metrics={},
        require_ood=False,
    )

    assert readiness["payload"]["ood_evidence"]["evaluation"]["require_ood"] is False


def test_ensure_notebook_checkpoint_manager_returns_existing_instance(tmp_path: Path):
    existing = TrainingCheckpointManager(tmp_path / "telemetry" / "run_1", retention=2)

    resolved = ensure_notebook_checkpoint_manager(existing, run_id="ignored", drive_root=tmp_path / "ignored")

    assert resolved is existing


def test_ensure_notebook_checkpoint_manager_builds_manager_when_missing(tmp_path: Path):
    resolved = ensure_notebook_checkpoint_manager(
        None,
        run_id="run_2",
        drive_root=tmp_path,
        retention=4,
    )

    assert isinstance(resolved, TrainingCheckpointManager)
    assert resolved.root_dir == tmp_path / "telemetry" / "run_2"
    assert resolved.retention == 4


def test_ensure_notebook_checkpoint_manager_generates_microsecond_run_ids(tmp_path: Path, monkeypatch):
    class _FakeDateTime:
        values = [
            datetime(2026, 3, 11, 12, 0, 0, 1),
            datetime(2026, 3, 11, 12, 0, 0, 2),
        ]

        @classmethod
        def now(cls):
            return cls.values.pop(0)

    monkeypatch.setattr("src.colab.training_helpers.datetime", _FakeDateTime)

    first = ensure_notebook_checkpoint_manager(None, drive_root=tmp_path)
    second = ensure_notebook_checkpoint_manager(None, drive_root=tmp_path)

    assert first.root_dir.name == "20260311_120000_000001"
    assert second.root_dir.name == "20260311_120000_000002"


class _FakeTelemetry:
    def __init__(self, summary_path: Path):
        self.local_summary_path = summary_path
        self.latest_payloads = []
        self.sync_calls = 0

    def update_latest(self, payload, **_kwargs):
        self.latest_payloads.append(dict(payload))

    def sync_pending(self):
        self.sync_calls += 1


def test_build_notebook_completion_report_marks_ready_when_outputs_exist(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    notebook_export_path = tmp_path / "notebooks" / "executed.ipynb"
    notebook_export_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_export_path.write_text("{}", encoding="utf-8")

    repo_run_exports = {}
    for name in ("outputs", "telemetry", "checkpoint_state"):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        repo_run_exports[name] = str(path)

    report = build_notebook_completion_report(
        state={
            "evaluation_artifacts": {"test": {"metric_gate": {}}},
            "behavioral_acceptance": {"status": "fail", "pass": False},
        },
        telemetry=_FakeTelemetry(summary_path),
        repo_run_exports=repo_run_exports,
        notebook_export_path=notebook_export_path,
    )

    assert report["ready"] is False
    assert report["missing"] == ["adapter_behavioral_acceptance_failed"]
    assert report["soft_missing"] == []
    assert report["evaluation_splits"] == ["test"]
    assert report["adapter_behavioral_acceptance_status"] == "fail"


def test_build_notebook_completion_report_treats_missing_notebook_export_as_soft_missing(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    repo_run_exports = {}
    for name in ("outputs", "telemetry", "checkpoint_state"):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        repo_run_exports[name] = str(path)

    report = build_notebook_completion_report(
        state={
            "evaluation_artifacts": {"test": {"metric_gate": {}}},
            "behavioral_acceptance": {"status": "pass", "pass": True},
        },
        telemetry=_FakeTelemetry(summary_path),
        repo_run_exports=repo_run_exports,
        notebook_export_path=tmp_path / "missing.ipynb",
    )

    assert report["ready"] is True
    assert report["missing"] == []
    assert report["soft_missing"] == ["executed_notebook_export"]
    assert report["checks"]["executed_notebook_export"] is False


def test_build_notebook_completion_report_rejects_unknown_behavioral_status(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    repo_run_exports = {}
    for name in ("outputs", "telemetry", "checkpoint_state"):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        repo_run_exports[name] = str(path)

    report = build_notebook_completion_report(
        state={
            "evaluation_artifacts": {"test": {"metric_gate": {}}},
            "behavioral_acceptance": {"status": "unknown_status", "pass": False},
        },
        telemetry=_FakeTelemetry(summary_path),
        repo_run_exports=repo_run_exports,
        notebook_export_path=tmp_path / "executed.ipynb",
    )

    assert report["ready"] is False
    assert report["checks"]["adapter_behavioral_acceptance"] is False
    assert report["missing"] == ["adapter_behavioral_acceptance"]


def test_maybe_auto_disconnect_colab_runtime_calls_unassign_when_ready(tmp_path: Path, monkeypatch):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    notebook_export_path = tmp_path / "executed.ipynb"
    notebook_export_path.write_text("{}", encoding="utf-8")

    repo_run_exports = {}
    for name in ("outputs", "telemetry", "checkpoint_state"):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        repo_run_exports[name] = str(path)

    telemetry = _FakeTelemetry(summary_path)
    completion_report = build_notebook_completion_report(
        state={
            "evaluation_artifacts": {"test": {"metric_gate": {}}},
            "behavioral_acceptance": {"status": "pass", "pass": True},
        },
        telemetry=telemetry,
        repo_run_exports=repo_run_exports,
        notebook_export_path=notebook_export_path,
    )

    calls = []

    class _FakeRuntime:
        def unassign(self):
            calls.append("unassign")

    monkeypatch.setattr(
        "src.colab.training_helpers._resolve_colab_runtime_api",
        lambda: _FakeRuntime(),
    )

    lines = []
    result = maybe_auto_disconnect_colab_runtime(
        enabled=True,
        grace_period_sec=0.0,
        telemetry=telemetry,
        completion_report=completion_report,
        print_fn=lines.append,
    )

    assert result["disconnect_requested"] is True
    assert calls == ["unassign"]
    assert telemetry.sync_calls == 1
    assert telemetry.latest_payloads[-1]["phase"] == "auto_disconnect_pending"
    assert lines == ["[COLAB] Work complete. Disconnecting runtime now to avoid idle credit use."]


def test_maybe_auto_disconnect_colab_runtime_skips_when_checks_are_incomplete(tmp_path: Path, monkeypatch):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    telemetry = _FakeTelemetry(summary_path)
    monkeypatch.setattr(
        "src.colab.training_helpers._resolve_colab_runtime_api",
        lambda: None,
    )

    lines = []
    result = maybe_auto_disconnect_colab_runtime(
        enabled=True,
        grace_period_sec=0.0,
        state={"evaluation_artifacts": {}, "behavioral_acceptance": {}},
        telemetry=telemetry,
        repo_run_exports={},
        notebook_export_path=tmp_path / "missing.ipynb",
        print_fn=lines.append,
    )

    assert result["ready"] is False
    assert result["disconnect_requested"] is not True
    assert telemetry.sync_calls == 1
    assert telemetry.latest_payloads[-1]["phase"] == "auto_disconnect_skipped"
    assert telemetry.latest_payloads[-1]["completion_missing"] == [
        "evaluation_artifacts",
        "adapter_behavioral_acceptance",
        "repo_exports",
    ]
    assert telemetry.latest_payloads[-1]["completion_soft_missing"] == ["executed_notebook_export"]
    assert lines == [
        (
            "[COLAB] Auto-disconnect skipped. Incomplete required checks: "
            "evaluation_artifacts, adapter_behavioral_acceptance, repo_exports"
        ),
        "[COLAB] Soft-missing checks: executed_notebook_export",
    ]


def test_maybe_auto_disconnect_colab_runtime_proceeds_when_only_notebook_export_is_soft_missing(
    tmp_path: Path,
    monkeypatch,
):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    repo_run_exports = {}
    for name in ("outputs", "telemetry", "checkpoint_state"):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        repo_run_exports[name] = str(path)

    telemetry = _FakeTelemetry(summary_path)
    completion_report = build_notebook_completion_report(
        state={
            "evaluation_artifacts": {"test": {"metric_gate": {}}},
            "behavioral_acceptance": {"status": "pass", "pass": True},
        },
        telemetry=telemetry,
        repo_run_exports=repo_run_exports,
        notebook_export_path=tmp_path / "missing.ipynb",
    )

    calls = []

    class _FakeRuntime:
        def unassign(self):
            calls.append("unassign")

    monkeypatch.setattr(
        "src.colab.training_helpers._resolve_colab_runtime_api",
        lambda: _FakeRuntime(),
    )

    lines = []
    result = maybe_auto_disconnect_colab_runtime(
        enabled=True,
        grace_period_sec=0.0,
        telemetry=telemetry,
        completion_report=completion_report,
        print_fn=lines.append,
    )

    assert result["ready"] is True
    assert result["soft_missing"] == ["executed_notebook_export"]
    assert result["disconnect_requested"] is True
    assert calls == ["unassign"]
    assert telemetry.latest_payloads[-1]["phase"] == "auto_disconnect_pending"
    assert telemetry.latest_payloads[-1]["completion_soft_missing"] == ["executed_notebook_export"]
    assert lines == [
        "[COLAB] Proceeding despite soft-missing checks: executed_notebook_export",
        "[COLAB] Work complete. Disconnecting runtime now to avoid idle credit use.",
    ]

def test_build_notebook_run_id_includes_crop_part_and_timestamp():
    run_id = build_notebook_run_id(
        "Tomato Leaf",
        "Upper Part",
        now=datetime(2026, 3, 23, 10, 11, 12),
    )

    assert run_id == "tomato_leaf_upper_part_2026-03-23_10-11-12"


def test_build_notebook_run_dir_groups_by_crop_and_part(tmp_path: Path):
    run_dir = build_notebook_run_dir(
        tmp_path,
        "Tomato Leaf",
        "Upper Part",
        "tomato_leaf_upper_part_2026-03-23_10-11-12",
    )

    assert run_dir == tmp_path / "runs" / "tomato_leaf" / "upper_part" / "tomato_leaf_upper_part_2026-03-23_10-11-12"


class _ArtifactMergingTelemetry:
    def __init__(self):
        self.copied = []
        self.catalog_entries = []
        self.summary_metadata = []

    def copy_artifact_file(self, source_path, relative_path):
        self.copied.append((Path(source_path), str(relative_path)))

    def merge_artifact_catalog(self, entries):
        self.catalog_entries.extend(list(entries))

    def merge_summary_metadata(self, payload):
        self.summary_metadata.append(dict(payload))


def test_merge_training_summary_fields_updates_summary_and_traceability(tmp_path: Path):
    telemetry = _ArtifactMergingTelemetry()
    artifact_root = tmp_path / "outputs" / "colab_notebook_training" / "artifacts"
    training_dir = artifact_root / "training"
    (artifact_root / "test").mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    (artifact_root / "production_readiness.json").write_text("{}", encoding="utf-8")
    (artifact_root / "test" / "metric_gate.json").write_text("{}", encoding="utf-8")
    (training_dir / "run_context.json").write_text(
        json.dumps(
            {
                "run_id": "tomato_leaf_2026-03-23_10-11-12",
                "created_at": "2026-03-23T10:11:12+00:00",
                "crop_name": "tomato",
                "part_name": "leaf",
                "resolved_config": {
                    "training": {
                        "continual": {
                            "backbone": {"model_name": "fake/backbone"},
                            "adapter": {"lora_r": 24, "lora_alpha": 24, "lora_dropout": 0.1},
                            "fusion": {"layers": [2, 5, 8, 11], "output_dim": 768, "dropout": 0.1, "gating": "softmax"},
                            "ood": {
                                "threshold_factor": 3.0,
                                "primary_score_method": "ensemble",
                                "radial_l2_enabled": True,
                                "radial_beta_range": [0.5, 2.0],
                                "radial_beta_steps": 16,
                                "sure_enabled": True,
                                "sure_semantic_percentile": 90.0,
                                "sure_confidence_percentile": 97.0,
                                "conformal_enabled": True,
                                "conformal_alpha": 0.05,
                                "conformal_method": "raps",
                                "conformal_raps_lambda": 0.2,
                                "conformal_raps_k_reg": 1,
                            },
                            "learning_rate": 0.0002,
                            "weight_decay": 0.01,
                            "num_epochs": 20,
                            "batch_size": 128,
                            "seed": 42,
                            "optimization": {
                                "loss_name": "logitnorm",
                                "logitnorm_tau": 1.0,
                                "grad_accumulation_steps": 4,
                                "mixed_precision": "bf16",
                                "max_grad_norm": 1.0,
                                "scheduler": {
                                    "name": "cosine",
                                    "warmup_ratio": 0.1,
                                    "min_lr": 1e-6,
                                    "step_on": "batch",
                                },
                            },
                            "data": {
                                "sampler": "auto",
                                "augmentation_policy": "randaugment",
                                "randaugment_num_ops": 2,
                                "randaugment_magnitude": 7,
                            },
                        }
                    }
                },
                "dataset": {
                    "crop_root": "/content/data/prepared_runtime_datasets/tomato__leaf",
                    "dataset_key": "tomato__leaf",
                    "resolution_source": "manifest:split_manifest.json",
                    "manifests": {
                        "split_manifest.json": {
                            "path": "/content/data/prepared_runtime_datasets/tomato__leaf/split_manifest.json",
                            "exists": True,
                            "sha256": "sha_summary_test",
                            "schema_version": "v1_grouped_runtime_layout",
                            "source_root": "/content/source/tomato_leaf",
                            "crop_name": "tomato",
                            "part_name": "leaf",
                            "dataset_key": "tomato__leaf",
                            "split_policy": "grouped_family_canonical_eval_60_20_20",
                            "ood": {
                                "source_root": "/content/ood/tomato_leaf",
                                "image_count": 20,
                                "image_fingerprint": "ood_fp_summary",
                            },
                        }
                    },
                },
                "training_runtime": {"train_sampler": {"resolved_sampler": "weighted"}},
            }
        ),
        encoding="utf-8",
    )
    (artifact_root / "test" / "classification_report.json").write_text(
        json.dumps(
            {
                "accuracy": 0.95,
                "macro avg": {"f1-score": 0.93},
                "weighted avg": {"f1-score": 0.92},
            }
        ),
        encoding="utf-8",
    )
    (artifact_root / "production_readiness.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "passed": False,
                "ood_evidence_source": "real_ood_split",
                "classification_evidence": {
                    "split_name": "test",
                    "metrics": {"accuracy": 0.95, "balanced_accuracy": 0.94, "macro_f1": 0.93},
                },
                "ood_evidence": {
                    "metrics": {
                        "ood_auroc": 0.91,
                        "ood_false_positive_rate": 0.04,
                        "sure_ds_f1": 0.90,
                        "conformal_empirical_coverage": 0.96,
                        "conformal_avg_set_size": 1.2,
                        "ood_samples": 20,
                        "in_distribution_samples": 30,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    payload = merge_training_summary_fields(
        root=tmp_path,
        telemetry=telemetry,
        payload={
            "run_id": "tomato_leaf_2026-03-23_10-11-12",
            "run_label": "tomato_leaf_2026-03-23_10-11-12",
            "crop_name": "tomato",
            "part_name": "leaf",
            "dataset_key": "tomato__leaf",
            "created_at": "2026-03-23T10:11:12+00:00",
            "notebook_surface": "2_train_continual_sd_lora_adapter.ipynb",
            "dataset_roots": {
                "runtime_dataset_key": "tomato__leaf",
                "runtime_dataset_root": "data/prepared_runtime_datasets/tomato__leaf",
            },
            "notebook_parameters": {"batch_size": 128, "learning_rate": 0.0002},
            "readiness_summary": {
                "status": "failed",
                "passed": False,
                "ood_evidence_source": "real_ood_split",
            },
        },
    )

    summary_path = artifact_root / "training" / "summary.json"
    experiment_manifest_path = artifact_root / "training" / "experiment_manifest.json"
    optimization_record_path = artifact_root / "training" / "optimization_record.json"

    assert summary_path.exists()
    assert payload["part_name"] == "leaf"
    assert experiment_manifest_path.exists()
    assert optimization_record_path.exists()
    assert any(relative_path == "training/summary.json" for _source, relative_path in telemetry.copied)

    experiment_manifest = json.loads(experiment_manifest_path.read_text(encoding="utf-8"))
    optimization_record = json.loads(optimization_record_path.read_text(encoding="utf-8"))
    assert experiment_manifest["surface"] == "notebook_2"
    assert experiment_manifest["notebook_context"]["dataset_roots"]["runtime_dataset_key"] == "tomato__leaf"
    assert optimization_record["surface"] == "notebook_2"
    assert optimization_record["notebook_context"]["notebook_parameters"]["batch_size"] == 128


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_split_manifest(dataset_root: Path, dataset_key: str) -> None:
    _write_json(
        dataset_root / "split_manifest.json",
        {
            "schema_version": "v1_grouped_runtime_layout",
            "crop_name": "tomato",
            "part_name": "leaf",
            "dataset_key": dataset_key,
            "split_policy": "grouped_family_canonical_eval_60_20_20",
        },
    )


def _write_canonical_run(
    *,
    runs_root: Path,
    run_name: str,
    dataset_lineage_key: str,
    learning_rate: float,
    macro_f1: float,
    auroc: float,
    fpr: float,
    backbone_model_name: str = "fake/backbone",
) -> None:
    artifact_root = runs_root / run_name / "training_metrics"
    manifest = {
        "schema_version": "v1_training_experiment_manifest",
        "record_quality": "canonical",
        "run_id": run_name,
        "run_label": run_name,
        "created_at": "2026-04-14T12:00:00+00:00",
        "surface": "workflow",
        "crop_name": "tomato",
        "part_name": "leaf",
        "dataset_key": "tomato__leaf",
        "dataset_lineage_key": dataset_lineage_key,
        "model_family": {"engine": "continual_sd_lora", "backbone_model_name": backbone_model_name},
        "artifacts": {"artifact_root": str(artifact_root)},
    }
    optimization = {
        "schema_version": "v1_training_optimization_record",
        "record_quality": "canonical",
        "run_id": run_name,
        "run_label": run_name,
        "created_at": "2026-04-14T12:00:00+00:00",
        "surface": "workflow",
        "crop_name": "tomato",
        "part_name": "leaf",
        "dataset_key": "tomato__leaf",
        "dataset_lineage_key": dataset_lineage_key,
        "comparability": {
            "dataset_lineage_key": dataset_lineage_key,
            "crop_name": "tomato",
            "part_name": "leaf",
            "engine": "continual_sd_lora",
            "backbone_model_name": backbone_model_name,
            "cohort_key": f"{dataset_lineage_key}::tomato::leaf::continual_sd_lora::{backbone_model_name}",
        },
        "status": {
            "readiness_status": "ready",
            "readiness_passed": True,
            "authoritative_split": "test",
            "ood_evidence_source": "real_ood_split",
        },
        "parameters": {
            "training.learning_rate": learning_rate,
            "training.weight_decay": 0.01,
            "training.num_epochs": 12,
            "training.batch_size": 128,
            "training.adapter.lora_r": 24,
            "training.adapter.lora_alpha": 24,
            "training.adapter.lora_dropout": 0.1,
            "training.fusion.dropout": 0.1,
            "training.ood.threshold_factor": 3.0,
            "training.ood.react_enabled": False,
            "training.ood.react_percentile": 0.99,
            "training.optimization.logitnorm_tau": 1.0,
            "training.optimization.label_smoothing": 0.0,
            "training.data.augmentation_policy": "randaugment",
            "training.data.randaugment_num_ops": 2,
            "training.data.randaugment_magnitude": 7,
            "training.data.augmix_severity": 3,
            "training.classifier_rebalance.enabled": False,
        },
        "objectives": {
            "classification.macro_f1": macro_f1,
            "ood.ood_auroc": auroc,
            "ood.ood_false_positive_rate": fpr,
        },
        "objective_directions": {
            "classification.macro_f1": "maximize",
            "ood.ood_auroc": "maximize",
            "ood.ood_false_positive_rate": "minimize",
        },
        "artifacts": {"artifact_root": str(artifact_root)},
    }
    _write_json(artifact_root / "training" / "experiment_manifest.json", manifest)
    _write_json(artifact_root / "training" / "optimization_record.json", optimization)


def test_resolve_notebook_optimization_campaign_bootstrap_pending_when_no_trials(tmp_path: Path):
    dataset_root = tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf"
    dataset_root.mkdir(parents=True, exist_ok=True)
    _write_split_manifest(dataset_root, "tomato__leaf")
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)

    campaign = resolve_notebook_optimization_campaign(
        root=tmp_path,
        runtime_dataset_root=dataset_root,
        dataset_key="tomato__leaf",
        crop_name="tomato",
        part_name="leaf",
        backbone_model_name="fake/backbone",
        notebook_parameters={"EPOCHS": 20, "BATCH_SIZE": 128, "LEARNING_RATE": 0.0002, "LORA_R": 24, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.1, "WEIGHT_DECAY": 0.01, "OOD_FACTOR": 3.0, "LOGITNORM_TAU": 1.0, "RANDAUGMENT_MAGNITUDE": 7},
        mode="continue",
    )

    assert campaign["status"] == "bootstrap_pending"
    assert Path(campaign["campaign_json"]).exists()
    assert summarize_notebook_optimization_campaign(campaign)["executed_run_count"] == 0


def test_resolve_notebook_optimization_campaign_falls_back_to_legacy_dataset_key(tmp_path: Path):
    dataset_root = tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf"
    dataset_root.mkdir(parents=True, exist_ok=True)
    _write_split_manifest(dataset_root, "tomato__leaf")

    # Legacy runs may only carry dataset_key as lineage and blank backbone metadata.
    _write_canonical_run(
        runs_root=tmp_path / "runs",
        run_name="run_legacy",
        dataset_lineage_key="tomato__leaf",
        learning_rate=0.00012,
        macro_f1=0.80,
        auroc=0.72,
        fpr=0.20,
        backbone_model_name="",
    )

    campaign = resolve_notebook_optimization_campaign(
        root=tmp_path,
        runtime_dataset_root=dataset_root,
        dataset_key="tomato__leaf",
        crop_name="tomato",
        part_name="leaf",
        backbone_model_name="fake/backbone",
        notebook_parameters={
            "EPOCHS": 20,
            "BATCH_SIZE": 128,
            "LEARNING_RATE": 0.0002,
            "LORA_R": 24,
            "LORA_ALPHA": 24,
            "LORA_DROPOUT": 0.1,
            "WEIGHT_DECAY": 0.01,
            "OOD_FACTOR": 3.0,
            "LOGITNORM_TAU": 1.0,
            "RANDAUGMENT_MAGNITUDE": 7,
        },
        mode="continue",
    )

    assert campaign["status"] == "active"
    assert campaign["cohort_match_mode"] == "legacy_dataset_key_blank_backbone"
    assert campaign["eligible_run_count"] == 1
    assert campaign["next_proposal"]["parameters"]


def test_apply_notebook_optimization_proposal_updates_visible_parameters(tmp_path: Path):
    campaign_path = tmp_path / "runs" / "_index" / "notebook_optimization_campaigns" / "campaign.json"
    campaign_path.parent.mkdir(parents=True, exist_ok=True)
    campaign = {
        "campaign_json": str(campaign_path),
        "status": "active",
        "next_proposal": {
            "rank": 1,
            "signature": "sig_1",
            "parameters": {
                "training.learning_rate": 0.00015,
                "training.num_epochs": 16,
                "training.adapter.lora_r": 32,
                "training.adapter.lora_dropout": 0.18,
                "training.fusion.dropout": 0.12,
                "training.ood.oe_loss_weight": 0.35,
                "training.ood.react_enabled": True,
                "training.ood.react_percentile": 0.995,
                "training.data.augmentation_policy": "augmix",
                "training.data.randaugment_num_ops": 3,
                "training.data.randaugment_magnitude": 9,
                "training.classifier_rebalance.enabled": True,
                "training.classifier_rebalance.logit_adjustment_tau": 1.4,
            },
        },
    }

    result = apply_notebook_optimization_proposal(
        notebook_parameters={
            "EPOCHS": 20,
            "BATCH_SIZE": 128,
            "LEARNING_RATE": 0.0002,
            "LORA_R": 24,
            "LORA_ALPHA": 24,
            "LORA_DROPOUT": 0.1,
            "FUSION_DROPOUT": 0.1,
            "WEIGHT_DECAY": 0.01,
            "OOD_FACTOR": 3.0,
            "OE_LOSS_WEIGHT": 0.2,
            "REACT_ENABLED": False,
            "REACT_PERCENTILE": 0.99,
            "AUGMENTATION_POLICY": "randaugment",
            "LOGITNORM_TAU": 1.0,
            "CLASSIFIER_REBALANCE_ENABLED": False,
            "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU": 1.0,
            "RANDAUGMENT_NUM_OPS": 2,
            "RANDAUGMENT_MAGNITUDE": 7,
        },
        campaign=campaign,
        print_fn=lambda _line: None,
    )

    assert result["applied"] is True
    assert result["notebook_parameters"]["LEARNING_RATE"] == 0.00015
    assert result["notebook_parameters"]["EPOCHS"] == 16
    assert result["notebook_parameters"]["LORA_R"] == 32
    assert result["notebook_parameters"]["LORA_DROPOUT"] == 0.18
    assert result["notebook_parameters"]["FUSION_DROPOUT"] == 0.12
    assert result["notebook_parameters"]["OE_LOSS_WEIGHT"] == 0.35
    assert result["notebook_parameters"]["REACT_ENABLED"] is True
    assert result["notebook_parameters"]["REACT_PERCENTILE"] == 0.995
    assert result["notebook_parameters"]["AUGMENTATION_POLICY"] == "augmix"
    assert result["notebook_parameters"]["CLASSIFIER_REBALANCE_ENABLED"] is True
    assert result["notebook_parameters"]["CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU"] == 1.4
    assert result["notebook_parameters"]["RANDAUGMENT_NUM_OPS"] == 3
    assert result["notebook_parameters"]["RANDAUGMENT_MAGNITUDE"] == 9


