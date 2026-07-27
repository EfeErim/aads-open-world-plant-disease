import hashlib
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from scripts.colab_checkpointing import TrainingCheckpointManager
from scripts.colab_notebook_helpers import (
    NotebookTrainingStatusPrinter,
    apply_notebook_optimization_proposal,
    build_notebook_completion_report,
    build_notebook_run_dir,
    build_notebook_run_id,
    cleanup_notebook_training_state,
    complete_notebook_training_run,
    ensure_notebook_checkpoint_manager,
    finalize_notebook_optimization_campaign,
    initialize_notebook_training_engine,
    maybe_auto_disconnect_colab_runtime,
    merge_training_summary_fields,
    persist_production_readiness_artifact,
    persist_validation_artifacts,
    prepare_notebook_access_and_dataset,
    resolve_notebook_optimization_campaign,
    run_notebook_training_session,
    summarize_notebook_optimization_campaign,
)
from src.training.types import EvaluationArtifactsPayload, ValidationReport


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

    monkeypatch.setattr("scripts.colab_notebook_helpers.datetime", _FakeDateTime)

    first = ensure_notebook_checkpoint_manager(None, drive_root=tmp_path)
    second = ensure_notebook_checkpoint_manager(None, drive_root=tmp_path)

    assert first.root_dir.name == "20260311_120000_000001"
    assert second.root_dir.name == "20260311_120000_000002"


def test_notebook_training_status_printer_throttles_batch_updates():
    lines = []
    printer = NotebookTrainingStatusPrinter(
        total_epochs=5,
        batch_interval=50,
        min_interval_sec=15.0,
        print_fn=lines.append,
    )

    printer.handle(
        "batch_end",
        {
            "epoch": 1,
            "batch": 1,
            "total_batches": 100,
            "loss": 0.8,
            "lr": 0.0003,
            "samples_per_sec": 42.5,
            "elapsed_sec": 2.0,
            "eta_sec": 100.0,
        },
    )
    printer.handle(
        "batch_end",
        {
            "epoch": 1,
            "batch": 2,
            "total_batches": 100,
            "loss": 0.7,
            "lr": 0.0003,
            "samples_per_sec": 43.0,
            "elapsed_sec": 8.0,
            "eta_sec": 94.0,
        },
    )
    printer.handle(
        "batch_end",
        {
            "epoch": 1,
            "batch": 3,
            "total_batches": 100,
            "loss": 0.6,
            "lr": 0.0003,
            "samples_per_sec": 44.0,
            "elapsed_sec": 18.0,
            "eta_sec": 84.0,
        },
    )
    printer.handle(
        "batch_end",
        {
            "epoch": 1,
            "batch": 50,
            "total_batches": 100,
            "loss": 0.5,
            "lr": 0.0003,
            "samples_per_sec": 45.0,
            "elapsed_sec": 20.0,
            "eta_sec": 80.0,
        },
    )

    assert len(lines) == 3
    assert lines[0].startswith("[LIVE] 1/5 batch=1/100")
    assert "elapsed=2s" in lines[0]
    assert lines[1].startswith("[LIVE] 1/5 batch=3/100")
    assert "eta=1m24s" in lines[1]
    assert lines[2].startswith("[LIVE] 1/5 batch=50/100")


def test_notebook_training_status_printer_emits_validation_best_and_stop():
    lines = []
    printer = NotebookTrainingStatusPrinter(total_epochs=4, print_fn=lines.append)

    printer.handle(
        "validation_end",
        {
            "epoch_done": 2,
            "val_loss": 0.21,
            "val_accuracy": 0.93,
            "macro_f1": 0.91,
            "balanced_accuracy": 0.92,
            "generalization_gap": 0.04,
        },
    )
    printer.handle(
        "best_metric_updated",
        {"epoch_done": 2, "best_metric_name": "val_loss", "best_metric_value": 0.21},
    )
    printer.handle(
        "stop_requested",
        {"epoch": 3, "global_step": 120, "reason": "early_stopping"},
    )

    assert lines == [
        "[VALID] 2/4 val_loss=0.2100 val_acc=0.9300 macro_f1=0.9100 bal_acc=0.9200 gap=0.0400",
        "[BEST] 2/4 val_loss=0.2100",
        "[STOP] epoch=3 step=120 reason=early_stopping",
    ]


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
        "scripts.colab_notebook_helpers._resolve_colab_runtime_api",
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
        "scripts.colab_notebook_helpers._resolve_colab_runtime_api",
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
        "scripts.colab_notebook_helpers._resolve_colab_runtime_api",
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


def test_merge_training_summary_fields_updates_summary_and_guided_outputs(tmp_path: Path):
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
    guided_dir = artifact_root / "guided"
    catalog = json.loads((guided_dir / "02_file_catalog.json").read_text(encoding="utf-8"))

    assert summary_path.exists()
    assert payload["part_name"] == "leaf"
    assert experiment_manifest_path.exists()
    assert optimization_record_path.exists()
    assert (guided_dir / "00_start_here.md").exists()
    assert (guided_dir / "01_run_overview.json").exists()
    assert any(entry["relative_path"] == "training/summary.json" for entry in catalog["entries"])
    assert any(entry["relative_path"] == "training/experiment_manifest.json" for entry in catalog["entries"])
    assert any(entry["relative_path"] == "training/optimization_record.json" for entry in catalog["entries"])
    assert any(relative_path == "training/summary.json" for _source, relative_path in telemetry.copied)
    assert telemetry.catalog_entries
    assert telemetry.summary_metadata

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


def test_finalize_notebook_optimization_campaign_records_completed_run(tmp_path: Path):
    dataset_root = tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf"
    dataset_root.mkdir(parents=True, exist_ok=True)
    _write_split_manifest(dataset_root, "tomato__leaf")
    dataset_lineage_key = f"tomato__leaf::{hashlib.sha256((dataset_root / 'split_manifest.json').read_bytes()).hexdigest()}"
    runs_root = tmp_path / "runs"
    _write_canonical_run(
        runs_root=runs_root,
        run_name="run_a",
        dataset_lineage_key=dataset_lineage_key,
        learning_rate=0.00010,
        macro_f1=0.82,
        auroc=0.76,
        fpr=0.18,
    )

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
    applied = apply_notebook_optimization_proposal(
        notebook_parameters={"EPOCHS": 20, "BATCH_SIZE": 128, "LEARNING_RATE": 0.0002, "LORA_R": 24, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.1, "WEIGHT_DECAY": 0.01, "OOD_FACTOR": 3.0, "LOGITNORM_TAU": 1.0, "RANDAUGMENT_MAGNITUDE": 7},
        campaign=campaign,
        print_fn=lambda _line: None,
    )

    finalized = finalize_notebook_optimization_campaign(
        root=tmp_path,
        campaign=applied["campaign"],
        run_id="run_b",
    )

    assert "run_b" in finalized["executed_run_ids"]
    assert finalized["last_completed_run_id"] == "run_b"
    assert finalized["executed_proposal_signatures"]


def test_prepare_notebook_access_and_dataset_collects_access_and_runtime_dataset(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf"
    for split_name in ("continual", "val", "test", "ood"):
        (dataset_root / split_name / "healthy").mkdir(parents=True, exist_ok=True)

    access_report = {
        "github": {"read_access_mode": "ok"},
        "repo_updates": {"relation": "clean"},
        "huggingface": {"access_mode": "token"},
    }
    monkeypatch.setattr(
        "scripts.colab_repo_bootstrap.collect_notebook_access_report",
        lambda repo_root, hf_model_ids: access_report,
    )
    monkeypatch.setattr(
        "scripts.colab_repo_bootstrap.print_notebook_access_report",
        lambda report, print_fn=None: (print_fn or print)(f"[ACCESS] {report['github']['read_access_mode']}"),
    )

    lines = []
    result = prepare_notebook_access_and_dataset(
        root=tmp_path,
        base_config={"training": {"continual": {"backbone": {"model_name": "fake/backbone"}}}},
        crop_name="tomato",
        dataset_name="tomato__leaf",
        runtime_dataset_root="data/prepared_runtime_datasets",
        optimization_campaign_mode="continue",
        print_fn=lines.append,
    )

    assert result["validated"] is True
    assert result["runtime_dataset_key"] == "tomato__leaf"
    assert result["selected_dataset_root"] == dataset_root
    assert result["resolved_ood_root"] == str(dataset_root / "ood")
    assert result["access_report"] == access_report
    assert any("Bayesian campaign mode requested" in line for line in lines)


def test_initialize_notebook_training_engine_applies_campaign_proposal_and_builds_loader_state(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf"
    for split_name in ("continual", "val", "test"):
        (dataset_root / split_name / "Healthy Leaf").mkdir(parents=True, exist_ok=True)

    class _FakeParam:
        def __init__(self, count, requires_grad):
            self._count = count
            self.requires_grad = requires_grad

        def numel(self):
            return self._count

    class _FakeAdapter:
        def __init__(self, crop_name, device):
            self.crop_name = crop_name
            self.device = device
            self.engine_calls = []
            self.class_to_idx = {}

        def initialize_engine(self, class_names, config):
            self.engine_calls.append((list(class_names), dict(config)))
            self.class_to_idx = {name: index for index, name in enumerate(class_names)}

        def parameters(self):
            return [_FakeParam(10, True), _FakeParam(5, False)]

    monkeypatch.setattr(
        "scripts.colab_dataset_layout.resolve_notebook_training_classes",
        lambda available_classes, crop_name, taxonomy_path: {
            "selected_classes": ["healthy_leaf"],
            "used_taxonomy_filter": True,
            "reason": "matched",
            "matched_classes": ["healthy_leaf"],
            "unmatched_classes": [],
        },
    )
    monkeypatch.setattr(
        "src.adapter.independent_crop_adapter.IndependentCropAdapter",
        _FakeAdapter,
    )
    loader_calls = []

    class _FakeDataset:
        def __init__(self, class_names):
            self.class_to_idx = {name: index for index, name in enumerate(class_names)}

    class _FakeLoader:
        def __init__(self, class_names):
            self.dataset = _FakeDataset(class_names)

    def _fake_create_training_loaders(**kwargs):
        loader_calls.append(dict(kwargs))
        return {
            "train": _FakeLoader(kwargs.get("class_names", [])),
            "val": _FakeLoader(kwargs.get("class_names", [])),
            "test": _FakeLoader(kwargs.get("class_names", [])),
        }

    monkeypatch.setattr("src.data.loaders.create_training_loaders", _fake_create_training_loaders)
    monkeypatch.setattr(
        "scripts.colab_notebook_helpers.resolve_notebook_optimization_campaign",
        lambda **kwargs: {
            "status": "active",
            "next_proposal": {"rank": 1, "signature": "sig_1"},
            "frontier_run_ids": [],
            "campaign_json": str(tmp_path / "runs" / "_index" / "notebook_optimization_campaigns" / "campaign.json"),
        },
    )
    monkeypatch.setattr(
        "scripts.colab_notebook_helpers.apply_notebook_optimization_proposal",
        lambda **kwargs: {
            "applied": True,
            "notebook_parameters": dict(kwargs["notebook_parameters"], BATCH_SIZE=16, EPOCHS=14, RANDAUGMENT_NUM_OPS=4),
            "campaign": dict(kwargs["campaign"], selected_proposal={"rank": 1, "signature": "sig_1"}),
            "proposal": {"signature": "sig_1"},
        },
    )

    state = {
        "validated": True,
        "runtime_dataset_root": dataset_root.parent,
        "runtime_dataset_key": "tomato__leaf",
        "selected_dataset_name": "tomato__leaf",
        "resolved_ood_root": "",
    }
    result = initialize_notebook_training_engine(
        root=tmp_path,
        state=state,
        base_config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake/backbone"},
                    "adapter": {},
                    "ood": {},
                    "optimization": {"scheduler": {}},
                }
            }
        },
        crop_name="tomato",
        part_name="leaf",
        device="cpu",
        deterministic=True,
        notebook_parameters={
            "EPOCHS": 12,
            "BATCH_SIZE": 8,
            "LEARNING_RATE": 2e-4,
            "LORA_R": 24,
            "LORA_ALPHA": 24,
            "LORA_DROPOUT": 0.1,
            "WEIGHT_DECAY": 0.01,
            "OOD_FACTOR": 3.0,
            "LOGITNORM_TAU": 1.0,
            "RANDAUGMENT_NUM_OPS": 2,
            "RANDAUGMENT_MAGNITUDE": 7,
        },
        optimization_campaign_mode="continue",
        data_settings={
            "AUGMENTATION_POLICY": "randaugment",
            "RANDAUGMENT_NUM_OPS": 2,
            "ALLOW_UNDER_MIN_TRAINING": False,
        },
        loader_settings={
            "NUM_WORKERS": 0,
            "PREFETCH": 2,
            "USE_CACHE": False,
            "CACHE_SIZE": 0,
            "CACHE_TRAIN_SPLIT": True,
            "TARGET_SIZE": 224,
            "LOADER_ERROR_POLICY": "raise",
            "DATA_SAMPLER": "weighted",
            "SEED": 42,
            "VALIDATE_IMAGES_ON_INIT": True,
            "PIN_MEMORY": False,
        },
        ood_settings={
            "sure_semantic_percentile": 90.0,
            "sure_confidence_percentile": 97.0,
            "conformal_alpha": 0.05,
            "conformal_method": "raps",
            "conformal_raps_lambda": 0.2,
            "conformal_raps_k_reg": 1,
            "ber_enabled": False,
            "ber_lambda_old": 0.0,
            "ber_lambda_new": 0.0,
            "ber_warmup_steps": 0,
        },
        optimization_settings={
            "grad_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "mixed_precision": "bf16",
            "label_smoothing": 0.0,
            "loss_name": "logitnorm",
            "scheduler": {"name": "cosine", "warmup_ratio": 0.1, "min_lr": 1e-6, "step_on": "batch"},
        },
        early_stopping_settings={
            "EARLY_STOPPING_PATIENCE": 3,
            "EARLY_STOPPING_MIN_DELTA": 0.001,
        },
        print_fn=lambda _line: None,
    )

    assert result["notebook_parameters"]["BATCH_SIZE"] == 16
    assert state["class_names"] == ["Healthy Leaf"]
    assert state["continual_config"]["batch_size"] == 16
    assert state["continual_config"]["num_epochs"] == 14
    assert state["continual_config"]["data"]["randaugment_num_ops"] == 4
    assert state["adapter"].engine_calls[0][0] == ["Healthy Leaf"]
    assert loader_calls[0]["class_names"] == ["Healthy Leaf"]
    assert loader_calls[0]["randaugment_num_ops"] == 4
    assert state["adapter"].crop_name == "tomato"
    assert result["trainable_params"] == 10


def test_run_notebook_training_session_persists_history_and_marks_adapter_trained(tmp_path: Path):
    class _FakeHistory:
        def to_dict(self):
            return {"train_loss": [0.4], "val_loss": [0.3], "stopped_early": False}

    class _FakeSession:
        def __init__(self, observers):
            self._observers = observers

        def run(self):
            for observer in self._observers:
                observer({"event_type": "batch_end", "payload": {"epoch": 1, "batch": 1, "total_batches": 1, "loss": 0.4, "lr": 0.0002, "samples_per_sec": 5.0, "elapsed_sec": 1.0, "eta_sec": 0.0, "global_step": 1}})
                observer({"event_type": "epoch_end", "payload": {"epoch_done": 1, "epoch_loss": 0.4, "val_loss": 0.3, "val_accuracy": 0.9, "macro_f1": 0.88, "balanced_accuracy": 0.89, "generalization_gap": 0.02, "global_step": 1}})
            return _FakeHistory()

        def snapshot_state(self):
            return {"progress_state": {"epoch": 1, "global_step": 1}}

    class _FakeAdapter:
        def __init__(self):
            self.is_trained = False

        def build_training_session(self, _train_loader, **kwargs):
            return _FakeSession(kwargs["observers"])

    class _FakeCheckpointManager:
        def save_checkpoint(self, **kwargs):
            return {"path": str(tmp_path / "checkpoint.pt"), "epoch": 1, "global_step": 1}

    state = {
        "adapter": _FakeAdapter(),
        "loaders": {"train": object(), "val": object()},
        "checkpoint_manager": _FakeCheckpointManager(),
    }

    result = run_notebook_training_session(
        root=tmp_path,
        state=state,
        run_id="run_1",
        epochs=1,
        device="cpu",
        stdout_batch_interval=50,
        validation_every_n_epochs=1,
        checkpoint_every_n_steps=0,
        checkpoint_on_exception=True,
        print_fn=lambda _line: None,
    )

    assert state["adapter"].is_trained is True
    assert result["history"]["stopped_early"] is False
    assert Path(tmp_path / "outputs" / "colab_notebook_training" / "artifacts" / "training" / "history.json").exists()
    assert state["resume_state"]["progress_state"]["epoch"] == 1


def test_complete_notebook_training_run_finalizes_outputs_with_helper_callbacks(tmp_path: Path, monkeypatch):
    class _FakeModule:
        def eval(self):
            return None

    class _FakeTrainer:
        def __init__(self):
            self.adapter_model = _FakeModule()
            self.classifier = _FakeModule()
            self.fusion = _FakeModule()

    class _FakeDataset:
        def __len__(self):
            return 2

    class _FakeLoader:
        def __init__(self, name="loader"):
            self.name = name
            self.dataset = _FakeDataset()

    class _FakeTelemetry:
        def __init__(self):
            self.artifacts_dir = tmp_path / "telemetry_artifacts"
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            self.latest = []
            self.summary = []
            self.closed = []

        def update_latest(self, payload, **_kwargs):
            self.latest.append(dict(payload))

        def emit_event(self, *_args, **_kwargs):
            return None

        def merge_summary_metadata(self, payload):
            self.summary.append(dict(payload))

        def close(self, payload):
            self.closed.append(dict(payload))

    evaluation = EvaluationArtifactsPayload(
        report=ValidationReport(
            val_loss=0.1,
            val_accuracy=1.0,
            macro_precision=1.0,
            macro_recall=1.0,
            macro_f1=1.0,
            weighted_f1=1.0,
            balanced_accuracy=1.0,
            per_class_accuracy={"healthy": 1.0},
            per_class_support={"healthy": 2},
            worst_classes=[],
        ),
        y_true=[0, 0],
        y_pred=[0, 0],
        prediction_rows=[],
        ood_labels=[0, 1],
        ood_scores=[0.1, 0.9],
        ood_scores_by_method={"ensemble": [0.1, 0.9]},
        sure_ds_f1=0.8,
        conformal_empirical_coverage=0.9,
        conformal_avg_set_size=1.2,
        context={"extra": "value"},
    )
    evaluation_calls = []

    def _evaluate(_trainer, loader, ood_loader=None):
        evaluation_calls.append((loader.name, None if ood_loader is None else ood_loader.name))
        return evaluation

    monkeypatch.setattr(
        "src.training.validation.evaluate_model_with_artifact_metrics",
        _evaluate,
    )
    monkeypatch.setattr(
        "scripts.colab_notebook_helpers.persist_validation_artifacts",
        lambda **kwargs: {
            "metric_gate": {"metrics": {"ood_auroc": 0.91, "sure_ds_f1": 0.8, "conformal_empirical_coverage": 0.9}},
            "report_dict": {"accuracy": 0.95},
        },
    )
    monkeypatch.setattr(
        "scripts.colab_notebook_helpers.persist_production_readiness_artifact",
        lambda **kwargs: {
            "payload": {"status": "ready", "passed": True, "ood_evidence_source": kwargs["ood_evidence_source"]}
        },
    )
    monkeypatch.setattr(
        "scripts.colab_notebook_helpers.merge_training_summary_fields",
        lambda **kwargs: dict(kwargs["payload"]),
    )
    monkeypatch.setattr(
        "scripts.colab_notebook_helpers.finalize_notebook_optimization_campaign",
        lambda **kwargs: {
            "status": "active",
            "mode": "continue",
            "frontier_count": 1,
            "eligible_run_count": 2,
            "executed_run_ids": ["run_1"],
            "next_proposal": {"rank": 2, "signature": "sig_2"},
            "campaign_json": str(tmp_path / "runs" / "_index" / "campaign.json"),
        },
    )
    monkeypatch.setattr(
        "scripts.colab_notebook_helpers.build_notebook_completion_report",
        lambda **kwargs: {"checks": {"repo_exports": True}, "ready": True, "soft_missing": [], "missing": []},
    )
    disconnect_calls = []
    monkeypatch.setattr(
        "scripts.colab_notebook_helpers.maybe_auto_disconnect_colab_runtime",
        lambda **kwargs: disconnect_calls.append(dict(kwargs)),
    )

    telemetry = _FakeTelemetry()
    state = {
        "adapter": SimpleNamespace(
            _trainer=_FakeTrainer(),
            class_to_idx={"healthy": 0},
        ),
        "loaders": {
            "val": _FakeLoader("val"),
            "test": _FakeLoader("test"),
            "ood_dev": _FakeLoader("ood_dev"),
            "ood": _FakeLoader("ood_test"),
        },
        "ood_policy_selection": {
            "requested_primary_score_method": "auto",
            "selected_primary_score_method": "knn",
            "selection_source": "real_ood_dev",
            "selected_threshold": 0.4,
            "selection": {"selected_threshold": 0.4},
        },
        "continual_config": {
            "batch_size": 8,
            "learning_rate": 2e-4,
            "adapter": {"lora_r": 24},
            "ood": {"threshold_factor": 3.0},
            "optimization": {"mixed_precision": "bf16"},
        },
        "runtime_dataset_key": "tomato__leaf",
        "selected_dataset_name": "tomato__leaf",
        "selected_dataset_root": tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf",
        "resolved_ood_root": str(tmp_path / "data" / "prepared_runtime_datasets" / "tomato__leaf" / "ood"),
        "runtime_dataset_root": tmp_path / "data" / "prepared_runtime_datasets",
        "access_report": {"github": {"read_access_mode": "ok"}},
        "optimization_campaign": {"status": "active", "mode": "continue"},
        "loader_settings": {"NUM_WORKERS": 0},
        "training_runtime": {"checkpoint_every_n_steps": 0},
    }
    repo_run_dir = tmp_path / "runs" / "run_1"
    repo_output_dir = repo_run_dir / "outputs"
    repo_telemetry_dir = repo_run_dir / "telemetry"
    repo_checkpoint_state_dir = repo_run_dir / "checkpoint_state"
    repo_notebook_output_path = repo_run_dir / "executed.ipynb"
    for path in (repo_output_dir, repo_telemetry_dir, repo_checkpoint_state_dir):
        path.mkdir(parents=True, exist_ok=True)

    save_calls = []

    def _save_run_outputs():
        save_calls.append("save")
        return {
            "outputs": str(repo_output_dir),
            "telemetry": str(repo_telemetry_dir),
            "checkpoint_state": str(repo_checkpoint_state_dir),
        }

    def _export_notebook(target: Path):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
        return str(target)

    result = complete_notebook_training_run(
        root=tmp_path,
        state=state,
        base_config={"training": {"continual": {"evaluation": {"ood_benchmark_auto_run": False}}}},
        crop_name="tomato",
        part_name="leaf",
        run_id="run_1",
        device="cpu",
        epochs=12,
        runtime_dataset_root="data/prepared_runtime_datasets",
        repo_run_dir=repo_run_dir,
        repo_output_dir=repo_output_dir,
        repo_telemetry_dir=repo_telemetry_dir,
        repo_checkpoint_state_dir=repo_checkpoint_state_dir,
        repo_notebook_output_path=repo_notebook_output_path,
        auto_push_to_github=False,
        auto_push_remote_name="origin",
        auto_push_branch="master",
        auto_disconnect_runtime=True,
        auto_disconnect_grace_seconds=0.0,
        save_run_outputs_to_repo_fn=_save_run_outputs,
        export_current_colab_notebook_fn=_export_notebook,
        push_repo_run_to_github_fn=lambda *args, **kwargs: {"enabled": True, "pushed": True},
        telemetry=telemetry,
        print_fn=lambda _line: None,
    )

    assert "test" in result["evaluation_artifacts"]
    assert result["production_readiness"]["status"] == "ready"
    assert save_calls == ["save", "save", "save"]
    assert state["git_push_report"]["enabled"] is False
    assert state["auto_disconnect_report"]["checks"]["repo_exports"] is True
    assert telemetry.closed[0]["status"] == "ok"
    assert disconnect_calls
    assert evaluation_calls == [("val", "ood_dev"), ("test", "ood_test")]
