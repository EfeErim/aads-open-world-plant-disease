import json
from pathlib import Path

import pytest

from src.training.types import EvaluationArtifactsPayload, ValidationReport
from src.workflows.training import TrainingWorkflow


class FakeDataset:
    def __init__(self, classes):
        self.classes = list(classes)

    def __len__(self):
        return len(self.classes)


class FakeLoader(list):
    def __init__(self, classes, *, split_name=""):
        super().__init__([{"images": 1, "labels": 1}])
        self.dataset = FakeDataset(classes)
        self.split_name = str(split_name)


class FakeHistory:
    def to_dict(self):
        return {"train_loss": [0.1], "val_loss": [0.2], "global_step": 1, "optimizer_steps": 1}


class FakeSession:
    def __init__(self, observers, trainer=None):
        self.observers = list(observers)
        self.trainer = trainer if trainer is not None else object()
        self.best_state_restored = False

    def snapshot_state(self):
        return {
            "progress_state": {
                "epoch": 1,
                "batch": 1,
                "total_batches": 1,
                "global_step": 2,
            },
            "history": {"train_loss": [0.1]},
        }

    def run(self):
        for observer in self.observers:
            observer(
                {
                    "event_type": "checkpoint_requested",
                    "payload": {"reason": "batch_interval", "mark_best": False, "val_loss": 0.2},
                }
            )
        return FakeHistory()

    def restore_best_model_state(self):
        self.best_state_restored = True
        if hasattr(self.trainer, "best_state_restored"):
            self.trainer.best_state_restored = True
        return True


class FakeAdapter:
    last_export_metadata = None

    def __init__(self, crop_name, part_name="unspecified", model_name="model", device="cpu"):
        self.crop_name = crop_name
        self.model_name = model_name
        self.device = device
        self.initialized = None
        self.export_metadata = None

    def initialize_engine(self, *, class_names=None, config=None):
        self.initialized = {"class_names": list(class_names or []), "config": dict(config or {})}
        return {"status": "initialized"}

    def build_training_session(self, train_loader, **kwargs):
        evaluation = self.initialized.get("config", {}).get("training", {}).get("continual", {}).get("evaluation", {})
        trainer_config = type(
            "FakeTrainerConfig",
            (),
            {
                "evaluation_require_ood_for_gate": bool(evaluation.get("require_ood_for_gate", True)),
                "evaluation_emit_ood_gate": bool(evaluation.get("emit_ood_gate", True)),
            },
        )()
        trainer = type("FakeTrainer", (), {"config": trainer_config, "best_state_restored": False})()
        return FakeSession(kwargs.get("observers", []), trainer=trainer)

    def calibrate_ood(self, loader):
        return {"status": "calibrated", "ood_calibration": {"version": 1}}

    def set_export_metadata(self, *, ood_calibration=None, adapter_runtime=None):
        self.export_metadata = {
            "ood_calibration": dict(ood_calibration or {}),
            "adapter_runtime": dict(adapter_runtime or {}),
        }
        type(self).last_export_metadata = dict(self.export_metadata)

    def save_adapter(self, output_dir):
        path = Path(output_dir) / "continual_sd_lora_adapter"
        path.mkdir(parents=True, exist_ok=True)
        return path


class FakeCheckpointManager:
    def __init__(self):
        self.calls = []

    def save_checkpoint(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"name": "ckpt_1", "reason": kwargs["reason"]}


class FakeEmptyLoader(list):
    def __init__(self, classes):
        super().__init__([])
        self.dataset = FakeDataset(classes)


def _fake_evaluation_result(include_ood: bool) -> EvaluationArtifactsPayload:
    ood_labels = [0] * 30 + [1] * 30 if include_ood else None
    ood_scores = (
        [0.10 + (index % 5) * 0.01 for index in range(30)]
        + [0.80 + (index % 5) * 0.02 for index in range(30)]
        if include_ood
        else None
    )
    ood_type_breakdown = (
        {
            "field": {
                "sample_count": 30,
                "method_metrics": {
                    "ensemble": {
                        "ood_auroc": 1.0,
                        "ood_false_positive_rate": 0.0,
                        "in_distribution_samples": 30,
                        "ood_samples": 30,
                    }
                },
            }
        }
        if include_ood
        else {}
    )
    report = ValidationReport(
        val_loss=0.05,
        val_accuracy=1.0,
        macro_precision=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        weighted_f1=1.0,
        balanced_accuracy=1.0,
        per_class_accuracy={"healthy": 1.0, "disease_a": 1.0},
        per_class_support={"healthy": 2, "disease_a": 2},
        worst_classes=[],
    )
    return EvaluationArtifactsPayload(
        report=report,
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        ood_labels=ood_labels,
        ood_scores=ood_scores,
        ood_type_breakdown=ood_type_breakdown,
        sure_ds_f1=0.95,
        conformal_empirical_coverage=0.97,
        conformal_avg_set_size=1.0,
    )


def test_training_workflow_runs_adapter_session_and_checkpoint(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.workflows.training.create_training_loaders",
        lambda **kwargs: {
            "train": FakeLoader(["healthy", "disease_a"]),
            "val": FakeLoader(["healthy"]),
            "test": FakeLoader(["healthy"]),
        },
    )
    monkeypatch.setattr("src.workflows.training.IndependentCropAdapter", FakeAdapter)

    checkpoint_manager = FakeCheckpointManager()
    workflow = TrainingWorkflow(
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake"},
                    "batch_size": 2,
                    "seed": 7,
                    "data": {"target_size": 224, "cache_size": 10, "loader_error_policy": "tolerant"},
                }
            },
            "colab": {"training": {"num_workers": 0, "pin_memory": False, "checkpoint_every_n_steps": 1}},
        },
        device="cpu",
    )

    result = workflow.run(
        crop_name="tomato",
        data_dir=tmp_path / "runtime_data",
        output_dir=tmp_path / "outputs",
        checkpoint_manager=checkpoint_manager,
    )

    assert result.class_names == ["healthy", "disease_a"]
    assert result.history["train_loss"] == [0.1]
    assert result.adapter_dir.exists()
    assert result.adapter_dir == tmp_path / "outputs" / "tomato" / "unspecified" / "continual_sd_lora_adapter"
    assert result.artifact_dir == tmp_path / "outputs" / "tomato" / "unspecified" / "training_metrics"
    assert result.artifact_dir is not None and result.artifact_dir.exists()
    assert (result.artifact_dir / "training" / "results.png").exists()
    assert (result.artifact_dir / "training" / "experiment_manifest.json").exists()
    assert (result.artifact_dir / "training" / "optimization_record.json").exists()
    assert checkpoint_manager.calls
    assert result.checkpoint_records[0]["reason"] == "batch_interval"

    experiment_manifest = json.loads((result.artifact_dir / "training" / "experiment_manifest.json").read_text(encoding="utf-8"))
    optimization_record = json.loads((result.artifact_dir / "training" / "optimization_record.json").read_text(encoding="utf-8"))
    assert experiment_manifest["surface"] == "workflow"
    assert experiment_manifest["part_name"] == "unspecified"
    assert optimization_record["comparability"]["engine"] == "continual_sd_lora"
    assert "authoritative_split" in optimization_record["status"]


def test_training_workflow_prefers_real_ood_evidence(monkeypatch, tmp_path: Path):
    benchmark_calls = []
    monkeypatch.setattr(
        "src.workflows.training.create_training_loaders",
        lambda **kwargs: {
            "train": FakeLoader(["healthy", "disease_a"]),
            "val": FakeLoader(["healthy", "disease_a"]),
            "test": FakeLoader(["healthy", "disease_a"]),
            "ood": FakeLoader(["unknown"]),
        },
    )
    monkeypatch.setattr("src.workflows.training.IndependentCropAdapter", FakeAdapter)
    monkeypatch.setattr(
        "src.workflows.training.evaluate_model_with_artifact_metrics",
        lambda trainer, loader, *, ood_loader=None: _fake_evaluation_result(include_ood=ood_loader is not None),
    )
    monkeypatch.setattr(
        "src.workflows.training.run_leave_one_class_out_benchmark",
        lambda **kwargs: benchmark_calls.append(dict(kwargs)) or {},
    )

    workflow = TrainingWorkflow(
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake"},
                    "batch_size": 2,
                    "seed": 7,
                    "data": {"target_size": 224, "cache_size": 10, "loader_error_policy": "tolerant"},
                    "evaluation": {"require_ood_for_gate": True},
                }
            },
            "colab": {"training": {"num_workers": 0, "pin_memory": False}},
        },
        device="cpu",
    )

    result = workflow.run(
        crop_name="tomato",
        data_dir=tmp_path / "runtime_data",
        output_dir=tmp_path / "outputs",
    )

    assert benchmark_calls == []
    assert result.ood_evidence_source == "real_ood_split"
    assert result.production_readiness["status"] == "ready"
    assert result.production_readiness["passed"] is True
    assert result.production_readiness["policy_passed"] is True
    assert result.production_readiness["missing_deployment_requirements"] == []
    assert (result.artifact_dir / "production_readiness.json").exists()


def test_training_workflow_accepts_exact_runtime_dataset_root(monkeypatch, tmp_path: Path):
    dataset_root = tmp_path / "runtime_data" / "tomato__fruit"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "split_manifest.json").write_text(
        json.dumps(
            {
                "crop_name": "tomato",
                "part_name": "fruit",
                "dataset_key": "tomato__fruit",
                "classes": [
                    {"class_name": "healthy", "image_count": 260},
                    {"class_name": "disease_a", "image_count": 120},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    loader_calls = []

    def _capture_loader_kwargs(**kwargs):
        loader_calls.append(dict(kwargs))
        return {
            "train": FakeLoader(["healthy", "disease_a"], split_name="continual"),
            "val": FakeLoader(["healthy", "disease_a"], split_name="val"),
            "test": FakeLoader(["healthy", "disease_a"], split_name="test"),
        }

    monkeypatch.setattr("src.workflows.training.create_training_loaders", _capture_loader_kwargs)
    monkeypatch.setattr("src.workflows.training.IndependentCropAdapter", FakeAdapter)
    monkeypatch.setattr(
        "src.workflows.training.evaluate_model_with_artifact_metrics",
        lambda *_args, **_kwargs: _fake_evaluation_result(include_ood=False),
    )
    monkeypatch.setattr("src.workflows.training.run_leave_one_class_out_benchmark", lambda **kwargs: {})

    workflow = TrainingWorkflow(
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake"},
                    "batch_size": 2,
                    "seed": 7,
                    "data": {"target_size": 224, "cache_size": 10, "loader_error_policy": "tolerant"},
                    "evaluation": {"require_ood_for_gate": False},
                }
            },
            "colab": {"training": {"num_workers": 0, "pin_memory": False}},
        },
        device="cpu",
    )

    result = workflow.run(
        crop_name="tomato",
        data_dir=dataset_root,
        output_dir=tmp_path / "outputs",
    )

    assert loader_calls[0]["data_dir"] == str(dataset_root.parent)
    assert loader_calls[0]["crop"] == "tomato__fruit"
    run_context = json.loads((result.artifact_dir / "training" / "run_context.json").read_text(encoding="utf-8"))
    assert run_context["dataset"]["dataset_key"] == "tomato__fruit"
    assert run_context["dataset"]["resolution_source"] == "exact_root"
    assert Path(run_context["dataset"]["crop_root"]) == dataset_root.resolve()


def test_training_workflow_rejects_ambiguous_runtime_dataset_parent(monkeypatch, tmp_path: Path):
    runtime_root = tmp_path / "runtime_data"
    (runtime_root / "tomato" / "continual" / "healthy").mkdir(parents=True, exist_ok=True)
    dataset_root = runtime_root / "tomato__fruit"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "split_manifest.json").write_text(
        json.dumps({"crop_name": "tomato", "part_name": "fruit", "dataset_key": "tomato__fruit"}),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.workflows.training.IndependentCropAdapter", FakeAdapter)

    workflow = TrainingWorkflow(
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake"},
                    "batch_size": 2,
                    "seed": 7,
                    "data": {"target_size": 224, "cache_size": 10, "loader_error_policy": "tolerant"},
                }
            },
            "colab": {"training": {"num_workers": 0, "pin_memory": False}},
        },
        device="cpu",
    )

    with pytest.raises(ValueError, match="Multiple runtime datasets matched crop 'tomato'"):
        workflow.run(
            crop_name="tomato",
            data_dir=runtime_root,
            output_dir=tmp_path / "outputs",
        )


def test_training_workflow_uses_held_out_benchmark_when_real_ood_is_missing(monkeypatch, tmp_path: Path):
    benchmark_calls = []
    monkeypatch.setattr(
        "src.workflows.training.create_training_loaders",
        lambda **kwargs: {
            "train": FakeLoader(["healthy", "disease_a", "disease_b"]),
            "val": FakeLoader(["healthy", "disease_a", "disease_b"]),
            "test": FakeLoader(["healthy", "disease_a", "disease_b"]),
        },
    )
    monkeypatch.setattr("src.workflows.training.IndependentCropAdapter", FakeAdapter)
    monkeypatch.setattr(
        "src.workflows.training.evaluate_model_with_artifact_metrics",
        lambda trainer, loader, *, ood_loader=None: _fake_evaluation_result(include_ood=ood_loader is not None),
    )
    monkeypatch.setattr(
        "src.workflows.training.run_leave_one_class_out_benchmark",
        lambda **kwargs: (
            benchmark_calls.append(dict(kwargs))
            or {
                "status": "completed",
                "passed": True,
                "metrics": {
                    "ood_auroc": 0.96,
                    "ood_false_positive_rate": 0.03,
                    "ood_samples": 30,
                    "in_distribution_samples": 30,
                    "sure_ds_f1": 0.94,
                    "conformal_empirical_coverage": 0.97,
                    "conformal_avg_set_size": 1.0,
                },
                "paths": {
                    "summary_json": str(
                        tmp_path
                        / "outputs"
                        / "tomato"
                        / "unspecified"
                        / "training_metrics"
                        / "ood_benchmark"
                        / "summary.json"
                    )
                },
            }
        ),
    )

    workflow = TrainingWorkflow(
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake"},
                    "batch_size": 2,
                    "seed": 7,
                    "data": {"target_size": 224, "cache_size": 10, "loader_error_policy": "tolerant"},
                    "evaluation": {
                        "require_ood_for_gate": True,
                        "ood_benchmark_min_classes": 3,
                    },
                }
            },
            "colab": {"training": {"num_workers": 0, "pin_memory": False}},
        },
        device="cpu",
    )

    result = workflow.run(
        crop_name="tomato",
        data_dir=tmp_path / "runtime_data",
        output_dir=tmp_path / "outputs",
    )

    assert len(benchmark_calls) == 1
    assert result.ood_evidence_source == "held_out_benchmark"
    assert result.production_readiness["status"] == "provisional"
    assert result.production_readiness["passed"] is False
    assert result.production_readiness["policy_passed"] is True
    assert result.production_readiness["ood_evidence_source"] == "held_out_benchmark"
    assert result.production_readiness["missing_deployment_requirements"] == ["real_ood_evidence"]


def test_training_workflow_restores_best_state_before_export(monkeypatch, tmp_path: Path):
    restored_flags = []

    class RecordingAdapter(FakeAdapter):
        def build_training_session(self, train_loader, **kwargs):
            session = super().build_training_session(train_loader, **kwargs)
            self._session = session
            return session

        def calibrate_ood(self, loader):
            restored_flags.append(getattr(self._session, "best_state_restored", False))
            return super().calibrate_ood(loader)

        def save_adapter(self, output_dir):
            restored_flags.append(getattr(self._session, "best_state_restored", False))
            return super().save_adapter(output_dir)

    monkeypatch.setattr(
        "src.workflows.training.create_training_loaders",
        lambda **kwargs: {
            "train": FakeLoader(["healthy", "disease_a"]),
            "val": FakeLoader(["healthy", "disease_a"]),
            "test": FakeLoader(["healthy", "disease_a"]),
        },
    )
    monkeypatch.setattr("src.workflows.training.IndependentCropAdapter", RecordingAdapter)
    monkeypatch.setattr(
        "src.workflows.training.evaluate_model_with_artifact_metrics",
        lambda trainer, loader, *, ood_loader=None: _fake_evaluation_result(include_ood=False),
    )
    monkeypatch.setattr("src.workflows.training.run_leave_one_class_out_benchmark", lambda **kwargs: {})

    workflow = TrainingWorkflow(
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake"},
                    "batch_size": 2,
                    "seed": 7,
                    "data": {"target_size": 224, "cache_size": 10, "loader_error_policy": "tolerant"},
                    "evaluation": {
                        "require_ood_for_gate": False,
                    },
                }
            },
            "colab": {"training": {"num_workers": 0, "pin_memory": False}},
        },
        device="cpu",
    )

    workflow.run(
        crop_name="tomato",
        data_dir=tmp_path / "runtime_data",
        output_dir=tmp_path / "outputs",
    )

    assert restored_flags == [True, True]


def test_training_workflow_requires_isolated_eval_split_when_val_used_for_calibration(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.workflows.training.create_training_loaders",
        lambda **kwargs: {
            "train": FakeLoader(["healthy", "disease_a"]),
            "val": FakeLoader(["healthy", "disease_a"]),
            "test": FakeEmptyLoader([]),
        },
    )
    monkeypatch.setattr("src.workflows.training.IndependentCropAdapter", FakeAdapter)
    monkeypatch.setattr(
        "src.workflows.training.evaluate_model_with_artifact_metrics",
        lambda trainer, loader, *, ood_loader=None: _fake_evaluation_result(include_ood=False),
    )
    monkeypatch.setattr("src.workflows.training.run_leave_one_class_out_benchmark", lambda **kwargs: {})

    workflow = TrainingWorkflow(
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake"},
                    "batch_size": 2,
                    "seed": 7,
                    "data": {"target_size": 224, "cache_size": 10, "loader_error_policy": "tolerant"},
                    "evaluation": {
                        "require_ood_for_gate": False,
                    },
                }
            },
            "colab": {"training": {"num_workers": 0, "pin_memory": False}},
        },
        device="cpu",
    )

    result = workflow.run(
        crop_name="tomato",
        data_dir=tmp_path / "runtime_data",
        output_dir=tmp_path / "outputs",
    )

    assert result.production_readiness["passed"] is False
    assert "accuracy" in result.production_readiness["missing_requirements"]


def test_training_workflow_does_not_write_readiness_when_adapter_export_fails(monkeypatch, tmp_path: Path):
    class FailingExportAdapter(FakeAdapter):
        def save_adapter(self, output_dir):
            raise RuntimeError("simulated export failure")

    monkeypatch.setattr(
        "src.workflows.training.create_training_loaders",
        lambda **kwargs: {
            "train": FakeLoader(["healthy", "disease_a"]),
            "val": FakeLoader(["healthy", "disease_a"]),
            "test": FakeLoader(["healthy", "disease_a"]),
        },
    )
    monkeypatch.setattr("src.workflows.training.IndependentCropAdapter", FailingExportAdapter)
    monkeypatch.setattr(
        "src.workflows.training.evaluate_model_with_artifact_metrics",
        lambda trainer, loader, *, ood_loader=None: _fake_evaluation_result(include_ood=False),
    )
    monkeypatch.setattr("src.workflows.training.run_leave_one_class_out_benchmark", lambda **kwargs: {})

    workflow = TrainingWorkflow(
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake"},
                    "batch_size": 2,
                    "seed": 7,
                    "data": {"target_size": 224, "cache_size": 10, "loader_error_policy": "tolerant"},
                    "evaluation": {"require_ood_for_gate": False},
                }
            },
            "colab": {"training": {"num_workers": 0, "pin_memory": False}},
        },
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="simulated export failure"):
        workflow.run(
            crop_name="tomato",
            data_dir=tmp_path / "runtime_data",
            output_dir=tmp_path / "outputs",
        )

    assert not (
        tmp_path / "outputs" / "tomato" / "unspecified" / "training_metrics" / "production_readiness.json"
    ).exists()
