import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.training.services.ood_benchmark import (
    _build_benchmark_summary_payload,
    _build_resume_key,
    _loader_size,
    run_leave_one_class_out_benchmark,
)
from src.training.types import EvaluationArtifactsPayload, ValidationReport


class ToyDataset(Dataset):
    def __init__(self, classes, items_per_class):
        self.classes = [str(name) for name in classes]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.samples = []
        self.labels = []
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            for item_idx in range(int(items_per_class)):
                self.samples.append(torch.tensor([float(class_idx), float(item_idx)], dtype=torch.float32))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]


class FakeSession:
    def __init__(self, trainer):
        self.trainer = trainer

    def run(self):
        return None


class FakeAdapter:
    def __init__(self, crop_name, model_name="model", device="cpu"):
        self.crop_name = crop_name
        self.model_name = model_name
        self.device = device
        self._trainer = type("FakeTrainer", (), {})()

    def initialize_engine(self, *, class_names=None, config=None):
        del config
        self._trainer.class_names = list(class_names or [])
        return {"status": "initialized"}

    def build_training_session(self, train_loader, **kwargs):
        del train_loader, kwargs
        return FakeSession(self._trainer)

    def calibrate_ood(self, loader):
        del loader
        return {"status": "calibrated"}


class FakeTelemetry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.latest_payloads = []
        self.logs = []
        self.copied_paths = []

    def update_latest(self, payload):
        self.latest_payloads.append(dict(payload))

    def emit_log(self, message, *, phase="runtime", level="info"):
        self.logs.append({"message": str(message), "phase": str(phase), "level": str(level)})

    def copy_artifact_file(self, source_path, relative_path):
        target = self.root / Path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source_path).read_bytes())
        self.copied_paths.append(str(relative_path).replace("\\", "/"))
        return target


def _fake_evaluate_model_with_artifact_metrics(_trainer, loader, *, ood_loader=None):
    y_true = [int(label) for label in getattr(loader.dataset, "labels", [])]
    y_pred = list(y_true)
    per_class_accuracy = {name: 1.0 for name in getattr(loader.dataset, "classes", [])}
    per_class_support = {name: 1 for name in getattr(loader.dataset, "classes", [])}
    report = ValidationReport(
        val_loss=0.05,
        val_accuracy=1.0,
        macro_precision=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        weighted_f1=1.0,
        balanced_accuracy=1.0,
        per_class_accuracy=per_class_accuracy,
        per_class_support=per_class_support,
        worst_classes=[],
    )
    ood_count = len(getattr(getattr(ood_loader, "dataset", None), "labels", [])) if ood_loader is not None else 0
    return EvaluationArtifactsPayload(
        report=report,
        y_true=y_true,
        y_pred=y_pred,
        ood_labels=([0] * len(y_true)) + ([1] * ood_count) if ood_loader is not None else None,
        ood_scores=([0.1] * len(y_true)) + ([0.9] * ood_count) if ood_loader is not None else None,
        ood_primary_score_method="ensemble",
        ood_scores_by_method={
            "ensemble": ([0.1] * len(y_true)) + ([0.9] * ood_count) if ood_loader is not None else [],
            "energy": ([0.2] * len(y_true)) + ([0.8] * ood_count) if ood_loader is not None else [],
            "knn": ([0.15] * len(y_true)) + ([0.85] * ood_count) if ood_loader is not None else [],
        },
        sure_ds_f1=0.95,
        conformal_empirical_coverage=0.97,
        conformal_avg_set_size=1.0,
    )


def _build_loaders(classes, *, train_items=3, eval_items=5):
    return {
        "train": DataLoader(ToyDataset(classes, train_items), batch_size=2, shuffle=False),
        "val": DataLoader(ToyDataset(classes, eval_items), batch_size=2, shuffle=False),
        "test": DataLoader(ToyDataset(classes, eval_items), batch_size=2, shuffle=False),
    }


def test_loader_size_raises_for_dataset_runtime_errors():
    class _BrokenDataset:
        def __len__(self):
            raise RuntimeError("boom")

    class _Loader:
        dataset = _BrokenDataset()

    with pytest.raises(RuntimeError, match="dataset length"):
        _loader_size(_Loader())


def test_loader_size_falls_back_to_loader_len_for_lenless_dataset():
    class _LenlessDataset:
        def __len__(self):
            raise TypeError("len unavailable")

    class _Loader:
        dataset = _LenlessDataset()

        def __len__(self):
            return 7

    assert _loader_size(_Loader()) == 7


def test_run_leave_one_class_out_benchmark_writes_aggregate_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.training.services.ood_benchmark.evaluate_model_with_artifact_metrics",
        _fake_evaluate_model_with_artifact_metrics,
    )

    summary = run_leave_one_class_out_benchmark(
        crop_name="tomato",
        class_names=["healthy", "disease_a", "disease_b"],
        loaders=_build_loaders(["healthy", "disease_a", "disease_b"], eval_items=30),
        config={"training": {"continual": {"backbone": {"model_name": "fake"}}}},
        device="cpu",
        artifact_root=tmp_path / "training_metrics",
        adapter_factory=FakeAdapter,
        run_id="run_123",
        min_classes=3,
    )

    assert summary["status"] == "completed"
    assert summary["passed"] is True
    assert summary["successful_folds"] == 3
    assert summary["failed_folds"] == 0
    assert summary["requested_primary_score_method"] == "auto"
    assert summary["primary_score_method"] == "ensemble"
    assert {"ensemble", "energy", "knn"} <= set(summary["method_comparison_metrics"].keys())
    assert summary["method_comparison"]["selected_primary_score_method"] == "ensemble"
    assert summary["method_comparison"]["selection_source"] == "held_out_benchmark"
    assert summary["method_comparison"]["methods"]["ensemble"]["worst_fold"]["held_out_class"] in {
        "healthy",
        "disease_a",
        "disease_b",
    }
    assert all(fold["status"] == "completed" for fold in summary["folds"])
    assert (tmp_path / "training_metrics" / "ood_benchmark" / "summary.json").exists()
    assert (tmp_path / "training_metrics" / "ood_benchmark" / "per_fold.csv").exists()


def test_run_leave_one_class_out_benchmark_auto_selects_best_method(monkeypatch, tmp_path: Path):
    def _energy_wins(_trainer, loader, *, ood_loader=None):
        y_true = [int(label) for label in getattr(loader.dataset, "labels", [])]
        y_pred = list(y_true)
        ood_count = len(getattr(getattr(ood_loader, "dataset", None), "labels", [])) if ood_loader is not None else 0
        report = ValidationReport.from_dict({"val_accuracy": 1.0})
        return EvaluationArtifactsPayload(
            report=report,
            y_true=y_true,
            y_pred=y_pred,
            ood_labels=([0] * len(y_true)) + ([1] * ood_count) if ood_loader is not None else None,
            ood_scores=([0.7] * len(y_true)) + ([0.6] * ood_count) if ood_loader is not None else None,
            ood_primary_score_method="ensemble",
            ood_scores_by_method={
                "ensemble": ([0.7] * len(y_true)) + ([0.6] * ood_count) if ood_loader is not None else [],
                "energy": ([0.1] * len(y_true)) + ([0.9] * ood_count) if ood_loader is not None else [],
                "knn": ([0.3] * len(y_true)) + ([0.7] * ood_count) if ood_loader is not None else [],
            },
        )

    monkeypatch.setattr(
        "src.training.services.ood_benchmark.evaluate_model_with_artifact_metrics",
        _energy_wins,
    )

    summary = run_leave_one_class_out_benchmark(
        crop_name="tomato",
        class_names=["healthy", "disease_a", "disease_b"],
        loaders=_build_loaders(["healthy", "disease_a", "disease_b"]),
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake"},
                    "ood": {"primary_score_method": "auto"},
                }
            }
        },
        device="cpu",
        artifact_root=tmp_path / "training_metrics",
        adapter_factory=FakeAdapter,
        run_id="run_auto_pick",
        min_classes=3,
    )

    assert summary["requested_primary_score_method"] == "auto"
    assert summary["primary_score_method"] == "energy"
    assert summary["primary_score_selection_source"] == "held_out_benchmark"
    assert summary["metrics"] == summary["method_comparison_metrics"]["energy"]
    assert summary["method_comparison"]["selected_primary_score_method"] == "energy"
    assert summary["method_comparison"]["methods"]["energy"]["worst_fold"]["held_out_class"] in {
        "healthy",
        "disease_a",
        "disease_b",
    }


def test_benchmark_summary_auto_selection_prefers_lower_worst_fold_fpr_before_pooled_auroc():
    summary = _build_benchmark_summary_payload(
        folds=[
            {
                "held_out_class": "healthy",
                "status": "completed",
                "metrics": {
                    "accuracy": 1.0,
                    "ood_auroc": 0.95,
                    "ood_false_positive_rate": 0.05,
                    "ood_samples": 5,
                    "in_distribution_samples": 5,
                    "sure_ds_f1": 0.95,
                    "conformal_empirical_coverage": 0.97,
                    "conformal_avg_set_size": 1.0,
                },
                "method_metrics": {
                    "ensemble": {
                        "ood_auroc": 0.99,
                        "ood_false_positive_rate": 0.18,
                        "ood_samples": 5,
                        "in_distribution_samples": 5,
                        "sure_ds_f1": 0.95,
                        "conformal_empirical_coverage": 0.97,
                        "conformal_avg_set_size": 1.0,
                    },
                    "energy": {
                        "ood_auroc": 0.96,
                        "ood_false_positive_rate": 0.05,
                        "ood_samples": 5,
                        "in_distribution_samples": 5,
                        "sure_ds_f1": 0.95,
                        "conformal_empirical_coverage": 0.97,
                        "conformal_avg_set_size": 1.0,
                    },
                },
            },
            {
                "held_out_class": "disease_a",
                "status": "completed",
                "metrics": {
                    "accuracy": 1.0,
                    "ood_auroc": 0.94,
                    "ood_false_positive_rate": 0.06,
                    "ood_samples": 5,
                    "in_distribution_samples": 5,
                    "sure_ds_f1": 0.94,
                    "conformal_empirical_coverage": 0.96,
                    "conformal_avg_set_size": 1.1,
                },
                "method_metrics": {
                    "ensemble": {
                        "ood_auroc": 0.98,
                        "ood_false_positive_rate": 0.16,
                        "ood_samples": 5,
                        "in_distribution_samples": 5,
                        "sure_ds_f1": 0.94,
                        "conformal_empirical_coverage": 0.96,
                        "conformal_avg_set_size": 1.1,
                    },
                    "energy": {
                        "ood_auroc": 0.95,
                        "ood_false_positive_rate": 0.04,
                        "ood_samples": 5,
                        "in_distribution_samples": 5,
                        "sure_ds_f1": 0.94,
                        "conformal_empirical_coverage": 0.96,
                        "conformal_avg_set_size": 1.1,
                    },
                },
            },
        ],
        primary_score_method="ensemble",
        requested_primary_score_method="auto",
        target_values={
            "accuracy": 0.93,
            "ood_auroc": 0.92,
            "ood_false_positive_rate": 0.10,
            "ood_samples": 5,
            "in_distribution_samples": 5,
            "sure_ds_f1": 0.90,
            "conformal_empirical_coverage": 0.95,
            "conformal_avg_set_size": 2.0,
        },
        base_context={"crop_name": "tomato"},
    )

    assert summary["primary_score_method"] == "energy"
    assert summary["primary_score_selection_source"] == "held_out_benchmark"
    assert summary["method_comparison"]["methods"]["ensemble"]["worst_fold"]["held_out_class"] == "healthy"
    assert summary["method_comparison"]["methods"]["ensemble"]["worst_fold"]["metrics"]["ood_false_positive_rate"] == 0.18
    assert summary["method_comparison"]["methods"]["energy"]["worst_fold"]["metrics"]["ood_false_positive_rate"] == 0.05



def test_benchmark_summary_prefers_real_fpr_when_worst_fold_has_missing_fpr():
    summary = _build_benchmark_summary_payload(
        folds=[
            {
                "held_out_class": "healthy",
                "status": "completed",
                "metrics": {"accuracy": 1.0},
                "method_metrics": {
                    "ensemble": {
                        "ood_auroc": 0.30,
                        "ood_false_positive_rate": None,
                        "ood_samples": 5,
                        "in_distribution_samples": 5,
                    },
                },
            },
            {
                "held_out_class": "disease_a",
                "status": "completed",
                "metrics": {"accuracy": 1.0},
                "method_metrics": {
                    "ensemble": {
                        "ood_auroc": 0.90,
                        "ood_false_positive_rate": 0.18,
                        "ood_samples": 5,
                        "in_distribution_samples": 5,
                    },
                },
            },
        ],
        primary_score_method="ensemble",
        requested_primary_score_method="auto",
        target_values={
            "accuracy": 0.93,
            "ood_auroc": 0.92,
            "ood_false_positive_rate": 0.10,
            "ood_samples": 5,
            "in_distribution_samples": 5,
            "sure_ds_f1": 0.90,
            "conformal_empirical_coverage": 0.95,
            "conformal_avg_set_size": 2.0,
        },
        base_context={"crop_name": "tomato"},
    )

    assert summary["method_comparison"]["methods"]["ensemble"]["worst_fold"]["held_out_class"] == "disease_a"
    assert summary["method_comparison"]["methods"]["ensemble"]["worst_fold"]["metrics"]["ood_false_positive_rate"] == 0.18
def test_run_leave_one_class_out_benchmark_fails_when_class_count_is_too_small(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.training.services.ood_benchmark.evaluate_model_with_artifact_metrics",
        _fake_evaluate_model_with_artifact_metrics,
    )

    summary = run_leave_one_class_out_benchmark(
        crop_name="tomato",
        class_names=["healthy", "disease_a"],
        loaders=_build_loaders(["healthy", "disease_a"]),
        config={"training": {"continual": {"backbone": {"model_name": "fake"}}}},
        device="cpu",
        artifact_root=tmp_path / "training_metrics",
        adapter_factory=FakeAdapter,
        run_id="run_456",
        min_classes=3,
    )

    assert summary["status"] == "failed"
    assert summary["passed"] is False
    assert summary["reason"] == "insufficient_classes_for_fallback"
    assert summary["successful_folds"] == 0


def test_run_leave_one_class_out_benchmark_writes_progress_artifact(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.training.services.ood_benchmark.evaluate_model_with_artifact_metrics",
        _fake_evaluate_model_with_artifact_metrics,
    )
    telemetry = FakeTelemetry(tmp_path / "telemetry_copy")

    summary = run_leave_one_class_out_benchmark(
        crop_name="tomato",
        class_names=["healthy", "disease_a", "disease_b"],
        loaders=_build_loaders(["healthy", "disease_a", "disease_b"]),
        config={"training": {"continual": {"backbone": {"model_name": "fake"}}}},
        device="cpu",
        artifact_root=tmp_path / "training_metrics",
        adapter_factory=FakeAdapter,
        run_id="run_progress",
        telemetry=telemetry,
        min_classes=3,
    )

    progress_path = tmp_path / "training_metrics" / "ood_benchmark" / "progress.json"
    assert summary["status"] == "completed"
    assert progress_path.exists()
    progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress_payload["status"] == "completed"
    assert progress_payload["stage"] == "benchmark_completed"
    assert telemetry.latest_payloads[-1]["stage"] == "benchmark_completed"
    assert "ood_benchmark/progress.json" in telemetry.copied_paths


def test_run_leave_one_class_out_benchmark_persists_compact_success_fold_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.training.services.ood_benchmark.evaluate_model_with_artifact_metrics",
        _fake_evaluate_model_with_artifact_metrics,
    )
    telemetry = FakeTelemetry(tmp_path / "telemetry_copy")

    summary = run_leave_one_class_out_benchmark(
        crop_name="tomato",
        class_names=["healthy", "disease_a", "disease_b"],
        loaders=_build_loaders(["healthy", "disease_a", "disease_b"]),
        config={"training": {"continual": {"backbone": {"model_name": "fake"}}}},
        device="cpu",
        artifact_root=tmp_path / "training_metrics",
        adapter_factory=FakeAdapter,
        run_id="run_compact",
        telemetry=telemetry,
        min_classes=3,
    )

    first_fold = summary["folds"][0]
    metric_gate_path = Path(first_fold["paths"]["metric_gate_json"])
    assert metric_gate_path.exists()
    assert not (metric_gate_path.parent / "classification_report.txt").exists()
    assert not (metric_gate_path.parent / "confusion_matrix.png").exists()
    assert "ood_benchmark/folds/healthy/metric_gate.json" in telemetry.copied_paths
    assert not any(path.endswith("classification_report.txt") for path in telemetry.copied_paths)
    assert not any(path.endswith("confusion_matrix.png") for path in telemetry.copied_paths)


def test_run_leave_one_class_out_benchmark_persists_traceback_for_fold_failures(monkeypatch, tmp_path: Path):
    def _raise_during_evaluation(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("eval boom")

    monkeypatch.setattr(
        "src.training.services.ood_benchmark.evaluate_model_with_artifact_metrics",
        _raise_during_evaluation,
    )

    summary = run_leave_one_class_out_benchmark(
        crop_name="tomato",
        class_names=["healthy", "disease_a", "disease_b"],
        loaders=_build_loaders(["healthy", "disease_a", "disease_b"]),
        config={"training": {"continual": {"backbone": {"model_name": "fake"}}}},
        device="cpu",
        artifact_root=tmp_path / "training_metrics",
        adapter_factory=FakeAdapter,
        run_id="run_failures",
        min_classes=3,
    )

    assert summary["status"] == "failed"
    assert summary["failed_folds"] == 3
    first_fold = summary["folds"][0]
    assert first_fold["status"] == "failed"
    assert first_fold["diagnostics"]["failed_stage"] == "fold_evaluating"
    traceback_path = Path(first_fold["paths"]["failure_traceback_txt"])
    failure_json_path = Path(first_fold["paths"]["failure_json"])
    assert traceback_path.exists()
    assert failure_json_path.exists()
    assert "RuntimeError: eval boom" in traceback_path.read_text(encoding="utf-8")


def test_run_leave_one_class_out_benchmark_resumes_completed_folds(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.training.services.ood_benchmark.evaluate_model_with_artifact_metrics",
        _fake_evaluate_model_with_artifact_metrics,
    )

    call_count = {"count": 0}

    class CountingSession(FakeSession):
        def run(self):
            call_count["count"] += 1
            return None

    class CountingAdapter(FakeAdapter):
        def build_training_session(self, train_loader, **kwargs):
            del train_loader, kwargs
            return CountingSession(self._trainer)

    kwargs = {
        "crop_name": "tomato",
        "class_names": ["healthy", "disease_a", "disease_b"],
        "loaders": _build_loaders(["healthy", "disease_a", "disease_b"]),
        "config": {
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake"},
                    "ood": {"primary_score_method": "ensemble"},
                }
            }
        },
        "device": "cpu",
        "artifact_root": tmp_path / "training_metrics",
        "adapter_factory": CountingAdapter,
        "run_id": "run_resume",
        "min_classes": 3,
    }

    first = run_leave_one_class_out_benchmark(**kwargs)
    assert first["successful_folds"] == 3
    assert call_count["count"] == 3

    second = run_leave_one_class_out_benchmark(**kwargs)
    assert second["successful_folds"] == 3
    assert call_count["count"] == 3
    assert all(fold["diagnostics"].get("resume_hit") for fold in second["folds"])


def test_benchmark_summary_fails_when_any_completed_fold_misses_targets():
    summary = _build_benchmark_summary_payload(
        folds=[
            {
                "held_out_class": "healthy",
                "status": "completed",
                "metrics": {
                    "ood_auroc": 1.0,
                    "ood_false_positive_rate": 0.0,
                    "ood_samples": 5,
                    "in_distribution_samples": 5,
                    "sure_ds_f1": 1.0,
                    "conformal_empirical_coverage": 1.0,
                    "conformal_avg_set_size": 1.0,
                },
                "method_metrics": {},
            },
            {
                "held_out_class": "disease_a",
                "status": "completed",
                "metrics": {
                    "ood_auroc": 0.85,
                    "ood_false_positive_rate": 0.0,
                    "ood_samples": 5,
                    "in_distribution_samples": 5,
                    "sure_ds_f1": 1.0,
                    "conformal_empirical_coverage": 1.0,
                    "conformal_avg_set_size": 1.0,
                },
                "method_metrics": {},
            },
        ],
        primary_score_method="ensemble",
        requested_primary_score_method="ensemble",
        target_values={
            "accuracy": 0.93,
            "ood_auroc": 0.92,
            "ood_false_positive_rate": 0.05,
            "ood_samples": 5,
            "in_distribution_samples": 5,
            "sure_ds_f1": 0.90,
            "conformal_empirical_coverage": 0.95,
            "conformal_avg_set_size": 2.0,
        },
        base_context={},
    )

    assert summary["status"] == "completed"
    assert summary["passed"] is False
    assert summary["metrics"]["ood_auroc"] == 0.925
    assert len(summary["fold_target_failures"]) == 1
    assert summary["fold_target_failures"][0]["held_out_class"] == "disease_a"
    assert summary["fold_target_failures"][0]["primary_score_method"] == "ensemble"
    assert summary["fold_target_failures"][0]["missing_requirements"] == ["ood_auroc"]
    assert summary["fold_target_failures"][0]["metrics"]["ood_auroc"] == 0.85
    assert summary["fold_target_failures"][0]["evaluation"]["checks"]["ood_auroc"]["passed"] is False


def test_resume_key_changes_when_training_config_changes():
    common = {
        "crop_name": "tomato",
        "held_out_class": "healthy",
        "seen_classes": ["disease_a", "disease_b"],
        "sample_counts": {
            "train_samples": 12,
            "calibration_samples": 4,
            "eval_in_distribution_samples": 4,
            "eval_ood_samples": 2,
        },
        "dataset_fingerprints": {
            "train": {"label_sequence_sha256": "train", "path_signature_sha256": "train_paths"},
            "calibration": {"label_sequence_sha256": "val", "path_signature_sha256": "val_paths"},
            "evaluation": {"label_sequence_sha256": "test", "path_signature_sha256": "test_paths"},
        },
        "device": "cpu",
        "num_epochs": 5,
    }
    first = _build_resume_key(
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "model_a"},
                    "batch_size": 8,
                    "seed": 7,
                    "ood": {"primary_score_method": "auto"},
                }
            }
        },
        **common,
    )
    second = _build_resume_key(
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "model_b"},
                    "batch_size": 64,
                    "seed": 999,
                    "ood": {"primary_score_method": "auto"},
                }
            }
        },
        **common,
    )

    assert first != second


def test_resume_key_changes_when_dataset_fingerprint_changes():
    common = {
        "crop_name": "tomato",
        "held_out_class": "healthy",
        "seen_classes": ["disease_a", "disease_b"],
        "sample_counts": {
            "train_samples": 12,
            "calibration_samples": 4,
            "eval_in_distribution_samples": 4,
            "eval_ood_samples": 2,
        },
        "config": {
            "training": {
                "continual": {
                    "backbone": {"model_name": "model_a"},
                    "batch_size": 8,
                    "seed": 7,
                    "ood": {"primary_score_method": "auto"},
                }
            }
        },
        "device": "cpu",
        "num_epochs": 5,
    }
    first = _build_resume_key(
        dataset_fingerprints={
            "train": {"label_sequence_sha256": "train", "path_signature_sha256": "train_paths_v1"},
            "calibration": {"label_sequence_sha256": "val", "path_signature_sha256": "val_paths"},
            "evaluation": {"label_sequence_sha256": "test", "path_signature_sha256": "test_paths"},
        },
        **common,
    )
    second = _build_resume_key(
        dataset_fingerprints={
            "train": {"label_sequence_sha256": "train", "path_signature_sha256": "train_paths_v2"},
            "calibration": {"label_sequence_sha256": "val", "path_signature_sha256": "val_paths"},
            "evaluation": {"label_sequence_sha256": "test", "path_signature_sha256": "test_paths"},
        },
        **common,
    )

    assert first != second
