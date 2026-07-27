import json
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.adapter import independent_crop_adapter as adapter_module
from src.training.types import EvaluationArtifactsPayload, TrainBatchStats, ValidationReport
from src.workflows.inference import InferenceWorkflow
from src.workflows.training import TrainingWorkflow

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "offline_workflow"


class FakeCalibrationStats:
    def __init__(self) -> None:
        self.mean = torch.tensor([0.1, 0.2], dtype=torch.float32)
        self.var = torch.tensor([1.0, 1.5], dtype=torch.float32)
        self.mahalanobis_mu = 0.3
        self.mahalanobis_sigma = 0.4
        self.energy_mu = 0.5
        self.energy_sigma = 0.6
        self.threshold = 0.8
        self.energy_threshold = 0.8
        self.knn_distance_mu = 0.2
        self.knn_distance_sigma = 0.1
        self.knn_threshold = 0.8
        self.knn_bank = [[0.1, 0.2], [0.2, 0.1]]
        self.knn_k = 10
        self.sure_semantic_threshold = 0.7
        self.sure_confidence_threshold = 0.8


class FakeOODDetector:
    def __init__(self) -> None:
        self.threshold_factor = 2.0
        self.primary_score_method = "ensemble"
        self.calibration_version = 6
        self.class_stats = {}
        self.knn_k = 10
        self.knn_bank_cap = 256
        self.knn_backend = "auto"
        self.knn_chunk_size = 2048
        self.radial_l2_enabled = False
        self.radial_beta = None
        self.radial_beta_range = [0.5, 2.0]
        self.radial_beta_steps = 16
        self.sure_enabled = False
        self.sure_semantic_percentile = 95.0
        self.sure_confidence_percentile = 90.0
        self.conformal_enabled = True
        self.conformal_alpha = 0.05
        self.conformal_method = "threshold"
        self.conformal_raps_lambda = 0.0
        self.conformal_raps_k_reg = 1
        self.conformal_qhat = 0.05
        self.energy_temperature = 1.0
        self.energy_temperature_mode = "fixed"
        self.energy_temperature_range = [0.5, 3.0]
        self.energy_temperature_steps = 16

    def calibration_issue(self) -> str | None:
        if not self.class_stats:
            return "OOD detector has no calibrated class statistics."
        if self.calibration_version <= 0:
            return "OOD detector calibration version is unset."
        return None


class FakeTrainerConfig:
    def __init__(self, payload: dict | None = None) -> None:
        training = dict(payload or {})
        backbone = dict(training.get("backbone", {}))
        fusion = dict(training.get("fusion", {}))
        self.backbone_model_name = str(backbone.get("model_name", "fake/dinov3"))
        self.fusion_layers = list(fusion.get("layers", [2]))
        self.fusion_output_dim = int(fusion.get("output_dim", 4))
        self.fusion_dropout = float(fusion.get("dropout", 0.1))
        self.fusion_gating = str(fusion.get("gating", "softmax"))
        self.num_epochs = int(training.get("num_epochs", 1))
        self.grad_accumulation_steps = 1
        self.scheduler_step_on = "batch"
        self.mixed_precision = False
        self.evaluation_best_metric = "val_loss"
        self.early_stopping_enabled = False

    @classmethod
    def from_training_config(cls, payload):
        return cls(payload)

    def as_contract_dict(self) -> dict:
        return {
            "backbone": {"model_name": self.backbone_model_name},
            "fusion": {
                "layers": list(self.fusion_layers),
                "output_dim": int(self.fusion_output_dim),
                "dropout": float(self.fusion_dropout),
                "gating": str(self.fusion_gating),
            },
            "num_epochs": int(self.num_epochs),
        }


class FakeTrainer:
    def __init__(self, config: FakeTrainerConfig) -> None:
        self.config = config
        self.class_to_idx: dict[str, int] = {}
        self.target_modules_resolved = ["transformer.block.0.linear"]
        self.ood_detector = FakeOODDetector()
        self.current_epoch = 0
        self.optimizer_steps = 0
        self.best_metric_state: dict[str, object] = {}
        self.adapter_model = nn.Identity()
        self.classifier = nn.Linear(int(config.fusion_output_dim), 2)
        self.fusion = nn.Identity()
        self._config_hash = "offline-fake-config"
        self._peft_available = False
        self._adapter_wrapped = False

    def initialize_engine(self, class_to_idx=None) -> None:
        self.class_to_idx = dict(class_to_idx or {})
        self.classifier = nn.Linear(int(self.config.fusion_output_dim), max(1, len(self.class_to_idx)))

    def set_preferred_ood_calibration_loader(self, _loader) -> None:
        return None

    def configure_training_plan(self, *, total_batches: int, num_epochs: int) -> None:
        self._planned_batches = int(total_batches)
        self._planned_epochs = int(num_epochs)

    def set_train_mode(self) -> None:
        self.adapter_model.train()
        self.classifier.train()
        self.fusion.train()

    def set_eval_mode(self) -> None:
        self.adapter_model.eval()
        self.classifier.eval()
        self.fusion.eval()

    def train_batch(self, batch) -> TrainBatchStats:
        self.optimizer_steps += 1
        batch_size = int(batch["images"].shape[0])
        loss = 0.25 / float(self.optimizer_steps)
        return TrainBatchStats(
            loss=loss,
            lr=1e-3,
            grad_norm=0.1,
            step_time_sec=0.001,
            samples_per_sec=float(batch_size) / 0.001,
            batch_size=batch_size,
            accumulation_step=1,
            optimizer_steps=self.optimizer_steps,
            optimizer_step_applied=True,
        )

    def calibrate_ood(self, loader) -> dict:
        del loader
        self.ood_detector.calibration_version = 7
        self.ood_detector.class_stats = {0: FakeCalibrationStats(), 1: FakeCalibrationStats()}
        return {"num_classes": float(len(self.class_to_idx))}

    def save_adapter(self, output_dir: str) -> Path:
        root = Path(output_dir) / "continual_sd_lora_adapter"
        root.mkdir(parents=True, exist_ok=True)
        (root / "classifier.pth").write_bytes(b"offline")
        (root / "fusion.pth").write_bytes(b"offline")
        return root

    def load_adapter(self, adapter_dir: str) -> dict:
        meta = json.loads((Path(adapter_dir) / "adapter_meta.json").read_text(encoding="utf-8"))
        self.class_to_idx = {str(key): int(value) for key, value in dict(meta.get("class_to_idx", {})).items()}
        calibration = dict(meta.get("ood_calibration", {}))
        self.ood_detector.calibration_version = int(calibration.get("version", 0))
        self.ood_detector.class_stats = {0: FakeCalibrationStats(), 1: FakeCalibrationStats()}
        return meta

    def predict_with_ood(self, image) -> dict:
        del image
        return {
            "status": "success",
            "disease": {"class_index": 0, "name": "healthy", "confidence": 0.91},
            "ood_analysis": {
                "score_method": "ensemble",
                "primary_score": 0.12,
                "decision_threshold": 0.8,
                "is_ood": False,
                "calibration_version": int(self.ood_detector.calibration_version),
                "conformal_set": ["healthy"],
            },
        }


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def _write_runtime_dataset(runtime_root: Path) -> Path:
    crop_root = runtime_root / "tomato"
    image_specs = {
        "continual/healthy/healthy_train_0.png": (0, 160, 0),
        "continual/disease_a/disease_train_0.png": (180, 0, 0),
        "val/healthy/healthy_val_0.png": (0, 180, 0),
        "val/disease_a/disease_val_0.png": (200, 0, 0),
        "test/healthy/healthy_test_0.png": (0, 200, 0),
        "test/disease_a/disease_test_0.png": (220, 0, 0),
        "ood/field/field_ood_0.png": (0, 0, 200),
        "ood/field/field_ood_1.png": (0, 0, 220),
    }
    for relative_path, color in image_specs.items():
        _write_image(crop_root / relative_path, color)

    manifest_rows = []
    for split_name in ("continual", "val", "test"):
        for class_name in ("healthy", "disease_a"):
            filename = f"{class_name}_{'train' if split_name == 'continual' else split_name}_0.png"
            manifest_rows.append(
                {
                    "split": split_name,
                    "raw_class_name": class_name,
                    "normalized_class_name": class_name,
                    "relative_path": f"{class_name}/{filename}",
                    "runtime_relative_path": f"{split_name}/{class_name}/{filename}",
                    "source_hint": "generated",
                }
            )
    (crop_root / "split_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "v1_runtime_split_manifest",
                "crop_name": "tomato",
                "classes": [
                    {"class_name": "healthy", "image_count": 120},
                    {"class_name": "disease_a", "image_count": 130},
                ],
                "rows": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return crop_root


def _validation_report() -> ValidationReport:
    return ValidationReport(
        val_loss=0.05,
        val_accuracy=1.0,
        macro_precision=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        weighted_f1=1.0,
        balanced_accuracy=1.0,
        per_class_accuracy={"healthy": 1.0, "disease_a": 1.0},
        per_class_support={"healthy": 1, "disease_a": 1},
        worst_classes=[],
    )


def _prediction_rows(loader, split_name: str) -> list[dict]:
    dataset = loader.dataset
    rows: list[dict] = []
    for image_path, label in zip(dataset.image_paths, dataset.labels):
        label_name = str(dataset.classes[int(label)])
        rows.append(
            {
                "sample_origin": "in_distribution",
                "split_name": split_name,
                "image_path": str(image_path),
                "true_index": int(label),
                "pred_index": int(label),
                "true_label": label_name,
                "pred_label": label_name,
                "is_correct": True,
                "class_confidence": 0.99,
            }
        )
    return rows


def _evaluate_with_artifacts(_trainer, loader, *, ood_loader=None) -> EvaluationArtifactsPayload:
    split_name = str(getattr(loader.dataset, "split", "test") or "test")
    y_true = [int(label) for label in list(loader.dataset.labels)]
    y_pred = list(y_true)
    include_ood = ood_loader is not None
    ood_labels = None
    ood_scores = None
    ood_scores_by_method: dict[str, list[float]] = {}
    ood_type_breakdown: dict[str, object] = {}
    if include_ood:
        ood_labels = ([0] * 30) + ([1] * 30)
        ood_scores = [0.10 + (index % 5) * 0.01 for index in range(30)] + [
            0.82 + (index % 5) * 0.02 for index in range(30)
        ]
        ood_scores_by_method = {
            "ensemble": list(ood_scores),
        }
        ood_type_breakdown = {
            "field": {
                "sample_count": 30,
                "metrics": {
                    "ood_auroc": 1.0,
                    "ood_false_positive_rate": 0.0,
                    "ood_samples": 30,
                    "in_distribution_samples": 30,
                },
                "method_metrics": {
                    "ensemble": {
                        "ood_auroc": 1.0,
                        "ood_false_positive_rate": 0.0,
                        "ood_samples": 30,
                        "in_distribution_samples": 30,
                    }
                },
            }
        }
    return EvaluationArtifactsPayload(
        report=_validation_report(),
        y_true=y_true,
        y_pred=y_pred,
        prediction_rows=_prediction_rows(loader, split_name),
        ood_labels=ood_labels,
        ood_scores=ood_scores,
        ood_primary_score_method="ensemble",
        ood_scores_by_method=ood_scores_by_method,
        ood_type_breakdown=ood_type_breakdown,
        sure_ds_f1=0.95 if include_ood else None,
        conformal_empirical_coverage=1.0 if include_ood else None,
        conformal_avg_set_size=1.0 if include_ood else None,
        context={
            "crop_name": "tomato",
            "split_name": split_name,
            "ood_requested_primary_score_method": "auto",
            "ood_primary_score_method": "ensemble",
            "ood_primary_score_selection_source": "real_ood_guardrail_no_dev",
            "ood_score_methods": ["ensemble"] if include_ood else [],
        },
    )


def _evaluation_report(_trainer, _loader) -> ValidationReport:
    return _validation_report()


def _load_golden(name: str) -> dict:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8-sig"))


def _adapter_meta_snapshot(adapter_meta: dict) -> dict:
    calibration = dict(adapter_meta.get("ood_calibration", {}))
    runtime = dict(adapter_meta.get("adapter_runtime", {}))
    return {
        "schema_version": str(adapter_meta.get("schema_version", "")),
        "engine": str(adapter_meta.get("engine", "")),
        "crop_name": str(adapter_meta.get("crop_name", "")),
        "class_to_idx": dict(adapter_meta.get("class_to_idx", {})),
        "ood_calibration": {
            "version": int(calibration.get("version", 0)),
            "source_split": str(calibration.get("source_split", "")),
            "source_loader_size": int(calibration.get("source_loader_size", 0)),
            "authoritative_classification_split": str(calibration.get("authoritative_classification_split", "")),
            "ood_evidence_source": str(calibration.get("ood_evidence_source", "")),
            "primary_score_method": str(calibration.get("primary_score_method", "")),
            "selection_source": str(calibration.get("selection_source", "")),
        },
        "adapter_runtime": {
            "best_state_restored": bool(runtime.get("best_state_restored", False)),
        },
    }


def _production_readiness_snapshot(payload: dict) -> dict:
    classification = dict(payload.get("classification_evidence", {}))
    ood_evidence = dict(payload.get("ood_evidence", {}))
    context = dict(payload.get("context", {}))
    return {
        "status": str(payload.get("status", "")),
        "passed": bool(payload.get("passed", False)),
        "policy_passed": bool(payload.get("policy_passed", False)),
        "ood_evidence_source": str(payload.get("ood_evidence_source", "")),
        "classification_split": str(classification.get("split_name", "")),
        "missing_requirements": list(payload.get("missing_requirements", [])),
        "missing_deployment_requirements": list(payload.get("missing_deployment_requirements", [])),
        "ood_metrics": {
            "ood_auroc": float(ood_evidence.get("metrics", {}).get("ood_auroc", 0.0)),
            "ood_false_positive_rate": float(ood_evidence.get("metrics", {}).get("ood_false_positive_rate", 0.0)),
            "sure_ds_f1": float(ood_evidence.get("metrics", {}).get("sure_ds_f1", 0.0)),
            "conformal_empirical_coverage": float(
                ood_evidence.get("metrics", {}).get("conformal_empirical_coverage", 0.0)
            ),
            "conformal_avg_set_size": float(ood_evidence.get("metrics", {}).get("conformal_avg_set_size", 0.0)),
        },
        "context": {
            "calibration_split_name": str(context.get("calibration_split_name", "")),
            "ood_primary_score_method": str(context.get("ood_primary_score_method", "")),
            "ood_primary_score_selection_source": str(context.get("ood_primary_score_selection_source", "")),
        },
    }


def _inference_payload_snapshot(payload: dict) -> dict:
    router = dict(payload.get("router", {}))
    ood_analysis = dict(payload.get("ood_analysis", {}))
    return {
        "status": str(payload.get("status", "")),
        "crop": payload.get("crop"),
        "part": payload.get("part"),
        "diagnosis": payload.get("diagnosis"),
        "confidence": float(payload.get("confidence", 0.0)),
        "conformal_set": list(payload.get("conformal_set", [])),
        "router": {
            "status": str(router.get("status", "")),
            "message": str(router.get("message", "")),
            "detections_count": int(router.get("detections_count", 0)),
            "primary_detection": dict(router.get("primary_detection", {})),
        },
        "ood_analysis": {
            "score_method": str(ood_analysis.get("score_method", "")),
            "primary_score": float(ood_analysis.get("primary_score", 0.0)),
            "decision_threshold": float(ood_analysis.get("decision_threshold", 0.0)),
            "is_ood": bool(ood_analysis.get("is_ood", False)),
            "calibration_version": int(ood_analysis.get("calibration_version", 0)),
            "conformal_set": list(ood_analysis.get("conformal_set", [])),
        },
    }


@pytest.fixture
def offline_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    runtime_root = tmp_path / "runtime_data"
    _write_runtime_dataset(runtime_root)
    monkeypatch.setattr(adapter_module, "_trainer_types", lambda: (FakeTrainerConfig, FakeTrainer))
    monkeypatch.setattr("src.training.session.evaluate_model", _evaluation_report)
    monkeypatch.setattr("src.workflows.training.evaluate_model_with_artifact_metrics", _evaluate_with_artifacts)
    monkeypatch.setattr("src.workflows.training.run_leave_one_class_out_benchmark", lambda **_kwargs: {})

    training_config = {
        "training": {
            "continual": {
                "backbone": {"model_name": "fake/dinov3"},
                "fusion": {"layers": [2], "output_dim": 4, "dropout": 0.1, "gating": "softmax"},
                "ood": {"primary_score_method": "auto"},
                "batch_size": 1,
                "seed": 7,
                "num_epochs": 1,
                "data": {
                    "target_size": 8,
                    "cache_size": 0,
                    "loader_error_policy": "strict",
                    "validate_images_on_init": True,
                },
                "evaluation": {"require_ood_for_gate": True},
                "optimization": {"loss_name": "cross_entropy", "logitnorm_tau": 1.0},
            }
        },
        "colab": {"training": {"num_workers": 0, "pin_memory": False}},
    }

    workflow = TrainingWorkflow(config=training_config, device="cpu")
    result = workflow.run(
        crop_name="tomato",
        data_dir=runtime_root,
        output_dir=tmp_path / "training_outputs",
    )

    deployment_root = tmp_path / "models" / "adapters" / "tomato" / "unspecified" / "continual_sd_lora_adapter"
    deployment_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(result.adapter_dir, deployment_root, dirs_exist_ok=True)

    inference_workflow = InferenceWorkflow(
        config={
            "inference": {"adapter_root": str((tmp_path / "models" / "adapters").resolve()), "target_size": 8}
        },
        device="cpu",
    )
    inference_payload = inference_workflow.predict(
        Image.new("RGB", (8, 8), color=(0, 160, 0)),
        crop_hint="tomato",
        part_hint="leaf",
        trust_crop_hint=True,
    )

    adapter_meta = json.loads((deployment_root / "adapter_meta.json").read_text(encoding="utf-8"))
    production_readiness = json.loads((result.artifact_dir / "production_readiness.json").read_text(encoding="utf-8"))
    return {
        "adapter_meta_snapshot": _adapter_meta_snapshot(adapter_meta),
        "production_readiness_snapshot": _production_readiness_snapshot(production_readiness),
        "inference_payload_snapshot": _inference_payload_snapshot(inference_payload),
    }


def test_offline_training_workflow_matches_golden_artifacts(offline_roundtrip: dict) -> None:
    assert offline_roundtrip["adapter_meta_snapshot"] == _load_golden("adapter_meta_snapshot.json")
    assert offline_roundtrip["production_readiness_snapshot"] == _load_golden("production_readiness_snapshot.json")


def test_offline_inference_workflow_matches_golden_payload(offline_roundtrip: dict) -> None:
    assert offline_roundtrip["inference_payload_snapshot"] == _load_golden("inference_payload_snapshot.json")
