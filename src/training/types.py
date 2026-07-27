"""Typed payloads for continual training orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainBatchStats:
    loss: float
    lr: float
    grad_norm: float
    step_time_sec: float
    samples_per_sec: float
    batch_size: int
    accumulation_step: int = 1
    optimizer_steps: int = 0
    optimizer_step_applied: bool = True
    # --- BER loss components (None when BER is disabled) ---
    ber_ce_loss: Optional[float] = None
    ber_old_loss: Optional[float] = None
    ber_new_loss: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingProgressState:
    epoch: int = 0
    batch: int = 0
    total_batches: int = 0
    global_step: int = 0
    optimizer_steps: int = 0
    elapsed_sec: float = 0.0
    eta_sec: float = 0.0
    resume_start_epoch: int = 0
    stopped_early: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "batch": int(self.batch),
            "total_batches": int(self.total_batches),
            "global_step": int(self.global_step),
            "optimizer_steps": int(self.optimizer_steps),
            "elapsed_sec": float(self.elapsed_sec),
            "eta_sec": float(self.eta_sec),
            "resume_start_epoch": int(self.resume_start_epoch),
            "stopped_early": bool(self.stopped_early),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "TrainingProgressState":
        data = dict(payload or {})
        return cls(
            epoch=int(data.get("epoch", 0)),
            batch=int(data.get("batch", 0)),
            total_batches=int(data.get("total_batches", 0)),
            global_step=int(data.get("global_step", 0)),
            optimizer_steps=int(data.get("optimizer_steps", data.get("global_step", 0))),
            elapsed_sec=float(data.get("elapsed_sec", 0.0)),
            eta_sec=float(data.get("eta_sec", 0.0)),
            resume_start_epoch=int(data.get("resume_start_epoch", data.get("epoch", 0) if data else 0)),
            stopped_early=bool(data.get("stopped_early", False)),
        )


@dataclass
class ValidationReport:
    val_loss: float
    val_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    per_class_accuracy: Dict[str, float]
    per_class_support: Dict[str, int]
    worst_classes: List[Dict[str, Any]]
    generalization_gap: Optional[float] = None
    epoch_advisory: str = ""
    epoch_severity: str = "info"

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "val_loss": float(self.val_loss),
            "val_accuracy": float(self.val_accuracy),
            "macro_precision": float(self.macro_precision),
            "macro_recall": float(self.macro_recall),
            "macro_f1": float(self.macro_f1),
            "weighted_f1": float(self.weighted_f1),
            "balanced_accuracy": float(self.balanced_accuracy),
            "per_class_accuracy": dict(self.per_class_accuracy),
            "per_class_support": dict(self.per_class_support),
            "worst_classes": list(self.worst_classes),
            "epoch_advisory": str(self.epoch_advisory),
            "epoch_severity": str(self.epoch_severity),
        }
        if self.generalization_gap is not None:
            payload["generalization_gap"] = float(self.generalization_gap)
        return payload

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "ValidationReport":
        data = dict(payload or {})
        return cls(
            val_loss=float(data.get("val_loss", 0.0)),
            val_accuracy=float(data.get("val_accuracy", 0.0)),
            macro_precision=float(data.get("macro_precision", 0.0)),
            macro_recall=float(data.get("macro_recall", 0.0)),
            macro_f1=float(data.get("macro_f1", 0.0)),
            weighted_f1=float(data.get("weighted_f1", 0.0)),
            balanced_accuracy=float(data.get("balanced_accuracy", 0.0)),
            per_class_accuracy={str(k): float(v) for k, v in dict(data.get("per_class_accuracy", {})).items()},
            per_class_support={str(k): int(v) for k, v in dict(data.get("per_class_support", {})).items()},
            worst_classes=[dict(item) for item in list(data.get("worst_classes", [])) if isinstance(item, dict)],
            generalization_gap=(
                None if data.get("generalization_gap") is None else float(data.get("generalization_gap", 0.0))
            ),
            epoch_advisory=str(data.get("epoch_advisory", "")),
            epoch_severity=str(data.get("epoch_severity", "info")),
        )


@dataclass
class EvaluationArtifactsPayload:
    report: ValidationReport
    y_true: List[int]
    y_pred: List[int]
    prediction_rows: List[Dict[str, Any]] = field(default_factory=list)
    ood_labels: Optional[List[int]] = None
    ood_scores: Optional[List[float]] = None
    ood_primary_score_method: str = "ensemble"
    ood_scores_by_method: Dict[str, List[float]] = field(default_factory=dict)
    ood_type_breakdown: Dict[str, Any] = field(default_factory=dict)
    sure_ds_f1: Optional[float] = None
    conformal_empirical_coverage: Optional[float] = None
    conformal_avg_set_size: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    macro_precision: List[float] = field(default_factory=list)
    macro_recall: List[float] = field(default_factory=list)
    macro_f1: List[float] = field(default_factory=list)
    weighted_f1: List[float] = field(default_factory=list)
    balanced_accuracy: List[float] = field(default_factory=list)
    generalization_gap: List[float] = field(default_factory=list)
    per_class_accuracy: List[Dict[str, float]] = field(default_factory=list)
    worst_classes: List[List[Dict[str, Any]]] = field(default_factory=list)
    stopped_early: bool = False
    global_step: int = 0
    optimizer_steps: int = 0
    resume_start_epoch: int = 0
    best_metric_name: str = ""
    best_metric_value: Optional[float] = None
    best_epoch: int = 0

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "train_loss": list(self.train_loss),
            "val_loss": list(self.val_loss),
            "val_accuracy": list(self.val_accuracy),
            "macro_precision": list(self.macro_precision),
            "macro_recall": list(self.macro_recall),
            "macro_f1": list(self.macro_f1),
            "weighted_f1": list(self.weighted_f1),
            "balanced_accuracy": list(self.balanced_accuracy),
            "generalization_gap": list(self.generalization_gap),
            "per_class_accuracy": [dict(item) for item in self.per_class_accuracy],
            "worst_classes": [[dict(row) for row in group] for group in self.worst_classes],
            "stopped_early": bool(self.stopped_early),
            "global_step": int(self.global_step),
            "optimizer_steps": int(self.optimizer_steps),
            "resume_start_epoch": int(self.resume_start_epoch),
            "best_metric_name": str(self.best_metric_name),
            "best_epoch": int(self.best_epoch),
        }
        if self.best_metric_value is not None:
            payload["best_metric_value"] = float(self.best_metric_value)
        return payload

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "TrainingHistory":
        data = dict(payload or {})
        best_metric_value = data.get("best_metric_value")

        def _to_float_list(key: str) -> List[float]:
            return [float(v) for v in list(data.get(key, []))]

        def _to_per_class_accuracy(key: str) -> List[Dict[str, float]]:
            return [
                {str(k): float(v) for k, v in dict(item).items()}
                for item in list(data.get(key, []))
                if isinstance(item, dict)
            ]

        def _to_worst_classes(key: str) -> List[List[Dict[str, Any]]]:
            return [
                [dict(row) for row in group if isinstance(row, dict)]
                for group in list(data.get(key, []))
                if isinstance(group, list)
            ]

        return cls(
            train_loss=_to_float_list("train_loss"),
            val_loss=_to_float_list("val_loss"),
            val_accuracy=_to_float_list("val_accuracy"),
            macro_precision=_to_float_list("macro_precision"),
            macro_recall=_to_float_list("macro_recall"),
            macro_f1=_to_float_list("macro_f1"),
            weighted_f1=_to_float_list("weighted_f1"),
            balanced_accuracy=_to_float_list("balanced_accuracy"),
            generalization_gap=_to_float_list("generalization_gap"),
            per_class_accuracy=_to_per_class_accuracy("per_class_accuracy"),
            worst_classes=_to_worst_classes("worst_classes"),
            stopped_early=bool(data.get("stopped_early", False)),
            global_step=int(data.get("global_step", 0)),
            optimizer_steps=int(data.get("optimizer_steps", data.get("global_step", 0))),
            resume_start_epoch=int(data.get("resume_start_epoch", 0)),
            best_metric_name=str(data.get("best_metric_name", "")),
            best_metric_value=(None if best_metric_value is None else float(best_metric_value)),
            best_epoch=int(data.get("best_epoch", 0)),
        )

    def append_validation(self, report: ValidationReport) -> None:
        self.val_loss.append(float(report.val_loss))
        self.val_accuracy.append(float(report.val_accuracy))
        self.macro_precision.append(float(report.macro_precision))
        self.macro_recall.append(float(report.macro_recall))
        self.macro_f1.append(float(report.macro_f1))
        self.weighted_f1.append(float(report.weighted_f1))
        self.balanced_accuracy.append(float(report.balanced_accuracy))
        if report.generalization_gap is not None:
            self.generalization_gap.append(float(report.generalization_gap))
        self.per_class_accuracy.append(dict(report.per_class_accuracy))
        self.worst_classes.append(list(report.worst_classes))


@dataclass
class TrainingCheckpointPayload:
    schema_version: str
    created_at: str
    trainer_config: Dict[str, Any]
    class_to_idx: Dict[str, int]
    target_modules_resolved: List[str]
    model_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]]
    ood_state: Dict[str, Any]
    rng_state: Dict[str, Any]
    config_hash: str = ""
    scheduler_state: Optional[Dict[str, Any]] = None
    scaler_state: Optional[Dict[str, Any]] = None
    best_metric_state: Dict[str, Any] = field(default_factory=dict)
    gradient_state: Dict[str, Any] = field(default_factory=dict)
    current_epoch: int = 0
    optimizer_steps: int = 0
    accumulation_counter: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "created_at": str(self.created_at),
            "trainer_config": dict(self.trainer_config),
            "config_hash": str(self.config_hash),
            "class_to_idx": {str(k): int(v) for k, v in self.class_to_idx.items()},
            "target_modules_resolved": [str(v) for v in self.target_modules_resolved],
            "model_state": dict(self.model_state),
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "scaler_state": self.scaler_state,
            "ood_state": dict(self.ood_state),
            "rng_state": dict(self.rng_state),
            "best_metric_state": dict(self.best_metric_state),
            "gradient_state": dict(self.gradient_state),
            "current_epoch": int(self.current_epoch),
            "optimizer_steps": int(self.optimizer_steps),
            "accumulation_counter": int(self.accumulation_counter),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainingCheckpointPayload":
        data = dict(payload or {})
        return cls(
            schema_version=str(data.get("schema_version", "v6_training_checkpoint")),
            created_at=str(data.get("created_at", "")),
            trainer_config=dict(data.get("trainer_config", {})),
            config_hash=str(data.get("config_hash", "")),
            class_to_idx={str(k): int(v) for k, v in dict(data.get("class_to_idx", {})).items()},
            target_modules_resolved=[str(v) for v in list(data.get("target_modules_resolved", []))],
            model_state=dict(data.get("model_state", {})),
            optimizer_state=data.get("optimizer_state"),
            scheduler_state=data.get("scheduler_state"),
            scaler_state=data.get("scaler_state"),
            ood_state=dict(data.get("ood_state", {})),
            rng_state=dict(data.get("rng_state", {})),
            best_metric_state=dict(data.get("best_metric_state", {})),
            gradient_state=dict(data.get("gradient_state", {})),
            current_epoch=int(data.get("current_epoch", 0)),
            optimizer_steps=int(data.get("optimizer_steps", 0)),
            accumulation_counter=int(data.get("accumulation_counter", 0)),
        )
