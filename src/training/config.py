"""ContinualSDLoRAConfig: the validated runtime configuration for continual SD-LoRA training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.training.quantization import assert_no_prohibited_4bit_flags
from src.training.services.config_surface import (
    DEFAULT_BACKBONE_MODEL_NAME,
    normalize_continual_training_config,
)
from src.training.services.ood_score_selection import (
    SUPPORTED_REQUESTED_OOD_SCORE_METHODS,
    normalize_requested_primary_score_method,
)

_CONFIG_EXTRA_EXCLUDED_KEYS = {
    "backbone",
    "adapter",
    "fusion",
    "ood",
    "optimization",
    "early_stopping",
    "evaluation",
    "learning_rate",
    "weight_decay",
    "num_epochs",
    "batch_size",
    "device",
    "strict_model_loading",
    "seed",
    "deterministic",
}


def _resolve_radial_beta_range(ood: Dict[str, Any]) -> tuple[float, float]:
    raw_range = ood.get("radial_beta_range", [0.5, 2.0])
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) < 2:
        raw_range = [0.5, 2.0]
    return float(raw_range[0]), float(raw_range[1])


def _resolve_early_stopping_surface(
    early_stopping: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> tuple[str, str]:
    metric = str(early_stopping.get("metric", evaluation.get("best_metric", "val_loss")))
    mode = str(
        early_stopping.get(
            "mode",
            "min" if metric in {"val_loss", "generalization_gap"} else "max",
        )
    )
    return metric, mode


def _collect_extra_training_fields(normalized: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in normalized.items()
        if key not in _CONFIG_EXTRA_EXCLUDED_KEYS
    }


@dataclass
class ContinualSDLoRAConfig:
    """Runtime configuration for v6 continual SD-LoRA training."""

    backbone_model_name: str = DEFAULT_BACKBONE_MODEL_NAME
    target_modules_strategy: str = "all_linear_transformer"
    fusion_layers: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    fusion_output_dim: int = 768
    fusion_dropout: float = 0.1
    fusion_gating: str = "softmax"
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    batch_size: int = 8
    device: str = "cuda"
    strict_model_loading: bool = False
    ood_threshold_factor: float = 2.0
    ood_primary_score_method: str = "auto"
    seed: int = 42
    deterministic: bool = True
    grad_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    mixed_precision: str = "auto"
    enable_torch_compile: bool = False
    loss_name: str = "logitnorm"
    logitnorm_tau: float = 1.0
    label_smoothing: float = 0.0
    scheduler_name: str = "cosine"
    scheduler_warmup_ratio: float = 0.1
    scheduler_min_lr: float = 1e-6
    scheduler_step_on: str = "batch"
    early_stopping_enabled: bool = True
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    evaluation_best_metric: str = "val_loss"
    evaluation_emit_ood_gate: bool = True
    evaluation_require_ood_for_gate: bool = True
    evaluation_ood_benchmark_min_classes: int = 3
    evaluation_min_in_distribution_samples: int = 30
    evaluation_min_ood_samples: int = 30
    evaluation_min_ood_samples_per_type: int = 5
    evaluation_gate_auxiliary_ood_diagnostics: bool = False
    # --- Bi-directional Energy Regularization (BER) ---
    ber_enabled: bool = False
    ber_lambda_old: float = 0.1
    ber_lambda_new: float = 0.1
    ber_warmup_steps: int = 50
    energy_temperature_mode: str = "auto"
    energy_temperature: float = 1.0
    energy_temperature_min: float = 0.5
    energy_temperature_max: float = 3.0
    energy_temperature_steps: int = 16
    react_enabled: bool = False
    react_percentile: float = 0.99
    react_apply_during_calibration: bool = True
    react_apply_during_inference: bool = True
    # --- Radially Scaled L2 Normalization ---
    radial_l2_enabled: bool = True
    radial_beta_min: float = 0.5
    radial_beta_max: float = 2.0
    radial_beta_steps: int = 16
    knn_backend: str = "auto"
    knn_chunk_size: int = 2048
    # --- SURE+/DS-F1-inspired Double Scoring ---
    sure_enabled: bool = True
    sure_semantic_percentile: float = 95.0
    sure_confidence_percentile: float = 90.0
    # --- Conformal Prediction ---
    conformal_enabled: bool = True
    conformal_alpha: float = 0.05
    conformal_method: str = "threshold"
    conformal_raps_lambda: float = 0.0
    conformal_raps_k_reg: int = 1
    oe_enabled: bool = True
    oe_loss_weight: float = 0.5
    oe_target: str = "uniform"
    oe_root: str = ""
    classifier_rebalance_enabled: bool = False
    classifier_rebalance_epochs: int = 3
    classifier_rebalance_learning_rate: float = 5e-5
    classifier_rebalance_weight_decay: float = 0.0
    classifier_rebalance_sampler: str = "weighted"
    classifier_rebalance_objective: str = "logit_adjusted_cross_entropy"
    classifier_rebalance_logit_adjustment_tau: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.target_modules_strategy != "all_linear_transformer":
            raise ValueError("v6 requires target_modules_strategy='all_linear_transformer'.")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("lora_r and lora_alpha must be positive.")
        if not self.fusion_layers:
            raise ValueError("fusion_layers must not be empty.")
        if self.grad_accumulation_steps <= 0:
            raise ValueError("grad_accumulation_steps must be positive.")
        if self.max_grad_norm < 0.0:
            raise ValueError("max_grad_norm must be non-negative.")
        if self.loss_name not in {"cross_entropy", "logitnorm"}:
            raise ValueError("optimization.loss_name must be one of: cross_entropy, logitnorm.")
        if self.logitnorm_tau <= 0.0:
            raise ValueError("optimization.logitnorm_tau must be positive.")
        if self.label_smoothing < 0.0:
            raise ValueError("label_smoothing must be non-negative.")
        if self.mixed_precision not in {"off", "auto", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of: off, auto, fp16, bf16.")
        if self.scheduler_name not in {"none", "linear", "cosine"}:
            raise ValueError("scheduler.name must be one of: none, linear, cosine.")
        if self.scheduler_step_on not in {"batch", "epoch"}:
            raise ValueError("scheduler.step_on must be 'batch' or 'epoch'.")
        if self.early_stopping_mode not in {"min", "max"}:
            raise ValueError("early_stopping.mode must be 'min' or 'max'.")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping.patience must be non-negative.")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping.min_delta must be non-negative.")
        if self.evaluation_ood_benchmark_min_classes < 1:
            raise ValueError("evaluation_ood_benchmark_min_classes must be at least 1.")
        if self.evaluation_min_in_distribution_samples < 0:
            raise ValueError("evaluation_min_in_distribution_samples must be non-negative.")
        if self.evaluation_min_ood_samples < 0:
            raise ValueError("evaluation_min_ood_samples must be non-negative.")
        if self.evaluation_min_ood_samples_per_type < 0:
            raise ValueError("evaluation_min_ood_samples_per_type must be non-negative.")
        if self.ood_primary_score_method not in SUPPORTED_REQUESTED_OOD_SCORE_METHODS:
            raise ValueError(
                "ood.primary_score_method must be one of: " + ", ".join(SUPPORTED_REQUESTED_OOD_SCORE_METHODS) + "."
            )
        if self.ber_lambda_old < 0.0 or self.ber_lambda_new < 0.0:
            raise ValueError("BER lambda values must be non-negative.")
        if self.ber_enabled and self.loss_name == "logitnorm":
            raise ValueError(
                "training.continual.ood.ber_enabled is incompatible with "
                "optimization.loss_name='logitnorm'."
            )
        if self.ber_warmup_steps < 0:
            raise ValueError("ber_warmup_steps must be non-negative.")
        if self.energy_temperature_mode not in {"fixed", "auto"}:
            raise ValueError("energy_temperature_mode must be 'fixed' or 'auto'.")
        if self.energy_temperature <= 0.0:
            raise ValueError("energy_temperature must be positive.")
        if self.energy_temperature_min <= 0.0 or self.energy_temperature_max <= 0.0:
            raise ValueError("energy_temperature_range values must be positive.")
        if self.energy_temperature_min > self.energy_temperature_max:
            raise ValueError("energy_temperature_min must be less than or equal to energy_temperature_max.")
        if self.energy_temperature_steps < 2:
            raise ValueError("energy_temperature_steps must be at least 2.")
        if not (0.0 < self.react_percentile <= 1.0):
            raise ValueError("react_percentile must be in (0, 1].")
        if self.radial_beta_min <= 0.0 or self.radial_beta_max <= 0.0:
            raise ValueError("Radial beta range values must be positive.")
        if self.radial_beta_min >= self.radial_beta_max:
            raise ValueError("radial_beta_min must be less than radial_beta_max.")
        if self.radial_beta_steps < 2:
            raise ValueError("radial_beta_steps must be at least 2.")
        if self.knn_backend not in {"auto", "cdist", "chunked", "faiss"}:
            raise ValueError("knn_backend must be one of: auto, cdist, chunked, faiss.")
        if self.knn_chunk_size < 1:
            raise ValueError("knn_chunk_size must be positive.")
        if not (0.0 < self.conformal_alpha < 1.0):
            raise ValueError("conformal_alpha must be in (0, 1).")
        if self.conformal_method not in {"threshold", "aps", "raps"}:
            raise ValueError("conformal_method must be one of: threshold, aps, raps.")
        if self.conformal_raps_lambda < 0.0:
            raise ValueError("conformal_raps_lambda must be non-negative.")
        if self.conformal_raps_k_reg < 1:
            raise ValueError("conformal_raps_k_reg must be at least 1.")
        if self.oe_loss_weight < 0.0:
            raise ValueError("oe_loss_weight must be non-negative.")
        if self.oe_target not in {"uniform"}:
            raise ValueError("oe_target must be 'uniform'.")
        if self.classifier_rebalance_epochs < 1:
            raise ValueError("classifier_rebalance_epochs must be at least 1.")
        if self.classifier_rebalance_learning_rate <= 0.0:
            raise ValueError("classifier_rebalance_learning_rate must be positive.")
        if self.classifier_rebalance_sampler not in {"weighted", "shuffle", "auto"}:
            raise ValueError("classifier_rebalance_sampler must be one of: auto, shuffle, weighted.")
        if self.classifier_rebalance_objective not in {"cross_entropy", "logit_adjusted_cross_entropy"}:
            raise ValueError(
                "classifier_rebalance_objective must be one of: cross_entropy, logit_adjusted_cross_entropy."
            )
        if self.classifier_rebalance_logit_adjustment_tau < 0.0:
            raise ValueError("classifier_rebalance_logit_adjustment_tau must be non-negative.")
        if not (0.0 < self.sure_semantic_percentile <= 100.0):
            raise ValueError("sure_semantic_percentile must be in (0, 100].")
        if not (0.0 < self.sure_confidence_percentile <= 100.0):
            raise ValueError("sure_confidence_percentile must be in (0, 100].")

    def as_contract_dict(self) -> Dict[str, Any]:
        """Return normalized config payload used in metadata persistence."""
        payload = {
            "backbone": {"model_name": self.backbone_model_name},
            "adapter": {
                "target_modules_strategy": self.target_modules_strategy,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
            },
            "fusion": {
                "layers": self.fusion_layers,
                "output_dim": self.fusion_output_dim,
                "dropout": self.fusion_dropout,
                "gating": self.fusion_gating,
            },
            "ood": {
                "threshold_factor": self.ood_threshold_factor,
                "primary_score_method": self.ood_primary_score_method,
                "ber_enabled": self.ber_enabled,
                "ber_lambda_old": self.ber_lambda_old,
                "ber_lambda_new": self.ber_lambda_new,
                "ber_warmup_steps": self.ber_warmup_steps,
                "energy_temperature_mode": self.energy_temperature_mode,
                "energy_temperature": self.energy_temperature,
                "energy_temperature_range": [self.energy_temperature_min, self.energy_temperature_max],
                "energy_temperature_steps": self.energy_temperature_steps,
                "react_enabled": self.react_enabled,
                "react_percentile": self.react_percentile,
                "react_apply_during_calibration": self.react_apply_during_calibration,
                "react_apply_during_inference": self.react_apply_during_inference,
                "radial_l2_enabled": self.radial_l2_enabled,
                "radial_beta_range": [self.radial_beta_min, self.radial_beta_max],
                "radial_beta_steps": self.radial_beta_steps,
                "knn_backend": self.knn_backend,
                "knn_chunk_size": self.knn_chunk_size,
                "sure_enabled": self.sure_enabled,
                "sure_semantic_percentile": self.sure_semantic_percentile,
                "sure_confidence_percentile": self.sure_confidence_percentile,
                "conformal_enabled": self.conformal_enabled,
                "conformal_alpha": self.conformal_alpha,
                "conformal_method": self.conformal_method,
                "conformal_raps_lambda": self.conformal_raps_lambda,
                "conformal_raps_k_reg": self.conformal_raps_k_reg,
                "oe_enabled": self.oe_enabled,
                "oe_loss_weight": self.oe_loss_weight,
                "oe_target": self.oe_target,
                "oe_root": self.oe_root,
            },
            "classifier_rebalance": {
                "enabled": self.classifier_rebalance_enabled,
                "epochs": self.classifier_rebalance_epochs,
                "learning_rate": self.classifier_rebalance_learning_rate,
                "weight_decay": self.classifier_rebalance_weight_decay,
                "sampler": self.classifier_rebalance_sampler,
                "objective": self.classifier_rebalance_objective,
                "logit_adjustment_tau": self.classifier_rebalance_logit_adjustment_tau,
            },
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "device": self.device,
            "strict_model_loading": self.strict_model_loading,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "optimization": {
                "grad_accumulation_steps": self.grad_accumulation_steps,
                "max_grad_norm": self.max_grad_norm,
                "mixed_precision": self.mixed_precision,
                "loss_name": self.loss_name,
                "logitnorm_tau": self.logitnorm_tau,
                "label_smoothing": self.label_smoothing,
                "scheduler": {
                    "name": self.scheduler_name,
                    "warmup_ratio": self.scheduler_warmup_ratio,
                    "min_lr": self.scheduler_min_lr,
                    "step_on": self.scheduler_step_on,
                },
            },
            "early_stopping": {
                "enabled": self.early_stopping_enabled,
                "metric": self.early_stopping_metric,
                "mode": self.early_stopping_mode,
                "patience": self.early_stopping_patience,
                "min_delta": self.early_stopping_min_delta,
            },
            "evaluation": {
                "best_metric": self.evaluation_best_metric,
                "emit_ood_gate": self.evaluation_emit_ood_gate,
                "require_ood_for_gate": self.evaluation_require_ood_for_gate,
                "ood_benchmark_min_classes": self.evaluation_ood_benchmark_min_classes,
                "min_in_distribution_samples": self.evaluation_min_in_distribution_samples,
                "min_ood_samples": self.evaluation_min_ood_samples,
                "min_ood_samples_per_type": self.evaluation_min_ood_samples_per_type,
                "gate_auxiliary_ood_diagnostics": self.evaluation_gate_auxiliary_ood_diagnostics,
            },
        }
        if self.extra:
            payload.update(dict(self.extra))
        return payload

    @classmethod
    def from_training_config(cls, training_continual: Dict[str, Any]) -> "ContinualSDLoRAConfig":
        """Build from `training.continual` dictionary."""
        normalized = normalize_continual_training_config(
            training_continual,
            model_name=str(training_continual.get("backbone", {}).get("model_name", DEFAULT_BACKBONE_MODEL_NAME)),
            device=training_continual.get("device", "cuda"),
        )
        assert_no_prohibited_4bit_flags(normalized)
        backbone = normalized.get("backbone", {})
        adapter = normalized.get("adapter", {})
        fusion = normalized.get("fusion", {})
        ood = normalized.get("ood", {})
        optimization = normalized.get("optimization", {})
        scheduler = optimization.get("scheduler", {}) if isinstance(optimization, dict) else {}
        early_stopping = normalized.get("early_stopping", {})
        evaluation = normalized.get("evaluation", {})
        classifier_rebalance = normalized.get("classifier_rebalance", {})
        radial_beta_min, radial_beta_max = _resolve_radial_beta_range(ood)
        raw_energy_temperature_range = list(ood.get("energy_temperature_range", [0.5, 3.0]))
        if len(raw_energy_temperature_range) < 2:
            raw_energy_temperature_range = [0.5, 3.0]
        early_stopping_metric, early_stopping_mode = _resolve_early_stopping_surface(
            early_stopping,
            evaluation,
        )

        config = cls(
            backbone_model_name=str(backbone.get("model_name", DEFAULT_BACKBONE_MODEL_NAME)),
            target_modules_strategy=str(adapter.get("target_modules_strategy", "all_linear_transformer")),
            fusion_layers=[int(v) for v in fusion.get("layers", [2, 5, 8, 11])],
            fusion_output_dim=int(fusion.get("output_dim", 768)),
            fusion_dropout=float(fusion.get("dropout", 0.1)),
            fusion_gating=str(fusion.get("gating", "softmax")),
            lora_r=int(adapter.get("lora_r", 16)),
            lora_alpha=int(adapter.get("lora_alpha", 16)),
            lora_dropout=float(adapter.get("lora_dropout", 0.1)),
            learning_rate=float(normalized.get("learning_rate", 1e-4)),
            weight_decay=float(normalized.get("weight_decay", 0.01)),
            num_epochs=int(normalized.get("num_epochs", 10)),
            batch_size=int(normalized.get("batch_size", 8)),
            device=str(normalized.get("device", "cuda")),
            strict_model_loading=bool(normalized.get("strict_model_loading", False)),
            ood_threshold_factor=float(ood.get("threshold_factor", 2.0)),
            ood_primary_score_method=normalize_requested_primary_score_method(
                ood.get("primary_score_method", "auto")
            ),
            ber_enabled=bool(ood.get("ber_enabled", False)),
            ber_lambda_old=float(ood.get("ber_lambda_old", 0.1)),
            ber_lambda_new=float(ood.get("ber_lambda_new", 0.1)),
            ber_warmup_steps=int(ood.get("ber_warmup_steps", 50)),
            energy_temperature_mode=str(ood.get("energy_temperature_mode", "fixed")),
            energy_temperature=float(ood.get("energy_temperature", 1.0)),
            energy_temperature_min=float(raw_energy_temperature_range[0]),
            energy_temperature_max=float(raw_energy_temperature_range[1]),
            energy_temperature_steps=int(ood.get("energy_temperature_steps", 16)),
            react_enabled=bool(ood.get("react_enabled", False)),
            react_percentile=float(ood.get("react_percentile", 0.99)),
            react_apply_during_calibration=bool(ood.get("react_apply_during_calibration", True)),
            react_apply_during_inference=bool(ood.get("react_apply_during_inference", True)),
            radial_l2_enabled=bool(ood.get("radial_l2_enabled", True)),
            radial_beta_min=radial_beta_min,
            radial_beta_max=radial_beta_max,
            radial_beta_steps=int(ood.get("radial_beta_steps", 16)),
            knn_backend=str(ood.get("knn_backend", "auto")),
            knn_chunk_size=int(ood.get("knn_chunk_size", 2048)),
            sure_enabled=bool(ood.get("sure_enabled", True)),
            sure_semantic_percentile=float(ood.get("sure_semantic_percentile", 95.0)),
            sure_confidence_percentile=float(ood.get("sure_confidence_percentile", 90.0)),
            conformal_enabled=bool(ood.get("conformal_enabled", True)),
            conformal_alpha=float(ood.get("conformal_alpha", 0.05)),
            conformal_method=str(ood.get("conformal_method", "threshold")),
            conformal_raps_lambda=float(ood.get("conformal_raps_lambda", 0.0)),
            conformal_raps_k_reg=int(ood.get("conformal_raps_k_reg", 1)),
            oe_enabled=bool(ood.get("oe_enabled", True)),
            oe_loss_weight=float(ood.get("oe_loss_weight", 0.5)),
            oe_target=str(ood.get("oe_target", "uniform")),
            oe_root=str(ood.get("oe_root", "") or ""),
            classifier_rebalance_enabled=bool(classifier_rebalance.get("enabled", False)),
            classifier_rebalance_epochs=int(classifier_rebalance.get("epochs", 3)),
            classifier_rebalance_learning_rate=float(classifier_rebalance.get("learning_rate", 5e-5)),
            classifier_rebalance_weight_decay=float(classifier_rebalance.get("weight_decay", 0.0)),
            classifier_rebalance_sampler=str(classifier_rebalance.get("sampler", "weighted")),
            classifier_rebalance_objective=str(
                classifier_rebalance.get("objective", "logit_adjusted_cross_entropy")
            ),
            classifier_rebalance_logit_adjustment_tau=float(
                classifier_rebalance.get("logit_adjustment_tau", 1.0)
            ),
            seed=int(normalized.get("seed", 42)),
            deterministic=bool(normalized.get("deterministic", True)),
            grad_accumulation_steps=int(optimization.get("grad_accumulation_steps", 4)),
            max_grad_norm=float(optimization.get("max_grad_norm", 1.0)),
            mixed_precision=str(optimization.get("mixed_precision", "auto")),
            loss_name=str(optimization.get("loss_name", "logitnorm")),
            logitnorm_tau=float(optimization.get("logitnorm_tau", 1.0)),
            label_smoothing=float(optimization.get("label_smoothing", 0.0)),
            scheduler_name=str(scheduler.get("name", "cosine")),
            scheduler_warmup_ratio=float(scheduler.get("warmup_ratio", 0.1)),
            scheduler_min_lr=float(scheduler.get("min_lr", 1e-6)),
            scheduler_step_on=str(scheduler.get("step_on", "batch")),
            early_stopping_enabled=bool(early_stopping.get("enabled", True)),
            early_stopping_metric=early_stopping_metric,
            early_stopping_mode=early_stopping_mode,
            early_stopping_patience=int(early_stopping.get("patience", 5)),
            early_stopping_min_delta=float(early_stopping.get("min_delta", 0.0)),
            evaluation_best_metric=str(evaluation.get("best_metric", "val_loss")),
            evaluation_emit_ood_gate=bool(evaluation.get("emit_ood_gate", True)),
            evaluation_require_ood_for_gate=bool(evaluation.get("require_ood_for_gate", True)),
            evaluation_ood_benchmark_min_classes=int(evaluation.get("ood_benchmark_min_classes", 3)),
            evaluation_min_in_distribution_samples=int(evaluation.get("min_in_distribution_samples", 30)),
            evaluation_min_ood_samples=int(evaluation.get("min_ood_samples", 30)),
            evaluation_min_ood_samples_per_type=int(evaluation.get("min_ood_samples_per_type", 5)),
            evaluation_gate_auxiliary_ood_diagnostics=bool(
                evaluation.get("gate_auxiliary_ood_diagnostics", False)
            ),
            extra=_collect_extra_training_fields(normalized),
        )
        config.validate()
        return config
