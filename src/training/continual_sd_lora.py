#!/usr/bin/env python3
"""v6 continual SD-LoRA training surfaces."""

from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, cast

import torch
import torch.nn as nn

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency in some local sandboxes
    np = None  # type: ignore[assignment]

from src.adapter.multi_scale_fusion import MultiScaleFeatureFusion, select_multiscale_features
from src.ood.continual_ood import ContinualOODDetector, normalize_primary_score_method
from src.training.ber_loss import BERLoss
from src.training.quantization import assert_no_prohibited_4bit_flags
from src.training.services.config_surface import (
    DEFAULT_BACKBONE_MODEL_NAME,
    normalize_continual_training_config,
)
from src.training.services.ood_calibration import calibrate_trainer_ood
from src.training.services.ood_score_selection import (
    SUPPORTED_REQUESTED_OOD_SCORE_METHODS,
    normalize_requested_primary_score_method,
    resolve_runtime_primary_score_method,
)
from src.training.services.persistence import (
    build_trainer_metadata_payload,
    compute_config_hash,
    load_trainer_adapter,
    save_trainer_adapter,
)
from src.training.services.persistence import (
    restore_training_state as restore_trainer_training_state,
)
from src.training.services.persistence import (
    snapshot_training_state as snapshot_trainer_training_state,
)
from src.training.services.runtime import (
    autocast_context,
    build_adamw_optimizer,
    build_grad_scaler,
    build_idx_to_class,
    compute_grad_norm,
    configure_runtime_reproducibility,
    configure_training_plan_state,
    ensure_scheduler,
    resolve_runtime_device,
    step_scheduler,
)
from src.training.services.runtime import (
    setup_optimizer as setup_trainer_optimizer,
)
from src.training.services.trainer_runtime import (
    add_trainer_classes,
    execute_train_batch,
    flush_pending_gradients,
    has_pending_gradients,
    initialize_trainer_engine,
    predict_with_ood_result,
    predict_with_ood_results,
    refresh_optimizer_after_model_change,
)
from src.training.types import TrainBatchStats, TrainingCheckpointPayload

AUTO_MODEL_FACTORY: Any = None
try:
    from transformers import AutoModel as _loaded_auto_model

    AUTO_MODEL_FACTORY = _loaded_auto_model
except ImportError:  # pragma: no cover - test fallback
    pass

# Preserve the historical module-level names so tests and callers can monkeypatch
# dependency injection points directly.
AutoModel = AUTO_MODEL_FACTORY

PEFT_LORA_CONFIG: Any = None
PEFT_MODEL_CLASS: Any = None
PEFT_GET_MODEL: Any = None
try:
    from peft import LoraConfig as _loaded_lora_config
    from peft import PeftModel as _loaded_peft_model
    from peft import get_peft_model as _loaded_peft_get_model

    PEFT_LORA_CONFIG = _loaded_lora_config
    PEFT_MODEL_CLASS = _loaded_peft_model
    PEFT_GET_MODEL = _loaded_peft_get_model
except ImportError:  # pragma: no cover - test fallback
    pass

LoraConfig = PEFT_LORA_CONFIG
PeftModel = PEFT_MODEL_CLASS
get_peft_model = PEFT_GET_MODEL


logger = logging.getLogger(__name__)

EXCLUDED_TARGET_TOKEN = ("classifier", "router", "head", "pooler")
PREFERRED_TARGET_TOKEN = (
    "transformer",
    "encoder",
    "block",
    "layer",
    "attention",
    "attn",
    "mlp",
    "ffn",
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
_META_TENSOR_COPY_ERROR = "Cannot copy out of meta tensor; no data!"


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


def _normalize_logits_for_logitnorm(logits: torch.Tensor, *, tau: float, eps: float = 1e-7) -> torch.Tensor:
    norms = torch.linalg.vector_norm(logits, ord=2, dim=1, keepdim=True).clamp_min(float(eps))
    return logits / norms / float(max(float(tau), float(eps)))


def compute_logitnorm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    tau: float,
    weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    normalized_logits = _normalize_logits_for_logitnorm(logits, tau=float(tau))
    return nn.functional.cross_entropy(
        normalized_logits,
        labels,
        weight=weight,
        label_smoothing=float(label_smoothing),
    )


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


class ContinualSDLoRATrainer:
    """Single-engine continual SD-LoRA trainer for v6 runtime."""

    def __init__(self, config: ContinualSDLoRAConfig):
        self.config = config
        self.config.validate()

        self.device = resolve_runtime_device(self.config.device)
        self.backbone: Optional[nn.Module] = None
        self.adapter_model: Optional[nn.Module] = None
        self.classifier: Optional[nn.Linear] = None
        self.fusion: Optional[MultiScaleFeatureFusion] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.LRScheduler] = None
        self.scaler = build_grad_scaler(self.device, self.config.mixed_precision)
        self.current_epoch = 0
        self.optimizer_steps = 0
        self.class_to_idx: Dict[str, int] = {}
        self.target_modules_resolved: List[str] = []
        self.ood_detector = ContinualOODDetector(
            threshold_factor=self.config.ood_threshold_factor,
            primary_score_method=resolve_runtime_primary_score_method(self.config.ood_primary_score_method),
            knn_backend=self.config.knn_backend,
            knn_chunk_size=self.config.knn_chunk_size,
            radial_l2_enabled=self.config.radial_l2_enabled,
            radial_beta_range=(self.config.radial_beta_min, self.config.radial_beta_max),
            radial_beta_steps=self.config.radial_beta_steps,
            sure_enabled=self.config.sure_enabled,
            sure_semantic_percentile=self.config.sure_semantic_percentile,
            sure_confidence_percentile=self.config.sure_confidence_percentile,
            conformal_enabled=self.config.conformal_enabled,
            conformal_alpha=self.config.conformal_alpha,
            conformal_method=self.config.conformal_method,
            conformal_raps_lambda=self.config.conformal_raps_lambda,
            conformal_raps_k_reg=self.config.conformal_raps_k_reg,
            energy_temperature=self.config.energy_temperature,
            energy_temperature_mode=self.config.energy_temperature_mode,
            energy_temperature_range=(self.config.energy_temperature_min, self.config.energy_temperature_max),
            energy_temperature_steps=self.config.energy_temperature_steps,
            react_enabled=self.config.react_enabled,
            react_percentile=self.config.react_percentile,
            react_apply_during_calibration=self.config.react_apply_during_calibration,
            react_apply_during_inference=self.config.react_apply_during_inference,
        )
        self.ber_loss: Optional[BERLoss] = None
        self._is_initialized = False
        self._contract = self.config.as_contract_dict()
        self._config_hash = compute_config_hash(self._contract)
        self._peft_available = LoraConfig is not None
        self._adapter_wrapped = False
        self._planned_scheduler_steps = 0
        self._planned_epochs = int(max(1, self.config.num_epochs))
        self._accumulation_counter = 0
        self._last_grad_norm = 0.0
        self.best_metric_state: Dict[str, Any] = {}
        self._idx_to_class: Dict[int, str] = {}
        self._trainable_params_cache: Optional[List[torch.nn.Parameter]] = None
        self._ood_calibration_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None
        self._oe_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None
        self._oe_loader_iter: Optional[Iterator[Dict[str, torch.Tensor]]] = None
        self._active_training_stage = "main"
        self._optimizer_override: Optional[Dict[str, float]] = None
        self.class_balance_runtime = (
            dict(self.config.extra.get("class_balance", {}))
            if isinstance(self.config.extra.get("class_balance"), dict)
            else {}
        )
        self.class_balance_weights: Optional[torch.Tensor] = None
        self._classifier_rebalance_log_priors: Optional[torch.Tensor] = None
        configure_runtime_reproducibility(self.config, np_module=np)
        self._refresh_class_index_cache()

    def set_ood_primary_score_method(self, primary_score_method: str) -> str:
        resolved = normalize_primary_score_method(primary_score_method)
        self.config.ood_primary_score_method = resolved
        self.ood_detector.primary_score_method = resolved
        self._contract = self.config.as_contract_dict()
        self._config_hash = compute_config_hash(self._contract)
        return resolved

    def _refresh_class_balance_weights(self) -> None:
        weights_by_class = dict(self.class_balance_runtime.get("weights_by_class", {}))
        if not self.class_to_idx or not weights_by_class:
            self.class_balance_weights = None
            return
        ordered_names = [
            class_name
            for class_name, _idx in sorted(self.class_to_idx.items(), key=lambda item: int(item[1]))
        ]
        if any(class_name not in weights_by_class for class_name in ordered_names):
            self.class_balance_weights = None
            return
        self.class_balance_weights = torch.tensor(
            [float(weights_by_class[class_name]) for class_name in ordered_names],
            dtype=torch.float32,
            device=self.device,
        )

    def _refresh_class_index_cache(self) -> None:
        self._idx_to_class = build_idx_to_class(self.class_to_idx)
        self._refresh_class_balance_weights()

    def _class_index_cache_stale(self) -> bool:
        if len(self._idx_to_class) != len(self.class_to_idx):
            return True
        return any(self._idx_to_class.get(idx) != name for name, idx in self.class_to_idx.items())

    def _refresh_optimizer_after_model_change(self) -> None:
        refresh_optimizer_after_model_change(self)

    def _reported_grad_norm(self, *, gradients_unscaled: bool) -> float:
        del gradients_unscaled
        # Non-step microbatches only need a stable telemetry value. Reuse the
        # last measured post-step norm instead of scanning all gradients again.
        return float(getattr(self, "_last_grad_norm", 0.0))

    def set_preferred_ood_calibration_loader(self, loader: Iterable[Dict[str, torch.Tensor]]) -> None:
        self._ood_calibration_loader = loader

    def set_oe_loader(self, loader: Optional[Iterable[Dict[str, torch.Tensor]]]) -> None:
        self._oe_loader = loader
        self._oe_loader_iter = None

    def _next_oe_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self.config.oe_enabled or self._oe_loader is None:
            return None
        if self._oe_loader_iter is None:
            self._oe_loader_iter = iter(self._oe_loader)
        try:
            batch = next(self._oe_loader_iter)
        except StopIteration:
            self._oe_loader_iter = iter(self._oe_loader)
            try:
                batch = next(self._oe_loader_iter)
            except StopIteration:
                return None
        return dict(batch)

    def _ensure_ood_calibrated(self, *, operation: str) -> None:
        issue = self.ood_detector.calibration_issue()
        if issue is None:
            return
        if self._ood_calibration_loader is not None:
            self.calibrate_ood(self._ood_calibration_loader)
            issue = self.ood_detector.calibration_issue()
            if issue is None:
                return
            raise RuntimeError(
                f"{issue} Automatic OOD calibration before {operation} did not produce usable class statistics."
            )
        raise RuntimeError(
            f"{issue} No calibration loader is available for automatic OOD calibration before {operation}."
        )

    def configure_training_plan(self, *, total_batches: int, num_epochs: Optional[int] = None) -> None:
        configure_training_plan_state(self, total_batches=total_batches, num_epochs=num_epochs)

    def _ensure_scheduler(self) -> None:
        ensure_scheduler(self)

    def _step_scheduler(self) -> None:
        step_scheduler(self)

    def initialize_engine(self, class_to_idx: Optional[Dict[str, int]] = None) -> None:
        initialize_trainer_engine(
            self,
            class_to_idx=class_to_idx,
            auto_model_factory=AutoModel,
            fusion_cls=MultiScaleFeatureFusion,
        )
        # Optionally attempt to compile the training_step with torch.compile.
        # This is guarded: enabled via config.extra['enable_torch_compile'] or
        # environment variable AADS_ENABLE_TORCH_COMPILE. Failures fall back
        # to the original uncompiled method.
        try_compile_flag = bool(self.config.extra.get("enable_torch_compile", False))
        env_flag = os.getenv("AADS_ENABLE_TORCH_COMPILE", "").lower() in ("1", "true", "yes")
        if self.config.enable_torch_compile or try_compile_flag or env_flag:
            if hasattr(torch, "compile") and callable(getattr(torch, "compile")):
                # On some developer machines (notably Windows without MSVC installed),
                # Torch Inductor will attempt to invoke a C++ compiler during
                # codegen and fail if `cl` is missing. Detect that situation and
                # skip compile to keep tests and lightweight runs stable.
                try:
                    import platform
                    import shutil

                    if platform.system().lower().startswith("windows") and shutil.which("cl") is None:
                        logger.debug("MSVC cl compiler not found; skipping torch.compile to avoid inductor build errors")
                        compile_available = False
                    else:
                        compile_available = True
                except Exception:
                    compile_available = True
                if not compile_available:
                    logger.debug("Skipping torch.compile based on platform/compiler checks.")
                else:
                    # First attempt default (likely inductor on Colab). If that fails,
                    # try a safer eager backend before giving up.
                    try:
                        # Compile the underlying function rather than the bound method so we can
                        # bind the compiled function to this instance reliably. Compiling the
                        # already-bound method can produce a callable that already has `self`
                        # captured which would make an additional MethodType binding pass the
                        # `self` argument twice (causing a TypeError at runtime).
                        unbound = type(self).training_step
                        compiled = torch.compile(unbound)
                        # Bind compiled unbound function with a lightweight wrapper that
                        # calls the compiled function with the instance as first arg.
                        setattr(self, "training_step", cast(Any, (lambda batch, _compiled=compiled, _self=self: _compiled(_self, batch))))
                        logger.info("torch.compile: training_step compiled successfully (default backend)")
                    except Exception as exc_default:  # pragma: no cover - runtime guard
                        logger.warning(
                            "torch.compile default backend failed for training_step: %s; attempting eager backend",
                            exc_default,
                        )
                        try:
                            unbound = type(self).training_step
                            compiled_eager = torch.compile(unbound, backend="eager")
                            setattr(self, "training_step", cast(Any, (lambda batch, _compiled=compiled_eager, _self=self: _compiled(_self, batch))))
                            logger.info("torch.compile: training_step compiled successfully (eager backend)")
                        except Exception as exc_eager:  # pragma: no cover - runtime guard
                            logger.warning(
                                "torch.compile eager backend also failed; continuing without compile: %s",
                                exc_eager,
                            )
            else:
                logger.debug("torch.compile not available in this PyTorch build; skipping compile attempt.")

    def _raise_missing_peft(self) -> None:
        message = (
            "peft is required for SD-LoRA adapter wrapping but is not available. "
            "Install a compatible `peft` package and retry."
        )
        logger.error(message)
        raise RuntimeError(message)

    @staticmethod
    def _module_has_meta_tensors(module: nn.Module) -> bool:
        return any(param.is_meta for param in module.parameters()) or any(buffer.is_meta for buffer in module.buffers())

    @staticmethod
    def _is_dispatch_managed_module(module: nn.Module) -> bool:
        if getattr(module, "hf_device_map", None) is not None:
            return True
        if hasattr(module, "_hf_hook"):
            return True
        for child in module.modules():
            if child is module:
                continue
            if hasattr(child, "_hf_hook"):
                return True
        return False

    def _prepare_module_for_device(self, module: nn.Module, module_name: str) -> nn.Module:
        has_meta_tensors = self._module_has_meta_tensors(module)
        dispatch_managed = self._is_dispatch_managed_module(module)

        if has_meta_tensors and dispatch_managed:
            logger.warning(
                "%s has meta tensors with HF/Accelerate dispatch hooks; skipping explicit .to(%s).",
                module_name,
                self.device,
            )
            return module

        if has_meta_tensors:
            raise RuntimeError(
                f"{module_name} contains meta tensors before device move and is not dispatch-managed. "
                "Disable meta/offload loading or materialize weights before initialize_engine()."
            )

        try:
            return module.to(self.device)
        except NotImplementedError as exc:
            if _META_TENSOR_COPY_ERROR in str(exc) and self._is_dispatch_managed_module(module):
                logger.warning(
                    "%s raised meta tensor NotImplementedError during .to(%s) but is dispatch-managed; "
                    "leaving module placement unchanged.",
                    module_name,
                    self.device,
                )
                return module
            raise

    @staticmethod
    def _supports_low_cpu_mem_usage_kwarg() -> bool:
        try:
            signature = inspect.signature(get_peft_model)
        except (TypeError, ValueError):
            return False

        if "low_cpu_mem_usage" in signature.parameters:
            return True
        return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())

    def _apply_lora(self, model: nn.Module, target_modules: Sequence[str]) -> nn.Module:
        if LoraConfig is None or get_peft_model is None:
            self._raise_missing_peft()

        suffixes = sorted({name.split(".")[-1] for name in target_modules})
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=suffixes,
            bias="none",
        )

        supports_low_cpu_mem_usage = self._supports_low_cpu_mem_usage_kwarg()
        if supports_low_cpu_mem_usage:
            wrapped = cast(nn.Module, get_peft_model(model, lora_config, low_cpu_mem_usage=False))
        else:
            wrapped = cast(nn.Module, get_peft_model(model, lora_config))

        for name, param in wrapped.named_parameters():
            if "lora_" in name.lower():
                param.requires_grad = True
        self._adapter_wrapped = True
        return wrapped

    def resolve_target_modules(self, model: nn.Module) -> List[str]:
        """Resolve all transformer linear modules excluding classifier/router heads."""
        preferred: List[str] = []
        fallback: List[str] = []
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            lowered = module_name.lower()
            if any(token in lowered for token in EXCLUDED_TARGET_TOKEN):
                continue
            fallback.append(module_name)
            if any(token in lowered for token in PREFERRED_TARGET_TOKEN):
                preferred.append(module_name)

        resolved = sorted(set(preferred or fallback))
        if not resolved:
            raise RuntimeError("No linear target modules resolved for all_linear_transformer strategy.")
        return resolved

    def add_classes(self, new_class_names: Iterable[str]) -> Dict[str, int]:
        return add_trainer_classes(self, new_class_names)

    def setup_optimizer(self) -> None:
        setup_trainer_optimizer(self)

    def _extract_hidden_states(self, images: torch.Tensor) -> Sequence[torch.Tensor]:
        if self.adapter_model is None:
            raise RuntimeError("Adapter model not initialized.")

        output = self.adapter_model(images, output_hidden_states=True)
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states:
            return list(hidden_states)

        if hasattr(output, "last_hidden_state"):
            return [output.last_hidden_state]
        if torch.is_tensor(output):
            return [output]
        raise RuntimeError(
            "Backbone output does not expose hidden states, last_hidden_state, or tensor output. "
            "Cannot derive feature representations for fusion."
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.fusion is None:
            raise RuntimeError("Fusion module is not initialized.")
        if images.device != self.device:
            images = images.to(self.device, non_blocking=True)
        states = self._extract_hidden_states(images)
        selected = select_multiscale_features(states, self.config.fusion_layers)
        return self.fusion(selected)

    def prepare_features_for_scoring(self, features: torch.Tensor) -> torch.Tensor:
        return self.ood_detector.apply_inference_adjustments(features)

    def forward_logits(self, images: torch.Tensor, *, apply_scoring_adjustments: bool = False) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError("Classifier is not initialized.")
        features = self.encode(images)
        if apply_scoring_adjustments:
            features = self.prepare_features_for_scoring(features)
        return self.classifier(features)

    def _logit_adjusted_logits(self, logits: torch.Tensor) -> torch.Tensor:
        priors = self._classifier_rebalance_log_priors
        if priors is None:
            return logits
        adjustment = priors.to(device=logits.device, dtype=logits.dtype) * float(
            self.config.classifier_rebalance_logit_adjustment_tau
        )
        return logits + adjustment

    def _compute_oe_uniform_loss(self, aux_logits: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(aux_logits, dim=1)
        return -log_probs.mean(dim=1).mean()

    def _compute_classifier_rebalance_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        objective = str(self.config.classifier_rebalance_objective)
        adjusted_logits = (
            self._logit_adjusted_logits(logits)
            if objective == "logit_adjusted_cross_entropy"
            else logits
        )
        return nn.functional.cross_entropy(adjusted_logits, labels)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not configured. Call setup_optimizer().")
        images = batch["images"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        with autocast_context(self.device, self.config.mixed_precision):
            try:
                logits = self.forward_logits(images, apply_scoring_adjustments=False)
            except TypeError:
                logits = self.forward_logits(images)
            if self._active_training_stage == "classifier_rebalance":
                return self._compute_classifier_rebalance_loss(logits, labels)
            if self.ber_loss is not None:
                total_loss, self._last_ber_components = self.ber_loss(
                    logits,
                    labels,
                    label_smoothing=float(self.config.label_smoothing),
                    class_weight=self.class_balance_weights,
                )
                return total_loss
            if self.config.loss_name == "logitnorm":
                total_loss = compute_logitnorm_loss(
                    logits,
                    labels,
                    tau=float(self.config.logitnorm_tau),
                    weight=self.class_balance_weights,
                    label_smoothing=float(self.config.label_smoothing),
                )
            else:
                total_loss = nn.functional.cross_entropy(
                    logits,
                    labels,
                    weight=self.class_balance_weights,
                    label_smoothing=float(self.config.label_smoothing),
                )
            aux_batch = self._next_oe_batch()
            if aux_batch is None or float(self.config.oe_loss_weight) <= 0.0:
                return total_loss
            if self.classifier is None:
                raise RuntimeError("Classifier is not initialized.")
            aux_images = aux_batch["images"].to(self.device, non_blocking=True)
            aux_logits = self.classifier(self.encode(aux_images))
            oe_loss = self._compute_oe_uniform_loss(aux_logits)
            return total_loss + (float(self.config.oe_loss_weight) * oe_loss)

    def set_train_mode(self) -> None:
        self._set_module_training_mode(training=True)

    def set_eval_mode(self) -> None:
        self._set_module_training_mode(training=False)

    def _set_module_training_mode(self, *, training: bool) -> None:
        for module in (self.adapter_model, self.classifier, self.fusion):
            if module is not None:
                module.train(training)

    def _compute_grad_norm(self) -> float:
        return compute_grad_norm(self.optimizer)

    def train_batch(self, batch: Dict[str, torch.Tensor]) -> TrainBatchStats:
        return execute_train_batch(self, batch)

    def has_pending_gradients(self) -> bool:
        return has_pending_gradients(self)

    def flush_pending_gradients(self) -> Optional[float]:
        return flush_pending_gradients(self)

    def calibrate_ood(self, loader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float | str]:
        self._ood_calibration_loader = loader
        return calibrate_trainer_ood(self, loader)

    def configure_classifier_rebalance_stage(self, *, log_priors: Optional[torch.Tensor] = None) -> None:
        if self.adapter_model is None or self.classifier is None or self.fusion is None:
            raise RuntimeError("Classifier rebalance requires an initialized engine.")
        self._active_training_stage = "classifier_rebalance"
        for module in (self.adapter_model, self.fusion):
            for parameter in module.parameters():
                parameter.requires_grad = False
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True
        self._classifier_rebalance_log_priors = None if log_priors is None else log_priors.detach().clone()
        self._trainable_params_cache = None
        self.optimizer = None
        self.scheduler = None
        self._accumulation_counter = 0

    def set_stage_optimizer_override(self, *, learning_rate: float, weight_decay: float) -> None:
        self._optimizer_override = {
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
        }

    def setup_stage_optimizer(self) -> None:
        if self.classifier is None:
            raise RuntimeError("Classifier is not initialized.")
        override = dict(self._optimizer_override or {})
        lr = float(override.get("learning_rate", self.config.learning_rate))
        weight_decay = float(override.get("weight_decay", self.config.weight_decay))
        params = [parameter for parameter in self.classifier.parameters() if parameter.requires_grad]
        self.optimizer = build_adamw_optimizer(
            params,
            lr=lr,
            weight_decay=weight_decay,
            device=self.device,
        )
        self.scheduler = None
        self.scaler = build_grad_scaler(self.device, self.config.mixed_precision)
        self.optimizer.zero_grad(set_to_none=True)
        self._last_grad_norm = 0.0

    def predict_with_ood(self, images: torch.Tensor) -> Dict[str, Any]:
        return predict_with_ood_result(self, images)

    def predict_with_ood_batch(self, images: torch.Tensor) -> list[Dict[str, Any]]:
        return predict_with_ood_results(self, images)

    def _metadata_payload(self) -> Dict[str, Any]:
        return build_trainer_metadata_payload(self)

    def snapshot_training_state(self) -> TrainingCheckpointPayload:
        return snapshot_trainer_training_state(self, np_module=np)

    def restore_training_state(self, payload: TrainingCheckpointPayload | Dict[str, Any]) -> TrainingCheckpointPayload:
        return restore_trainer_training_state(self, payload, np_module=np)

    def save_adapter(self, output_dir: str) -> Path:
        self._ensure_ood_calibrated(operation="save_adapter()")
        return save_trainer_adapter(self, output_dir)

    def load_adapter(self, adapter_dir: str) -> Dict[str, Any]:
        return load_trainer_adapter(self, adapter_dir, peft_model_cls=PeftModel)
