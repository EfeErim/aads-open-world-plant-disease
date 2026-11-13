#!/usr/bin/env python3
"""v6 continual SD-LoRA training surfaces."""

from __future__ import annotations

import copy
import inspect
import logging
import os
import time
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
from src.training.config import ContinualSDLoRAConfig
from src.training.services.ood_calibration import calibrate_trainer_ood
from src.training.services.ood_score_selection import (
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
from src.training.services.prediction_payloads import build_ood_prediction_payloads
from src.training.services.runtime import (
    autocast_context,
    build_adamw_optimizer,
    build_grad_scaler,
    build_idx_to_class,
    build_lr_scheduler,
    build_train_batch_stats,
    clip_gradients,
    compute_grad_norm,
    configure_runtime_reproducibility,
    plan_optimizer_steps,
    resolve_runtime_device,
    trainable_parameters,
)
from src.training.types import TrainBatchStats, TrainingCheckpointPayload

AUTO_MODEL_FACTORY: Any = None
try:
    from transformers import AutoModel as _loaded_auto_model

    AUTO_MODEL_FACTORY = _loaded_auto_model
except ImportError:  # pragma: no cover - test fallback
    pass

def _resolve_auto_model_factory(auto_model_factory: Any) -> Any:
    if auto_model_factory is not None:
        return auto_model_factory
    try:
        from transformers import AutoModel as late_auto_model
    except ImportError as exc:
        raise RuntimeError("transformers AutoModel is unavailable for continual trainer initialization.") from exc
    return late_auto_model


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
_META_TENSOR_COPY_ERROR = "Cannot copy out of meta tensor; no data!"


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

    def _trainable_parameters(self) -> List[torch.nn.Parameter]:
        if self._trainable_params_cache is None:
            if self.adapter_model is None or self.classifier is None or self.fusion is None:
                raise RuntimeError("initialize_engine() must be called before setup_optimizer().")
            self._trainable_params_cache = trainable_parameters(self.adapter_model, self.classifier, self.fusion)
        return list(self._trainable_params_cache)

    def _refresh_optimizer_after_model_change(self) -> None:
        """Rebuild the optimizer over the current parameters, keeping hyperparameters, moment state and LR position."""
        if self.optimizer is None:
            return

        previous_optimizer = self.optimizer
        previous_scheduler_state = self.scheduler.state_dict() if self.scheduler is not None else None
        rebuilt_optimizer = build_adamw_optimizer(
            self._trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            device=self.device,
        )
        if len(previous_optimizer.param_groups) == len(rebuilt_optimizer.param_groups):
            for previous_group, rebuilt_group in zip(previous_optimizer.param_groups, rebuilt_optimizer.param_groups):
                for key, value in previous_group.items():
                    if key != "params":
                        rebuilt_group[key] = value
        for group in rebuilt_optimizer.param_groups:
            for param in group.get("params", []):
                if param in previous_optimizer.state:
                    rebuilt_optimizer.state[param] = copy.deepcopy(previous_optimizer.state[param])

        self.optimizer = rebuilt_optimizer
        self.scheduler = None
        if previous_scheduler_state is not None:
            self._ensure_scheduler()
            if self.scheduler is not None:
                self.scheduler.load_state_dict(previous_scheduler_state)
        self.optimizer.zero_grad(set_to_none=True)
        self._last_grad_norm = 0.0

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
        self._planned_epochs, self._planned_scheduler_steps = plan_optimizer_steps(
            self.config, total_batches=total_batches, num_epochs=num_epochs
        )
        if self.optimizer is not None:
            self._ensure_scheduler()

    def _ensure_scheduler(self) -> None:
        if self.optimizer is None or self.scheduler is not None or self.config.scheduler_name == "none":
            return
        total_units = self._planned_scheduler_steps if self._planned_scheduler_steps > 0 else self._planned_epochs
        self.scheduler = build_lr_scheduler(self.optimizer, self.config, total_units=total_units)

    def _step_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    def initialize_engine(self, class_to_idx: Optional[Dict[str, int]] = None) -> None:
        """Load the frozen backbone, apply LoRA wrappers and create the trainable fusion and classifier heads."""
        if class_to_idx:
            self.class_to_idx = dict(class_to_idx)

        loaded_backbone = _resolve_auto_model_factory(AutoModel).from_pretrained(self.config.backbone_model_name)
        self.backbone = self._prepare_module_for_device(loaded_backbone, module_name="backbone")
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_size = int(getattr(getattr(self.backbone, "config", None), "hidden_size", self.config.fusion_output_dim))
        self.fusion = MultiScaleFeatureFusion(
            input_dim=hidden_size,
            output_dim=self.config.fusion_output_dim,
            num_scales=max(1, len(self.config.fusion_layers)),
            dropout=self.config.fusion_dropout,
            gating=self.config.fusion_gating,
        ).to(self.device)

        self.target_modules_resolved = self.resolve_target_modules(self.backbone)
        adapter_model = self._apply_lora(self.backbone, self.target_modules_resolved)
        self.adapter_model = self._prepare_module_for_device(adapter_model, module_name="adapter_model")
        self.classifier = nn.Linear(self.config.fusion_output_dim, max(1, len(self.class_to_idx))).to(self.device)

        if self.config.ber_enabled:
            self.ber_loss = BERLoss(
                lambda_old=self.config.ber_lambda_old,
                lambda_new=self.config.ber_lambda_new,
                num_old_classes=0,
                warmup_steps=int(getattr(self.config, "ber_warmup_steps", 50)),
            ).to(self.device)
        else:
            self.ber_loss = None

        self._is_initialized = True
        self.optimizer = None
        self.scheduler = None
        self.scaler = build_grad_scaler(self.device, self.config.mixed_precision)
        self.optimizer_steps = 0
        self._accumulation_counter = 0
        self._last_grad_norm = 0.0
        self._trainable_params_cache = None
        self._refresh_class_index_cache()
        logger.info(
            "Continual engine initialized: backbone=%s, targets=%s",
            self.config.backbone_model_name,
            len(self.target_modules_resolved),
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
                import platform
                import shutil

                if platform.system().lower().startswith("windows") and shutil.which("cl") is None:
                    logger.debug("MSVC cl compiler not found; skipping torch.compile to avoid inductor build errors")
                    compile_available = False
                else:
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
                    # torch.compile raises backend-specific error types; every failure falls back to eager.
                    except Exception as exc_default:  # noqa: BLE001  # pragma: no cover - runtime guard
                        logger.warning(
                            "torch.compile default backend failed for training_step: %s; attempting eager backend",
                            exc_default,
                        )
                        try:
                            unbound = type(self).training_step
                            compiled_eager = torch.compile(unbound, backend="eager")
                            setattr(self, "training_step", cast(Any, (lambda batch, _compiled=compiled_eager, _self=self: _compiled(_self, batch))))
                            logger.info("torch.compile: training_step compiled successfully (eager backend)")
                        except Exception as exc_eager:  # noqa: BLE001  # pragma: no cover - runtime guard
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
        """Expand the classifier while preserving learned weights for existing classes."""
        num_old = len(self.class_to_idx)
        for name in new_class_names:
            if name not in self.class_to_idx:
                self.class_to_idx[name] = len(self.class_to_idx)
        self._refresh_class_index_cache()

        if self.ber_loss is not None:
            self.ber_loss.num_old_classes = num_old

        if self.classifier is None:
            return dict(self.class_to_idx)

        old_classifier = self.classifier
        old_out = int(getattr(old_classifier, "out_features", 0))
        new_out = max(1, len(self.class_to_idx))
        if new_out == old_out:
            return dict(self.class_to_idx)

        replacement = nn.Linear(old_classifier.in_features, new_out).to(self.device)
        if old_out > 0:
            replacement.weight.data[:old_out] = old_classifier.weight.data[:old_out]
            if old_classifier.bias is not None and replacement.bias is not None:
                replacement.bias.data[:old_out] = old_classifier.bias.data[:old_out]
        self.classifier = replacement
        self._trainable_params_cache = None
        self._refresh_optimizer_after_model_change()
        return dict(self.class_to_idx)

    def setup_optimizer(self) -> None:
        if not self._is_initialized or self.adapter_model is None or self.classifier is None or self.fusion is None:
            raise RuntimeError("initialize_engine() must be called before setup_optimizer().")
        self.optimizer = build_adamw_optimizer(
            self._trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            device=self.device,
        )
        self.scaler = build_grad_scaler(self.device, self.config.mixed_precision)
        self._ensure_scheduler()
        self.optimizer.zero_grad(set_to_none=True)
        self._last_grad_norm = 0.0

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
        """Run one microbatch; step the optimizer every `grad_accumulation_steps` microbatches."""
        if self.optimizer is None:
            self.setup_optimizer()
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not configured. Call setup_optimizer().")
        self.set_train_mode()
        step_started_at = time.perf_counter()
        accumulation_steps = int(max(1, self.config.grad_accumulation_steps))
        if self._accumulation_counter == 0:
            self.optimizer.zero_grad(set_to_none=True)

        loss = self.training_step(batch)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss encountered.")

        raw_loss_value = float(loss.detach().item())
        scaled_loss = loss / float(accumulation_steps)
        if self.scaler.is_enabled():
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self._accumulation_counter += 1
        should_step = self._accumulation_counter >= accumulation_steps
        if should_step:
            grad_norm = self._apply_optimizer_step()
        else:
            grad_norm = self._reported_grad_norm(gradients_unscaled=not self.scaler.is_enabled())

        ber_components = getattr(self, "_last_ber_components", {}) if self.ber_loss is not None else {}
        return build_train_batch_stats(
            batch=batch,
            optimizer=self.optimizer,
            config=self.config,
            loss=raw_loss_value,
            grad_norm=grad_norm,
            step_started_at=step_started_at,
            accumulation_counter=self._accumulation_counter,
            accumulation_steps=accumulation_steps,
            optimizer_steps=self.optimizer_steps,
            optimizer_step_applied=bool(should_step),
            ber_ce_loss=ber_components.get("ce"),
            ber_old_loss=ber_components.get("ber_old"),
            ber_new_loss=ber_components.get("ber_new"),
        )

    def _apply_optimizer_step(self) -> float:
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not configured. Call setup_optimizer().")
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        if self.config.max_grad_norm > 0.0:
            clip_gradients(self._trainable_parameters(), float(self.config.max_grad_norm))
        grad_norm = self._compute_grad_norm()
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer_steps += 1
        if self.config.scheduler_step_on == "batch":
            self._step_scheduler()
        self.optimizer.zero_grad(set_to_none=True)
        self._accumulation_counter = 0
        self._last_grad_norm = float(grad_norm)
        return float(grad_norm)

    def has_pending_gradients(self) -> bool:
        return self._accumulation_counter > 0

    def flush_pending_gradients(self) -> Optional[float]:
        """Apply an optimizer step for a partial accumulation window, e.g. at the end of an epoch."""
        if self.optimizer is None or not self.has_pending_gradients():
            return None
        return self._apply_optimizer_step()

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
        results = self.predict_with_ood_batch(images)
        if not results:
            raise RuntimeError("Cannot predict an empty image batch.")
        return results[0]

    def predict_with_ood_batch(self, images: torch.Tensor) -> list[Dict[str, Any]]:
        if self.adapter_model is None or self.classifier is None or self.fusion is None:
            raise RuntimeError("Cannot predict before adapter, classifier, and fusion are initialized.")
        if images.ndim == 3:
            images = images.unsqueeze(0)
        self._ensure_ood_calibrated(operation="predict_with_ood()")
        self.set_eval_mode()
        with torch.inference_mode():
            features = self.encode(images.to(self.device, non_blocking=True))
            score_features = self.prepare_features_for_scoring(features)
            logits = self.classifier(score_features)
            confidence, indices = torch.softmax(logits, dim=1).max(dim=1)
            ood = self.ood_detector.score(features=score_features, logits=logits, predicted_labels=indices)
        if self._class_index_cache_stale():
            self._refresh_class_index_cache()
        return build_ood_prediction_payloads(
            confidence=confidence,
            indices=indices,
            ood=ood,
            score_features=score_features,
            logits=logits,
            ood_detector=self.ood_detector,
            idx_to_class=self._idx_to_class,
        )

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
