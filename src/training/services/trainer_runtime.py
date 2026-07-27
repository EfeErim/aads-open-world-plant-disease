"""Internal trainer orchestration helpers used by the public trainer facade."""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn

from src.training.ber_loss import BERLoss
from src.training.services.runtime import (
    build_adamw_optimizer,
    build_grad_scaler,
    build_train_batch_stats,
    clip_gradients,
    collect_trainable_parameters,
)

logger = logging.getLogger(__name__)


def _resolve_auto_model_factory(auto_model_factory: Any) -> Any:
    if auto_model_factory is not None:
        return auto_model_factory
    try:
        from transformers import AutoModel as late_auto_model
    except Exception as exc:
        raise RuntimeError("transformers AutoModel is unavailable for continual trainer initialization.") from exc
    return late_auto_model

def refresh_optimizer_after_model_change(trainer: Any) -> None:
    if trainer.optimizer is None:
        return

    previous_optimizer = trainer.optimizer
    previous_scheduler_state = trainer.scheduler.state_dict() if trainer.scheduler is not None else None

    rebuilt_optimizer = build_adamw_optimizer(
        collect_trainable_parameters(trainer),
        lr=trainer.config.learning_rate,
        weight_decay=trainer.config.weight_decay,
        device=trainer.device,
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

    trainer.optimizer = rebuilt_optimizer
    trainer.scheduler = None
    if previous_scheduler_state is not None:
        trainer._ensure_scheduler()
        if trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(previous_scheduler_state)
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._last_grad_norm = 0.0


def initialize_trainer_engine(
    trainer: Any,
    *,
    class_to_idx: Dict[str, int] | None,
    auto_model_factory: Any,
    fusion_cls: type[Any],
) -> None:
    """Load frozen backbone, apply LoRA wrappers, and initialize trainable heads."""
    if class_to_idx:
        trainer.class_to_idx = dict(class_to_idx)

    auto_model_factory = _resolve_auto_model_factory(auto_model_factory)
    loaded_backbone = auto_model_factory.from_pretrained(trainer.config.backbone_model_name)
    trainer.backbone = trainer._prepare_module_for_device(loaded_backbone, module_name="backbone")
    for param in trainer.backbone.parameters():
        param.requires_grad = False

    hidden_size = int(
        getattr(getattr(trainer.backbone, "config", None), "hidden_size", trainer.config.fusion_output_dim)
    )
    trainer.fusion = fusion_cls(
        input_dim=hidden_size,
        output_dim=trainer.config.fusion_output_dim,
        num_scales=max(1, len(trainer.config.fusion_layers)),
        dropout=trainer.config.fusion_dropout,
        gating=trainer.config.fusion_gating,
    ).to(trainer.device)

    trainer.target_modules_resolved = trainer.resolve_target_modules(trainer.backbone)
    adapter_model = trainer._apply_lora(trainer.backbone, trainer.target_modules_resolved)
    trainer.adapter_model = trainer._prepare_module_for_device(adapter_model, module_name="adapter_model")
    trainer.classifier = nn.Linear(trainer.config.fusion_output_dim, max(1, len(trainer.class_to_idx))).to(
        trainer.device
    )

    if trainer.config.ber_enabled:
        trainer.ber_loss = BERLoss(
            lambda_old=trainer.config.ber_lambda_old,
            lambda_new=trainer.config.ber_lambda_new,
            num_old_classes=0,
            warmup_steps=int(getattr(trainer.config, "ber_warmup_steps", 50)),
        ).to(trainer.device)
    else:
        trainer.ber_loss = None

    trainer._is_initialized = True
    trainer.optimizer = None
    trainer.scheduler = None
    trainer.scaler = build_grad_scaler(trainer.device, trainer.config.mixed_precision)
    trainer.optimizer_steps = 0
    trainer._accumulation_counter = 0
    trainer._last_grad_norm = 0.0
    trainer._trainable_params_cache = None
    trainer._refresh_class_index_cache()
    logger.info(
        "Continual engine initialized: backbone=%s, targets=%s",
        trainer.config.backbone_model_name,
        len(trainer.target_modules_resolved),
    )


def add_trainer_classes(trainer: Any, new_class_names: Iterable[str]) -> Dict[str, int]:
    """Expand the classifier while preserving learned weights for existing classes."""
    num_old = len(trainer.class_to_idx)
    for name in new_class_names:
        if name not in trainer.class_to_idx:
            trainer.class_to_idx[name] = len(trainer.class_to_idx)
    trainer._refresh_class_index_cache()

    if trainer.ber_loss is not None:
        trainer.ber_loss.num_old_classes = num_old

    if trainer.classifier is None:
        return dict(trainer.class_to_idx)

    old_classifier = trainer.classifier
    old_out = int(getattr(old_classifier, "out_features", 0))
    new_out = max(1, len(trainer.class_to_idx))
    if new_out == old_out:
        return dict(trainer.class_to_idx)

    replacement = nn.Linear(old_classifier.in_features, new_out).to(trainer.device)
    if old_out > 0:
        replacement.weight.data[:old_out] = old_classifier.weight.data[:old_out]
        if old_classifier.bias is not None and replacement.bias is not None:
            replacement.bias.data[:old_out] = old_classifier.bias.data[:old_out]
    trainer.classifier = replacement
    trainer._trainable_params_cache = None
    refresh_optimizer_after_model_change(trainer)
    return dict(trainer.class_to_idx)


def has_pending_gradients(trainer: Any) -> bool:
    return int(getattr(trainer, "_accumulation_counter", 0)) > 0


def _apply_optimizer_step(trainer: Any) -> float:
    if trainer.optimizer is None:
        raise RuntimeError("Optimizer is not configured. Call setup_optimizer().")
    if trainer.scaler.is_enabled():
        trainer.scaler.unscale_(trainer.optimizer)
    clip_gradients(trainer)
    grad_norm = trainer._compute_grad_norm()
    if trainer.scaler.is_enabled():
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
    else:
        trainer.optimizer.step()
    trainer.optimizer_steps += 1
    if trainer.config.scheduler_step_on == "batch":
        trainer._step_scheduler()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._accumulation_counter = 0
    trainer._last_grad_norm = float(grad_norm)
    return float(grad_norm)


def flush_pending_gradients(trainer: Any) -> float | None:
    if trainer.optimizer is None or not has_pending_gradients(trainer):
        return None
    return _apply_optimizer_step(trainer)


def execute_train_batch(trainer: Any, batch: Dict[str, torch.Tensor]):
    if trainer.optimizer is None:
        trainer.setup_optimizer()
    if trainer.optimizer is None:
        raise RuntimeError("Optimizer is not configured. Call setup_optimizer().")
    trainer.set_train_mode()
    step_started_at = time.perf_counter()
    accumulation_steps = int(max(1, trainer.config.grad_accumulation_steps))
    if trainer._accumulation_counter == 0:
        trainer.optimizer.zero_grad(set_to_none=True)

    loss = trainer.training_step(batch)
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite training loss encountered.")

    raw_loss_value = float(loss.detach().item())
    scaled_loss = loss / float(accumulation_steps)
    if trainer.scaler.is_enabled():
        trainer.scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    trainer._accumulation_counter += 1
    should_step = trainer._accumulation_counter >= accumulation_steps
    if should_step:
        grad_norm = _apply_optimizer_step(trainer)
    else:
        grad_norm = trainer._reported_grad_norm(gradients_unscaled=not trainer.scaler.is_enabled())

    ber_components = getattr(trainer, "_last_ber_components", {}) if trainer.ber_loss is not None else {}
    return build_train_batch_stats(
        batch=batch,
        optimizer=trainer.optimizer,
        config=trainer.config,
        loss=raw_loss_value,
        grad_norm=grad_norm,
        step_started_at=step_started_at,
        accumulation_counter=trainer._accumulation_counter,
        accumulation_steps=accumulation_steps,
        optimizer_steps=trainer.optimizer_steps,
        optimizer_step_applied=bool(should_step),
        ber_ce_loss=ber_components.get("ce"),
        ber_old_loss=ber_components.get("ber_old"),
        ber_new_loss=ber_components.get("ber_new"),
    )


def _tensor_item(values: Any, index: int, default: float = 0.0) -> float:
    if torch.is_tensor(values) and values.numel() > index:
        return float(values[index].item())
    return float(default)


def _bool_tensor_item(values: Any, index: int, default: bool = False) -> bool:
    if torch.is_tensor(values) and values.numel() > index:
        return bool(values[index].item())
    return bool(default)


def _int_tensor_item(values: Any, index: int, default: int = 0) -> int:
    if torch.is_tensor(values) and values.numel() > index:
        return int(values[index].item())
    return int(default)


def _score_dict_for_index(values: Any, index: int) -> Dict[str, float]:
    return {
        name: _tensor_item(tensor, index)
        for name, tensor in dict(values or {}).items()
        if torch.is_tensor(tensor) and tensor.numel() > index
    }


def predict_with_ood_results(trainer: Any, images: torch.Tensor) -> list[Dict[str, Any]]:
    if trainer.adapter_model is None or trainer.classifier is None or trainer.fusion is None:
        raise RuntimeError("Cannot predict before adapter, classifier, and fusion are initialized.")
    if images.ndim == 3:
        images = images.unsqueeze(0)
    trainer._ensure_ood_calibrated(operation="predict_with_ood()")
    trainer.set_eval_mode()
    with torch.inference_mode():
        features = trainer.encode(images.to(trainer.device, non_blocking=True))
        score_features = trainer.prepare_features_for_scoring(features)
        logits = trainer.classifier(score_features)
        probs = torch.softmax(logits, dim=1)
        confidence, indices = probs.max(dim=1)
        ood = trainer.ood_detector.score(features=score_features, logits=logits, predicted_labels=indices)

    if trainer._class_index_cache_stale():
        trainer._refresh_class_index_cache()

    primary_score_method = str(
        ood.get("primary_score_method", getattr(trainer.ood_detector, "primary_score_method", "ensemble")) or "ensemble"
    )
    results: list[Dict[str, Any]] = []
    for index in range(int(indices.numel())):
        predicted_idx = int(indices[index].item())
        candidate_scores = _score_dict_for_index(ood.get("candidate_scores"), index)
        candidate_thresholds = _score_dict_for_index(ood.get("candidate_thresholds"), index)
        primary_score_tensor = ood.get("primary_score")
        if torch.is_tensor(primary_score_tensor):
            primary_score = _tensor_item(primary_score_tensor, index)
        elif primary_score_method in candidate_scores:
            primary_score = float(candidate_scores[primary_score_method])
        else:
            primary_score = float(candidate_scores.get("ensemble", 0.0))
        decision_threshold_tensor = ood.get("decision_threshold")
        if torch.is_tensor(decision_threshold_tensor):
            decision_threshold = _tensor_item(decision_threshold_tensor, index)
        elif primary_score_method in candidate_thresholds:
            decision_threshold = float(candidate_thresholds[primary_score_method])
        else:
            decision_threshold = float(candidate_thresholds.get("ensemble", 0.0))
        ood_analysis: Dict[str, Any] = {
            "score_method": primary_score_method,
            "primary_score": primary_score,
            "decision_threshold": decision_threshold,
            "is_ood": _bool_tensor_item(ood.get("is_ood"), index),
            "candidate_scores": candidate_scores,
            "candidate_thresholds": candidate_thresholds,
            "calibration_version": _int_tensor_item(ood.get("calibration_version"), index),
        }
        for optional_key in ("mahalanobis_z", "energy_z", "knn_distance"):
            if torch.is_tensor(ood.get(optional_key)) and ood[optional_key].numel() > index:
                ood_analysis[optional_key] = _tensor_item(ood[optional_key], index)

        if trainer.ood_detector.radial_beta is not None:
            ood_analysis["radial_beta"] = float(trainer.ood_detector.radial_beta)

        if trainer.ood_detector.sure_enabled and "sure_semantic_score" in ood:
            ood_analysis["sure_semantic_score"] = _tensor_item(ood.get("sure_semantic_score"), index)
            ood_analysis["sure_confidence_score"] = _tensor_item(ood.get("sure_confidence_score"), index)
            ood_analysis["sure_semantic_ood"] = _bool_tensor_item(ood.get("sure_semantic_ood"), index)
            ood_analysis["sure_confidence_reject"] = _bool_tensor_item(ood.get("sure_confidence_reject"), index)

        if trainer.ood_detector.conformal_enabled:
            with torch.inference_mode():
                conformal_set = trainer.ood_detector.build_conformal_set(
                    score_features[index],
                    logits[index],
                    trainer._idx_to_class,
                )
            ood_analysis["conformal_set"] = conformal_set
            ood_analysis["conformal_set_size"] = len(conformal_set)
            ood_analysis["conformal_coverage"] = 1.0 - trainer.ood_detector.conformal_alpha

        results.append(
            {
                "status": "success",
                "disease": {
                    "class_index": predicted_idx,
                    "name": trainer._idx_to_class.get(predicted_idx, str(predicted_idx)),
                    "confidence": _tensor_item(confidence, index),
                },
                "ood_analysis": ood_analysis,
            }
        )
    return results


def predict_with_ood_result(trainer: Any, images: torch.Tensor) -> Dict[str, Any]:
    results = predict_with_ood_results(trainer, images)
    if not results:
        raise RuntimeError("Cannot predict an empty image batch.")
    return results[0]
