"""Runtime helpers for AMP, optimizer/scheduler setup, and session defaults."""

from __future__ import annotations

import inspect
import logging
import math
import random
import time
from contextlib import nullcontext
from typing import Any, Iterable, Optional

import torch

from src.training.types import TrainBatchStats

logger = logging.getLogger(__name__)


def configure_runtime_reproducibility(config: Any, *, np_module: Any = None) -> None:
    seed = int(getattr(config, "seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if np_module is not None:
        np_module.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic = bool(getattr(config, "deterministic", False))
    try:
        torch.use_deterministic_algorithms(deterministic)
    except Exception as exc:
        if deterministic:
            raise RuntimeError(
                "Deterministic training was requested, but torch.use_deterministic_algorithms(...) failed."
            ) from exc
        logger.warning(
            "Failed to configure deterministic algorithms; continuing with best-effort runtime settings: %s",
            exc,
        )
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def resolve_runtime_device(requested_device: Any) -> torch.device:
    requested = str(requested_device or "cpu").strip() or "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{requested}' but CUDA is not available.")
    return torch.device(requested)


def resolve_amp_dtype(device: torch.device, mixed_precision: str) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    mode = str(mixed_precision).lower()
    if mode == "off":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device, mixed_precision: str) -> Any:
    dtype = resolve_amp_dtype(device, mixed_precision)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def amp_scaler_enabled(device: torch.device, mixed_precision: str) -> bool:
    return resolve_amp_dtype(device, mixed_precision) == torch.float16 and device.type == "cuda"


def build_grad_scaler(device: torch.device, mixed_precision: str):
    enabled = amp_scaler_enabled(device, mixed_precision)
    amp_namespace = getattr(torch, "amp", None)
    grad_scaler_cls = getattr(amp_namespace, "GradScaler", None)
    if grad_scaler_cls is not None:
        try:
            return grad_scaler_cls(device.type, enabled=enabled)
        except TypeError:
            try:
                return grad_scaler_cls(enabled=enabled)
            except TypeError:
                pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def resolve_session_num_epochs(config: Any, explicit_num_epochs: Optional[int], *, default: int = 1) -> int:
    if explicit_num_epochs is not None:
        return int(max(1, explicit_num_epochs))
    configured = getattr(config, "num_epochs", default)
    try:
        return int(max(1, int(configured)))
    except (TypeError, ValueError, OverflowError):
        return int(max(1, default))


def plan_optimizer_steps(config: Any, *, total_batches: int, num_epochs: Optional[int] = None) -> tuple[int, int]:
    """Return (epochs, scheduler steps) for a run: optimizer steps, or epochs when the scheduler steps per epoch."""
    epochs = int(max(1, num_epochs if num_epochs is not None else config.num_epochs))
    if config.scheduler_step_on == "epoch":
        return epochs, epochs
    optimizer_steps = max(
        1,
        epochs * math.ceil(max(1, int(total_batches)) / max(1, int(config.grad_accumulation_steps))),
    )
    return epochs, int(optimizer_steps)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, config: Any, *, total_units: int) -> Any:
    """Build the configured LR scheduler over `total_units` steps; None for unknown or "none" names."""
    total_units = max(1, int(total_units))
    if config.scheduler_name == "linear":
        warmup_steps = int(max(0, round(total_units * float(config.scheduler_warmup_ratio))))
        min_lr_scale = (
            float(config.scheduler_min_lr) / float(config.learning_rate)
            if config.learning_rate > 0
            else 0.0
        )
        min_lr_scale = max(0.0, min(1.0, min_lr_scale))

        def _lr_lambda(step_idx: int) -> float:
            step = int(max(0, step_idx))
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            remaining = max(1, total_units - warmup_steps)
            progress = float(step - warmup_steps) / float(remaining)
            progress = max(0.0, min(1.0, progress))
            return max(min_lr_scale, 1.0 - progress * (1.0 - min_lr_scale))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    if config.scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_units,
            eta_min=float(config.scheduler_min_lr),
        )
    return None


def build_idx_to_class(class_to_idx: dict[str, int]) -> dict[int, str]:
    return {idx: name for name, idx in class_to_idx.items()}


def trainable_parameters(*modules: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Parameters with requires_grad=True, in module order."""
    return [param for module in modules for param in module.parameters() if param.requires_grad]


def _optimizer_supports_kwarg(optimizer_cls: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(optimizer_cls)
    except (TypeError, ValueError):
        return False

    if keyword in signature.parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def build_adamw_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    resolved_parameters = list(parameters)
    base_kwargs = {
        "lr": float(lr),
        "weight_decay": float(weight_decay),
    }
    candidate_kwargs = []
    if device.type == "cuda":
        if _optimizer_supports_kwarg(torch.optim.AdamW, "fused"):
            candidate_kwargs.append({**base_kwargs, "fused": True})
        if _optimizer_supports_kwarg(torch.optim.AdamW, "foreach"):
            candidate_kwargs.append({**base_kwargs, "foreach": True})
    candidate_kwargs.append(base_kwargs)

    last_error: Exception | None = None
    for kwargs in candidate_kwargs:
        try:
            # Some torch stubs trigger false-positive arg-type errors for AdamW kwargs.
            # Call via an Any-typed reference to avoid mypy checking the call signature.
            optimizer_ctor: Any = torch.optim.AdamW
            return optimizer_ctor(resolved_parameters, **kwargs)
        except (TypeError, RuntimeError, ValueError) as exc:
            last_error = exc
            if kwargs == base_kwargs:
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to construct AdamW optimizer.")


def compute_grad_norm(optimizer: Optional[torch.optim.Optimizer]) -> float:
    if optimizer is None:
        return 0.0
    total_norm_sq = 0.0
    has_grad = False
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            if param is None or param.grad is None:
                continue
            grad = param.grad.detach()
            if grad.is_sparse:
                grad = grad.coalesce().values()
            grad_norm = float(torch.norm(grad, p=2).item())
            total_norm_sq += grad_norm * grad_norm
            has_grad = True
    if not has_grad:
        return 0.0
    return float(total_norm_sq ** 0.5)


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_grad_norm: float) -> None:
    """Clip the global gradient norm in place; a non-positive max_grad_norm disables clipping."""
    if max_grad_norm <= 0.0:
        return
    params = [param for param in parameters if param is not None]
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(max_grad_norm))


def build_train_batch_stats(
    *,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: Any,
    loss: float,
    grad_norm: float,
    step_started_at: float,
    accumulation_counter: int,
    accumulation_steps: int,
    optimizer_steps: int,
    optimizer_step_applied: bool,
    ber_ce_loss: float | None = None,
    ber_old_loss: float | None = None,
    ber_new_loss: float | None = None,
) -> TrainBatchStats:
    step_time_sec = float(max(1e-9, time.perf_counter() - step_started_at))
    batch_size = int(batch.get("labels", torch.empty(0)).shape[0]) if isinstance(batch, dict) else 0
    if batch_size <= 0 and isinstance(batch, dict) and "images" in batch:
        batch_size = int(batch["images"].shape[0])
    samples_per_sec = float(batch_size / step_time_sec) if batch_size > 0 else 0.0
    lr_value = float(optimizer.param_groups[0].get("lr", config.learning_rate))
    return TrainBatchStats(
        loss=float(loss),
        lr=lr_value,
        grad_norm=float(grad_norm),
        step_time_sec=step_time_sec,
        samples_per_sec=samples_per_sec,
        batch_size=int(batch_size),
        accumulation_step=int(accumulation_counter if accumulation_counter > 0 else accumulation_steps),
        optimizer_steps=int(optimizer_steps),
        optimizer_step_applied=bool(optimizer_step_applied),
        ber_ce_loss=ber_ce_loss,
        ber_old_loss=ber_old_loss,
        ber_new_loss=ber_new_loss,
    )
