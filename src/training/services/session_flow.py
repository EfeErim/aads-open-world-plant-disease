"""Session-only metric, checkpoint, and early-stop helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch

from src.training.types import TrainingHistory, ValidationReport

EmitFn = Callable[[str, Dict[str, Any]], None]

_MIN_MODE_METRICS = {"val_loss", "generalization_gap", "epoch_loss", "train_loss"}


def batch_advisory(loss: float, grad_norm: float, loss_window: List[float]) -> Dict[str, str]:
    advisory = ""
    severity = "info"
    if not torch.isfinite(torch.tensor(loss)).item():
        advisory = "Loss became non-finite (NaN/Inf). Consider stopping this run."
        severity = "critical"
    elif grad_norm > 1000.0:
        advisory = f"Gradient norm is very high ({grad_norm:.2f}). Training may be unstable."
        severity = "warning"
    elif len(loss_window) >= 5:
        baseline = sum(loss_window[:-1]) / max(1, len(loss_window) - 1)
        if baseline > 0 and loss > (baseline * 1.6):
            advisory = "Batch loss spiked above recent trend. Monitor closely."
            severity = "warning"
    return {"advisory": advisory, "severity": severity}


def decorate_validation(report: ValidationReport, epoch_loss: float) -> ValidationReport:
    generalization_gap = float(report.val_loss - epoch_loss)
    advisory = ""
    severity = "info"
    if generalization_gap > 0.5:
        advisory = (
            f"Validation loss is notably above training loss (gap={generalization_gap:.3f}). "
            "Potential overfitting."
        )
        severity = "warning"
    return ValidationReport(
        val_loss=report.val_loss,
        val_accuracy=report.val_accuracy,
        macro_precision=report.macro_precision,
        macro_recall=report.macro_recall,
        macro_f1=report.macro_f1,
        weighted_f1=report.weighted_f1,
        balanced_accuracy=report.balanced_accuracy,
        per_class_accuracy=dict(report.per_class_accuracy),
        per_class_support=dict(report.per_class_support),
        worst_classes=list(report.worst_classes),
        generalization_gap=generalization_gap,
        epoch_advisory=advisory,
        epoch_severity=severity,
    )


def metric_mode(metric_name: str) -> str:
    return "min" if str(metric_name) in _MIN_MODE_METRICS else "max"


def metric_value(metric_name: str, *, epoch_loss: float, report: Optional[ValidationReport]) -> Optional[float]:
    if metric_name in {"epoch_loss", "train_loss"}:
        return float(epoch_loss)
    if report is None:
        return None
    if hasattr(report, metric_name):
        value = getattr(report, metric_name)
        if value is None:
            return None
        return float(value)
    return None


def metric_improved(*, current: float, best: Optional[float], mode: str, min_delta: float = 0.0) -> bool:
    if best is None:
        return True
    if mode == "min":
        return float(current) < float(best) - float(min_delta)
    return float(current) > float(best) + float(min_delta)


def build_checkpoint_request_payload(
    *,
    reason: str,
    payload: Dict[str, Any],
    global_step: int,
    elapsed_sec: float,
    history: TrainingHistory,
    best_metric_state: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "reason": str(reason),
        "mark_best": bool(payload.get("mark_best", False)),
        "val_loss": payload.get("val_loss"),
        "epoch": int(payload.get("epoch_done", payload.get("epoch", 0))),
        "batch": int(payload.get("batch", 0)),
        "global_step": int(payload.get("global_step", global_step)),
        "elapsed_sec": float(payload.get("elapsed_sec", elapsed_sec)),
        "history_snapshot": history.to_dict(),
        "best_metric_state": dict(best_metric_state),
    }


def update_best_metric_state(
    *,
    trainer: Any,
    history: TrainingHistory,
    best_metric_state: Dict[str, Any],
    epoch_idx: int,
    epoch_loss: float,
    validation_report: Optional[ValidationReport],
    emit_fn: EmitFn,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    metric_name = str(getattr(trainer.config, "evaluation_best_metric", "val_loss"))
    current_mode = metric_mode(metric_name)
    current_value = metric_value(metric_name, epoch_loss=epoch_loss, report=validation_report)
    previous_best = best_metric_state.get("best_metric_value")
    improved = False
    if current_value is not None:
        improved = metric_improved(
            current=float(current_value),
            best=None if previous_best is None else float(previous_best),
            mode=current_mode,
            min_delta=0.0,
        )

    next_state = dict(best_metric_state)
    if improved and current_value is not None:
        next_state = {
            "best_metric_name": metric_name,
            "best_metric_value": float(current_value),
            "best_epoch": int(epoch_idx + 1),
            "mode": current_mode,
            "patience_bad_epochs": 0,
        }
        history.best_metric_name = metric_name
        history.best_metric_value = float(current_value)
        history.best_epoch = int(epoch_idx + 1)
        if hasattr(trainer, "best_metric_state"):
            trainer.best_metric_state = dict(next_state)
        emit_fn(
            "best_metric_updated",
            {
                "epoch_done": int(epoch_idx + 1),
                "best_metric_name": metric_name,
                "best_metric_value": float(current_value),
                "mode": current_mode,
            },
        )

    return next_state, {
        "best_metric_name": metric_name,
        "best_metric_value": current_value,
        "best_metric_mode": current_mode,
        "mark_best": bool(improved),
    }


def should_early_stop(
    *,
    trainer: Any,
    best_metric_state: Dict[str, Any],
    epoch_loss: float,
    validation_report: Optional[ValidationReport],
) -> tuple[Dict[str, Any], bool]:
    if not bool(getattr(trainer.config, "early_stopping_enabled", False)):
        return dict(best_metric_state), False

    metric_name = str(getattr(trainer.config, "early_stopping_metric", "val_loss"))
    current_mode = str(getattr(trainer.config, "early_stopping_mode", metric_mode(metric_name)))
    patience = int(getattr(trainer.config, "early_stopping_patience", 3))
    min_delta = float(getattr(trainer.config, "early_stopping_min_delta", 0.0))

    current_value = metric_value(metric_name, epoch_loss=epoch_loss, report=validation_report)
    if current_value is None:
        return dict(best_metric_state), False

    next_state = dict(best_metric_state)
    current_bad_epochs = int(next_state.get("patience_bad_epochs", 0))
    best_for_metric = (
        float(next_state["best_metric_value"])
        if next_state.get("best_metric_name") == metric_name and next_state.get("best_metric_value") is not None
        else None
    )
    if metric_improved(current=float(current_value), best=best_for_metric, mode=current_mode, min_delta=min_delta):
        current_bad_epochs = 0
    else:
        current_bad_epochs += 1

    next_state["patience_bad_epochs"] = int(current_bad_epochs)
    next_state["early_stopping_metric"] = metric_name
    next_state["early_stopping_mode"] = current_mode
    if hasattr(trainer, "best_metric_state"):
        trainer.best_metric_state = dict(next_state)

    return next_state, current_bad_epochs > patience
