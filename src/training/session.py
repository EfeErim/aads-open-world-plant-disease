from __future__ import annotations

import time
from collections.abc import Sized
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, cast

import torch

from src.training.types import TrainBatchStats, TrainingHistory, TrainingProgressState, ValidationReport
from src.training.validation import evaluate_model

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

Observer = Callable[[Dict[str, Any]], None]
StopPolicy = Callable[[], bool]


class ContinualTrainingSession:
    """Own epoch/batch orchestration around a trainer batch API."""

    def __init__(
        self,
        trainer: Any,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        num_epochs: int,
        *,
        val_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
        observers: Optional[List[Observer]] = None,
        stop_policy: Optional[StopPolicy] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        run_id: str = "",
        checkpoint_every_n_steps: int = 0,
        checkpoint_on_exception: bool = False,
        validation_every_n_epochs: int = 1,
    ) -> None:
        self.trainer = trainer
        self.train_loader = train_loader
        self.num_epochs = int(max(1, num_epochs))
        self.val_loader = val_loader
        self.observers = list(observers or [])
        self.stop_policy = stop_policy
        self.run_id = str(run_id)
        self.checkpoint_every_n_steps = int(max(0, checkpoint_every_n_steps))
        self.checkpoint_on_exception = bool(checkpoint_on_exception)
        self.validation_every_n_epochs = int(max(1, validation_every_n_epochs))

        resume_payload = dict(resume_state or {})
        history_payload = resume_payload.get("history", resume_payload.get("history_snapshot", {}))
        progress_payload = resume_payload.get("progress_state", {})

        self.history = TrainingHistory.from_dict(
            history_payload if isinstance(history_payload, dict) else {}
        )
        self.progress_state = TrainingProgressState.from_dict(
            progress_payload if isinstance(progress_payload, dict) else {}
        )
        self.best_metric_state = dict(resume_payload.get("best_metric_state", {}))
        self.history.resume_start_epoch = int(self.progress_state.resume_start_epoch)
        self._elapsed_before_resume = float(self.progress_state.elapsed_sec)
        preferred_loader = self.val_loader if self.val_loader is not None else self.train_loader
        set_loader = getattr(self.trainer, "set_preferred_ood_calibration_loader", None)
        if callable(set_loader):
            set_loader(preferred_loader)
        self._best_model_state: Optional[Dict[str, Any]] = None

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        event = {
            "event_type": str(event_type),
            "run_id": self.run_id,
            "payload": dict(payload),
        }
        for observer in self.observers:
            observer(event)

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "schema_version": "v6_training_session",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "progress_state": self.progress_state.to_dict(),
            "history": self.history.to_dict(),
            "best_metric_state": dict(self.best_metric_state),
        }

    def _emit_checkpoint_request(self, *, reason: str, payload: Dict[str, Any]) -> None:
        checkpoint_payload = build_checkpoint_request_payload(
            reason=reason,
            payload=payload,
            global_step=self.progress_state.global_step,
            elapsed_sec=self.progress_state.elapsed_sec,
            history=self.history,
            best_metric_state=self.best_metric_state,
        )
        self._emit("checkpoint_requested", checkpoint_payload)

    def _capture_best_model_state(self) -> None:
        modules = {
            "adapter_model": getattr(self.trainer, "adapter_model", None),
            "classifier": getattr(self.trainer, "classifier", None),
            "fusion": getattr(self.trainer, "fusion", None),
        }
        snapshot: Dict[str, Any] = {}
        for name, module in modules.items():
            if module is None:
                continue
            snapshot[name] = {
                key: value.detach().cpu().clone()
                for key, value in module.state_dict().items()
            }
        self._best_model_state = snapshot or None

    def restore_best_model_state(self) -> bool:
        if not self._best_model_state:
            return False
        modules = {
            "adapter_model": getattr(self.trainer, "adapter_model", None),
            "classifier": getattr(self.trainer, "classifier", None),
            "fusion": getattr(self.trainer, "fusion", None),
        }
        for name, module in modules.items():
            state = self._best_model_state.get(name)
            if module is None or state is None:
                continue
            module.load_state_dict(state)
        return True

    @staticmethod
    def _seed_loader_for_epoch(loader: Any, epoch_idx: int) -> None:
        seed_base = getattr(loader, "_seed_base", None)
        epoch_seed = None if seed_base is None else int(seed_base) + (int(epoch_idx) * 1000)
        generator = getattr(loader, "generator", None)
        if epoch_seed is not None and generator is not None and hasattr(generator, "manual_seed"):
            generator.manual_seed(int(epoch_seed))

        sampler_seed_base = getattr(loader, "_sampler_seed_base", seed_base)
        sampler = getattr(loader, "sampler", None)
        sampler_generator = getattr(sampler, "generator", None)
        if (
            sampler_seed_base is not None
            and sampler_generator is not None
            and hasattr(sampler_generator, "manual_seed")
        ):
            sampler_generator.manual_seed(int(sampler_seed_base) + (int(epoch_idx) * 1000))

    def _resolve_resume_position(self, total_batches: int) -> tuple[int, int]:
        completed_epoch_1based = int(self.progress_state.epoch)
        completed_batch = int(self.progress_state.batch)
        resolved_total_batches = int(self.progress_state.total_batches or total_batches)
        if completed_epoch_1based <= 0:
            return 0, 0
        if completed_batch <= 0:
            return completed_epoch_1based, 0
        if resolved_total_batches > 0 and completed_batch >= resolved_total_batches:
            return completed_epoch_1based, 0
        return max(0, completed_epoch_1based - 1), completed_batch

    def _flush_pending_batch_state(self, stats: Optional[TrainBatchStats] = None) -> bool:
        has_pending = getattr(self.trainer, "has_pending_gradients", None)
        if not callable(has_pending) or not bool(has_pending()):
            return False
        flush_pending = getattr(self.trainer, "flush_pending_gradients", None)
        if not callable(flush_pending):
            return False
        grad_norm = flush_pending()
        if stats is not None:
            stats.optimizer_step_applied = True
            stats.optimizer_steps = int(getattr(self.trainer, "optimizer_steps", stats.optimizer_steps))
            if grad_norm is not None:
                stats.grad_norm = float(grad_norm)
            config = getattr(self.trainer, "config", None)
            stats.accumulation_step = int(getattr(config, "grad_accumulation_steps", stats.accumulation_step))
        return True

    def run(self) -> TrainingHistory:
        global_step = int(self.progress_state.global_step)
        total_batches = len(self.train_loader) if isinstance(self.train_loader, Sized) else 0
        start_epoch, resume_batch = self._resolve_resume_position(total_batches)
        self.history.resume_start_epoch = start_epoch
        batch_loss_window: List[float] = []
        train_started_at = time.perf_counter() - max(0.0, self._elapsed_before_resume)

        if hasattr(self.trainer, "configure_training_plan"):
            self.trainer.configure_training_plan(total_batches=max(1, total_batches), num_epochs=self.num_epochs)

        if global_step > 0 or start_epoch > 0:
            self._emit(
                "resume_loaded",
                {
                    "resume_epoch": int(start_epoch + 1) if start_epoch > 0 or global_step > 0 else 0,
                    "global_step": int(global_step),
                    "optimizer_steps": int(self.progress_state.optimizer_steps),
                },
            )

        stopped_early = False
        try:
            for epoch_idx in range(start_epoch, self.num_epochs):
                self._seed_loader_for_epoch(self.train_loader, epoch_idx)
                self.trainer.set_train_mode()
                losses: List[float] = []
                for batch_idx, batch in enumerate(self.train_loader):
                    if epoch_idx == start_epoch and batch_idx < resume_batch:
                        continue
                    stats = self.trainer.train_batch(batch)
                    is_last_batch = total_batches > 0 and batch_idx == (total_batches - 1)
                    if is_last_batch:
                        self._flush_pending_batch_state(stats)
                    losses.append(float(stats.loss))
                    batch_loss_window.append(float(stats.loss))
                    if len(batch_loss_window) > 8:
                        batch_loss_window = batch_loss_window[-8:]

                    global_step += 1
                    elapsed = float(time.perf_counter() - train_started_at)
                    total_batches_planned = max(1, self.num_epochs * max(1, total_batches))
                    processed_batches = max(1, (epoch_idx * max(1, total_batches)) + (batch_idx + 1))
                    eta_sec = float((elapsed / processed_batches) * max(0, total_batches_planned - processed_batches))
                    advisory = batch_advisory(float(stats.loss), float(stats.grad_norm), batch_loss_window)

                    self.progress_state = TrainingProgressState(
                        epoch=epoch_idx + 1,
                        batch=batch_idx + 1,
                        total_batches=int(total_batches),
                        global_step=int(global_step),
                        optimizer_steps=int(stats.optimizer_steps),
                        elapsed_sec=elapsed,
                        eta_sec=eta_sec,
                        resume_start_epoch=int(start_epoch),
                        stopped_early=False,
                    )
                    self.history.global_step = int(global_step)
                    self.history.optimizer_steps = int(stats.optimizer_steps)
                    payload = {
                        **self.progress_state.to_dict(),
                        **stats.to_dict(),
                        **advisory,
                        "epoch_progress": float((batch_idx + 1) / max(1, total_batches)),
                    }
                    self._emit("batch_end", payload)

                    if self.checkpoint_every_n_steps > 0 and (global_step % self.checkpoint_every_n_steps == 0):
                        self._emit_checkpoint_request(reason="batch_interval", payload=payload)

                    if self.stop_policy is not None and bool(self.stop_policy()):
                        stopped_early = True
                        self.progress_state.stopped_early = True
                        self._emit("stop_requested", self.progress_state.to_dict())
                        break

                if not losses:
                    break

                if not stopped_early:
                    self._flush_pending_batch_state()

                epoch_loss = float(sum(losses) / max(1, len(losses)))
                self.history.train_loss.append(epoch_loss)
                self.trainer.current_epoch = int(epoch_idx + 1)
                validation_report: Optional[ValidationReport] = None
                epoch_payload: Dict[str, Any] = {
                    "epoch_done": int(epoch_idx + 1),
                    "epoch_loss": epoch_loss,
                    "global_step": int(global_step),
                    "optimizer_steps": int(
                        getattr(self.trainer, "optimizer_steps", self.progress_state.optimizer_steps)
                    ),
                    "stopped_early": bool(stopped_early),
                    "elapsed_sec": float(time.perf_counter() - train_started_at),
                }

                should_validate = (
                    self.val_loader is not None
                    and (
                        int(epoch_idx + 1) == int(self.num_epochs)
                        or (int(epoch_idx + 1) % self.validation_every_n_epochs) == 0
                    )
                )
                epoch_payload["validation_every_n_epochs"] = int(self.validation_every_n_epochs)
                epoch_payload["validation_skipped"] = not should_validate
                if should_validate:
                    validation_report = evaluate_model(
                        self.trainer, cast(Iterable[Dict[str, torch.Tensor]], self.val_loader)
                    )
                    if validation_report is not None:
                        decorated = decorate_validation(validation_report, epoch_loss)
                        validation_report = decorated
                        self.history.append_validation(decorated)
                        epoch_payload.update(decorated.to_dict())
                        self._emit("validation_end", {"epoch_done": int(epoch_idx + 1), **decorated.to_dict()})

                self.best_metric_state, best_metric_payload = update_best_metric_state(
                    trainer=self.trainer,
                    history=self.history,
                    best_metric_state=self.best_metric_state,
                    epoch_idx=epoch_idx,
                    epoch_loss=epoch_loss,
                    validation_report=validation_report,
                    emit_fn=self._emit,
                )
                epoch_payload.update(best_metric_payload)
                if bool(best_metric_payload.get("mark_best", False)):
                    self._capture_best_model_state()

                if (
                    hasattr(self.trainer, "config")
                    and getattr(self.trainer.config, "scheduler_step_on", "batch") == "epoch"
                ):
                    step_scheduler = getattr(self.trainer, "_step_scheduler", None)
                    if callable(step_scheduler):
                        step_scheduler()

                epoch_payload["history_snapshot"] = self.history.to_dict()
                self.trainer.set_train_mode()
                self._emit("epoch_end", epoch_payload)
                self._emit_checkpoint_request(reason="epoch_end", payload=epoch_payload)

                self.best_metric_state, should_stop = should_early_stop(
                    trainer=self.trainer,
                    best_metric_state=self.best_metric_state,
                    epoch_loss=epoch_loss,
                    validation_report=validation_report,
                )
                if should_stop:
                    stopped_early = True
                    self.progress_state.stopped_early = True
                    self._emit(
                        "stop_requested",
                        {
                            **self.progress_state.to_dict(),
                            "reason": "early_stopping",
                            "best_metric_state": dict(self.best_metric_state),
                        },
                    )

                if stopped_early:
                    break
        except Exception as exc:
            aborted_payload = {
                **self.progress_state.to_dict(),
                "error": str(exc),
                "history_snapshot": self.history.to_dict(),
            }
            self._emit("training_aborted", aborted_payload)
            if self.checkpoint_on_exception:
                self._emit_checkpoint_request(reason="exception", payload=aborted_payload)
            raise

        self.history.stopped_early = bool(stopped_early)
        self.history.global_step = int(global_step)
        self.history.optimizer_steps = int(
            getattr(self.trainer, "optimizer_steps", self.progress_state.optimizer_steps)
        )
        self.progress_state.global_step = int(global_step)
        self.progress_state.optimizer_steps = int(
            getattr(self.trainer, "optimizer_steps", self.progress_state.optimizer_steps)
        )
        self.progress_state.resume_start_epoch = int(start_epoch)
        self.progress_state.stopped_early = bool(stopped_early)
        self._emit(
            "training_completed",
            {
                **self.progress_state.to_dict(),
                "history_snapshot": self.history.to_dict(),
                "best_metric_state": dict(self.best_metric_state),
            },
        )
        return self.history
