#!/usr/bin/env python3
"""Notebook training sessions, checkpoints, artifacts and completion handling."""

from __future__ import annotations

import gc
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib

from src.guided_artifacts import refresh_training_guided_artifacts
from src.shared.json_utils import deep_merge, read_json, write_json
from src.shared.string_utils import slug_label_component
from src.training.services.reporting import (
    persist_behavioral_acceptance_artifact as persist_behavioral_acceptance_artifact_core,
)
from src.training.services.reporting import (
    persist_behavioral_dev_report_artifact as persist_behavioral_dev_report_artifact_core,
)
from src.training.services.reporting import (
    persist_production_readiness_artifact as persist_production_readiness_artifact_core,
)
from src.training.services.reporting import (
    persist_training_history_artifacts as persist_training_history_artifacts_core,
)
from src.training.services.reporting import (
    persist_training_run_context_artifact as persist_training_run_context_artifact_core,
)
from src.training.services.reporting import (
    persist_validation_artifacts as persist_validation_artifacts_core,
)
from src.training.services.traceability import (
    build_experiment_manifest,
    build_optimization_record,
    load_authoritative_artifacts_from_root,
    persist_traceability_artifacts,
)

matplotlib.use("Agg")


_EXPECTED_REPO_EXPORTS = ("outputs", "telemetry", "checkpoint_state")


def build_notebook_run_id(crop_name: str, part_name: str = "unspecified", *, now: Optional[datetime] = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    crop = slug_label_component(crop_name, default="crop")
    part = slug_label_component(part_name, default="unspecified")
    return f"{crop}_{part}_{stamp}"


def build_notebook_run_dir(root: Path, crop_name: str, part_name: str, run_id: str) -> Path:
    crop = slug_label_component(crop_name, default="crop")
    part = slug_label_component(part_name, default="unspecified")
    return Path(root) / "runs" / crop / part / str(run_id)


def merge_training_summary_fields(
    *,
    root: Path,
    payload: Dict[str, Any],
    telemetry: Any = None,
    extra_entries: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    summary_path = _artifact_dir(root, "training") / "summary.json"
    current = read_json(summary_path, default={}, expect_type=dict)
    merged = deep_merge(dict(current), dict(payload or {}))
    write_json(summary_path, merged, ensure_ascii=False, sort_keys=False)
    if telemetry is not None and hasattr(telemetry, "copy_artifact_file"):
        telemetry.copy_artifact_file(summary_path, "training/summary.json")
    _refresh_traceability_records(root=_artifact_dir(root), summary_payload=merged, telemetry=telemetry)
    refresh_training_guided_artifacts(
        _artifact_dir(root),
        telemetry=telemetry,
        overview_updates=merged,
        extra_entries=list(extra_entries or []),
        generated_by="scripts.colab_notebook_helpers",
    )
    return merged


def _resolve_traceability_surface(summary_payload: Dict[str, Any]) -> str:
    notebook_surface = str(summary_payload.get("notebook_surface", "") or "")
    if notebook_surface.endswith(("2_train_continual_sd_lora_adapter.ipynb", "2_interactive_adapter_training.ipynb")):
        return "notebook_2"
    return str(summary_payload.get("surface", "") or "workflow")


def _refresh_traceability_records(*, root: Path, summary_payload: Dict[str, Any], telemetry: Any = None) -> None:
    run_context_path = root / "training" / "run_context.json"
    if not run_context_path.exists():
        return
    run_context = read_json(run_context_path, default={}, expect_type=dict)
    production_readiness = read_json(root / "production_readiness.json", default={}, expect_type=dict)
    classification_split = str(
        production_readiness.get("classification_evidence", {}).get("split_name", "")
        if isinstance(production_readiness.get("classification_evidence"), dict)
        else ""
    )
    authoritative_artifacts = load_authoritative_artifacts_from_root(
        root,
        classification_split=classification_split,
    )
    resolved_surface = _resolve_traceability_surface(summary_payload)
    experiment_manifest = build_experiment_manifest(
        summary_payload=summary_payload,
        run_context_payload=run_context,
        artifact_root=root,
        explicit_surface=resolved_surface,
        created_at=str(summary_payload.get("created_at", "") or run_context.get("created_at", "") or ""),
        record_quality="canonical",
    )
    optimization_record = build_optimization_record(
        summary_payload=summary_payload,
        run_context_payload=run_context,
        production_readiness_payload=production_readiness,
        authoritative_artifacts=authoritative_artifacts,
        artifact_root=root,
        explicit_surface=resolved_surface,
        created_at=str(summary_payload.get("created_at", "") or run_context.get("created_at", "") or ""),
        record_quality="canonical",
    )
    notebook_parameters = (
        dict(summary_payload.get("notebook_parameters", {}))
        if isinstance(summary_payload.get("notebook_parameters"), dict)
        else {}
    )
    persist_traceability_artifacts(
        artifact_root=root,
        experiment_manifest=experiment_manifest,
        optimization_record=optimization_record,
        telemetry=telemetry,
        enable_bayesian_proposals=bool(notebook_parameters.get("enable_bayesian_optimization", True)),
    )


def _artifact_dir(root: Path, *parts: str) -> Path:
    target = root / "outputs" / "colab_notebook_training" / "artifacts"
    for part in parts:
        target /= part
    target.mkdir(parents=True, exist_ok=True)
    return target


def notebook_artifact_root(root: Path) -> Path:
    return _artifact_dir(root)


def ensure_notebook_checkpoint_manager(
    checkpoint_manager: Any = None,
    *,
    run_id: Optional[str] = None,
    drive_root: Optional[str | Path] = None,
    retention: int = 3,
    now_fn: Optional[Callable[[], datetime]] = None,
) -> Any:
    if checkpoint_manager is not None:
        return checkpoint_manager

    from src.training.colab_checkpointing import TrainingCheckpointManager

    resolved_run_id = str(run_id or (now_fn or datetime.now)().strftime("%Y%m%d_%H%M%S_%f"))
    resolved_drive_root = Path(drive_root or os.environ.get("AADS_DRIVE_LOG_ROOT", "/content/drive/MyDrive/aads_ulora"))
    return TrainingCheckpointManager(resolved_drive_root / "telemetry" / resolved_run_id, retention=retention)


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds or 0.0))))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def _path_exists(path_like: Optional[str | Path]) -> bool:
    return bool(path_like and Path(path_like).expanduser().exists())


def _call_if_present(target: Any, method_name: str, *args, **kwargs) -> None:
    method = getattr(target, method_name, None)
    if not callable(method):
        return
    try:
        method(*args, **kwargs)
    except Exception:
        import logging

        logging.exception("Unhandled exception")
        raise


class NotebookTrainingStatusPrinter:
    """Emit low-frequency, notebook-friendly training status lines."""

    def __init__(
        self,
        *,
        total_epochs: int,
        batch_interval: int = 50,
        min_interval_sec: float = 15.0,
        print_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.total_epochs = int(max(1, total_epochs))
        self.batch_interval = int(max(0, batch_interval))
        self.min_interval_sec = float(max(1.0, min_interval_sec))
        self.print_fn = print if print_fn is None else print_fn
        self._last_batch_emit_elapsed = -1.0

    def _emit(self, message: str) -> None:
        self.print_fn(str(message))

    @staticmethod
    def _append_advisory(
        parts: List[str],
        payload: Dict[str, Any],
        *,
        message_key: str,
        severity_key: str,
    ) -> None:
        advisory = str(payload.get(message_key, "")).strip()
        severity = str(payload.get(severity_key, "")).strip().lower()
        if advisory and severity in {"warning", "critical"}:
            parts.append(f"{severity}={advisory}")

    def _metric_fragment(self, payload: Dict[str, Any], key: str, label: str) -> Optional[str]:
        value = payload.get(key)
        if value is None:
            return None
        return f"{label}={float(value):.4f}"

    def handle(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        event_name = str(event_type or "")
        event = dict(payload or {})
        handler = {
            "batch_end": self._handle_batch_end,
            "validation_end": self._handle_validation_end,
            "best_metric_updated": self._handle_best_metric,
            "stop_requested": self._handle_stop_requested,
        }.get(event_name)
        if handler is not None:
            handler(event)

    def _handle_batch_end(self, payload: Dict[str, Any]) -> None:
        batch = int(payload.get("batch", 0))
        if batch <= 0:
            return
        total_batches = int(payload.get("total_batches", 0))
        elapsed_sec = float(payload.get("elapsed_sec", 0.0))
        emit_due_to_interval = self.batch_interval > 0 and (batch % self.batch_interval == 0)
        emit_due_to_time = (
            self._last_batch_emit_elapsed < 0 or (elapsed_sec - self._last_batch_emit_elapsed) >= self.min_interval_sec
        )
        emit_due_to_terminal_batch = total_batches > 0 and batch >= total_batches
        if not (batch == 1 or emit_due_to_interval or emit_due_to_time or emit_due_to_terminal_batch):
            return

        self._last_batch_emit_elapsed = elapsed_sec
        epoch = int(payload.get("epoch", 0))
        parts = [
            f"[LIVE] {epoch}/{self.total_epochs}",
            f"batch={batch}/{total_batches or '?'}",
            f"loss={float(payload.get('loss', 0.0)):.4f}",
            f"lr={float(payload.get('lr', 0.0)):.6f}",
            f"throughput={float(payload.get('samples_per_sec', 0.0)):.1f}/s",
            f"elapsed={_format_duration(elapsed_sec)}",
            f"eta={_format_duration(float(payload.get('eta_sec', 0.0)))}",
        ]
        self._append_advisory(parts, payload, message_key="advisory", severity_key="severity")
        self._emit(" ".join(parts))

    def _handle_validation_end(self, payload: Dict[str, Any]) -> None:
        epoch_done = int(payload.get("epoch_done", 0))
        parts = [f"[VALID] {epoch_done}/{self.total_epochs}"]
        for key, label in (
            ("val_loss", "val_loss"),
            ("val_accuracy", "val_acc"),
            ("macro_f1", "macro_f1"),
            ("balanced_accuracy", "bal_acc"),
            ("generalization_gap", "gap"),
        ):
            metric = self._metric_fragment(payload, key, label)
            if metric is not None:
                parts.append(metric)
        self._append_advisory(parts, payload, message_key="epoch_advisory", severity_key="epoch_severity")
        self._emit(" ".join(parts))

    def _handle_best_metric(self, payload: Dict[str, Any]) -> None:
        metric_name = str(payload.get("best_metric_name", "metric"))
        metric_value = payload.get("best_metric_value")
        if metric_value is None:
            return
        epoch_done = int(payload.get("epoch_done", 0))
        self._emit(f"[BEST] {epoch_done}/{self.total_epochs} {metric_name}={float(metric_value):.4f}")

    def _handle_stop_requested(self, payload: Dict[str, Any]) -> None:
        reason = str(payload.get("reason", "requested"))
        epoch = int(payload.get("epoch", 0))
        step = int(payload.get("global_step", 0))
        self._emit(f"[STOP] epoch={epoch} step={step} reason={reason}")


def build_history_snapshot(
    *,
    state_history: Optional[Dict[str, Any]] = None,
    session_history: Optional[Dict[str, Any]] = None,
    train_loss_curve: List[float],
    val_loss_curve: List[float],
    val_acc_curve: List[float],
    macro_f1_curve: List[float],
    weighted_f1_curve: List[float],
    balanced_acc_curve: List[float],
    gap_curve: List[float],
) -> Dict[str, Any]:
    if session_history:
        merged = dict(session_history)
        merged.setdefault("per_class_accuracy", list((state_history or {}).get("per_class_accuracy", [])))
        merged.setdefault("worst_classes", list((state_history or {}).get("worst_classes", [])))
        return merged

    baseline = state_history or {}
    return {
        "train_loss": list(train_loss_curve),
        "val_loss": list(val_loss_curve),
        "val_accuracy": list(val_acc_curve),
        "macro_precision": list(baseline.get("macro_precision", [])),
        "macro_recall": list(baseline.get("macro_recall", [])),
        "macro_f1": list(macro_f1_curve),
        "weighted_f1": list(weighted_f1_curve),
        "balanced_accuracy": list(balanced_acc_curve),
        "generalization_gap": list(gap_curve),
        "per_class_accuracy": list(baseline.get("per_class_accuracy", [])),
        "worst_classes": list(baseline.get("worst_classes", [])),
    }


def persist_training_history_artifacts(
    *,
    root: Path,
    history_snapshot: Dict[str, Any],
    telemetry: Any = None,
) -> Dict[str, Path]:
    return persist_training_history_artifacts_core(
        artifact_root=_artifact_dir(root),
        history_snapshot=history_snapshot,
        telemetry=telemetry,
    )


def persist_training_curve_figure(*, root: Path, epoch_done: int, telemetry: Any = None) -> Dict[str, Path]:
    import matplotlib.pyplot as plt

    train_dir = _artifact_dir(root, "training")
    latest_curve = train_dir / "training_curves_latest.png"
    epoch_curve = train_dir / f"training_curves_epoch_{int(epoch_done):03d}.png"
    plt.savefig(latest_curve, dpi=150)
    plt.savefig(epoch_curve, dpi=150)
    if telemetry is not None:
        telemetry.copy_artifact_file(latest_curve, "training/training_curves_latest.png")
        telemetry.copy_artifact_file(epoch_curve, f"training/training_curves_epoch_{int(epoch_done):03d}.png")
    return {"latest_curve": latest_curve, "epoch_curve": epoch_curve}


def save_notebook_checkpoint(
    *,
    checkpoint_manager: Any,
    adapter: Any,
    session: Any,
    reason: str,
    run_id: str,
    telemetry: Any = None,
    mark_best: bool = False,
    val_loss: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if checkpoint_manager is None:
        return None
    record = checkpoint_manager.save_checkpoint(
        adapter=adapter,
        session=session,
        reason=reason,
        run_id=run_id,
        mark_best=bool(mark_best),
        val_loss=(float(val_loss) if val_loss is not None else None),
    )
    if telemetry is not None:
        telemetry.emit_event("checkpoint_saved", dict(record), phase="checkpoint")
    return record


def run_notebook_training_session(
    *,
    root: Path,
    state: Dict[str, Any],
    run_id: str,
    epochs: int,
    device: str,
    stdout_batch_interval: int,
    validation_every_n_epochs: int,
    checkpoint_every_n_steps: int,
    checkpoint_on_exception: bool,
    resume_mode: str = "fresh",
    telemetry: Any = None,
    print_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    emit = print if print_fn is None else print_fn
    if state.get("adapter") is None or state.get("loaders") is None:
        raise RuntimeError("Once engine init hucresini calistirin.")

    adapter = state["adapter"]
    loaders = state["loaders"]
    checkpoint_manager = state.get("checkpoint_manager")
    val_loader = loaders.get("val")
    resolved_telemetry = telemetry if telemetry is not None else state.get("telemetry")

    resume_state = None
    if str(resume_mode or "").strip().lower() == "resume" and isinstance(state.get("resume_manifest"), dict):
        checkpoint_path = str(state["resume_manifest"].get("path", "")).strip()
        if checkpoint_path:
            try:
                resume_state = adapter.load_training_checkpoint(checkpoint_path)
                state["resume_state"] = resume_state
                progress_state = resume_state.get("progress_state") or {}
                emit(f"[RESUME] epoch={progress_state.get('epoch', 0)} step={progress_state.get('global_step', 0)}")
            except Exception as exc:
                emit(f"[RESUME] Basarisiz: {exc}")

    existing_history = (resume_state or {}).get("history", (resume_state or {}).get("history_snapshot", {}))
    train_loss_curve = list(existing_history.get("train_loss", []))
    val_loss_curve = list(existing_history.get("val_loss", []))
    val_acc_curve = list(existing_history.get("val_accuracy", []))
    macro_f1_curve = list(existing_history.get("macro_f1", []))
    weighted_f1_curve = list(existing_history.get("weighted_f1", []))
    balanced_acc_curve = list(existing_history.get("balanced_accuracy", []))
    gap_curve = list(existing_history.get("generalization_gap", []))

    start_time = time.time()
    session = None
    last_checkpoint_step = -1
    best_val_loss = float(state["best_val_loss"]) if state.get("best_val_loss") is not None else None
    status_printer = NotebookTrainingStatusPrinter(
        total_epochs=int(epochs),
        batch_interval=int(stdout_batch_interval),
        print_fn=emit,
    )

    emit(f"[TRAIN] epochs={int(epochs)} device={device} batch_interval={int(stdout_batch_interval)}")
    state["training_runtime"] = {
        "checkpoint_every_n_steps": int(checkpoint_every_n_steps),
        "validation_every_n_epochs": int(validation_every_n_epochs),
        "stdout_batch_interval": int(stdout_batch_interval),
        "resume_mode": str(resume_mode or "fresh"),
    }
    _call_if_present(
        resolved_telemetry,
        "update_latest",
        {
            "phase": "training_started",
            "epochs": int(epochs),
            "batch_interval": int(stdout_batch_interval),
        },
    )

    def _history_snapshot() -> Dict[str, Any]:
        return build_history_snapshot(
            state_history=state.get("history"),
            train_loss_curve=train_loss_curve,
            val_loss_curve=val_loss_curve,
            val_acc_curve=val_acc_curve,
            macro_f1_curve=macro_f1_curve,
            weighted_f1_curve=weighted_f1_curve,
            balanced_acc_curve=balanced_acc_curve,
            gap_curve=gap_curve,
        )

    def _persist_history() -> Dict[str, Any]:
        snapshot = _history_snapshot()
        state["history"] = dict((state.get("history") or {}), **snapshot)
        persist_training_history_artifacts(
            root=root,
            history_snapshot=state["history"],
            telemetry=resolved_telemetry,
        )
        return snapshot

    def _checkpoint(
        reason: str, event: Dict[str, Any], *, mark_best: bool = False, val_loss: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        if session is None:
            return None
        record = save_notebook_checkpoint(
            checkpoint_manager=checkpoint_manager,
            adapter=adapter,
            session=session,
            reason=reason,
            run_id=run_id,
            telemetry=resolved_telemetry,
            mark_best=bool(mark_best),
            val_loss=(float(val_loss) if val_loss is not None else None),
        )
        if record is not None:
            state["resume_manifest"] = record
            emit(f"[CKPT] {reason} epoch={record.get('epoch', '?')} step={record.get('global_step', '?')}")
        return record

    def session_observer(record: Dict[str, Any]) -> None:
        nonlocal last_checkpoint_step, best_val_loss
        event_type = str(record.get("event_type", "") or "")
        event = dict(record.get("payload", {}) or {})
        if event_type == "stop_requested":
            status_printer.handle("stop_requested", event)
            return
        if event_type == "batch_end":
            status_printer.handle(
                "batch_end",
                dict(
                    event,
                    loss=event.get("loss", event.get("batch_loss", 0.0)),
                ),
            )
            step = int(event.get("global_step", 0))
            if (
                int(checkpoint_every_n_steps) > 0
                and step > 0
                and (step % int(checkpoint_every_n_steps) == 0)
                and step != last_checkpoint_step
            ):
                _checkpoint(f"batch_{int(checkpoint_every_n_steps)}", event)
                last_checkpoint_step = step
            return
        if event_type != "epoch_end":
            return

        train_loss_curve.append(float(event.get("epoch_loss", 0.0)))
        for key, curve in [
            ("val_loss", val_loss_curve),
            ("val_accuracy", val_acc_curve),
            ("macro_f1", macro_f1_curve),
            ("weighted_f1", weighted_f1_curve),
            ("balanced_accuracy", balanced_acc_curve),
            ("generalization_gap", gap_curve),
        ]:
            if key in event:
                curve.append(float(event[key]))

        val_loss = float(event["val_loss"]) if "val_loss" in event else None
        mark_best = False
        if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
            best_val_loss = val_loss
            state["best_val_loss"] = best_val_loss
            mark_best = True

        status_printer.handle("validation_end", event)
        if mark_best:
            status_printer.handle(
                "best_metric_updated",
                {
                    "epoch_done": int(event.get("epoch_done", 0)),
                    "best_metric_name": "val_loss",
                    "best_metric_value": val_loss,
                },
            )

        should_persist_curve = (
            mark_best
            or int(event["epoch_done"]) == 1
            or int(event["epoch_done"]) == int(epochs)
            or bool(event.get("stopped_early", False))
            or (int(event["epoch_done"]) % 5 == 0)
        )
        if should_persist_curve:
            _fig, axes = plt.subplots(1, 3, figsize=(13, 3))
            axes[0].plot(range(1, len(train_loss_curve) + 1), train_loss_curve, marker="o", label="Train")
            if val_loss_curve:
                axes[0].plot(range(1, len(val_loss_curve) + 1), val_loss_curve, marker="s", label="Val")
            for values, label, marker in [
                (val_acc_curve, "Acc", "^"),
                (macro_f1_curve, "MacroF1", "d"),
                (weighted_f1_curve, "WtdF1", "x"),
                (balanced_acc_curve, "BalAcc", "*"),
            ]:
                if values:
                    axes[1].plot(range(1, len(values) + 1), values, marker=marker, label=label)
            if gap_curve:
                axes[2].plot(range(1, len(gap_curve) + 1), gap_curve, marker="o", label="Gap")
            axes[1].set_ylim(0, 1)
            axes[2].axhline(0, color="black", lw=1, alpha=0.5)
            for axis, title, ylabel in zip(axes, ("Loss", "Metrics", "Gen. Gap"), ("Loss", "Score", "Gap")):
                axis.set(xlabel="Epoch", ylabel=ylabel, title=title)
                axis.grid(True, alpha=0.3)
                axis.legend()
            plt.tight_layout()
            persist_training_curve_figure(
                root=root,
                epoch_done=int(event["epoch_done"]),
                telemetry=resolved_telemetry,
            )
            plt.close("all")

        _persist_history()
        if (
            mark_best
            or bool(event.get("stopped_early", False))
            or int(event["epoch_done"]) == int(epochs)
            or int(checkpoint_every_n_steps) <= 0
        ):
            _checkpoint("epoch_end", event, mark_best=mark_best, val_loss=val_loss)

        _call_if_present(
            resolved_telemetry,
            "update_latest",
            {
                "phase": "training",
                "epoch_done": int(event["epoch_done"]),
                "global_step": int(event.get("global_step", 0)),
                "best_val_loss": best_val_loss,
            },
        )

    session = adapter.build_training_session(
        loaders["train"],
        num_epochs=int(epochs),
        val_loader=val_loader,
        observers=[session_observer],
        resume_state=resume_state,
        run_id=run_id,
        validation_every_n_epochs=int(validation_every_n_epochs),
    )

    try:
        history = session.run()
        adapter.is_trained = True
    except Exception as exc:
        emit(f"[TRAIN] Exception: {exc}")
        _call_if_present(resolved_telemetry, "emit_log", f"Training exception: {exc}", phase="train", level="error")
        if checkpoint_on_exception:
            try:
                _checkpoint(
                    "exception",
                    {
                        "epoch": 0,
                        "batch": 0,
                        "global_step": int((state.get("history") or {}).get("global_step", 0)),
                        "elapsed_sec": time.time() - start_time,
                    },
                )
            except Exception as exc:
                import logging

                logging.exception("Unhandled exception")
                raise
        raise

    elapsed_total = time.time() - start_time
    state["history"] = history.to_dict()
    state["resume_state"] = session.snapshot_state()
    _persist_history()
    _call_if_present(
        resolved_telemetry,
        "update_latest",
        {
            "phase": "training_complete",
            "elapsed_sec": round(elapsed_total, 3),
            "stopped_early": bool(state["history"].get("stopped_early", False)),
        },
    )
    emit(f"[TRAIN] Complete. elapsed={elapsed_total:.1f}s stopped_early={state['history'].get('stopped_early', False)}")
    return {
        "state": state,
        "history": state["history"],
        "resume_state": state.get("resume_state"),
        "elapsed_sec": elapsed_total,
    }


def persist_validation_artifacts(
    *,
    root: Path,
    y_true: List[int],
    y_pred: List[int],
    classes: List[str],
    telemetry: Any = None,
    artifact_subdir: str = "validation",
    telemetry_subdir: Optional[str] = None,
    gate_targets: Optional[Dict[str, float]] = None,
    require_ood: bool = False,
    emit_metric_gate: bool = True,
    ood_labels: Optional[List[int]] = None,
    ood_scores: Optional[List[float]] = None,
    ood_scores_by_method: Optional[Dict[str, List[float]]] = None,
    sure_ds_f1: Optional[float] = None,
    conformal_empirical_coverage: Optional[float] = None,
    conformal_avg_set_size: Optional[float] = None,
    ood_type_breakdown: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    prediction_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return persist_validation_artifacts_core(
        artifact_root=_artifact_dir(root),
        y_true=y_true,
        y_pred=y_pred,
        classes=classes,
        telemetry=telemetry,
        artifact_subdir=artifact_subdir,
        telemetry_subdir=telemetry_subdir,
        gate_targets=gate_targets,
        require_ood=require_ood,
        emit_metric_gate=emit_metric_gate,
        ood_labels=ood_labels,
        ood_scores=ood_scores,
        ood_scores_by_method=ood_scores_by_method,
        sure_ds_f1=sure_ds_f1,
        conformal_empirical_coverage=conformal_empirical_coverage,
        conformal_avg_set_size=conformal_avg_set_size,
        ood_type_breakdown=ood_type_breakdown,
        context=context,
        prediction_rows=prediction_rows,
    )


def persist_production_readiness_artifact(
    *,
    root: Path,
    classification_metric_gate: Dict[str, Any] | None,
    classification_split: str,
    ood_evidence_source: str | None,
    ood_metrics: Dict[str, Any] | None,
    targets: Optional[Dict[str, float]] = None,
    context: Optional[Dict[str, Any]] = None,
    require_ood: bool = True,
    telemetry: Any = None,
) -> Dict[str, Any]:
    return persist_production_readiness_artifact_core(
        artifact_root=_artifact_dir(root),
        classification_metric_gate=classification_metric_gate,
        classification_split=classification_split,
        ood_evidence_source=ood_evidence_source,
        ood_metrics=ood_metrics,
        targets=targets,
        context=context,
        require_ood=require_ood,
        telemetry=telemetry,
    )


def persist_training_run_context_artifact(
    *, root: Path, context_payload: Dict[str, Any], telemetry: Any = None
) -> Dict[str, Path]:
    return persist_training_run_context_artifact_core(
        artifact_root=_artifact_dir(root),
        context_payload=context_payload,
        telemetry=telemetry,
    )


def persist_behavioral_acceptance_artifact(
    *,
    root: Path,
    classification_metrics: Dict[str, Any] | None,
    prediction_rows: List[Dict[str, Any]] | None,
    context: Optional[Dict[str, Any]] = None,
    telemetry: Any = None,
) -> Dict[str, Any]:
    return persist_behavioral_acceptance_artifact_core(
        artifact_root=_artifact_dir(root),
        classification_metrics=classification_metrics,
        prediction_rows=prediction_rows,
        context=context,
        telemetry=telemetry,
    )


def persist_behavioral_dev_report_artifact(
    *,
    root: Path,
    classification_metrics: Dict[str, Any] | None,
    prediction_rows: List[Dict[str, Any]] | None,
    context: Optional[Dict[str, Any]] = None,
    telemetry: Any = None,
) -> Dict[str, Any]:
    return persist_behavioral_dev_report_artifact_core(
        artifact_root=_artifact_dir(root),
        classification_metrics=classification_metrics,
        prediction_rows=prediction_rows,
        context=context,
        telemetry=telemetry,
    )


def _resolve_colab_runtime_api() -> Any:
    try:
        from google.colab import runtime
    except Exception:
        import logging

        logging.exception("Unhandled exception")
        raise
    return runtime


def build_notebook_completion_report(
    *,
    state: Optional[Dict[str, Any]] = None,
    telemetry: Any = None,
    repo_run_exports: Optional[Dict[str, str]] = None,
    notebook_export_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    resolved_state = dict(state or {})
    resolved_exports = dict(repo_run_exports or {})

    evaluation_artifacts = resolved_state.get("evaluation_artifacts")
    evaluation_splits = sorted(evaluation_artifacts.keys()) if isinstance(evaluation_artifacts, dict) else []
    behavioral = resolved_state.get("behavioral_acceptance") or {}
    behavioral_status = str(behavioral.get("status", "") or "")

    summary_path = getattr(telemetry, "local_summary_path", None)
    repo_export_checks = {name: _path_exists(resolved_exports.get(name)) for name in _EXPECTED_REPO_EXPORTS}
    repo_exports_complete = bool(resolved_exports) and all(
        repo_export_checks.get(name, False) for name in _EXPECTED_REPO_EXPORTS
    )

    checks = {
        "evaluation_artifacts": bool(evaluation_splits),
        "adapter_behavioral_acceptance": isinstance(behavioral, dict)
        and bool(behavioral)
        and behavioral_status in {"pass", "fail"},
        "telemetry_summary": _path_exists(summary_path),
        "repo_exports": repo_exports_complete,
        "executed_notebook_export": _path_exists(notebook_export_path),
    }
    blocking_check_names = (
        "evaluation_artifacts",
        "adapter_behavioral_acceptance",
        "telemetry_summary",
        "repo_exports",
    )
    soft_check_names = ("executed_notebook_export",)
    missing = [name for name in blocking_check_names if not checks.get(name, False)]
    soft_missing = [name for name in soft_check_names if not checks.get(name, False)]
    completion_ready = not missing and behavioral_status == "pass" and behavioral.get("pass") is True
    if not completion_ready and behavioral_status == "fail" and "adapter_behavioral_acceptance_failed" not in missing:
        missing = [*missing, "adapter_behavioral_acceptance_failed"]
    return {
        "ready": completion_ready,
        "checks": checks,
        "missing": missing,
        "soft_missing": soft_missing,
        "blocking_checks": {name: checks.get(name, False) for name in blocking_check_names},
        "soft_checks": {name: checks.get(name, False) for name in soft_check_names},
        "evaluation_splits": evaluation_splits,
        "repo_exports": repo_export_checks,
        "adapter_behavioral_acceptance_status": behavioral_status,
        "decision_authority": "adapter_behavioral_acceptance",
    }


def maybe_auto_disconnect_colab_runtime(
    *,
    enabled: bool,
    grace_period_sec: float = 20.0,
    state: Optional[Dict[str, Any]] = None,
    telemetry: Any = None,
    repo_run_exports: Optional[Dict[str, str]] = None,
    notebook_export_path: Optional[str | Path] = None,
    completion_report: Optional[Dict[str, Any]] = None,
    print_fn: Optional[Callable[[str], None]] = None,
    runtime_api_resolver: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    emit = print if print_fn is None else print_fn
    report = (
        completion_report
        if completion_report is not None
        else build_notebook_completion_report(
            state=state,
            telemetry=telemetry,
            repo_run_exports=repo_run_exports,
            notebook_export_path=notebook_export_path,
        )
    )
    report["auto_disconnect_enabled"] = bool(enabled)
    report.setdefault("disconnect_requested", False)
    report.setdefault("missing", [])
    report.setdefault("soft_missing", [])

    def _publish_status(phase: str, **extra: Any) -> None:
        payload: Dict[str, Any] = {
            "phase": str(phase),
            "auto_disconnect": bool(enabled),
            "disconnect_requested": bool(report.get("disconnect_requested", False)),
            "completion_checks": dict(report.get("checks", {})),
            "completion_missing": list(report.get("missing", [])),
            "completion_soft_missing": list(report.get("soft_missing", [])),
        }
        payload.update(extra)
        _call_if_present(telemetry, "update_latest", payload)
        _call_if_present(telemetry, "sync_pending")

    if not enabled:
        emit("[COLAB] Auto-disconnect disabled.")
        _publish_status("auto_disconnect_disabled")
        return report

    if not bool(report.get("ready")):
        missing = ", ".join(str(item) for item in report.get("missing", [])) or "unknown"
        emit(f"[COLAB] Auto-disconnect skipped. Incomplete required checks: {missing}")
        soft_missing = ", ".join(str(item) for item in report.get("soft_missing", []))
        if soft_missing:
            emit(f"[COLAB] Soft-missing checks: {soft_missing}")
        _publish_status("auto_disconnect_skipped")
        return report

    runtime_api = (runtime_api_resolver or _resolve_colab_runtime_api)()
    if runtime_api is None or not hasattr(runtime_api, "unassign"):
        emit("[COLAB] Auto-disconnect skipped. google.colab.runtime.unassign is unavailable.")
        _publish_status("auto_disconnect_unavailable")
        return report

    soft_missing = ", ".join(str(item) for item in report.get("soft_missing", []))
    if soft_missing:
        emit(f"[COLAB] Proceeding despite soft-missing checks: {soft_missing}")

    delay = max(0.0, float(grace_period_sec or 0.0))
    report["disconnect_requested"] = True
    report["grace_period_sec"] = delay

    _publish_status(
        "auto_disconnect_pending",
        auto_disconnect=True,
        grace_period_sec=delay,
    )

    if delay > 0:
        emit(f"[COLAB] Work complete. Disconnecting runtime in {delay:.0f}s to avoid idle credit use.")
        time.sleep(delay)
    else:
        emit("[COLAB] Work complete. Disconnecting runtime now to avoid idle credit use.")

    try:
        runtime_api.unassign()
    except Exception as exc:
        report["disconnect_requested"] = False
        report["disconnect_error"] = f"{exc.__class__.__name__}: {exc}"
        emit(f"[COLAB] Auto-disconnect failed: {report['disconnect_error']}")
        _publish_status("auto_disconnect_failed", disconnect_error=report["disconnect_error"])
    return report


def cleanup_notebook_training_state(
    shared_globals: Dict[str, Any],
    *,
    label: str = "training target",
    print_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Release per-target notebook objects before the next sequential training run."""
    emit = print if print_fn is None else print_fn
    state = shared_globals.get("STATE")
    loaders = state.get("loaders") if isinstance(state, dict) else None
    if isinstance(loaders, dict):
        for loader in loaders.values():
            iterator = getattr(loader, "_iterator", None)
            shutdown_workers = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown_workers):
                try:
                    shutdown_workers()
                except Exception as exc:
                    emit(f"[CLEANUP] DataLoader worker shutdown skipped: {exc}")
            try:
                loader._iterator = None
            except Exception:
                pass
    if isinstance(state, dict):
        for key in (
            "adapter",
            "loaders",
            "history",
            "calibration",
            "evaluation_artifacts",
            "ood_benchmark",
            "optimization_campaign",
            "recommendation_report",
        ):
            state.pop(key, None)
    for name in (
        "adapter",
        "loaders",
        "trainer",
        "session",
        "test_loader",
        "val_loader",
        "ood_loader",
        "ood_dev_loader",
        "ood_test_loader",
        "results",
        "benchmark_summary",
        "selected_artifacts",
        "evaluation",
    ):
        shared_globals.pop(name, None)
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            emit(
                "[CLEANUP] CUDA "
                f"allocated={torch.cuda.memory_allocated() / (1024**3):.2f}GiB "
                f"reserved={torch.cuda.memory_reserved() / (1024**3):.2f}GiB"
            )
    except Exception as exc:
        emit(f"[CLEANUP] CUDA cleanup skipped: {exc}")
    gc.collect()
    emit(f"[CLEANUP] Released runtime state after {label}.")
