#!/usr/bin/env python3
"""Notebook training sessions, checkpoints, artifacts and completion handling."""

from __future__ import annotations

import contextlib
import gc
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib

from src.shared.artifacts import ensure_artifact_dir
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

logger = logging.getLogger(__name__)

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
) -> Dict[str, Any]:
    summary_path = _artifact_dir(root, "training") / "summary.json"
    current = read_json(summary_path, default={}, expect_type=dict)
    merged = deep_merge(dict(current), dict(payload or {}))
    write_json(summary_path, merged, ensure_ascii=False, sort_keys=False)
    if telemetry is not None and hasattr(telemetry, "copy_artifact_file"):
        telemetry.copy_artifact_file(summary_path, "training/summary.json")
    _refresh_traceability_records(root=_artifact_dir(root), summary_payload=merged, telemetry=telemetry)
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
    return ensure_artifact_dir(Path(root) / "outputs" / "colab_notebook_training" / "artifacts", *parts)


def notebook_artifact_root(root: Path) -> Path:
    return _artifact_dir(root)


def _path_exists(path_like: Optional[str | Path]) -> bool:
    return bool(path_like and Path(path_like).expanduser().exists())


def _call_if_present(target: Any, method_name: str, *args, **kwargs) -> None:
    method = getattr(target, method_name, None)
    if not callable(method):
        return
    try:
        method(*args, **kwargs)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Telemetry call %s failed: %s", method_name, exc)


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


def persist_validation_artifacts(
    *,
    root: Path,
    y_true: List[int],
    y_pred: List[int],
    classes: List[str],
    telemetry: Any = None,
    artifact_subdir: str = "validation",
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
    except ImportError:
        return None
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
            # Some loader wrappers expose a read-only `_iterator`; leaving it set is harmless.
            with contextlib.suppress(AttributeError):
                loader._iterator = None
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
    import matplotlib.pyplot as plt

    plt.close("all")
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
