"""Canonical training workflow for the supported adapter-training path."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

import torch

from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.core.config_manager import get_config
from src.data.dataset_lineage import build_local_legacy_lineage
from src.data.dataset_release import canonical_sha256
from src.data.loaders import create_training_loaders
from src.shared.adapter_paths import build_adapter_bundle_root
from src.shared.dict_utils import nested_dict as _nested_dict
from src.shared.json_utils import read_json
from src.shared.telemetry import emit_event as _emit_event
from src.shared.telemetry import update_latest as _update_latest
from src.training.services.ood_benchmark import run_leave_one_class_out_benchmark
from src.training.services.ood_score_selection import (
    build_real_ood_dev_selection,
    is_auto_primary_score_method,
    normalize_requested_primary_score_method,
    resolve_runtime_primary_score_method,
    select_best_ood_score_method,
)
from src.training.services.reporting import (
    BatchMetricsRecorder,
    persist_behavioral_acceptance_artifact,
    persist_production_readiness_artifact,
    persist_training_run_context_artifact,
    persist_training_summary_artifact,
)
from src.training.services.traceability import (
    build_experiment_manifest,
    build_optimization_record,
    persist_traceability_artifacts,
)
from src.training.validation import evaluate_model_with_artifact_metrics
from src.utils.training_helpers import loader_size
from src.workflows import training_utils
from src.workflows.training_readiness import (
    build_production_readiness_context as build_production_readiness_context_payload,
)
from src.workflows.training_readiness import (
    build_training_summary_payload as build_training_summary_payload_dict,
)
from src.workflows.training_readiness import (
    record_adapter_export_metadata as record_adapter_export_metadata_payload,
)
from src.workflows.training_readiness import (
    record_primary_score_selection as record_primary_score_selection_payload,
)
from src.workflows.training_readiness import (
    select_authoritative_artifacts as select_authoritative_artifacts_payload,
)
from src.workflows.training_readiness import (
    select_authoritative_evaluation as select_authoritative_evaluation_payload,
)
from src.workflows.training_support import (
    build_artifact_payload,
    prepare_training_run,
    stringify_paths,
)

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]
Observer = Callable[[JsonDict], None]


class TelemetrySink(Protocol):
    def emit_event(
        self,
        event_type: str,
        payload: JsonDict,
        *,
        phase: str,
        force_sync: bool = ...,
    ) -> None: ...

    def update_latest(self, payload: JsonDict) -> None: ...


class CheckpointManagerLike(Protocol):
    def save_checkpoint(
        self,
        *,
        adapter: IndependentCropAdapter,
        session: "TrainingSessionLike",
        reason: str,
        run_id: str,
        mark_best: bool,
        val_loss: Any,
    ) -> JsonDict: ...


class HistoryLike(Protocol):
    def to_dict(self) -> JsonDict: ...


class TrainingSessionLike(Protocol):
    trainer: Any

    def run(self) -> HistoryLike: ...

    def restore_best_model_state(self) -> bool: ...


def _last_history_value(history_payload: Dict[str, Any], key: str) -> Optional[float]:
    values = history_payload.get(key, [])
    if not isinstance(values, list) or not values:
        return None
    return float(values[-1])


_FINAL_HISTORY_METRICS = (
    "train_loss",
    "val_loss",
    "val_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "generalization_gap",
)


def _collect_final_metrics(history_payload: Dict[str, Any]) -> Dict[str, float]:
    final_metrics = {key: _last_history_value(history_payload, key) for key in _FINAL_HISTORY_METRICS}
    return {key: value for key, value in final_metrics.items() if value is not None}


def _evaluation_metrics_summary(evaluation_payload: Any) -> Dict[str, float]:
    report = getattr(evaluation_payload, "report", None)
    if report is None:
        return {}
    return {
        "val_loss": float(getattr(report, "val_loss", 0.0)),
        "val_accuracy": float(getattr(report, "val_accuracy", 0.0)),
        "macro_precision": float(getattr(report, "macro_precision", 0.0)),
        "macro_recall": float(getattr(report, "macro_recall", 0.0)),
        "macro_f1": float(getattr(report, "macro_f1", 0.0)),
        "weighted_f1": float(getattr(report, "weighted_f1", 0.0)),
        "balanced_accuracy": float(getattr(report, "balanced_accuracy", 0.0)),
    }











def _resolve_ood_method_comparison_from_artifacts(artifacts: JsonDict) -> JsonDict:
    metric_gate = dict(artifacts.get("metric_gate", {})) if isinstance(artifacts, dict) else {}
    context = _nested_dict(metric_gate, "context")
    comparison = _nested_dict(context, "ood_method_comparison")
    if comparison:
        return comparison
    paths = dict(artifacts.get("paths", {})) if isinstance(artifacts, dict) else {}
    comparison_path = paths.get("ood_method_comparison_json")
    if comparison_path:
        resolved = read_json(Path(str(comparison_path)), default={}, expect_type=dict)
        if isinstance(resolved, dict):
            return resolved
    return {}


def _build_ood_method_comparison_context(
    *,
    authoritative_artifacts: Dict[str, Any],
    ood_evidence_source: str,
    ood_benchmark: Dict[str, Any],
) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {}
    if str(ood_evidence_source or "") == "held_out_benchmark":
        comparison = dict(ood_benchmark.get("method_comparison", {})) if isinstance(ood_benchmark, dict) else {}
        if not comparison and isinstance(ood_benchmark, dict):
            comparison_metrics = dict(ood_benchmark.get("method_comparison_metrics", {}))
            if comparison_metrics:
                comparison = {
                    "requested_primary_score_method": str(
                        ood_benchmark.get("requested_primary_score_method", "") or ""
                    ),
                    "selected_primary_score_method": str(ood_benchmark.get("primary_score_method", "") or ""),
                    "selection_source": str(ood_benchmark.get("primary_score_selection_source", "") or ""),
                    "methods": {
                        str(method_name): {"pooled_metrics": dict(metrics or {})}
                        for method_name, metrics in comparison_metrics.items()
                    },
                }
    else:
        comparison = _resolve_ood_method_comparison_from_artifacts(authoritative_artifacts)
    if not comparison:
        return {}
    method_payloads = dict(comparison.get("methods", {})) if isinstance(comparison.get("methods"), dict) else {}
    summarized_methods: Dict[str, Any] = {}
    for method_name, payload in method_payloads.items():
        details = dict(payload or {}) if isinstance(payload, dict) else {}
        summary_payload: Dict[str, Any] = {
            "pooled_metrics": dict(details.get("pooled_metrics", {}))
            if isinstance(details.get("pooled_metrics"), dict)
            else {},
        }
        if "pooled_gate_eligible" in details:
            summary_payload["pooled_gate_eligible"] = bool(details.get("pooled_gate_eligible"))
        if isinstance(details.get("worst_slice"), dict):
            summary_payload["worst_slice"] = dict(details.get("worst_slice", {}))
        if isinstance(details.get("worst_fold"), dict):
            summary_payload["worst_fold"] = dict(details.get("worst_fold", {}))
        if isinstance(details.get("metric_std"), dict):
            summary_payload["metric_std"] = dict(details.get("metric_std", {}))
        summarized_methods[str(method_name)] = summary_payload
    return {
        "requested_primary_score_method": str(comparison.get("requested_primary_score_method", "") or ""),
        "selected_primary_score_method": str(
            comparison.get("selected_primary_score_method", comparison.get("primary_score_method", "")) or ""
        ),
        "selection_source": str(
            comparison.get("selection_source", comparison.get("primary_score_selection_source", "")) or ""
        ),
        "split_name": str(comparison.get("split_name", "") or ""),
        "methods": summarized_methods,
    }


@dataclass
class TrainingWorkflowResult:
    run_id: str
    crop_name: str
    part_name: str
    class_names: List[str]
    history: Dict[str, Any]
    loader_sizes: Dict[str, int]
    adapter_dir: Path
    artifact_dir: Optional[Path] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    ood_calibration: Dict[str, Any] = field(default_factory=dict)
    checkpoint_records: List[Dict[str, Any]] = field(default_factory=list)
    ood_evidence_source: str = ""
    ood_benchmark: Dict[str, Any] = field(default_factory=dict)
    behavioral_acceptance: Dict[str, Any] = field(default_factory=dict)
    production_readiness: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "crop_name": self.crop_name,
            "part_name": self.part_name,
            "class_names": list(self.class_names),
            "history": dict(self.history),
            "loader_sizes": {str(k): int(v) for k, v in self.loader_sizes.items()},
            "adapter_dir": str(self.adapter_dir),
            "artifact_dir": ("" if self.artifact_dir is None else str(self.artifact_dir)),
            "artifacts": stringify_paths(self.artifacts),
            "ood_calibration": dict(self.ood_calibration),
            "checkpoint_records": [dict(item) for item in self.checkpoint_records],
            "ood_evidence_source": str(self.ood_evidence_source),
            "ood_benchmark": stringify_paths(self.ood_benchmark),
            "behavioral_acceptance": dict(self.behavioral_acceptance),
            "production_readiness": dict(self.production_readiness),
        }


@dataclass(frozen=True)
class _SplitEvaluationStage:
    val: Any
    test: Any

    def as_dict(self) -> Dict[str, Any]:
        return {
            "val": self.val,
            "test": self.test,
        }


@dataclass(frozen=True)
class _PrimaryScoreSelectionStage:
    requested_method: str
    selected_method: str
    selection_source: str
    selection_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _OODEvidenceStage:
    source: str
    metrics: Dict[str, Any]
    benchmark: Dict[str, Any]


@dataclass(frozen=True)
class _TrainingExecutionStage:
    history_payload: JsonDict
    best_state_restored: bool
    calibration_split_name: str
    calibration_loader: Any
    ood_calibration: JsonDict
    trainer: Any


class TrainingWorkflow:
    """Stable app-facing entrypoint for continual adapter training."""

    def __init__(
        self,
        *,
        config: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = "colab",
        device: str = "cuda",
    ) -> None:
        self.config = dict(config or get_config(environment=environment))
        self.device = str(device)

    @staticmethod
    def _emit_telemetry(
        telemetry: Optional[TelemetrySink],
        event_type: str,
        payload: Dict[str, Any],
        *,
        phase: str,
        force_sync: Optional[bool] = None,
    ) -> None:
        _emit_event(telemetry, event_type, payload, phase=phase, force_sync=force_sync)

    def _build_workflow_observer(
        self,
        *,
        adapter: IndependentCropAdapter,
        run_id: str,
        telemetry: Any,
        checkpoint_manager: Optional[CheckpointManagerLike],
        batch_recorder: Optional[BatchMetricsRecorder],
        checkpoint_records: List[JsonDict],
        session_holder: JsonDict,
    ) -> Observer:
        def _workflow_observer(event: Dict[str, Any]) -> None:
            event_type = str(event.get("event_type", "training_event"))
            payload = dict(event.get("payload", {}))
            if event_type == "batch_end" and batch_recorder is not None:
                batch_recorder.append(payload)

            self._emit_telemetry(
                telemetry,
                event_type,
                payload,
                phase="training",
                force_sync=False,
            )
            if (
                telemetry is not None
                and event_type
                in {
                    "batch_end",
                    "validation_end",
                    "epoch_end",
                    "training_completed",
                }
            ):
                _update_latest(telemetry, {"event_type": event_type, **payload})

            if checkpoint_manager is None or event_type != "checkpoint_requested":
                return
            session = session_holder.get("session")
            if session is None:
                return
            record = checkpoint_manager.save_checkpoint(
                adapter=adapter,
                session=session,
                reason=str(payload.get("reason", "event")),
                run_id=run_id,
                mark_best=bool(payload.get("mark_best", False)),
                val_loss=payload.get("val_loss"),
            )
            checkpoint_records.append(dict(record))

        return _workflow_observer

    def _create_training_session(
        self,
        *,
        adapter: IndependentCropAdapter,
        loaders: JsonDict,
        num_epochs: Optional[int],
        colab_cfg: JsonDict,
        run_id: str,
        telemetry: Optional[TelemetrySink],
        checkpoint_manager: Optional[CheckpointManagerLike],
        observers: Optional[Iterable[Observer]],
        stop_policy: Optional[Callable[[], bool]],
        checkpoint_every_n_steps: Optional[int],
        checkpoint_on_exception: Optional[bool],
        validation_every_n_epochs: Optional[int],
        artifact_dir: Path,
    ) -> tuple[TrainingSessionLike, BatchMetricsRecorder, List[JsonDict]]:
        batch_recorder = BatchMetricsRecorder(artifact_root=artifact_dir)
        checkpoint_records: List[JsonDict] = []
        session_observers: List[Observer] = list(observers or [])
        session_holder: JsonDict = {}
        session_observers.append(
            self._build_workflow_observer(
                adapter=adapter,
                run_id=run_id,
                telemetry=telemetry,
                checkpoint_manager=checkpoint_manager,
                batch_recorder=batch_recorder,
                checkpoint_records=checkpoint_records,
                session_holder=session_holder,
            )
        )
        session = adapter.build_training_session(
            train_loader=loaders["train"],
            num_epochs=num_epochs,
            val_loader=loaders.get("val"),
            observers=session_observers,
            stop_policy=stop_policy,
            run_id=run_id,
            checkpoint_every_n_steps=int(
                checkpoint_every_n_steps
                if checkpoint_every_n_steps is not None
                else colab_cfg.get("checkpoint_every_n_steps", 0)
            ),
            checkpoint_on_exception=bool(
                colab_cfg.get("checkpoint_on_exception", True)
                if checkpoint_on_exception is None
                else checkpoint_on_exception
            ),
            validation_every_n_epochs=int(
                validation_every_n_epochs
                if validation_every_n_epochs is not None
                else colab_cfg.get("validation_every_n_epochs", 1)
            ),
        )
        session_holder["session"] = session
        return session, batch_recorder, checkpoint_records

    def _execute_training_session(
        self,
        *,
        session: TrainingSessionLike,
        batch_recorder: BatchMetricsRecorder,
        adapter: IndependentCropAdapter,
        loaders: JsonDict,
        loader_sizes: Dict[str, int],
    ) -> _TrainingExecutionStage:
        from src.workflows.training_runtime import execute_training_session_logic as _exec_logic

        (
            history_payload,
            best_state_restored,
            calibration_split_name,
            calibration_loader,
            ood_calibration,
            trainer,
        ) = _exec_logic(session, batch_recorder, adapter, loaders, loader_sizes)
        return _TrainingExecutionStage(
            history_payload=history_payload,
            best_state_restored=best_state_restored,
            calibration_split_name=calibration_split_name,
            calibration_loader=calibration_loader,
            ood_calibration=ood_calibration,
            trainer=trainer,
        )

    @staticmethod
    def _configure_optional_training_pools(*, trainer: Any, loaders: Dict[str, Any]) -> None:
        if trainer is None:
            return
        set_oe_loader = getattr(trainer, "set_oe_loader", None)
        if callable(set_oe_loader):
            set_oe_loader(loaders.get("oe"))

    @staticmethod
    def _build_oe_context(*, trainer: Any, loaders: Dict[str, Any]) -> Dict[str, Any]:
        if trainer is None:
            return {}
        config = getattr(trainer, "config", None)
        if config is None:
            return {}
        oe_loader = loaders.get("oe")
        dataset = getattr(oe_loader, "dataset", None) if oe_loader is not None else None
        split_root = getattr(dataset, "split_root", None)
        return {
            "enabled": bool(getattr(config, "oe_enabled", False)),
            "loss_weight": float(getattr(config, "oe_loss_weight", 0.0)),
            "target": str(getattr(config, "oe_target", "uniform")),
            "sample_count": int(loader_size(oe_loader)) if oe_loader is not None else 0,
            "source_root": str(split_root) if split_root is not None else str(getattr(config, "oe_root", "") or ""),
        }

    @staticmethod
    def _build_classifier_rebalance_log_priors(trainer: Any) -> Optional[torch.Tensor]:
        runtime = dict(getattr(trainer, "class_balance_runtime", {}) or {})
        counts_by_class = dict(runtime.get("resolved_class_counts", {}) or {})
        class_to_idx = dict(getattr(trainer, "class_to_idx", {}) or {})
        if not counts_by_class or not class_to_idx:
            return None
        ordered_counts: List[float] = []
        for class_name, _idx in sorted(class_to_idx.items(), key=lambda item: int(item[1])):
            count = float(counts_by_class.get(class_name, 0) or 0)
            if count <= 0.0:
                return None
            ordered_counts.append(count)
        priors = torch.tensor(ordered_counts, dtype=torch.float32)
        priors = priors / priors.sum().clamp_min(1e-6)
        return torch.log(priors.clamp_min(1e-6))

    def _run_classifier_rebalance(
        self,
        *,
        adapter: IndependentCropAdapter,
        trainer: Any,
        loaders: Dict[str, Any],
        training_cfg: Dict[str, Any],
        colab_cfg: Dict[str, Any],
        telemetry: Any,
        run_id: str,
    ) -> tuple[Any, Dict[str, Any], str, Any, Dict[str, Any], bool]:
        from src.workflows.training_runtime import run_classifier_rebalance_logic as _rebalance_logic

        (
            trainer,
            rebalance_history_payload,
            calibration_split_name,
            calibration_loader,
            rebalance_calibration,
            rebalance_best_state_restored,
        ) = _rebalance_logic(
            adapter=adapter,
            trainer=trainer,
            loaders=loaders,
            training_cfg=training_cfg,
            colab_cfg=colab_cfg,
            telemetry=telemetry,
            run_id=run_id,
        )
        if rebalance_history_payload:
            return (
                trainer,
                dict(rebalance_history_payload),
                calibration_split_name,
                calibration_loader,
                dict(rebalance_calibration),
                rebalance_best_state_restored,
            )
        return trainer, {}, calibration_split_name, calibration_loader, {}, False

    def _persist_training_artifacts(
        self,
        *,
        artifact_dir: Path,
        history_payload: Dict[str, Any],
        batch_metrics_csv: Path,
        telemetry: Any,
    ) -> Dict[str, Any]:
        from src.workflows.training_subroutines import persist_training_artifacts as _persist_training_artifacts_fn
        return _persist_training_artifacts_fn(
            artifact_dir=artifact_dir,
            history_payload=history_payload,
            batch_metrics_csv=batch_metrics_csv,
            telemetry=telemetry,
        )

    def _persist_evaluation_artifacts(
        self,
        *,
        artifact_dir: Path,
        trainer: Any,
        loader: Any,
        ood_loader: Any,
        detected_classes: List[str],
        telemetry: Any,
        run_id: str,
        crop_name: str,
        loader_sizes: Dict[str, int],
        split_name: str,
        artifact_subdir: str,
        telemetry_subdir: Optional[str] = None,
        evaluation_result: Any = None,
        requested_primary_score_method: str = "auto",
        selected_primary_score_method: str = "auto",
        selection_source: str = "",
        gate_targets: Optional[Dict[str, float]] = None,
        gate_auxiliary_ood_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        from src.workflows.training_subroutines import persist_evaluation_artifacts as _persist_evaluation_artifacts_fn
        return _persist_evaluation_artifacts_fn(
            artifact_dir=artifact_dir,
            trainer=trainer,
            loader=loader,
            ood_loader=ood_loader,
            detected_classes=detected_classes,
            telemetry=telemetry,
            run_id=run_id,
            crop_name=crop_name,
            loader_sizes=loader_sizes,
            split_name=split_name,
            artifact_subdir=artifact_subdir,
            telemetry_subdir=telemetry_subdir,
            evaluation_result=evaluation_result,
            requested_primary_score_method=requested_primary_score_method,
            selected_primary_score_method=selected_primary_score_method,
            selection_source=selection_source,
            gate_targets=gate_targets,
            gate_auxiliary_ood_diagnostics=gate_auxiliary_ood_diagnostics,
        )

    @staticmethod
    def _evaluate_split(
        *,
        trainer: Any,
        loader: Any,
        ood_loader: Any,
    ) -> Any:
        if trainer is None or loader_size(loader) <= 0:
            return None
        return evaluate_model_with_artifact_metrics(trainer, loader, ood_loader=ood_loader)

    @classmethod
    def _collect_split_evaluations(
        cls,
        *,
        trainer: Any,
        loaders: Dict[str, Any],
    ) -> _SplitEvaluationStage:
        return _SplitEvaluationStage(
            val=cls._evaluate_split(trainer=trainer, loader=loaders.get("val"), ood_loader=loaders.get("ood")),
            test=cls._evaluate_split(trainer=trainer, loader=loaders.get("test"), ood_loader=loaders.get("ood")),
        )

    def _persist_split_artifacts(
        self,
        *,
        artifact_dir: Path,
        trainer: Any,
        loaders: Dict[str, Any],
        detected_classes: List[str],
        telemetry: Any,
        run_id: str,
        crop_name: str,
        loader_sizes: Dict[str, int],
        evaluation_results: Dict[str, Any],
        requested_primary_score_method: str,
        selected_primary_score_method: str,
        selection_source: str,
        gate_targets: Optional[Dict[str, float]] = None,
        gate_auxiliary_ood_diagnostics: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        from src.workflows.training_subroutines import persist_split_artifacts as _persist_split_artifacts_fn
        return _persist_split_artifacts_fn(
            artifact_dir=artifact_dir,
            trainer=trainer,
            loaders=loaders,
            detected_classes=detected_classes,
            telemetry=telemetry,
            run_id=run_id,
            crop_name=crop_name,
            loader_sizes=loader_sizes,
            evaluation_results=evaluation_results,
            requested_primary_score_method=requested_primary_score_method,
            selected_primary_score_method=selected_primary_score_method,
            selection_source=selection_source,
            gate_targets=gate_targets,
            gate_auxiliary_ood_diagnostics=gate_auxiliary_ood_diagnostics,
        )

    @staticmethod
    def _apply_primary_score_method_to_trainer(trainer: Any, primary_score_method: str) -> str:
        from src.workflows.training_subroutines import apply_primary_score_method_to_trainer as _apply_primary
        return _apply_primary(trainer, primary_score_method)

    @staticmethod
    def _apply_score_threshold_override_to_trainer(
        trainer: Any,
        *,
        primary_score_method: str,
        selected_threshold: Any,
    ) -> None:
        from src.workflows.training_subroutines import apply_score_threshold_override_to_trainer as _apply_threshold
        return _apply_threshold(trainer, primary_score_method=primary_score_method, selected_threshold=selected_threshold)

    @staticmethod
    def _build_evaluation_gate_targets(evaluation_cfg: Dict[str, Any]) -> Dict[str, float]:
        from src.workflows.training_subroutines import build_evaluation_gate_targets as _build_targets
        return _build_targets(evaluation_cfg)

    def _select_from_real_ood_dev(
        self,
        *,
        trainer: Any,
        loaders: Dict[str, Any],
        requested_method: str,
        fallback_method: str,
        target_fpr: float,
    ) -> Optional[_PrimaryScoreSelectionStage]:
        from src.workflows.training_subroutines import select_and_apply_real_ood_dev_policy

        selection = select_and_apply_real_ood_dev_policy(
            trainer=trainer,
            loaders=loaders,
            requested_method=requested_method,
            fallback_method=fallback_method,
            target_fpr=target_fpr,
            evaluate_fn=evaluate_model_with_artifact_metrics,
            selection_builder=build_real_ood_dev_selection,
        )
        if not selection:
            return None
        selected_method = str(selection["selected_primary_score_method"])
        return _PrimaryScoreSelectionStage(
            requested_method=requested_method,
            selected_method=selected_method,
            selection_source="real_ood_dev",
            selection_metadata=dict(selection),
        )

    def _resolve_ood_evidence(
        self,
        *,
        crop_name: str,
        detected_classes: List[str],
        loaders: Dict[str, Any],
        authoritative_artifacts: Dict[str, Any],
        artifact_dir: Path,
        run_id: str,
        num_epochs: Optional[int],
        telemetry: Any,
        min_classes: int,
    ) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        real_ood_present = loader_size(loaders.get("ood")) > 0
        if real_ood_present:
            ood_metrics = (
                dict(authoritative_artifacts.get("metric_gate", {}).get("metrics", {}))
                if authoritative_artifacts
                else {}
            )
            return "real_ood_split", ood_metrics, {}

        self._emit_telemetry(
            telemetry,
            "ood_benchmark_planned",
            {
                "run_id": run_id,
                "crop_name": crop_name,
                "estimated_fold_trainings": int(len(detected_classes)),
                "class_count": int(len(detected_classes)),
                "primary_score_method": str(
                    self.config.get("training", {})
                    .get("continual", {})
                    .get("ood", {})
                    .get("primary_score_method", "auto")
                ),
                "min_classes": int(min_classes),
            },
            phase="evaluation",
        )

        ood_benchmark = run_leave_one_class_out_benchmark(
            crop_name=crop_name,
            class_names=detected_classes,
            loaders=loaders,
            config=self.config,
            device=self.device,
            artifact_root=artifact_dir,
            adapter_factory=IndependentCropAdapter,
            run_id=run_id,
            num_epochs=num_epochs,
            telemetry=telemetry,
            emit_event=lambda event_type, payload: self._emit_telemetry(
                telemetry,
                event_type,
                payload,
                phase="evaluation",
            ),
            min_classes=int(min_classes),
        )
        return "held_out_benchmark", dict(ood_benchmark.get("metrics", {})), ood_benchmark

    def run(
        self,
        *,
        crop_name: str,
        part_name: Optional[str] = None,
        data_dir: str | Path,
        output_dir: str | Path,
        class_names: Optional[Sequence[str]] = None,
        num_epochs: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        use_cache: bool = True,
        checkpoint_manager: Any = None,
        telemetry: Any = None,
        observers: Optional[Iterable[Observer]] = None,
        stop_policy: Optional[Callable[[], bool]] = None,
        checkpoint_every_n_steps: Optional[int] = None,
        checkpoint_on_exception: Optional[bool] = None,
        validation_every_n_epochs: Optional[int] = None,
        sampler: Optional[str] = None,
        error_policy: Optional[str] = None,
        run_id: str = "",
        dataset_lineage: Optional[Mapping[str, Any]] = None,
        require_immutable_dataset_lineage: bool = False,
    ) -> TrainingWorkflowResult:
        crop_name = str(crop_name).strip().lower()
        if not crop_name:
            raise ValueError("crop_name must not be empty")
        resolved_data_dir = Path(data_dir)
        resolved_output_dir = Path(output_dir)

        run_setup = prepare_training_run(
            config=self.config,
            device=self.device,
            crop_name=crop_name,
            part_name=training_utils.normalize_part_name(part_name) if part_name is not None else None,
            data_dir=resolved_data_dir,
            class_names=class_names,
            num_workers=num_workers,
            pin_memory=pin_memory,
            use_cache=use_cache,
            sampler=sampler,
            error_policy=error_policy,
            run_id=run_id,
            loader_factory=create_training_loaders,
            adapter_factory=IndependentCropAdapter,
        )
        training_cfg = run_setup.training_cfg
        colab_cfg = run_setup.colab_cfg
        run_id = run_setup.run_id
        loaders = run_setup.loaders
        loader_sizes = dict(run_setup.loader_sizes)
        loader_batch_counts = dict(run_setup.loader_batch_counts)
        detected_classes = list(run_setup.detected_classes)
        runtime_dataset_key = str(run_setup.runtime_dataset_key)
        runtime_crop_root = Path(run_setup.runtime_crop_root)
        runtime_dataset_resolution_source = str(run_setup.runtime_dataset_resolution_source)
        runtime_manifest_payload = training_utils.read_dataset_manifest_payload(runtime_crop_root)
        runtime_manifest_path = runtime_crop_root / "split_manifest.json"
        runtime_manifest_context = training_utils.collect_dataset_manifest_context(runtime_crop_root)
        split_manifest_context = dict(runtime_manifest_context.get("split_manifest.json", {}))
        split_manifest_sha256 = str(split_manifest_context.get("sha256", "") or "")
        if not split_manifest_sha256:
            split_manifest_sha256 = canonical_sha256(
                {"dataset_key": runtime_dataset_key, "split_manifest": "missing_local_legacy"}
            )
        resolved_dataset_lineage = dict(dataset_lineage or {})
        if not resolved_dataset_lineage:
            resolved_dataset_lineage = build_local_legacy_lineage(
                dataset_key=runtime_dataset_key,
                split_manifest_sha256=split_manifest_sha256,
                compatibility_reason="local_workflow_compatibility",
            )
            if not runtime_manifest_path.is_file():
                resolved_dataset_lineage["split_manifest_missing"] = True
        if str(resolved_dataset_lineage.get("dataset_key", "") or "") != runtime_dataset_key:
            raise ValueError("dataset_lineage dataset_key does not match the resolved runtime dataset")
        if str(resolved_dataset_lineage.get("split_manifest_sha256", "") or "") != split_manifest_sha256:
            raise ValueError(f"dataset_lineage does not match {runtime_manifest_path}")
        adapter = run_setup.adapter
        resolved_part_name = (
            training_utils.normalize_part_name(part_name)
            if part_name is not None
            else training_utils.resolve_part_name(
                runtime_dataset_key=runtime_dataset_key,
                manifest_payload=runtime_manifest_payload,
            )
        )
        run_output_dir = build_adapter_bundle_root(resolved_output_dir, crop_name, resolved_part_name)
        if hasattr(adapter, "part_name"):
            adapter.part_name = resolved_part_name
        run_created_at = datetime.now(timezone.utc).isoformat()
        split_class_counts = {
            str(split_name): {str(class_name): int(count) for class_name, count in counts.items()}
            for split_name, counts in dict(run_setup.split_class_counts).items()
        }
        sampler_runtime = dict(run_setup.sampler_runtime)
        class_balance_runtime = dict(run_setup.class_balance_runtime)
        artifact_dir = run_output_dir / "training_metrics"
        session, batch_recorder, checkpoint_records = self._create_training_session(
            adapter=adapter,
            loaders=loaders,
            num_epochs=num_epochs,
            colab_cfg=colab_cfg,
            run_id=run_id,
            telemetry=telemetry,
            checkpoint_manager=checkpoint_manager,
            observers=observers,
            stop_policy=stop_policy,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            checkpoint_on_exception=checkpoint_on_exception,
            validation_every_n_epochs=validation_every_n_epochs,
            artifact_dir=artifact_dir,
        )
        trainer_for_session = getattr(session, "trainer", None)
        self._configure_optional_training_pools(trainer=trainer_for_session, loaders=loaders)
        oe_context = self._build_oe_context(trainer=trainer_for_session, loaders=loaders)
        if oe_context.get("enabled") and int(oe_context.get("sample_count", 0)) <= 0:
            raise ValueError(
                "training.continual.ood.oe_enabled requires a separate OE pool under "
                "runtime_dataset/oe or an explicit training.continual.ood.oe_root."
            )

        self._emit_telemetry(
            telemetry,
            "training_workflow_started",
            {
                "run_id": run_id,
                "crop_name": crop_name,
                "data_dir": str(resolved_data_dir),
                "output_dir": str(run_output_dir),
            },
            phase="training",
        )

        execution_stage = self._execute_training_session(
            session=session,
            batch_recorder=batch_recorder,
            adapter=adapter,
            loaders=loaders,
            loader_sizes=loader_sizes,
        )
        history_payload = execution_stage.history_payload
        best_state_restored = execution_stage.best_state_restored
        calibration_split_name = execution_stage.calibration_split_name
        calibration_loader = execution_stage.calibration_loader
        ood_calibration = execution_stage.ood_calibration

        training_artifacts = self._persist_training_artifacts(
            artifact_dir=artifact_dir,
            history_payload=history_payload,
            batch_metrics_csv=batch_recorder.output_path,
            telemetry=telemetry,
        )
        trainer_for_artifacts = execution_stage.trainer
        split_evaluation_stage = self._collect_split_evaluations(
            trainer=trainer_for_artifacts,
            loaders=loaders,
        )
        split_evaluations = split_evaluation_stage.as_dict()
        pre_rebalance_metrics = {
            split_name: _evaluation_metrics_summary(payload)
            for split_name, payload in split_evaluations.items()
            if payload is not None
        }
        classifier_rebalance_cfg = dict(training_cfg.get("classifier_rebalance", {}))
        classifier_rebalance_context: Dict[str, Any] = {
            "enabled": bool(classifier_rebalance_cfg.get("enabled", False)),
            "objective": str(classifier_rebalance_cfg.get("objective", "") or ""),
            "sampler": str(classifier_rebalance_cfg.get("sampler", "") or ""),
            "epochs": int(classifier_rebalance_cfg.get("epochs", 0) or 0),
            "pre_rebalance_metrics": pre_rebalance_metrics,
        }
        if classifier_rebalance_context["enabled"]:
            (
                trainer_for_artifacts,
                rebalance_history_payload,
                calibration_split_name,
                calibration_loader,
                ood_calibration,
                rebalance_best_state_restored,
            ) = self._run_classifier_rebalance(
                adapter=adapter,
                trainer=trainer_for_artifacts,
                loaders=loaders,
                training_cfg=training_cfg,
                colab_cfg=colab_cfg,
                telemetry=telemetry,
                run_id=run_id,
            )
            if rebalance_history_payload:
                classifier_rebalance_context["history"] = rebalance_history_payload
            split_evaluation_stage = self._collect_split_evaluations(
                trainer=trainer_for_artifacts,
                loaders=loaders,
            )
            split_evaluations = split_evaluation_stage.as_dict()
            classifier_rebalance_context["post_rebalance_metrics"] = {
                split_name: _evaluation_metrics_summary(payload)
                for split_name, payload in split_evaluations.items()
                if payload is not None
            }
            best_state_restored = bool(best_state_restored or rebalance_best_state_restored)
        real_ood_present = loader_size(loaders.get("ood")) > 0
        ood_cfg = dict(training_cfg.get("ood", {}))
        evaluation_cfg = dict(training_cfg.get("evaluation", {}))
        gate_targets = self._build_evaluation_gate_targets(evaluation_cfg)
        gate_auxiliary_ood_diagnostics = bool(
            evaluation_cfg.get("gate_auxiliary_ood_diagnostics", False)
        )
        primary_score_stage = _PrimaryScoreSelectionStage(
            requested_method=normalize_requested_primary_score_method(
                ood_cfg.get("primary_score_method", "auto")
            ),
            selected_method=resolve_runtime_primary_score_method(
                ood_cfg.get("primary_score_method", "auto")
            ),
            selection_source="configured",
        )
        selected_primary_score_method = self._apply_primary_score_method_to_trainer(
            trainer_for_artifacts,
            primary_score_stage.selected_method,
        )
        primary_score_stage = _PrimaryScoreSelectionStage(
            requested_method=primary_score_stage.requested_method,
            selected_method=selected_primary_score_method,
            selection_source=primary_score_stage.selection_source,
            selection_metadata=dict(primary_score_stage.selection_metadata),
        )
        if real_ood_present and is_auto_primary_score_method(primary_score_stage.requested_method):
            real_dev_stage = None
            if bool(ood_cfg.get("real_dev_selection_enabled", True)):
                real_dev_stage = self._select_from_real_ood_dev(
                    trainer=trainer_for_artifacts,
                    loaders=loaders,
                    requested_method=primary_score_stage.requested_method,
                    fallback_method=primary_score_stage.selected_method,
                    target_fpr=float(ood_cfg.get("real_dev_target_fpr", gate_targets.get("ood_false_positive_rate", 0.05))),
                )
            if real_dev_stage is not None:
                primary_score_stage = real_dev_stage
                split_evaluation_stage = self._collect_split_evaluations(
                    trainer=trainer_for_artifacts,
                    loaders=loaders,
                )
                split_evaluations = split_evaluation_stage.as_dict()
            else:
                primary_score_stage = _PrimaryScoreSelectionStage(
                    requested_method=primary_score_stage.requested_method,
                    selected_method=primary_score_stage.selected_method,
                    selection_source="real_ood_guardrail_no_dev",
                    selection_metadata=dict(primary_score_stage.selection_metadata),
                )

        ood_stage = _OODEvidenceStage(source="", metrics={}, benchmark={})
        if not real_ood_present:
            ood_evidence_source, ood_evidence_metrics, ood_benchmark = self._resolve_ood_evidence(
                crop_name=crop_name,
                detected_classes=detected_classes,
                loaders=loaders,
                authoritative_artifacts={},
                artifact_dir=artifact_dir,
                run_id=run_id,
                num_epochs=num_epochs,
                telemetry=telemetry,
                min_classes=int(evaluation_cfg.get("ood_benchmark_min_classes", 3)),
            )
            ood_stage = _OODEvidenceStage(
                source=ood_evidence_source,
                metrics=dict(ood_evidence_metrics),
                benchmark=dict(ood_benchmark),
            )
            if ood_stage.source == "held_out_benchmark" and is_auto_primary_score_method(
                primary_score_stage.requested_method
            ):
                benchmark_method_comparison = dict(ood_stage.benchmark.get("method_comparison", {}))
                if not benchmark_method_comparison:
                    benchmark_method_comparison = dict(ood_stage.benchmark.get("method_comparison_metrics", {}))
                if benchmark_method_comparison:
                    selected_primary_score_method = self._apply_primary_score_method_to_trainer(
                        trainer_for_artifacts,
                        select_best_ood_score_method(
                            benchmark_method_comparison,
                            fallback=primary_score_stage.selected_method,
                        ),
                    )
                    primary_score_stage = _PrimaryScoreSelectionStage(
                        requested_method=primary_score_stage.requested_method,
                        selected_method=selected_primary_score_method,
                        selection_source="held_out_benchmark",
                        selection_metadata=dict(primary_score_stage.selection_metadata),
                    )

        requested_primary_score_method = primary_score_stage.requested_method
        selection_source = primary_score_stage.selection_source
        record_primary_score_selection_payload(
            ood_calibration,
            requested_primary_score_method=primary_score_stage.requested_method,
            selected_primary_score_method=primary_score_stage.selected_method,
            selection_source=primary_score_stage.selection_source,
            selection_metadata=dict(primary_score_stage.selection_metadata),
        )
        split_artifacts = self._persist_split_artifacts(
            artifact_dir=artifact_dir,
            trainer=trainer_for_artifacts,
            loaders=loaders,
            detected_classes=detected_classes,
            telemetry=telemetry,
            run_id=run_id,
            crop_name=crop_name,
            loader_sizes=loader_sizes,
            evaluation_results=split_evaluations,
            requested_primary_score_method=requested_primary_score_method,
            selected_primary_score_method=primary_score_stage.selected_method,
            selection_source=primary_score_stage.selection_source,
            gate_targets=gate_targets,
            gate_auxiliary_ood_diagnostics=gate_auxiliary_ood_diagnostics,
        )
        validation_artifacts = split_artifacts["val"]
        test_artifacts = split_artifacts["test"]
        authoritative_split, authoritative_artifacts = select_authoritative_artifacts_payload(
            validation_artifacts,
            test_artifacts,
            calibration_split_name=calibration_split_name,
        )
        authoritative_evaluation_split, authoritative_evaluation = select_authoritative_evaluation_payload(
            split_evaluations.get("val"),
            split_evaluations.get("test"),
            calibration_split_name=calibration_split_name,
        )
        if real_ood_present:
            ood_evidence_source, ood_evidence_metrics, ood_benchmark = self._resolve_ood_evidence(
                crop_name=crop_name,
                detected_classes=detected_classes,
                loaders=loaders,
                authoritative_artifacts=authoritative_artifacts,
                artifact_dir=artifact_dir,
                run_id=run_id,
                num_epochs=num_epochs,
                telemetry=telemetry,
                min_classes=int(evaluation_cfg.get("ood_benchmark_min_classes", 3)),
            )
            ood_stage = _OODEvidenceStage(
                source=ood_evidence_source,
                metrics=dict(ood_evidence_metrics),
                benchmark=dict(ood_benchmark),
            )
        ood_evidence_source = ood_stage.source
        ood_evidence_metrics = dict(ood_stage.metrics)
        ood_benchmark = dict(ood_stage.benchmark)
        ood_method_comparison_context = _build_ood_method_comparison_context(
            authoritative_artifacts=authoritative_artifacts,
            ood_evidence_source=ood_evidence_source,
            ood_benchmark=ood_benchmark,
        )
        readiness_context = build_production_readiness_context_payload(
            run_id=run_id,
            crop_name=crop_name,
            loader_sizes=loader_sizes,
            loader_batch_counts=loader_batch_counts,
            split_class_counts=split_class_counts,
            calibration_split_name=calibration_split_name,
            best_state_restored=best_state_restored,
            classification_split=authoritative_split,
            requested_primary_score_method=requested_primary_score_method,
            selected_primary_score_method=primary_score_stage.selected_method,
            selection_source=primary_score_stage.selection_source,
            selection_metadata=dict(primary_score_stage.selection_metadata),
            ood_benchmark=ood_benchmark,
            ood_method_comparison=ood_method_comparison_context,
            oe_context=oe_context,
            classifier_rebalance=classifier_rebalance_context,
        )
        readiness_context["gate_auxiliary_ood_diagnostics"] = gate_auxiliary_ood_diagnostics
        readiness_context["dataset_lineage"] = dict(resolved_dataset_lineage)
        readiness_context["require_immutable_dataset_lineage"] = bool(require_immutable_dataset_lineage)
        record_adapter_export_metadata_payload(
            adapter,
            ood_calibration=ood_calibration,
            calibration_split_name=calibration_split_name,
            calibration_loader_size=loader_size(calibration_loader),
            authoritative_split=authoritative_split,
            ood_evidence_source=ood_evidence_source,
            crop_name=crop_name,
            part_name=resolved_part_name,
            requested_primary_score_method=requested_primary_score_method,
            selected_primary_score_method=primary_score_stage.selected_method,
            selection_source=primary_score_stage.selection_source,
            best_state_restored=best_state_restored,
            selection_metadata=dict(primary_score_stage.selection_metadata),
            dataset_lineage=resolved_dataset_lineage,
        )
        adapter_dir = adapter.save_adapter(str(run_output_dir))
        readiness_artifacts = persist_production_readiness_artifact(
            artifact_root=artifact_dir,
            classification_metric_gate=(
                dict(authoritative_artifacts.get("metric_gate", {}))
                if isinstance(authoritative_artifacts, dict)
                else None
            ),
            classification_split=authoritative_split,
            ood_evidence_source=ood_evidence_source,
            ood_metrics=ood_evidence_metrics,
            targets=gate_targets,
            context=readiness_context,
            require_ood=bool(evaluation_cfg.get("require_ood_for_gate", True)),
            telemetry=telemetry,
        )
        production_readiness = dict(readiness_artifacts.get("payload", {}))
        self._emit_telemetry(
            telemetry,
            "production_readiness_ready",
            {
                "run_id": run_id,
                "crop_name": crop_name,
                "status": production_readiness.get("status"),
                "passed": production_readiness.get("passed"),
                "ood_evidence_source": production_readiness.get("ood_evidence_source"),
            },
            phase="artifact",
        )
        behavioral_artifacts = persist_behavioral_acceptance_artifact(
            artifact_root=artifact_dir,
            classification_metrics=(
                dict(authoritative_artifacts.get("metric_gate", {}).get("metrics", {}))
                if isinstance(authoritative_artifacts, dict)
                else None
            ),
            prediction_rows=(
                list(authoritative_evaluation.prediction_rows) if authoritative_evaluation is not None else []
            ),
            context={
                "run_id": run_id,
                "crop_name": crop_name,
                "part_name": resolved_part_name,
                "classification_split": authoritative_split,
                "ood_scope": "same_crop_same_part_unsupported_disease",
                "router_scope_excluded": ["wrong_part", "other_plant", "non_plant"],
                "production_readiness_status": production_readiness.get("status"),
            },
            telemetry=telemetry,
        )
        behavioral_acceptance = dict(behavioral_artifacts.get("payload", {}))
        self._emit_telemetry(
            telemetry,
            "adapter_behavioral_acceptance_ready",
            {
                "run_id": run_id,
                "crop_name": crop_name,
                "part_name": resolved_part_name,
                "status": behavioral_acceptance.get("status"),
                "passed": behavioral_acceptance.get("pass"),
            },
            phase="artifact",
        )

        summary_payload = build_training_summary_payload_dict(
            run_id=run_id,
            crop_name=crop_name,
            part_name=resolved_part_name,
            detected_classes=detected_classes,
            dataset_key=runtime_dataset_key,
            loader_sizes=loader_sizes,
            loader_batch_counts=loader_batch_counts,
            split_class_counts=split_class_counts,
            class_balance=class_balance_runtime,
            adapter_dir=adapter_dir,
            artifact_dir=str(artifact_dir),
            created_at=run_created_at,
            surface="workflow",
            checkpoint_records=checkpoint_records,
            ood_calibration=ood_calibration,
            history_payload=history_payload,
            calibration_split_name=calibration_split_name,
            ood_evidence_source=ood_evidence_source,
            ood_benchmark=ood_benchmark,
            behavioral_acceptance=behavioral_acceptance,
            production_readiness=production_readiness,
            best_state_restored=best_state_restored,
            requested_primary_score_method=requested_primary_score_method,
            selected_primary_score_method=primary_score_stage.selected_method,
            primary_score_selection_source=selection_source,
            primary_score_selection=dict(primary_score_stage.selection_metadata),
            loss_name=str(training_cfg.get("optimization", {}).get("loss_name", "cross_entropy")),
            logitnorm_tau=float(training_cfg.get("optimization", {}).get("logitnorm_tau", 1.0)),
            final_metrics=_collect_final_metrics(history_payload),
            classifier_rebalance=classifier_rebalance_context,
        )
        summary_artifacts = persist_training_summary_artifact(
            artifact_root=artifact_dir,
            summary_payload=summary_payload,
            telemetry=telemetry,
        )
        repo_root = Path(__file__).resolve().parents[2]
        run_context_payload = {
            "schema_version": "v1_training_run_context",
            "run_id": run_id,
            "created_at": run_created_at,
            "surface": "workflow",
            "crop_name": crop_name,
            "part_name": resolved_part_name,
            "device": self.device,
            "python_version": sys.version.split()[0],
            "data_dir": str(resolved_data_dir),
            "output_dir": str(run_output_dir),
            "output_root": str(resolved_output_dir),
            "artifact_dir": str(artifact_dir),
            "adapter_dir": str(adapter_dir),
            "resolved_config": dict(self.config),
            "git": training_utils.collect_git_context(repo_root),
            "package_versions": training_utils.collect_package_versions(),
            "dataset": {
                "crop_root": str(runtime_crop_root.resolve()),
                "crop_name": crop_name,
                "part_name": resolved_part_name,
                "dataset_key": runtime_dataset_key,
                "resolution_source": runtime_dataset_resolution_source,
                "manifests": runtime_manifest_context,
                "lineage": dict(resolved_dataset_lineage),
            },
            "loader_sizes": dict(loader_sizes),
            "loader_batch_counts": dict(loader_batch_counts),
            "split_class_counts": dict(split_class_counts),
            "class_balance": dict(class_balance_runtime),
            "training_runtime": {
                "calibration_split_name": calibration_split_name,
                "best_state_restored": bool(best_state_restored),
                "ood_requested_primary_score_method": requested_primary_score_method,
                "ood_primary_score_method": primary_score_stage.selected_method,
                "ood_primary_score_selection_source": selection_source,
                "ood_primary_score_selection": dict(primary_score_stage.selection_metadata),
                "ood_evidence_source": ood_evidence_source,
                "train_sampler": sampler_runtime,
                "oe": oe_context,
                "classifier_rebalance": classifier_rebalance_context,
            },
            "production_readiness_status": production_readiness.get("status"),
            "adapter_behavioral_acceptance_status": behavioral_acceptance.get("status"),
            "adapter_behavioral_acceptance_pass": behavioral_acceptance.get("pass"),
        }
        run_context_artifacts = persist_training_run_context_artifact(
            artifact_root=artifact_dir,
            context_payload=run_context_payload,
            telemetry=telemetry,
        )
        experiment_manifest = build_experiment_manifest(
            summary_payload=summary_payload,
            run_context_payload=run_context_payload,
            artifact_root=artifact_dir,
            explicit_surface="workflow",
            created_at=run_created_at,
            record_quality="canonical",
        )
        optimization_record = build_optimization_record(
            summary_payload=summary_payload,
            run_context_payload=run_context_payload,
            production_readiness_payload=production_readiness,
            authoritative_artifacts=authoritative_artifacts,
            artifact_root=artifact_dir,
            explicit_surface="workflow",
            created_at=run_created_at,
            record_quality="canonical",
        )
        traceability_artifacts = persist_traceability_artifacts(
            artifact_root=artifact_dir,
            experiment_manifest=experiment_manifest,
            optimization_record=optimization_record,
            telemetry=telemetry,
        )
        training_artifacts = {**training_artifacts, **run_context_artifacts, **traceability_artifacts}
        artifact_payload = build_artifact_payload(
            training_artifacts=training_artifacts,
            validation_artifacts=validation_artifacts,
            test_artifacts=test_artifacts,
            ood_benchmark=ood_benchmark,
            readiness_artifacts=readiness_artifacts,
            behavioral_artifacts=behavioral_artifacts,
            summary_artifacts=summary_artifacts,
        )

        result = TrainingWorkflowResult(
            run_id=run_id,
            crop_name=crop_name,
            part_name=resolved_part_name,
            class_names=detected_classes,
            history=history_payload,
            loader_sizes=loader_sizes,
            adapter_dir=adapter_dir,
            artifact_dir=artifact_dir,
            artifacts=artifact_payload,
            ood_calibration=ood_calibration,
            checkpoint_records=checkpoint_records,
            ood_evidence_source=ood_evidence_source,
            ood_benchmark=ood_benchmark,
            behavioral_acceptance=behavioral_acceptance,
            production_readiness=production_readiness,
        )

        self._emit_telemetry(
            telemetry,
            "training_artifacts_ready",
            {"artifact_dir": str(artifact_dir), "artifacts": artifact_payload},
            phase="artifact",
        )
        self._emit_telemetry(
            telemetry,
            "training_workflow_completed",
            result.to_dict(),
            phase="training",
        )

        return result
