from pathlib import Path
from typing import Any, Dict, List, Optional

from src.training.services.ood_score_selection import (
    apply_primary_score_method_to_evaluation,
    build_real_ood_dev_selection,
    is_auto_primary_score_method,
    resolve_runtime_primary_score_method,
)
from src.training.services.reporting import (
    load_batch_metrics_history,
    persist_batch_metrics_artifacts,
    persist_training_history_artifacts,
    persist_training_results_figure,
    persist_validation_artifacts,
)
from src.training.validation import evaluate_model_with_artifact_metrics
from src.utils.training_helpers import loader_size


def persist_training_artifacts(*, artifact_dir: Path, history_payload: Dict[str, Any], batch_metrics_csv: Path, telemetry: Any) -> Dict[str, Any]:
    batch_history = load_batch_metrics_history(batch_metrics_csv)
    return {
        **persist_training_history_artifacts(
            artifact_root=artifact_dir,
            history_snapshot=history_payload,
            telemetry=telemetry,
        ),
        **persist_batch_metrics_artifacts(
            artifact_root=artifact_dir,
            batch_metrics_csv=batch_metrics_csv,
            telemetry=telemetry,
        ),
        **persist_training_results_figure(
            artifact_root=artifact_dir,
            history_snapshot=history_payload,
            batch_history=batch_history,
            telemetry=telemetry,
        ),
    }


def persist_evaluation_artifacts(
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
    if trainer is None or loader_size(loader) <= 0:
        return {}

    if evaluation_result is None:
        evaluation_result = evaluate_model_with_artifact_metrics(trainer, loader, ood_loader=ood_loader)
    if evaluation_result is None:
        return {}
    evaluation_result = apply_primary_score_method_to_evaluation(
        evaluation_result,
        selected_primary_score_method,
        requested_primary_score_method=requested_primary_score_method,
        selection_source=selection_source,
    )
    if evaluation_result is None:
        return {}

    require_ood = bool(getattr(getattr(trainer, "config", None), "evaluation_require_ood_for_gate", False))
    emit_metric_gate = bool(getattr(getattr(trainer, "config", None), "evaluation_emit_ood_gate", True))
    metric_context = {
        "run_id": run_id,
        "crop_name": crop_name,
        "split_name": split_name,
        "num_classes": len(detected_classes),
        "loader_sizes": loader_sizes,
        **dict(evaluation_result.context),
    }
    return persist_validation_artifacts(
        artifact_root=artifact_dir,
        y_true=evaluation_result.y_true,
        y_pred=evaluation_result.y_pred,
        classes=detected_classes,
        telemetry=telemetry,
        artifact_subdir=artifact_subdir,
        telemetry_subdir=telemetry_subdir,
        gate_targets=gate_targets,
        require_ood=require_ood,
        gate_auxiliary_ood_diagnostics=gate_auxiliary_ood_diagnostics,
        emit_metric_gate=emit_metric_gate,
        ood_labels=evaluation_result.ood_labels,
        ood_scores=evaluation_result.ood_scores,
        ood_scores_by_method=evaluation_result.ood_scores_by_method,
        sure_ds_f1=evaluation_result.sure_ds_f1,
        conformal_empirical_coverage=evaluation_result.conformal_empirical_coverage,
        conformal_avg_set_size=evaluation_result.conformal_avg_set_size,
        ood_type_breakdown=evaluation_result.ood_type_breakdown,
        context=metric_context,
        prediction_rows=evaluation_result.prediction_rows,
    )


def persist_split_artifacts(
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
    split_specs = (
        ("val", loaders.get("val"), "validation"),
        ("test", loaders.get("test"), "test"),
    )
    result = {}
    for split_name, loader, artifact_subdir in split_specs:
        result[split_name] = persist_evaluation_artifacts(
            artifact_dir=artifact_dir,
            trainer=trainer,
            loader=loader,
            ood_loader=loaders.get("ood"),
            detected_classes=detected_classes,
            telemetry=telemetry,
            run_id=run_id,
            crop_name=crop_name,
            loader_sizes=loader_sizes,
            split_name=split_name,
            artifact_subdir=artifact_subdir,
            evaluation_result=evaluation_results.get(split_name),
            requested_primary_score_method=requested_primary_score_method,
            selected_primary_score_method=selected_primary_score_method,
            selection_source=selection_source,
            gate_targets=gate_targets,
            gate_auxiliary_ood_diagnostics=gate_auxiliary_ood_diagnostics,
        )
    return result


def apply_primary_score_method_to_trainer(trainer: Any, primary_score_method: str) -> str:
    resolved = resolve_runtime_primary_score_method(primary_score_method)
    if trainer is None:
        return resolved
    setter = getattr(trainer, "set_ood_primary_score_method", None)
    if callable(setter):
        return str(setter(resolved))
    config = getattr(trainer, "config", None)
    if config is not None and hasattr(config, "ood_primary_score_method"):
        setattr(config, "ood_primary_score_method", resolved)
    detector = getattr(trainer, "ood_detector", None)
    if detector is not None and hasattr(detector, "primary_score_method"):
        setattr(detector, "primary_score_method", resolved)
    return resolved


def apply_score_threshold_override_to_trainer(trainer: Any, *, primary_score_method: str, selected_threshold: Any) -> None:
    if selected_threshold is None or trainer is None:
        return
    detector = getattr(trainer, "ood_detector", None)
    setter = getattr(detector, "set_score_threshold_override", None)
    if callable(setter):
        setter(primary_score_method, float(selected_threshold))


def select_and_apply_real_ood_dev_policy(
    *,
    trainer: Any,
    loaders: Dict[str, Any],
    requested_method: str,
    fallback_method: str,
    target_fpr: float,
    evaluate_fn: Any = None,
    selection_builder: Any = None,
) -> Optional[Dict[str, Any]]:
    """Select and apply an OOD method/threshold using only ID dev plus OOD dev evidence."""
    if trainer is None or not is_auto_primary_score_method(requested_method):
        return None
    ood_dev_loader = loaders.get("ood_dev")
    if loader_size(ood_dev_loader) <= 0:
        return None
    id_dev_loader = loaders.get("val") if loader_size(loaders.get("val")) > 0 else loaders.get("train")
    if loader_size(id_dev_loader) <= 0:
        return None
    evaluator = evaluate_model_with_artifact_metrics if evaluate_fn is None else evaluate_fn
    dev_evaluation = evaluator(
        trainer,
        id_dev_loader,
        ood_loader=ood_dev_loader,
    )
    builder = build_real_ood_dev_selection if selection_builder is None else selection_builder
    selection = builder(
        dev_evaluation,
        fallback=fallback_method,
        target_fpr=float(target_fpr),
    )
    if not selection:
        return None
    selected_method = apply_primary_score_method_to_trainer(
        trainer,
        str(selection.get("selected_primary_score_method", fallback_method) or fallback_method),
    )
    selection["selected_primary_score_method"] = selected_method
    apply_score_threshold_override_to_trainer(
        trainer,
        primary_score_method=selected_method,
        selected_threshold=selection.get("selected_threshold"),
    )
    return selection


def build_evaluation_gate_targets(evaluation_cfg: Dict[str, Any]) -> Dict[str, float]:
    targets: Dict[str, float] = {}
    for config_key, target_key in (
        ("min_in_distribution_samples", "in_distribution_samples"),
        ("min_ood_samples", "ood_samples"),
        ("min_ood_samples_per_type", "ood_samples_per_type"),
    ):
        if config_key in evaluation_cfg:
            targets[target_key] = float(evaluation_cfg[config_key])
    return targets
