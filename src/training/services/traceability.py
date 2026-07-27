"""Canonical experiment-traceability helpers for training runs."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.guided_artifacts import refresh_training_guided_artifacts
from src.shared.artifacts import ArtifactStore
from src.shared.dict_utils import nested_dict as _nested_dict
from src.shared.json_utils import read_json

JsonDict = Dict[str, Any]
logger = logging.getLogger(__name__)

EXPERIMENT_MANIFEST_SCHEMA = "v1_training_experiment_manifest"
OPTIMIZATION_RECORD_SCHEMA = "v1_training_optimization_record"
OPTIMIZATION_PROFILE = "accuracy_plus_ood"
TRAINING_ENGINE = "continual_sd_lora"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_to_telemetry(telemetry: Any, source_path: Path, relative_path: str) -> None:
    if telemetry is None or not hasattr(telemetry, "copy_artifact_file"):
        return
    telemetry.copy_artifact_file(source_path, relative_path)


def _merge_registry_to_telemetry(telemetry: Any, registry_result: Mapping[str, Any] | None) -> None:
    if telemetry is None or not isinstance(registry_result, Mapping):
        return
    paths = {
        "trials_jsonl": "runs/_index/trials.jsonl",
        "latest_registry_json": "runs/_index/latest_registry.json",
        "pareto_inputs_json": "runs/_index/pareto_inputs.json",
        "pareto_frontiers_json": "runs/_index/pareto_frontiers.json",
        "automatic_wins_markdown": "runs/_index/automatic_wins.md",
        "bayesian_recommendations_json": "runs/_index/bayesian_recommendations.json",
    }
    for key, relative_path in paths.items():
        source_path = registry_result.get(key)
        if source_path:
            _copy_to_telemetry(telemetry, Path(str(source_path)), relative_path)


def _artifact_dir(root: Path, *parts: str) -> Path:
    target = Path(root)
    for part in parts:
        target /= part
    target.mkdir(parents=True, exist_ok=True)
    return target


def _resolve_runs_root_for_registry(artifact_root: str | Path) -> Path | None:
    root = Path(artifact_root).resolve()
    for parent in (root, *root.parents):
        if parent.name == "runs":
            return parent
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "runs"
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


def _try_refresh_run_registry(
    *,
    artifact_root: str | Path,
    telemetry: Any = None,
    enable_bayesian_proposals: bool = True,
) -> Dict[str, Any]:
    runs_root = _resolve_runs_root_for_registry(artifact_root)
    if runs_root is None:
        return {}
    try:
        from src.training.services.run_registry import build_run_registry

        result = build_run_registry(
            runs_root=runs_root,
            enable_bayesian_proposals=bool(enable_bayesian_proposals),
        )
    except Exception as exc:
        logger.warning("Automatic run-registry refresh failed for artifact_root=%s: %s", artifact_root, exc)
        return {}
    _merge_registry_to_telemetry(telemetry, result)
    return {
        "runs_root": str(runs_root),
        "latest_registry_json": str(result.get("latest_registry_json", "")) if result.get("latest_registry_json") else "",
        "trials_jsonl": str(result.get("trials_jsonl", "")) if result.get("trials_jsonl") else "",
        "pareto_inputs_json": str(result.get("pareto_inputs_json", "")) if result.get("pareto_inputs_json") else "",
        "pareto_frontiers_json": str(result.get("pareto_frontiers_json", "")) if result.get("pareto_frontiers_json") else "",
        "automatic_wins_markdown": str(result.get("automatic_wins_markdown", "")) if result.get("automatic_wins_markdown") else "",
        "bayesian_recommendations_json": str(result.get("bayesian_recommendations_json", "")) if result.get("bayesian_recommendations_json") else "",
    }





def _value_from_candidates(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, "", False):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _resolve_surface(summary_payload: Mapping[str, Any] | None, explicit_surface: str | None = None) -> str:
    if explicit_surface:
        return str(explicit_surface)
    summary = dict(summary_payload or {})
    surface = str(summary.get("surface", "") or "").strip()
    if surface:
        return surface
    notebook_surface = str(summary.get("notebook_surface", "") or "").strip()
    if notebook_surface.endswith(
        ("2_train_continual_sd_lora_adapter.ipynb", "2_interactive_adapter_training.ipynb")
    ):
        return "notebook_2"
    return "workflow"


def _split_dataset_key(dataset_key: str) -> tuple[str, str]:
    key = str(dataset_key or "").strip().lower()
    if "__" not in key:
        return key, "unspecified"
    crop_name, part_name = key.split("__", 1)
    return crop_name, (part_name or "unspecified")


def _extract_dataset_manifest_entry(run_context_payload: Mapping[str, Any] | None) -> JsonDict:
    run_context = dict(run_context_payload or {})
    dataset = _nested_dict(run_context, "dataset")
    manifests = _nested_dict(dataset, "manifests")
    manifest = manifests.get("split_manifest.json")
    return dict(manifest) if isinstance(manifest, Mapping) else {}


def _resolve_dataset_identity(
    *,
    summary_payload: Mapping[str, Any] | None,
    run_context_payload: Mapping[str, Any] | None,
) -> JsonDict:
    summary = dict(summary_payload or {})
    run_context = dict(run_context_payload or {})
    dataset_context = _nested_dict(run_context, "dataset")
    dataset_lineage = _nested_dict(dataset_context, "lineage")
    manifest_entry = _extract_dataset_manifest_entry(run_context)
    dataset_roots = _nested_dict(summary, "dataset_roots")

    dataset_key = str(
        _value_from_candidates(
            summary.get("dataset_key"),
            dataset_context.get("dataset_key"),
            manifest_entry.get("dataset_key"),
            dataset_roots.get("runtime_dataset_key"),
            dataset_roots.get("selected_dataset_name"),
        )
        or ""
    )
    crop_guess, part_guess = _split_dataset_key(dataset_key)
    crop_name = str(
        _value_from_candidates(
            summary.get("crop_name"),
            run_context.get("crop_name"),
            dataset_context.get("crop_name"),
            manifest_entry.get("crop_name"),
            crop_guess,
        )
        or ""
    ).strip().lower()
    part_name = str(
        _value_from_candidates(
            summary.get("part_name"),
            dataset_context.get("part_name"),
            manifest_entry.get("part_name"),
            part_guess,
        )
        or "unspecified"
    ).strip().lower()
    manifest_sha256 = str(manifest_entry.get("sha256", "") or "").strip()
    release_aware_lineage_key = str(dataset_lineage.get("dataset_lineage_key", "") or "").strip()
    if release_aware_lineage_key:
        dataset_lineage_key = release_aware_lineage_key
    elif dataset_key and manifest_sha256:
        dataset_lineage_key = f"{dataset_key}::{manifest_sha256}"
    else:
        dataset_lineage_key = dataset_key or manifest_sha256
    return {
        "crop_name": crop_name,
        "part_name": part_name or "unspecified",
        "dataset_key": dataset_key,
        "dataset_lineage_key": dataset_lineage_key,
        "dataset_context": dataset_context,
        "dataset_lineage": dataset_lineage,
        "dataset_manifest": manifest_entry,
        "dataset_roots": dataset_roots,
    }


def _resolve_artifact_paths(artifact_root: str | Path | None) -> JsonDict:
    if artifact_root is None:
        return {}
    root = Path(artifact_root)
    return {
        "artifact_root": str(root),
        "training_summary_json": str(root / "training" / "summary.json"),
        "training_run_context_json": str(root / "training" / "run_context.json"),
        "experiment_manifest_json": str(root / "training" / "experiment_manifest.json"),
        "optimization_record_json": str(root / "training" / "optimization_record.json"),
        "production_readiness_json": str(root / "production_readiness.json"),
        "guided_run_overview_json": str(root / "guided" / "01_run_overview.json"),
    }


def _resolve_runtime_context(
    *,
    summary_payload: Mapping[str, Any] | None,
    run_context_payload: Mapping[str, Any] | None,
) -> JsonDict:
    summary = dict(summary_payload or {})
    run_context = dict(run_context_payload or {})
    return {
        "device": str(_value_from_candidates(summary.get("device"), run_context.get("device")) or ""),
        "python_version": str(run_context.get("python_version", "") or ""),
        "git": _nested_dict(run_context, "git"),
        "package_versions": _nested_dict(run_context, "package_versions"),
    }


def _resolve_model_family(run_context_payload: Mapping[str, Any] | None) -> JsonDict:
    run_context = dict(run_context_payload or {})
    config = _nested_dict(run_context, "resolved_config")
    backbone = _nested_dict(config, "training", "continual", "backbone")
    return {
        "engine": TRAINING_ENGINE,
        "backbone_model_name": str(backbone.get("model_name", "") or ""),
    }


def _build_notebook_context(summary_payload: Mapping[str, Any] | None) -> JsonDict:
    summary = dict(summary_payload or {})
    notebook_surface = str(summary.get("notebook_surface", "") or "")
    if not notebook_surface:
        return {}
    payload: JsonDict = {
        "notebook_surface": notebook_surface,
    }
    dataset_roots = _nested_dict(summary, "dataset_roots")
    notebook_parameters = _nested_dict(summary, "notebook_parameters")
    readiness_summary = _nested_dict(summary, "readiness_summary")
    export_paths = _nested_dict(summary, "export_paths")
    if dataset_roots:
        payload["dataset_roots"] = dataset_roots
    if notebook_parameters:
        payload["notebook_parameters"] = notebook_parameters
    if readiness_summary:
        payload["readiness_summary"] = readiness_summary
    if export_paths:
        payload["export_paths"] = export_paths
    return payload


def build_experiment_manifest(
    *,
    summary_payload: Mapping[str, Any] | None,
    run_context_payload: Mapping[str, Any] | None,
    artifact_root: str | Path | None,
    explicit_surface: str | None = None,
    created_at: str | None = None,
    record_quality: str = "canonical",
) -> JsonDict:
    summary = dict(summary_payload or {})
    run_context = dict(run_context_payload or {})
    dataset_identity = _resolve_dataset_identity(summary_payload=summary, run_context_payload=run_context)
    dataset_context = dict(dataset_identity["dataset_context"])
    dataset_manifest = dict(dataset_identity["dataset_manifest"])
    dataset_lineage = dict(dataset_identity["dataset_lineage"])
    manifest_ood = dict(dataset_manifest.get("ood", {})) if isinstance(dataset_manifest.get("ood"), Mapping) else {}
    manifest_oe = dict(dataset_manifest.get("oe", {})) if isinstance(dataset_manifest.get("oe"), Mapping) else {}
    model_family = _resolve_model_family(run_context)
    surface = _resolve_surface(summary, explicit_surface)
    resolved_created_at = str(_value_from_candidates(created_at, summary.get("created_at"), run_context.get("created_at")) or _utc_now_iso())
    payload: JsonDict = {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA,
        "record_quality": str(record_quality),
        "optimization_profile": OPTIMIZATION_PROFILE,
        "run_id": str(_value_from_candidates(summary.get("run_id"), run_context.get("run_id")) or ""),
        "run_label": str(_value_from_candidates(summary.get("run_label"), summary.get("run_id"), run_context.get("run_id")) or ""),
        "created_at": resolved_created_at,
        "surface": surface,
        "crop_name": dataset_identity["crop_name"],
        "part_name": dataset_identity["part_name"],
        "dataset_key": dataset_identity["dataset_key"],
        "dataset_lineage_key": dataset_identity["dataset_lineage_key"],
        "model_family": model_family,
        "dataset": {
            "crop_root": str(dataset_context.get("crop_root", "") or ""),
            "dataset_key": dataset_identity["dataset_key"],
            "part_name": dataset_identity["part_name"],
            "resolution_source": str(dataset_context.get("resolution_source", "") or ""),
            "lineage": dataset_lineage,
            "manifest": {
                "path": str(dataset_manifest.get("path", "") or ""),
                "exists": bool(dataset_manifest.get("exists", False)),
                "sha256": str(dataset_manifest.get("sha256", "") or ""),
                "schema_version": str(dataset_manifest.get("schema_version", "") or ""),
                "source_root": str(dataset_manifest.get("source_root", "") or ""),
                "crop_name": str(dataset_manifest.get("crop_name", "") or dataset_identity["crop_name"]),
                "part_name": str(dataset_manifest.get("part_name", "") or dataset_identity["part_name"]),
                "dataset_key": str(dataset_manifest.get("dataset_key", "") or dataset_identity["dataset_key"]),
                "split_policy": str(dataset_manifest.get("split_policy", "") or ""),
            },
            "ood": {
                "source_root": str(manifest_ood.get("source_root", "") or ""),
                "image_count": _coerce_int(manifest_ood.get("image_count")),
                "image_fingerprint": manifest_ood.get("image_fingerprint"),
            },
            "oe": {
                "source_root": str(manifest_oe.get("source_root", "") or ""),
                "image_count": _coerce_int(manifest_oe.get("image_count")),
                "image_fingerprint": manifest_oe.get("image_fingerprint"),
            },
        },
        "runtime": _resolve_runtime_context(summary_payload=summary, run_context_payload=run_context),
        "artifacts": _resolve_artifact_paths(artifact_root),
    }
    notebook_context = _build_notebook_context(summary)
    if notebook_context:
        payload["notebook_context"] = notebook_context
    return payload


def _config_value(config: Mapping[str, Any], *keys: str) -> Any:
    current: Any = dict(config)
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


_NOTEBOOK_PARAMETER_MAP: dict[str, tuple[str, Any]] = {
    "learning_rate": ("training.learning_rate", _coerce_float),
    "weight_decay": ("training.weight_decay", _coerce_float),
    "epochs": ("training.num_epochs", _coerce_int),
    "batch_size": ("training.batch_size", _coerce_int),
    "lora_r": ("training.adapter.lora_r", _coerce_int),
    "lora_alpha": ("training.adapter.lora_alpha", _coerce_int),
    "lora_dropout": ("training.adapter.lora_dropout", _coerce_float),
    "fusion_dropout": ("training.fusion.dropout", _coerce_float),
    "ood_factor": ("training.ood.threshold_factor", _coerce_float),
    "oe_loss_weight": ("training.ood.oe_loss_weight", _coerce_float),
    "react_enabled": ("training.ood.react_enabled", _coerce_bool),
    "react_percentile": ("training.ood.react_percentile", _coerce_float),
    "loss_name": ("training.optimization.loss_name", str),
    "logitnorm_tau": ("training.optimization.logitnorm_tau", _coerce_float),
    "label_smoothing": ("training.optimization.label_smoothing", _coerce_float),
    "grad_accum_steps": ("training.optimization.grad_accumulation_steps", _coerce_int),
    "mixed_precision": ("training.optimization.mixed_precision", str),
    "max_grad_norm": ("training.optimization.max_grad_norm", _coerce_float),
    "augmentation_policy": ("training.data.augmentation_policy", str),
    "randaugment_num_ops": ("training.data.randaugment_num_ops", _coerce_int),
    "randaugment_magnitude": ("training.data.randaugment_magnitude", _coerce_int),
    "augmix_severity": ("training.data.augmix_severity", _coerce_int),
    "classifier_rebalance_enabled": ("training.classifier_rebalance.enabled", _coerce_bool),
    "classifier_rebalance_logit_adjustment_tau": (
        "training.classifier_rebalance.logit_adjustment_tau",
        _coerce_float,
    ),
}


def _merge_notebook_parameters(parameters: JsonDict, notebook_parameters: Mapping[str, Any] | None) -> JsonDict:
    merged = dict(parameters)
    if not isinstance(notebook_parameters, Mapping):
        return merged
    normalized = {str(key).strip().lower(): value for key, value in dict(notebook_parameters).items()}
    for notebook_key, (parameter_key, coercer) in _NOTEBOOK_PARAMETER_MAP.items():
        value = normalized.get(notebook_key)
        if value in (None, "", []):
            continue
        if parameter_key in merged and merged[parameter_key] not in (None, "", []):
            continue
        coerced = coercer(value)
        if coerced not in (None, "", []):
            merged[parameter_key] = coerced
    return merged


def _extract_parameter_block(
    run_context_payload: Mapping[str, Any] | None,
    summary_payload: Mapping[str, Any] | None = None,
) -> JsonDict:
    run_context = dict(run_context_payload or {})
    config = _nested_dict(run_context, "resolved_config")
    training_cfg = _nested_dict(config, "training", "continual")
    adapter_cfg = _nested_dict(training_cfg, "adapter")
    fusion_cfg = _nested_dict(training_cfg, "fusion")
    ood_cfg = _nested_dict(training_cfg, "ood")
    classifier_rebalance_cfg = _nested_dict(training_cfg, "classifier_rebalance")
    optimization_cfg = _nested_dict(training_cfg, "optimization")
    scheduler_cfg = _nested_dict(optimization_cfg, "scheduler")
    data_cfg = _nested_dict(training_cfg, "data")
    sampler_runtime = _nested_dict(run_context, "training_runtime", "train_sampler")
    parameters: JsonDict = {
        "training.learning_rate": _coerce_float(training_cfg.get("learning_rate")),
        "training.weight_decay": _coerce_float(training_cfg.get("weight_decay")),
        "training.num_epochs": _coerce_int(training_cfg.get("num_epochs")),
        "training.batch_size": _coerce_int(training_cfg.get("batch_size")),
        "training.seed": _coerce_int(training_cfg.get("seed")),
        "training.adapter.lora_r": _coerce_int(adapter_cfg.get("lora_r")),
        "training.adapter.lora_alpha": _coerce_int(adapter_cfg.get("lora_alpha")),
        "training.adapter.lora_dropout": _coerce_float(adapter_cfg.get("lora_dropout")),
        "training.fusion.layers": list(fusion_cfg.get("layers", [])) if isinstance(fusion_cfg.get("layers"), list) else [],
        "training.fusion.output_dim": _coerce_int(fusion_cfg.get("output_dim")),
        "training.fusion.dropout": _coerce_float(fusion_cfg.get("dropout")),
        "training.fusion.gating": fusion_cfg.get("gating"),
        "training.optimization.loss_name": optimization_cfg.get("loss_name"),
        "training.optimization.logitnorm_tau": _coerce_float(optimization_cfg.get("logitnorm_tau")),
        "training.optimization.label_smoothing": _coerce_float(optimization_cfg.get("label_smoothing")),
        "training.optimization.grad_accumulation_steps": _coerce_int(optimization_cfg.get("grad_accumulation_steps")),
        "training.optimization.mixed_precision": optimization_cfg.get("mixed_precision"),
        "training.optimization.max_grad_norm": _coerce_float(optimization_cfg.get("max_grad_norm")),
        "training.optimization.scheduler.name": scheduler_cfg.get("name"),
        "training.optimization.scheduler.warmup_ratio": _coerce_float(scheduler_cfg.get("warmup_ratio")),
        "training.optimization.scheduler.min_lr": _coerce_float(scheduler_cfg.get("min_lr")),
        "training.optimization.scheduler.step_on": scheduler_cfg.get("step_on"),
        "training.ood.threshold_factor": _coerce_float(ood_cfg.get("threshold_factor")),
        "training.ood.primary_score_method": ood_cfg.get("primary_score_method"),
        "training.ood.energy_temperature_mode": ood_cfg.get("energy_temperature_mode"),
        "training.ood.energy_temperature": _coerce_float(ood_cfg.get("energy_temperature")),
        "training.ood.react_enabled": bool(ood_cfg.get("react_enabled", False)),
        "training.ood.react_percentile": _coerce_float(ood_cfg.get("react_percentile")),
        "training.ood.oe_enabled": bool(ood_cfg.get("oe_enabled", False)),
        "training.ood.oe_loss_weight": _coerce_float(ood_cfg.get("oe_loss_weight")),
        "training.ood.oe_target": ood_cfg.get("oe_target"),
        "training.ood.radial_l2_enabled": bool(ood_cfg.get("radial_l2_enabled", False)),
        "training.ood.radial_beta_range": list(ood_cfg.get("radial_beta_range", [])) if isinstance(ood_cfg.get("radial_beta_range"), list) else [],
        "training.ood.radial_beta_steps": _coerce_int(ood_cfg.get("radial_beta_steps")),
        "training.ood.real_split_enabled": bool(ood_cfg.get("real_split_enabled", False)),
        "training.ood.real_split_dev_fraction": _coerce_float(ood_cfg.get("real_split_dev_fraction")),
        "training.ood.real_split_min_per_slice": _coerce_int(ood_cfg.get("real_split_min_per_slice")),
        "training.ood.real_split_min_total": _coerce_int(ood_cfg.get("real_split_min_total")),
        "training.ood.real_split_manifest_name": ood_cfg.get("real_split_manifest_name"),
        "training.ood.enforce_oe_disjoint": bool(ood_cfg.get("enforce_oe_disjoint", True)),
        "training.ood.sure_enabled": bool(ood_cfg.get("sure_enabled", False)),
        "training.ood.sure_semantic_percentile": _coerce_float(ood_cfg.get("sure_semantic_percentile")),
        "training.ood.sure_confidence_percentile": _coerce_float(ood_cfg.get("sure_confidence_percentile")),
        "training.ood.conformal_enabled": bool(ood_cfg.get("conformal_enabled", False)),
        "training.ood.conformal_alpha": _coerce_float(ood_cfg.get("conformal_alpha")),
        "training.ood.conformal_method": ood_cfg.get("conformal_method"),
        "training.ood.conformal_raps_lambda": _coerce_float(ood_cfg.get("conformal_raps_lambda")),
        "training.ood.conformal_raps_k_reg": _coerce_int(ood_cfg.get("conformal_raps_k_reg")),
        "training.data.sampler": data_cfg.get("sampler"),
        "training.data.resolved_sampler": sampler_runtime.get("resolved_sampler"),
        "training.data.augmentation_policy": data_cfg.get("augmentation_policy"),
        "training.data.randaugment_num_ops": _coerce_int(data_cfg.get("randaugment_num_ops")),
        "training.data.randaugment_magnitude": _coerce_int(data_cfg.get("randaugment_magnitude")),
        "training.data.augmix_severity": _coerce_int(data_cfg.get("augmix_severity")),
        "training.data.augmix_width": _coerce_int(data_cfg.get("augmix_width")),
        "training.data.augmix_depth": _coerce_int(data_cfg.get("augmix_depth")),
        "training.data.augmix_alpha": _coerce_float(data_cfg.get("augmix_alpha")),
        "training.classifier_rebalance.enabled": bool(classifier_rebalance_cfg.get("enabled", False)),
        "training.classifier_rebalance.epochs": _coerce_int(classifier_rebalance_cfg.get("epochs")),
        "training.classifier_rebalance.learning_rate": _coerce_float(classifier_rebalance_cfg.get("learning_rate")),
        "training.classifier_rebalance.weight_decay": _coerce_float(classifier_rebalance_cfg.get("weight_decay")),
        "training.classifier_rebalance.sampler": classifier_rebalance_cfg.get("sampler"),
        "training.classifier_rebalance.objective": classifier_rebalance_cfg.get("objective"),
        "training.classifier_rebalance.logit_adjustment_tau": _coerce_float(
            classifier_rebalance_cfg.get("logit_adjustment_tau")
        ),
    }
    merged = _merge_notebook_parameters(parameters, _nested_dict(summary_payload, "notebook_parameters"))
    return {key: value for key, value in merged.items() if value not in (None, "", [])}


def _weighted_f1_from_report(report_dict: Mapping[str, Any] | None) -> Optional[float]:
    if not isinstance(report_dict, Mapping):
        return None
    weighted_avg = report_dict.get("weighted avg")
    if not isinstance(weighted_avg, Mapping):
        return None
    return _coerce_float(weighted_avg.get("f1-score"))


def _macro_f1_from_report(report_dict: Mapping[str, Any] | None) -> Optional[float]:
    if not isinstance(report_dict, Mapping):
        return None
    macro_avg = report_dict.get("macro avg")
    if not isinstance(macro_avg, Mapping):
        return None
    return _coerce_float(macro_avg.get("f1-score"))


def _resolve_classification_metrics(
    *,
    production_readiness_payload: Mapping[str, Any] | None,
    authoritative_artifacts: Mapping[str, Any] | None,
) -> JsonDict:
    readiness = dict(production_readiness_payload or {})
    authoritative = dict(authoritative_artifacts or {})
    classification = _nested_dict(readiness, "classification_evidence")
    metric_gate = dict(authoritative.get("metric_gate", {})) if isinstance(authoritative.get("metric_gate"), Mapping) else {}
    gate_metrics = _nested_dict(metric_gate, "metrics")
    report_dict = dict(authoritative.get("report_dict", {})) if isinstance(authoritative.get("report_dict"), Mapping) else {}
    accuracy = _coerce_float(_value_from_candidates(classification.get("metrics", {}).get("accuracy") if isinstance(classification.get("metrics"), Mapping) else None, gate_metrics.get("accuracy"), report_dict.get("accuracy")))
    balanced_accuracy = _coerce_float(_value_from_candidates(classification.get("metrics", {}).get("balanced_accuracy") if isinstance(classification.get("metrics"), Mapping) else None, gate_metrics.get("balanced_accuracy")))
    macro_f1 = _coerce_float(
        _value_from_candidates(
            classification.get("metrics", {}).get("macro_f1") if isinstance(classification.get("metrics"), Mapping) else None,
            gate_metrics.get("macro_f1"),
            _macro_f1_from_report(report_dict),
        )
    )
    weighted_f1 = _weighted_f1_from_report(report_dict)
    classification_samples = _coerce_int(_value_from_candidates(gate_metrics.get("classification_samples"), classification.get("metrics", {}).get("classification_samples") if isinstance(classification.get("metrics"), Mapping) else None))
    return {
        "classification.accuracy": accuracy,
        "classification.balanced_accuracy": balanced_accuracy,
        "classification.macro_f1": macro_f1,
        "classification.weighted_f1": weighted_f1,
        "classification.classification_samples": classification_samples,
    }


def build_optimization_record(
    *,
    summary_payload: Mapping[str, Any] | None,
    run_context_payload: Mapping[str, Any] | None,
    production_readiness_payload: Mapping[str, Any] | None = None,
    authoritative_artifacts: Mapping[str, Any] | None = None,
    artifact_root: str | Path | None,
    explicit_surface: str | None = None,
    created_at: str | None = None,
    record_quality: str = "canonical",
) -> JsonDict:
    summary = dict(summary_payload or {})
    run_context = dict(run_context_payload or {})
    production_readiness = dict(production_readiness_payload or {})
    dataset_identity = _resolve_dataset_identity(summary_payload=summary, run_context_payload=run_context)
    model_family = _resolve_model_family(run_context)
    surface = _resolve_surface(summary, explicit_surface)
    resolved_created_at = str(_value_from_candidates(created_at, summary.get("created_at"), run_context.get("created_at")) or _utc_now_iso())
    authoritative_split = str(
        _value_from_candidates(
            _nested_dict(production_readiness, "classification_evidence").get("split_name"),
            summary.get("classification_split"),
            _nested_dict(summary, "production_readiness").get("classification_split"),
        )
        or ""
    )
    readiness_status = str(_value_from_candidates(production_readiness.get("status"), _nested_dict(summary, "readiness_summary").get("status"), _nested_dict(summary, "production_readiness").get("status")) or "")
    readiness_passed = _value_from_candidates(production_readiness.get("passed"), _nested_dict(summary, "readiness_summary").get("passed"), _nested_dict(summary, "production_readiness").get("passed"))
    ood_evidence_source = str(_value_from_candidates(production_readiness.get("ood_evidence_source"), summary.get("ood_evidence_source"), _nested_dict(summary, "readiness_summary").get("ood_evidence_source")) or "")
    ood_metrics = _nested_dict(production_readiness, "ood_evidence", "metrics")
    classification_metrics = _resolve_classification_metrics(
        production_readiness_payload=production_readiness,
        authoritative_artifacts=authoritative_artifacts,
    )
    objective_directions: JsonDict = {
        "classification.accuracy": "maximize",
        "classification.balanced_accuracy": "maximize",
        "classification.macro_f1": "maximize",
        "classification.weighted_f1": "maximize",
        "ood.ood_auroc": "maximize",
        "ood.sure_ds_f1": "maximize",
        "ood.conformal_empirical_coverage": "maximize",
        "ood.ood_false_positive_rate": "minimize",
        "ood.conformal_avg_set_size": "minimize",
        "training_cost.optimizer_steps": "minimize",
        "training_cost.global_step": "minimize",
        "training_cost.checkpoint_count": "minimize",
    }
    objectives: JsonDict = {
        **classification_metrics,
        "ood.ood_auroc": _coerce_float(ood_metrics.get("ood_auroc")),
        "ood.ood_false_positive_rate": _coerce_float(ood_metrics.get("ood_false_positive_rate")),
        "ood.sure_ds_f1": _coerce_float(ood_metrics.get("sure_ds_f1")),
        "ood.conformal_empirical_coverage": _coerce_float(ood_metrics.get("conformal_empirical_coverage")),
        "ood.conformal_avg_set_size": _coerce_float(ood_metrics.get("conformal_avg_set_size")),
        "ood.ood_samples": _coerce_int(ood_metrics.get("ood_samples")),
        "ood.in_distribution_samples": _coerce_int(ood_metrics.get("in_distribution_samples")),
        "context.class_count": _coerce_int(summary.get("class_count")),
        "context.loader_sizes.train": _coerce_int(_nested_dict(summary, "loader_sizes").get("train")),
        "context.loader_sizes.val": _coerce_int(_nested_dict(summary, "loader_sizes").get("val")),
        "context.loader_sizes.test": _coerce_int(_nested_dict(summary, "loader_sizes").get("test")),
        "context.loader_sizes.ood": _coerce_int(_nested_dict(summary, "loader_sizes").get("ood")),
        "training_cost.optimizer_steps": _coerce_int(summary.get("optimizer_steps")),
        "training_cost.global_step": _coerce_int(summary.get("global_step")),
        "training_cost.checkpoint_count": _coerce_int(summary.get("checkpoint_count")),
    }
    objectives = {key: value for key, value in objectives.items() if value is not None}
    comparability = {
        "dataset_lineage_key": dataset_identity["dataset_lineage_key"],
        "crop_name": dataset_identity["crop_name"],
        "part_name": dataset_identity["part_name"],
        "engine": model_family["engine"],
        "backbone_model_name": model_family["backbone_model_name"],
    }
    cohort_key = "::".join(
        str(comparability.get(key, "") or "")
        for key in ("dataset_lineage_key", "crop_name", "part_name", "engine", "backbone_model_name")
    ).strip(":")
    comparability["cohort_key"] = cohort_key
    payload: JsonDict = {
        "schema_version": OPTIMIZATION_RECORD_SCHEMA,
        "record_quality": str(record_quality),
        "optimization_profile": OPTIMIZATION_PROFILE,
        "run_id": str(_value_from_candidates(summary.get("run_id"), run_context.get("run_id")) or ""),
        "run_label": str(_value_from_candidates(summary.get("run_label"), summary.get("run_id"), run_context.get("run_id")) or ""),
        "created_at": resolved_created_at,
        "surface": surface,
        "crop_name": dataset_identity["crop_name"],
        "part_name": dataset_identity["part_name"],
        "dataset_key": dataset_identity["dataset_key"],
        "dataset_lineage_key": dataset_identity["dataset_lineage_key"],
        "status": {
            "readiness_status": readiness_status,
            "readiness_passed": bool(readiness_passed) if readiness_passed is not None else None,
            "authoritative_split": authoritative_split,
            "ood_evidence_source": ood_evidence_source,
        },
        "model_family": model_family,
        "comparability": comparability,
        "parameters": _extract_parameter_block(run_context, summary),
        "objectives": objectives,
        "objective_directions": objective_directions,
        "dataset": build_experiment_manifest(
            summary_payload=summary,
            run_context_payload=run_context,
            artifact_root=artifact_root,
            explicit_surface=surface,
            created_at=resolved_created_at,
            record_quality=record_quality,
        )["dataset"],
        "artifacts": _resolve_artifact_paths(artifact_root),
    }
    notebook_context = _build_notebook_context(summary)
    if notebook_context:
        payload["notebook_context"] = notebook_context
    return payload


def persist_traceability_artifacts(
    *,
    artifact_root: str | Path,
    experiment_manifest: Mapping[str, Any],
    optimization_record: Mapping[str, Any],
    telemetry: Any = None,
    auto_refresh_registry: bool = True,
    enable_bayesian_proposals: bool = True,
) -> Dict[str, Any]:
    root = Path(artifact_root)
    training_dir = _artifact_dir(root, "training")
    store = ArtifactStore(training_dir)
    manifest_json = store.write_json("experiment_manifest.json", dict(experiment_manifest))
    optimization_json = store.write_json("optimization_record.json", dict(optimization_record))
    _copy_to_telemetry(telemetry, manifest_json, "training/experiment_manifest.json")
    _copy_to_telemetry(telemetry, optimization_json, "training/optimization_record.json")
    refresh_training_guided_artifacts(root, telemetry=telemetry)
    result: Dict[str, Any] = {
        "experiment_manifest_json": manifest_json,
        "optimization_record_json": optimization_json,
    }
    if auto_refresh_registry:
        registry_result = _try_refresh_run_registry(
            artifact_root=root,
            telemetry=telemetry,
            enable_bayesian_proposals=bool(enable_bayesian_proposals),
        )
        if registry_result:
            result["run_registry"] = registry_result
    return result


def load_authoritative_artifacts_from_root(
    artifact_root: str | Path,
    *,
    classification_split: str,
) -> JsonDict:
    root = Path(artifact_root)
    split_name = str(classification_split or "").strip().lower()
    if not split_name:
        return {}
    split_root = root / split_name
    if not split_root.exists():
        return {}
    report_dict = read_json(split_root / "classification_report.json", default={}, expect_type=dict)
    metric_gate = read_json(split_root / "metric_gate.json", default={}, expect_type=dict)
    return {
        "report_dict": dict(report_dict) if isinstance(report_dict, Mapping) else {},
        "metric_gate": dict(metric_gate) if isinstance(metric_gate, Mapping) else {},
    }
