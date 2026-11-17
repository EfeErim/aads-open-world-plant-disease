#!/usr/bin/env python3
"""Notebook 2 helper functions only."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib

from src.colab.training_session import (
    _call_if_present,
    _resolve_colab_runtime_api,
)
from src.colab.training_session import ensure_notebook_checkpoint_manager as _ensure_notebook_checkpoint_manager_impl
from src.colab.training_session import maybe_auto_disconnect_colab_runtime as _maybe_auto_disconnect_colab_runtime_impl
from src.shared.hash_utils import sha256_file
from src.shared.json_utils import read_json, write_json
from src.shared.string_utils import slug_label_component

matplotlib.use("Agg")

_NOTEBOOK_OPTIMIZATION_CAMPAIGN_SCHEMA = "v1_notebook_optimization_campaign"
_NOTEBOOK_OPTIMIZATION_SEARCH_SPACE = {
    "schema_version": "v1_notebook_optimization_search_space",
    "parameters": [
        {"name": "training.learning_rate", "type": "float", "low": 5e-5, "high": 3e-4, "scale": "log"},
        {"name": "training.weight_decay", "type": "float", "low": 1e-4, "high": 5e-2, "scale": "log"},
        {"name": "training.num_epochs", "type": "int", "low": 8, "high": 40, "step": 2},
        {
            "name": "training.batch_size",
            "type": "categorical",
            "values": [4, 8, 12, 16, 32, 40, 48, 52, 56, 64, 88, 96, 128],
        },
        {"name": "training.adapter.lora_r", "type": "categorical", "values": [8, 16, 20, 24, 26, 32]},
        {"name": "training.adapter.lora_alpha", "type": "categorical", "values": [8, 16, 20, 24, 26, 32]},
        {"name": "training.adapter.lora_dropout", "type": "float", "low": 0.0, "high": 0.25},
        {"name": "training.fusion.dropout", "type": "float", "low": 0.0, "high": 0.25},
        {"name": "training.ood.threshold_factor", "type": "float", "low": 1.5, "high": 4.5},
        {"name": "training.ood.oe_loss_weight", "type": "float", "low": 0.0, "high": 0.5},
        {"name": "training.ood.react_enabled", "type": "categorical", "values": [False, True]},
        {"name": "training.ood.react_percentile", "type": "float", "low": 0.95, "high": 0.999},
        {"name": "training.optimization.logitnorm_tau", "type": "float", "low": 0.5, "high": 2.0},
        {"name": "training.optimization.label_smoothing", "type": "float", "low": 0.0, "high": 0.2},
        {"name": "training.data.augmentation_policy", "type": "categorical", "values": ["randaugment", "augmix"]},
        {"name": "training.data.randaugment_num_ops", "type": "int", "low": 1, "high": 4, "step": 1},
        {"name": "training.data.randaugment_magnitude", "type": "int", "low": 3, "high": 12, "step": 1},
        {"name": "training.data.augmix_severity", "type": "int", "low": 1, "high": 5, "step": 1},
        {"name": "training.classifier_rebalance.enabled", "type": "categorical", "values": [False, True]},
        {"name": "training.classifier_rebalance.logit_adjustment_tau", "type": "float", "low": 0.5, "high": 2.0},
    ],
}
_NOTEBOOK_PARAM_MAP = {
    "training.learning_rate": "LEARNING_RATE",
    "training.weight_decay": "WEIGHT_DECAY",
    "training.num_epochs": "EPOCHS",
    "training.batch_size": "BATCH_SIZE",
    "training.adapter.lora_r": "LORA_R",
    "training.adapter.lora_alpha": "LORA_ALPHA",
    "training.adapter.lora_dropout": "LORA_DROPOUT",
    "training.fusion.dropout": "FUSION_DROPOUT",
    "training.ood.threshold_factor": "OOD_FACTOR",
    "training.ood.oe_loss_weight": "OE_LOSS_WEIGHT",
    "training.ood.react_enabled": "REACT_ENABLED",
    "training.ood.react_percentile": "REACT_PERCENTILE",
    "training.data.augmentation_policy": "AUGMENTATION_POLICY",
    "training.optimization.logitnorm_tau": "LOGITNORM_TAU",
    "training.optimization.label_smoothing": "LABEL_SMOOTHING",
    "training.data.randaugment_num_ops": "RANDAUGMENT_NUM_OPS",
    "training.data.randaugment_magnitude": "RANDAUGMENT_MAGNITUDE",
    "training.data.augmix_severity": "AUGMIX_SEVERITY",
    "training.classifier_rebalance.enabled": "CLASSIFIER_REBALANCE_ENABLED",
    "training.classifier_rebalance.logit_adjustment_tau": "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU",
}
_DEFAULT_PARETO_OBJECTIVES = (
    "classification.macro_f1",
    "ood.ood_auroc",
    "ood.ood_false_positive_rate",
)


def coerce_like_value(reference: Any, value: Any) -> Any:
    """Coerce a value to match the type of a reference value."""
    if isinstance(reference, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(value)
    if isinstance(reference, int) and not isinstance(reference, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return reference
    if isinstance(reference, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return reference
    return value


def ensure_notebook_checkpoint_manager(
    checkpoint_manager: Any = None,
    *,
    run_id: Optional[str] = None,
    drive_root: Optional[str | Path] = None,
    retention: int = 3,
) -> Any:
    return _ensure_notebook_checkpoint_manager_impl(
        checkpoint_manager,
        run_id=run_id,
        drive_root=drive_root,
        retention=retention,
        now_fn=datetime.now,
    )


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
) -> Dict[str, Any]:
    return _maybe_auto_disconnect_colab_runtime_impl(
        enabled=enabled,
        grace_period_sec=grace_period_sec,
        state=state,
        telemetry=telemetry,
        repo_run_exports=repo_run_exports,
        notebook_export_path=notebook_export_path,
        completion_report=completion_report,
        print_fn=print_fn,
        runtime_api_resolver=_resolve_colab_runtime_api,
    )


def _load_trials_jsonl(path: Path) -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    if not path.exists():
        return trials
    for line in path.read_text(encoding="utf-8").splitlines():
        text = str(line).strip()
        if not text:
            continue
        trials.append(read_json_text(text))
    return trials


def read_json_text(payload: str) -> Dict[str, Any]:

    loaded = json.loads(payload)
    return dict(loaded) if isinstance(loaded, dict) else {}


def _build_notebook_campaign_seed(
    *,
    dataset_lineage_key: str,
    crop_name: str,
    part_name: str,
    engine: str,
    backbone_model_name: str,
) -> str:
    identity = "||".join(
        [
            str(dataset_lineage_key or ""),
            str(crop_name or ""),
            str(part_name or ""),
            str(engine or ""),
            str(backbone_model_name or ""),
        ]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    parts = (
        slug_label_component(crop_name, default="crop"),
        slug_label_component(part_name, default="part"),
        digest,
    )
    return "__".join(parts)


def _notebook_campaign_root(index_root: Path) -> Path:
    target = index_root / "notebook_optimization_campaigns"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _build_campaign_snapshot(campaign: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = dict(campaign or {})
    next_proposal = dict(payload.get("next_proposal", {})) if isinstance(payload.get("next_proposal"), dict) else {}
    selected_proposal = (
        dict(payload.get("selected_proposal", {})) if isinstance(payload.get("selected_proposal"), dict) else {}
    )
    return {
        "status": str(payload.get("status", "") or ""),
        "mode": str(payload.get("mode", "") or ""),
        "campaign_json": str(payload.get("campaign_json", "") or ""),
        "campaign_seed": str(payload.get("campaign_seed", "") or ""),
        "cohort_key": str(payload.get("cohort_key", "") or ""),
        "dataset_lineage_key": str(payload.get("dataset_lineage_key", "") or ""),
        "frontier_count": int(payload.get("frontier_count", 0) or 0),
        "frontier_run_ids": list(payload.get("frontier_run_ids", [])),
        "eligible_run_count": int(payload.get("eligible_run_count", 0) or 0),
        "executed_run_count": int(len(payload.get("executed_run_ids", []))),
        "next_proposal_signature": str(next_proposal.get("signature", "") or ""),
        "next_proposal_rank": next_proposal.get("rank"),
        "selected_proposal_signature": str(selected_proposal.get("signature", "") or ""),
        "selected_proposal_rank": selected_proposal.get("rank"),
        "last_completed_run_id": str(payload.get("last_completed_run_id", "") or ""),
    }


def _select_next_campaign_proposal(
    recommendation_cohort: Dict[str, Any],
    executed_signatures: set[str],
) -> Dict[str, Any]:
    for proposal in recommendation_cohort.get("proposals", []):
        if not isinstance(proposal, dict):
            continue
        signature = str(proposal.get("signature", "") or "")
        if signature and signature in executed_signatures:
            continue
        return dict(proposal)
    return {}


def summarize_notebook_optimization_campaign(campaign: Dict[str, Any] | None) -> Dict[str, Any]:
    return _build_campaign_snapshot(campaign)


def print_notebook_optimization_campaign_status(
    campaign: Dict[str, Any] | None,
    *,
    print_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    emit = print if print_fn is None else print_fn
    snapshot = summarize_notebook_optimization_campaign(campaign)
    if not snapshot:
        emit("[OPT] campaign=unavailable")
        return {}
    emit(
        f"[OPT] mode={snapshot.get('mode', 'unknown')} status={snapshot.get('status', 'unknown')} "
        f"eligible={snapshot.get('eligible_run_count', 0)} frontier={snapshot.get('frontier_count', 0)} "
        f"executed={snapshot.get('executed_run_count', 0)}"
    )
    next_rank = snapshot.get("next_proposal_rank")
    next_signature = str(snapshot.get("next_proposal_signature", "") or "")
    if next_rank is not None or next_signature:
        emit(
            f"[OPT] next_proposal rank={next_rank if next_rank is not None else '?'} "
            f"signature={next_signature[:16] if next_signature else 'n/a'}"
        )
    if snapshot.get("frontier_run_ids"):
        emit(f"[OPT] frontier_runs={', '.join(str(item) for item in snapshot['frontier_run_ids'])}")
    return snapshot


def _resolve_runtime_dataset_identity(
    *,
    runtime_dataset_root: Path,
    dataset_key: str,
    crop_name: str,
    part_name: str,
) -> Dict[str, Any]:
    manifest_path = runtime_dataset_root / "split_manifest.json"
    manifest = read_json(manifest_path, default={}, expect_type=dict) if manifest_path.exists() else {}
    dataset_key_value = str(manifest.get("dataset_key", "") or dataset_key or "").strip().lower()
    manifest_sha = sha256_file(manifest_path) if manifest_path.exists() else ""
    dataset_lineage_key = (
        f"{dataset_key_value}::{manifest_sha}"
        if dataset_key_value and manifest_sha
        else dataset_key_value or manifest_sha
    )
    return {
        "dataset_key": dataset_key_value,
        "dataset_lineage_key": dataset_lineage_key,
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "manifest_sha256": manifest_sha,
        "crop_name": str(manifest.get("crop_name", "") or crop_name or "").strip().lower(),
        "part_name": str(manifest.get("part_name", "") or part_name or "unspecified").strip().lower(),
    }


def _select_notebook_campaign_trials(
    *,
    trials: List[Dict[str, Any]],
    select_trials_for_cohort_fn: Callable[..., List[Dict[str, Any]]],
    dataset_lineage_key: str,
    dataset_key: str,
    crop_name: str,
    part_name: str,
    backbone_model_name: str,
    engine: str,
) -> tuple[List[Dict[str, Any]], str]:
    # Keep campaign continuity across legacy records that predate lineage hashing or backbone recording.
    lineage = str(dataset_lineage_key or "")
    dataset = str(dataset_key or "")
    backbone = str(backbone_model_name or "")

    common = {
        "crop_name": crop_name,
        "part_name": part_name,
        "engine": engine,
    }
    attempts: List[tuple[str, Dict[str, Any]]] = []

    def add_attempt(label: str, **filters: Any) -> None:
        attempts.append((label, {**common, **filters}))

    if lineage:
        add_attempt("strict_lineage_backbone", dataset_lineage_key=lineage, backbone_model_name=backbone)
    if dataset:
        add_attempt("legacy_dataset_key_backbone", dataset_key=dataset, backbone_model_name=backbone)
    if backbone:
        if lineage:
            add_attempt("strict_lineage_blank_backbone", dataset_lineage_key=lineage, backbone_model_name="")
        if dataset:
            add_attempt("legacy_dataset_key_blank_backbone", dataset_key=dataset, backbone_model_name="")
        if lineage:
            add_attempt("strict_lineage_any_backbone", dataset_lineage_key=lineage)
        if dataset:
            add_attempt("legacy_dataset_key_any_backbone", dataset_key=dataset)

    seen = set()
    for label, filters in attempts:
        key = tuple(sorted((k, str(v)) for k, v in filters.items()))
        if key in seen:
            continue
        seen.add(key)
        selected = select_trials_for_cohort_fn(trials, **filters)
        if selected:
            return selected, label

    return [], "no_match"


def resolve_notebook_optimization_campaign(
    *,
    root: Path,
    runtime_dataset_root: Path,
    dataset_key: str,
    crop_name: str,
    part_name: str,
    backbone_model_name: str,
    notebook_parameters: Dict[str, Any],
    mode: str = "disabled",
    telemetry: Any = None,
    engine: str = "continual_sd_lora",
    objectives: Optional[List[str]] = None,
    dataset_lineage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from src.training.services.optimization import (
        build_bayesian_recommendations,
        build_pareto_frontiers,
        select_trials_for_cohort,
    )
    from src.training.services.run_registry import build_run_registry

    requested_mode = str(mode or "").strip().lower()
    resolved_mode = requested_mode if requested_mode in {"continue", "stop"} else "disabled"
    dataset_identity = _resolve_runtime_dataset_identity(
        runtime_dataset_root=Path(runtime_dataset_root),
        dataset_key=dataset_key,
        crop_name=crop_name,
        part_name=part_name,
    )
    if dataset_lineage:
        lineage = dict(dataset_lineage)
        if str(lineage.get("dataset_key") or "") != str(dataset_identity.get("dataset_key") or ""):
            raise ValueError("Optimization campaign dataset lineage target mismatch")
        dataset_identity["dataset_lineage_key"] = str(lineage.get("dataset_lineage_key") or "")
        dataset_identity["dataset_lineage"] = lineage
    index_root = root / "runs" / "_index"
    runs_root = root / "runs"
    campaign_seed = _build_notebook_campaign_seed(
        dataset_lineage_key=str(
            dataset_identity.get("dataset_lineage_key", "") or dataset_identity.get("dataset_key", "")
        ),
        crop_name=str(dataset_identity.get("crop_name", "") or crop_name),
        part_name=str(dataset_identity.get("part_name", "") or part_name),
        engine=engine,
        backbone_model_name=backbone_model_name,
    )
    campaign_path = _notebook_campaign_root(index_root) / f"{campaign_seed}.json"
    existing = read_json(campaign_path, default={}, expect_type=dict) if campaign_path.exists() else {}
    objective_names = list(objectives or _DEFAULT_PARETO_OBJECTIVES)

    build_run_registry(
        runs_root=runs_root,
        output_root=index_root,
        enable_bayesian_proposals=resolved_mode in {"continue", "stop"},
        search_space_payload=_NOTEBOOK_OPTIMIZATION_SEARCH_SPACE,
    )
    trials = _load_trials_jsonl(index_root / "trials.jsonl")
    selected_trials, cohort_match_mode = _select_notebook_campaign_trials(
        trials=trials,
        select_trials_for_cohort_fn=select_trials_for_cohort,
        dataset_lineage_key=str(dataset_identity.get("dataset_lineage_key", "") or ""),
        dataset_key=str(dataset_identity.get("dataset_key", "") or dataset_key),
        crop_name=str(dataset_identity.get("crop_name", "") or crop_name),
        part_name=str(dataset_identity.get("part_name", "") or part_name),
        backbone_model_name=backbone_model_name,
        engine=engine,
    )
    pareto_payload = build_pareto_frontiers(selected_trials, objective_names=objective_names)
    frontier_cohort = pareto_payload.get("cohorts", [{}])[0] if pareto_payload.get("cohorts") else {}
    recommendation_cohort = {
        "cohort_key": str(frontier_cohort.get("cohort_key", "") or ""),
        "eligible_run_count": 0,
        "search_strategy": "disabled",
        "best_observed_run_id": "",
        "best_observed_score": None,
        "proposals": [],
    }
    executed_signatures = set(str(item) for item in existing.get("executed_proposal_signatures", []))
    next_proposal = {}
    if resolved_mode == "continue" and selected_trials:
        recommendation_payload = build_bayesian_recommendations(
            selected_trials,
            objective_names=objective_names,
            search_space_payload=_NOTEBOOK_OPTIMIZATION_SEARCH_SPACE,
            proposal_count=5,
        )
        recommendation_cohort = (
            dict(recommendation_payload.get("cohorts", [{}])[0])
            if recommendation_payload.get("cohorts")
            else recommendation_cohort
        )
        next_proposal = _select_next_campaign_proposal(recommendation_cohort, executed_signatures)

    status = "disabled"
    if resolved_mode == "stop":
        status = "stopped"
        next_proposal = {}
    elif resolved_mode == "continue":
        status = "active" if next_proposal else ("complete" if selected_trials else "bootstrap_pending")

    campaign: Dict[str, Any] = {
        "schema_version": _NOTEBOOK_OPTIMIZATION_CAMPAIGN_SCHEMA,
        "campaign_seed": campaign_seed,
        "campaign_json": str(campaign_path),
        "mode": resolved_mode,
        "status": status,
        "dataset_key": dataset_identity.get("dataset_key"),
        "dataset_lineage_key": dataset_identity.get("dataset_lineage_key"),
        "dataset_lineage": dataset_identity.get("dataset_lineage", {}),
        "crop_name": dataset_identity.get("crop_name"),
        "part_name": dataset_identity.get("part_name"),
        "engine": engine,
        "backbone_model_name": backbone_model_name,
        "cohort_match_mode": cohort_match_mode,
        "cohort_key": str(recommendation_cohort.get("cohort_key", "") or frontier_cohort.get("cohort_key", "") or ""),
        "frontier_count": int(frontier_cohort.get("frontier_count", 0) or 0),
        "frontier_run_ids": list(frontier_cohort.get("frontier_run_ids", [])),
        "eligible_run_count": int(recommendation_cohort.get("eligible_run_count", 0) or 0),
        "executed_run_ids": list(existing.get("executed_run_ids", [])),
        "executed_proposal_signatures": list(executed_signatures),
        "proposal_history": list(existing.get("proposal_history", [])),
        "selected_proposal": dict(existing.get("selected_proposal", {}))
        if isinstance(existing.get("selected_proposal"), dict)
        else {},
        "next_proposal": next_proposal,
        "objective_names": objective_names,
        "manifest_path": dataset_identity.get("manifest_path"),
        "manifest_sha256": dataset_identity.get("manifest_sha256"),
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "notebook_parameters": dict(notebook_parameters or {}),
        "recommendation_generated_at": "",
    }
    if existing:
        campaign.setdefault("created_at", existing.get("created_at"))
    if not campaign.get("created_at"):
        campaign["created_at"] = datetime.utcnow().isoformat() + "Z"
    write_json(campaign_path, campaign, ensure_ascii=False)
    _call_if_present(
        telemetry,
        "update_latest",
        {
            "phase": "optimization_campaign_resolved",
            "optimization_campaign": _build_campaign_snapshot(campaign),
        },
    )
    return campaign


def apply_notebook_optimization_proposal(
    *,
    notebook_parameters: Dict[str, Any],
    campaign: Optional[Dict[str, Any]],
    telemetry: Any = None,
    print_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    resolved_campaign = dict(campaign or {})
    emit = print if print_fn is None else print_fn
    proposal = (
        dict(resolved_campaign.get("next_proposal", {}))
        if isinstance(resolved_campaign.get("next_proposal"), dict)
        else {}
    )
    updated = dict(notebook_parameters or {})
    changes: List[Dict[str, Any]] = []
    if str(resolved_campaign.get("status", "") or "") != "active" or not proposal:
        return {
            "applied": False,
            "notebook_parameters": updated,
            "changes": changes,
            "campaign": resolved_campaign,
            "proposal": proposal,
        }

    proposal_parameters = dict(proposal.get("parameters", {}))
    for flat_name, notebook_name in _NOTEBOOK_PARAM_MAP.items():
        if flat_name not in proposal_parameters or notebook_name not in updated:
            continue
        before = updated[notebook_name]
        after = coerce_like_value(before, proposal_parameters.get(flat_name))
        if before == after:
            continue
        updated[notebook_name] = after
        changes.append(
            {
                "parameter_key": flat_name,
                "notebook_name": notebook_name,
                "before": before,
                "after": after,
            }
        )

    resolved_campaign["selected_proposal"] = proposal
    write_json(Path(str(resolved_campaign.get("campaign_json", ""))), resolved_campaign, ensure_ascii=False)
    if changes:
        emit(f"[OPT] Applying proposal rank={proposal.get('rank')} signature={proposal.get('signature', '')[:16]}...")
        for change in changes:
            emit(
                f"[OPT] {change['notebook_name']}: {change['before']} -> {change['after']} ({change['parameter_key']})"
            )
    _call_if_present(
        telemetry,
        "update_latest",
        {
            "phase": "optimization_proposal_applied",
            "optimization_campaign": _build_campaign_snapshot(resolved_campaign),
            "optimization_change_count": len(changes),
        },
    )
    return {
        "applied": bool(changes),
        "notebook_parameters": updated,
        "changes": changes,
        "campaign": resolved_campaign,
        "proposal": proposal,
    }


def select_notebook_ood_policy(
    *,
    trainer: Any,
    loaders: Dict[str, Any],
    continual_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Select and apply the notebook OOD policy without touching final OOD evidence."""
    from src.training.services.ood_score_selection import (
        is_auto_primary_score_method,
        normalize_requested_primary_score_method,
        resolve_runtime_primary_score_method,
    )
    from src.training.validation import evaluate_model_with_artifact_metrics
    from src.workflows.run_setup import loader_size
    from src.workflows.stages import (
        apply_primary_score_method_to_trainer,
        select_and_apply_real_ood_dev_policy,
    )

    ood_cfg = dict(continual_config.get("ood", {}))
    requested_method = normalize_requested_primary_score_method(ood_cfg.get("primary_score_method", "auto"))
    selected_method = apply_primary_score_method_to_trainer(
        trainer,
        resolve_runtime_primary_score_method(requested_method),
    )
    selection_source = "configured"
    selection: Dict[str, Any] = {}

    real_ood_present = loader_size(loaders.get("ood")) > 0
    if real_ood_present and is_auto_primary_score_method(requested_method):
        ood_dev_loader = loaders.get("ood_dev")
        id_dev_loader = loaders.get("val") if loader_size(loaders.get("val")) > 0 else loaders.get("train")
        if (
            bool(ood_cfg.get("real_dev_selection_enabled", True))
            and loader_size(ood_dev_loader) > 0
            and loader_size(id_dev_loader) > 0
        ):
            selection = select_and_apply_real_ood_dev_policy(
                trainer=trainer,
                loaders=loaders,
                requested_method=requested_method,
                fallback_method=selected_method,
                target_fpr=float(ood_cfg.get("real_dev_target_fpr", 0.05)),
                evaluate_fn=evaluate_model_with_artifact_metrics,
            )
            if selection:
                selected_method = str(selection["selected_primary_score_method"])
                selection_source = "real_ood_dev"
            else:
                selection_source = "real_ood_guardrail_no_dev"
        else:
            selection_source = "real_ood_guardrail_no_dev"

    return {
        "requested_primary_score_method": requested_method,
        "selected_primary_score_method": selected_method,
        "selection_source": selection_source,
        "selected_threshold": selection.get("selected_threshold"),
        "selection": selection,
    }


def calibrate_notebook_ood_policy(
    *,
    adapter: Any,
    loaders: Dict[str, Any],
    continual_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Calibrate on ID validation, then select method/threshold on disjoint OOD dev."""
    from src.workflows.readiness import record_primary_score_selection
    from src.workflows.run_setup import loader_size

    val_loader = loaders.get("val")
    if loader_size(val_loader) <= 0:
        raise RuntimeError("Validation loader bos; OOD kalibrasyonu yapilamaz.")
    calibration = adapter.calibrate_ood(val_loader)
    trainer = getattr(adapter, "_trainer", None)
    if trainer is None:
        raise RuntimeError("Trainer hazir degil.")
    policy = select_notebook_ood_policy(
        trainer=trainer,
        loaders=loaders,
        continual_config=continual_config,
    )
    record_primary_score_selection(
        calibration,
        requested_primary_score_method=policy["requested_primary_score_method"],
        selected_primary_score_method=policy["selected_primary_score_method"],
        selection_source=policy["selection_source"],
        selection_metadata=dict(policy.get("selection") or {}),
    )
    metadata_setter = getattr(adapter, "set_export_metadata", None)
    if callable(metadata_setter):
        metadata_setter(ood_calibration=dict(calibration.get("ood_calibration") or {}))
    return {"calibration": calibration, "policy": policy}


