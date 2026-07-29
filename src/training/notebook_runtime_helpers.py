#!/usr/bin/env python3
"""Notebook 2 helper functions only."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib

from src.shared.adapter_paths import build_adapter_bundle_root
from src.shared.hash_utils import sha256_file
from src.shared.json_utils import clone_jsonable, coerce_like_value, read_json, write_json
from src.shared.string_utils import normalize_notebook_identifier, slug_label_component
from src.training.notebook_runtime_session import (
    NotebookTrainingStatusPrinter as NotebookTrainingStatusPrinter,
)
from src.training.notebook_runtime_session import _artifact_dir as _artifact_dir
from src.training.notebook_runtime_session import _call_if_present as _call_if_present
from src.training.notebook_runtime_session import (
    _resolve_colab_runtime_api as _resolve_colab_runtime_api,
)
from src.training.notebook_runtime_session import build_history_snapshot as build_history_snapshot
from src.training.notebook_runtime_session import (
    build_notebook_completion_report as build_notebook_completion_report,
)
from src.training.notebook_runtime_session import build_notebook_run_dir as build_notebook_run_dir
from src.training.notebook_runtime_session import build_notebook_run_id as build_notebook_run_id
from src.training.notebook_runtime_session import (
    cleanup_notebook_training_state as cleanup_notebook_training_state,
)
from src.training.notebook_runtime_session import (
    ensure_notebook_checkpoint_manager as _ensure_notebook_checkpoint_manager_impl,
)
from src.training.notebook_runtime_session import (
    maybe_auto_disconnect_colab_runtime as _maybe_auto_disconnect_colab_runtime_impl,
)
from src.training.notebook_runtime_session import merge_training_summary_fields as merge_training_summary_fields
from src.training.notebook_runtime_session import notebook_artifact_root as notebook_artifact_root
from src.training.notebook_runtime_session import (
    persist_behavioral_acceptance_artifact as persist_behavioral_acceptance_artifact,
)
from src.training.notebook_runtime_session import (
    persist_behavioral_dev_report_artifact as persist_behavioral_dev_report_artifact,
)
from src.training.notebook_runtime_session import (
    persist_production_readiness_artifact as persist_production_readiness_artifact,
)
from src.training.notebook_runtime_session import persist_training_curve_figure as persist_training_curve_figure
from src.training.notebook_runtime_session import (
    persist_training_history_artifacts as persist_training_history_artifacts,
)
from src.training.notebook_runtime_session import (
    persist_training_run_context_artifact as persist_training_run_context_artifact,
)
from src.training.notebook_runtime_session import persist_validation_artifacts as persist_validation_artifacts
from src.training.notebook_runtime_session import run_notebook_training_session as run_notebook_training_session
from src.training.notebook_runtime_session import save_notebook_checkpoint as save_notebook_checkpoint

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
    import json

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


def prepare_notebook_access_and_dataset(
    *,
    root: Path,
    base_config: Dict[str, Any],
    crop_name: str,
    dataset_name: str,
    runtime_dataset_root: str | Path,
    ood_root: str | Path = "",
    oe_root: str | Path = "",
    ask_for_ood_root: bool = False,
    optimization_campaign_mode: str = "disabled",
    telemetry: Any = None,
    print_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    from src.data.dataset_layout import (
        list_repo_dataset_directories,
        resolve_direct_repo_dataset_root,
        resolve_repo_relative_root,
    )
    from src.notebook.repo_bootstrap import (
        collect_notebook_access_report,
        print_notebook_access_report,
    )

    emit = print if print_fn is None else print_fn
    backbone_model = str(
        dict(dict(base_config or {}).get("training", {}).get("continual", {})).get("backbone", {}).get("model_name", "")
    ).strip()
    access_report = collect_notebook_access_report(
        repo_root=root,
        hf_model_ids=[backbone_model] if backbone_model else [],
    )
    print_notebook_access_report(access_report, print_fn=emit)
    _call_if_present(
        telemetry,
        "update_latest",
        {
            "phase": "access_checked",
            "github_read_access": access_report.get("github", {}).get("read_access_mode"),
            "repo_update_relation": access_report.get("repo_updates", {}).get("relation"),
            "hf_access_mode": access_report.get("huggingface", {}).get("access_mode"),
        },
    )

    crop_key = normalize_notebook_identifier(crop_name)
    if not crop_key:
        raise RuntimeError("CROP_NAME bos olmayan bir crop anahtarina cozulmeli.")

    runtime_parent = resolve_repo_relative_root(repo_root=root, repo_relative_root=runtime_dataset_root)
    direct_runtime_dataset = resolve_direct_repo_dataset_root(
        repo_root=root,
        repo_relative_root=runtime_dataset_root,
    )
    runtime_dirs = (
        []
        if direct_runtime_dataset is not None
        else list_repo_dataset_directories(
            repo_root=root,
            repo_relative_root=runtime_dataset_root,
        )
    )
    candidates: List[Dict[str, Any]] = []
    if direct_runtime_dataset is not None:
        direct_runtime_name, direct_runtime_path = direct_runtime_dataset
        candidates.append({"name": direct_runtime_name, "path": direct_runtime_path, "parent": runtime_parent})
    else:
        candidates.extend({"name": path.name, "path": path, "parent": runtime_parent} for path in runtime_dirs)
    if not candidates:
        raise RuntimeError(
            "No prepared runtime datasets were found under RUNTIME_DATASET_ROOT. Notebook 0'u once calistirin."
        )

    requested_dataset_name = str(dataset_name).strip()
    if requested_dataset_name:
        matches = [item for item in candidates if item["name"] == requested_dataset_name]
        if not matches:
            available_options = [item["name"] for item in candidates]
            raise RuntimeError(
                f"Requested dataset '{requested_dataset_name}' was not found. Available options: {available_options}"
            )
        selected = matches[0]
    elif len(candidates) == 1:
        selected = candidates[0]
        emit(f"[DATASET] Yalnizca bir runtime dataset bulundu, otomatik secildi: {selected['name']}")
    else:
        emit("[DATASET] Kullanilabilir runtime dataset secenekleri:")
        for index, item in enumerate(candidates, start=1):
            emit(f"  [{index}] {item['name']} ({item['path']})")
        raw_choice = str(input(f"Kullanilacak dataset icin numara ya da isim girin (1-{len(candidates)}): ")).strip()
        if not raw_choice:
            raise RuntimeError("Dataset secimi bos birakilamaz.")
        if raw_choice.isdigit():
            selected_index = int(raw_choice) - 1
            if selected_index < 0 or selected_index >= len(candidates):
                raise RuntimeError(f"Dataset secim index'i aralik disi: {raw_choice}")
            selected = candidates[selected_index]
        else:
            matches = [item for item in candidates if item["name"] == raw_choice]
            if not matches:
                raise RuntimeError(f"Dataset secimi bulunamadi: {raw_choice}")
            selected = matches[0]

    selected_dataset_name = str(selected["name"])
    selected_dataset_root = Path(selected["path"])
    if not selected_dataset_name.startswith(crop_key):
        raise RuntimeError(f"Secilen runtime dataset CROP_NAME ile uyusmuyor: {selected_dataset_name} vs {crop_key}")
    missing_splits = [name for name in ("continual", "val", "test") if not (selected_dataset_root / name).is_dir()]
    if missing_splits:
        raise RuntimeError(f"Prepared runtime dataset is missing split folder(s): {missing_splits}")
    class_names = sorted(d.name for d in (selected_dataset_root / "continual").iterdir() if d.is_dir())
    if not class_names:
        raise RuntimeError(f"No class subdirectories in prepared runtime split: {selected_dataset_root / 'continual'}")

    runtime_root = selected_dataset_root.parent
    default_ood_root = selected_dataset_root / "ood"
    default_oe_root = selected_dataset_root / "oe"
    requested_ood_root = str(ood_root or "").strip()
    requested_oe_root = str(oe_root or "").strip()
    if ask_for_ood_root and not requested_ood_root:
        default_hint = str(default_ood_root) if default_ood_root.is_dir() else ""
        prompt = "OOD klasoru yolunu girin"
        if default_hint:
            prompt += f" [Enter={default_hint}]"
        requested_ood_root = str(input(prompt + ": ")).strip()
        if not requested_ood_root and default_hint:
            requested_ood_root = default_hint

    if requested_ood_root:
        resolved_ood_root = Path(requested_ood_root).expanduser()
        if not resolved_ood_root.is_absolute():
            resolved_ood_root = (root / resolved_ood_root).resolve()
        if not resolved_ood_root.is_dir():
            raise RuntimeError(f"OOD klasoru bulunamadi veya klasor degil: {resolved_ood_root}")
        emit(f"[OOD] explicit ood root={resolved_ood_root}")
        resolved_ood_root_value = str(resolved_ood_root)
    elif default_ood_root.is_dir():
        emit(f"[OOD] runtime ood root={default_ood_root}")
        resolved_ood_root_value = str(default_ood_root)
    else:
        emit("[OOD] Gercek OOD split secilmedi; fallback held-out benchmark kullanilabilir.")
        resolved_ood_root_value = ""
    if requested_oe_root:
        resolved_oe_root = Path(requested_oe_root).expanduser()
        if not resolved_oe_root.is_absolute():
            resolved_oe_root = (root / resolved_oe_root).resolve()
        if not resolved_oe_root.is_dir():
            raise RuntimeError(f"OE klasoru bulunamadi veya klasor degil: {resolved_oe_root}")
        emit(f"[OE] explicit oe root={resolved_oe_root}")
        resolved_oe_root_value = str(resolved_oe_root)
    elif default_oe_root.is_dir():
        emit(f"[OE] runtime oe root={default_oe_root}")
        resolved_oe_root_value = str(default_oe_root)
    else:
        resolved_oe_root_value = ""
    emit(f"[DATASET] runtime root={selected_dataset_root} classes={len(class_names)}: {class_names}")

    _call_if_present(
        telemetry,
        "update_latest",
        {
            "phase": "dataset_validated",
            "dataset_root": str(selected_dataset_root),
            "runtime_dataset_root": str(runtime_root),
            "runtime_dataset_key": selected_dataset_name,
            "selected_dataset_name": selected_dataset_name,
            "resolved_ood_root": resolved_ood_root_value,
            "resolved_oe_root": resolved_oe_root_value,
            "class_count": len(class_names),
        },
    )

    resolved_mode = str(optimization_campaign_mode or "").strip().lower()
    if resolved_mode in {"continue", "stop"}:
        emit("[OPT] Bayesian campaign mode requested; proposals update after run traceability refresh.")
    else:
        emit(
            f"[OPT] Bayesian campaign automation is disabled. dataset_lineage={selected_dataset_root / 'split_manifest.json'}"
        )

    return {
        "access_report": access_report,
        "class_names": class_names,
        "validated": True,
        "runtime_dataset_root": runtime_root,
        "runtime_dataset_key": selected_dataset_name,
        "selected_dataset_name": selected_dataset_name,
        "selected_dataset_root": selected_dataset_root,
        "resolved_ood_root": resolved_ood_root_value,
        "resolved_oe_root": resolved_oe_root_value,
    }


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


def finalize_notebook_optimization_campaign(
    *,
    root: Path,
    campaign: Optional[Dict[str, Any]],
    run_id: str,
    telemetry: Any = None,
) -> Dict[str, Any]:
    from src.training.services.optimization import (
        build_bayesian_recommendations,
        build_pareto_frontiers,
        select_trials_for_cohort,
    )
    from src.training.services.run_registry import build_run_registry

    payload = dict(campaign or {})
    campaign_json = str(payload.get("campaign_json", "") or "")
    if not campaign_json:
        return payload
    campaign_path = Path(campaign_json)
    if campaign_path.exists():
        stored = read_json(campaign_path, default={}, expect_type=dict)
        if isinstance(stored, dict):
            payload = dict(stored)

    runs_root = root / "runs"
    index_root = root / "runs" / "_index"
    resolved_mode = str(payload.get("mode", "") or "").strip().lower()
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
        dataset_lineage_key=str(payload.get("dataset_lineage_key", "") or ""),
        dataset_key=str(payload.get("dataset_key", "") or ""),
        crop_name=str(payload.get("crop_name", "") or ""),
        part_name=str(payload.get("part_name", "") or ""),
        backbone_model_name=str(payload.get("backbone_model_name", "") or ""),
        engine=str(payload.get("engine", "") or "continual_sd_lora"),
    )
    objective_names = list(payload.get("objective_names", _DEFAULT_PARETO_OBJECTIVES))
    pareto_payload = build_pareto_frontiers(selected_trials, objective_names=objective_names)
    frontier_cohort = pareto_payload.get("cohorts", [{}])[0] if pareto_payload.get("cohorts") else {}
    recommendation_cohort = {
        "cohort_key": str(frontier_cohort.get("cohort_key", "") or payload.get("cohort_key", "")),
        "eligible_run_count": 0,
        "search_strategy": "disabled",
        "best_observed_run_id": "",
        "best_observed_score": None,
        "proposals": [],
    }

    executed_run_ids = list(payload.get("executed_run_ids", []))
    if run_id and run_id not in executed_run_ids:
        executed_run_ids.append(run_id)
    executed_signatures = list(payload.get("executed_proposal_signatures", []))
    selected_proposal = (
        dict(payload.get("selected_proposal", {})) if isinstance(payload.get("selected_proposal"), dict) else {}
    )
    selected_signature = str(selected_proposal.get("signature", "") or "")
    if selected_signature and selected_signature not in executed_signatures:
        executed_signatures.append(selected_signature)

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
    next_proposal = _select_next_campaign_proposal(recommendation_cohort, set(executed_signatures))

    payload.update(
        {
            "cohort_key": str(
                recommendation_cohort.get("cohort_key", "")
                or frontier_cohort.get("cohort_key", "")
                or payload.get("cohort_key", "")
            ),
            "cohort_match_mode": cohort_match_mode,
            "frontier_count": int(frontier_cohort.get("frontier_count", 0) or 0),
            "frontier_run_ids": list(frontier_cohort.get("frontier_run_ids", [])),
            "eligible_run_count": int(recommendation_cohort.get("eligible_run_count", 0) or 0),
            "executed_run_ids": executed_run_ids,
            "executed_proposal_signatures": executed_signatures,
            "last_completed_run_id": str(run_id or payload.get("last_completed_run_id", "") or ""),
            "next_proposal": next_proposal,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "recommendation_generated_at": "",
        }
    )
    if resolved_mode == "stop":
        payload["status"] = "stopped"
        payload["next_proposal"] = {}
    elif resolved_mode == "continue" and payload.get("next_proposal"):
        payload["status"] = "active"
    elif resolved_mode == "continue":
        payload["status"] = "complete" if selected_trials else "bootstrap_pending"
    else:
        payload["status"] = "disabled"
        payload["next_proposal"] = {}

    if selected_signature:
        history = list(payload.get("proposal_history", []))
        history.append(
            {
                "run_id": str(run_id or ""),
                "signature": selected_signature,
                "rank": selected_proposal.get("rank"),
                "completed_at": datetime.utcnow().isoformat() + "Z",
            }
        )
        payload["proposal_history"] = history

    write_json(campaign_path, payload, ensure_ascii=False)
    _call_if_present(
        telemetry,
        "update_latest",
        {
            "phase": "optimization_campaign_finalized",
            "optimization_campaign": _build_campaign_snapshot(payload),
        },
    )
    return payload


def _build_notebook_continual_config(
    *,
    base_config: Dict[str, Any],
    device: str,
    deterministic: bool,
    notebook_parameters: Dict[str, Any],
    data_settings: Dict[str, Any],
    ood_settings: Dict[str, Any],
    optimization_settings: Dict[str, Any],
    early_stopping_settings: Dict[str, Any],
) -> Dict[str, Any]:
    continual_cfg = clone_jsonable(dict(base_config or {}).get("training", {}).get("continual", {}))
    continual_cfg["device"] = device
    continual_cfg["num_epochs"] = int(notebook_parameters["EPOCHS"])
    continual_cfg["batch_size"] = int(notebook_parameters["BATCH_SIZE"])
    continual_cfg["learning_rate"] = float(notebook_parameters["LEARNING_RATE"])
    continual_cfg["weight_decay"] = float(notebook_parameters["WEIGHT_DECAY"])
    continual_cfg["deterministic"] = bool(deterministic)

    adapter_cfg = continual_cfg.setdefault("adapter", {})
    adapter_cfg["lora_r"] = int(notebook_parameters["LORA_R"])
    adapter_cfg["lora_alpha"] = int(notebook_parameters["LORA_ALPHA"])
    adapter_cfg["lora_dropout"] = float(notebook_parameters["LORA_DROPOUT"])

    fusion_cfg = continual_cfg.setdefault("fusion", {})
    if "FUSION_DROPOUT" in notebook_parameters:
        fusion_cfg["dropout"] = float(notebook_parameters["FUSION_DROPOUT"])

    data_cfg = continual_cfg.setdefault("data", {})
    data_cfg["augmentation_policy"] = str(
        notebook_parameters.get("AUGMENTATION_POLICY", data_settings["AUGMENTATION_POLICY"])
    )
    data_cfg["randaugment_num_ops"] = int(
        notebook_parameters.get("RANDAUGMENT_NUM_OPS", data_settings["RANDAUGMENT_NUM_OPS"])
    )
    data_cfg["randaugment_magnitude"] = int(notebook_parameters["RANDAUGMENT_MAGNITUDE"])
    data_cfg["augmix_severity"] = int(
        notebook_parameters.get("AUGMIX_SEVERITY", data_settings.get("AUGMIX_SEVERITY", 3))
    )
    data_cfg["augmix_width"] = int(data_settings.get("AUGMIX_WIDTH", 3))
    data_cfg["augmix_depth"] = int(data_settings.get("AUGMIX_DEPTH", -1))
    data_cfg["augmix_alpha"] = float(data_settings.get("AUGMIX_ALPHA", 1.0))
    data_cfg["allow_under_min_training"] = bool(data_settings["ALLOW_UNDER_MIN_TRAINING"])

    ood_cfg = continual_cfg.setdefault("ood", {})
    ood_cfg["threshold_factor"] = float(notebook_parameters["OOD_FACTOR"])
    for key, value in ood_settings.items():
        ood_cfg[key] = value
    if "OE_LOSS_WEIGHT" in notebook_parameters:
        ood_cfg["oe_loss_weight"] = float(notebook_parameters["OE_LOSS_WEIGHT"])
    if "REACT_ENABLED" in notebook_parameters:
        ood_cfg["react_enabled"] = bool(notebook_parameters["REACT_ENABLED"])
    if "REACT_PERCENTILE" in notebook_parameters:
        ood_cfg["react_percentile"] = float(notebook_parameters["REACT_PERCENTILE"])

    classifier_rebalance_cfg = continual_cfg.setdefault("classifier_rebalance", {})
    if "CLASSIFIER_REBALANCE_ENABLED" in notebook_parameters:
        classifier_rebalance_cfg["enabled"] = bool(notebook_parameters["CLASSIFIER_REBALANCE_ENABLED"])
    if "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU" in notebook_parameters:
        classifier_rebalance_cfg["logit_adjustment_tau"] = float(
            notebook_parameters["CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU"]
        )

    optimization_cfg = continual_cfg.setdefault("optimization", {})
    for key, value in optimization_settings.items():
        if key == "scheduler":
            continue
        optimization_cfg[key] = value
    optimization_cfg["grad_accumulation_steps"] = int(optimization_cfg["grad_accumulation_steps"])
    optimization_cfg["max_grad_norm"] = float(optimization_cfg["max_grad_norm"])
    optimization_cfg["mixed_precision"] = str(optimization_cfg["mixed_precision"])
    optimization_cfg["label_smoothing"] = float(
        notebook_parameters.get("LABEL_SMOOTHING", optimization_cfg["label_smoothing"])
    )
    optimization_cfg["loss_name"] = str(optimization_cfg["loss_name"]).strip().lower()
    optimization_cfg["logitnorm_tau"] = float(notebook_parameters["LOGITNORM_TAU"])
    scheduler_cfg = optimization_cfg.setdefault("scheduler", {})
    scheduler_payload = dict(optimization_settings.get("scheduler", {}))
    scheduler_cfg["name"] = str(scheduler_payload.get("name", scheduler_cfg.get("name", "")))
    scheduler_cfg["warmup_ratio"] = float(scheduler_payload.get("warmup_ratio", scheduler_cfg.get("warmup_ratio", 0.0)))
    scheduler_cfg["min_lr"] = float(scheduler_payload.get("min_lr", scheduler_cfg.get("min_lr", 0.0)))
    scheduler_cfg["step_on"] = str(scheduler_payload.get("step_on", scheduler_cfg.get("step_on", "batch")))

    early_stopping_cfg = continual_cfg.setdefault("early_stopping", {})
    early_stopping_cfg["enabled"] = True
    early_stopping_cfg["patience"] = int(early_stopping_settings["EARLY_STOPPING_PATIENCE"])
    early_stopping_cfg["min_delta"] = float(early_stopping_settings["EARLY_STOPPING_MIN_DELTA"])
    return continual_cfg


def initialize_notebook_training_engine(
    *,
    root: Path,
    state: Dict[str, Any],
    base_config: Dict[str, Any],
    crop_name: str,
    part_name: str,
    device: str,
    deterministic: bool,
    notebook_parameters: Dict[str, Any],
    optimization_campaign_mode: str,
    data_settings: Dict[str, Any],
    loader_settings: Dict[str, Any],
    ood_settings: Dict[str, Any],
    optimization_settings: Dict[str, Any],
    early_stopping_settings: Dict[str, Any],
    telemetry: Any = None,
    print_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    from src.adapter.independent_crop_adapter import IndependentCropAdapter
    from src.data.dataset_layout import resolve_notebook_training_classes
    from src.data.loaders import create_training_loaders

    emit = print if print_fn is None else print_fn
    if not state.get("validated"):
        raise RuntimeError("Once dataset validation hucresini calistirin.")

    resolved_crop_name = str(crop_name).strip().lower()
    dataset_key = str(state.get("runtime_dataset_key", "") or "").strip()
    if not dataset_key:
        raise RuntimeError("Runtime dataset key cozulmedi. Once dataset validation hucresini calistirin.")
    runtime_data_root = Path(state.get("runtime_dataset_root") or "")
    if not runtime_data_root.is_absolute():
        runtime_data_root = (root / runtime_data_root).resolve()
    class_root = runtime_data_root / dataset_key / "continual"
    if not class_root.is_dir():
        raise RuntimeError(f"Prepared runtime continual split not found: {class_root}")

    raw_class_names = sorted(path.name for path in class_root.iterdir() if path.is_dir())
    normalized_to_raw: Dict[str, str] = {}
    for raw_name in raw_class_names:
        normalized_name = normalize_notebook_identifier(raw_name)
        if not normalized_name:
            continue
        existing_raw = normalized_to_raw.get(normalized_name)
        if existing_raw is not None and existing_raw != raw_name:
            raise RuntimeError(f"Class names collapse to the same normalized key: {existing_raw!r} and {raw_name!r}")
        normalized_to_raw[normalized_name] = raw_name
    available = sorted(normalized_to_raw)
    resolved_ood_root = str(state.get("resolved_ood_root") or "").strip()
    resolved_oe_root = str(state.get("resolved_oe_root") or "").strip()
    if resolved_ood_root:
        emit(f"[OOD] selected ood root={resolved_ood_root}")
    if resolved_oe_root:
        emit(f"[OE] selected oe root={resolved_oe_root}")

    class_resolution = resolve_notebook_training_classes(
        available_classes=available,
        crop_name=resolved_crop_name,
        taxonomy_path=root / "config" / "plant_taxonomy.json",
    )
    aligned = list(class_resolution.get("selected_classes", available))
    if not aligned:
        raise RuntimeError(f"No usable classes for crop '{resolved_crop_name}'. Available: {available}")
    final_class_names = [normalized_to_raw[name] for name in aligned if name in normalized_to_raw]
    if len(final_class_names) != len(aligned):
        missing = sorted(set(aligned) - set(normalized_to_raw))
        raise RuntimeError(f"Resolved class names are not present on disk after normalization: {missing}")

    emit(f"[CLASSES] {final_class_names}")
    emit(
        f"[CLASSES] mode={'taxonomy_filter' if class_resolution.get('used_taxonomy_filter') else 'dataset_fallback'} "
        f"reason={class_resolution.get('reason', 'unknown')} "
        f"matched={len(class_resolution.get('matched_classes', []))} "
        f"unmatched={len(class_resolution.get('unmatched_classes', []))}"
    )
    if class_resolution.get("unmatched_classes"):
        emit(f"[CLASSES] taxonomy-unmatched classes kept: {class_resolution['unmatched_classes']}")

    resolved_parameters = dict(notebook_parameters or {})
    resolved_optimization_campaign_mode = str(optimization_campaign_mode or "").strip().lower()
    if resolved_optimization_campaign_mode not in {"continue", "stop"}:
        resolved_optimization_campaign_mode = "disabled"
    optimization_campaign = resolve_notebook_optimization_campaign(
        root=root,
        runtime_dataset_root=runtime_data_root / dataset_key,
        dataset_key=dataset_key,
        crop_name=crop_name,
        part_name=part_name,
        backbone_model_name=str(
            dict(base_config or {}).get("training", {}).get("continual", {}).get("backbone", {}).get("model_name", "")
        ),
        notebook_parameters=resolved_parameters,
        mode=resolved_optimization_campaign_mode,
        telemetry=telemetry,
    )
    print_notebook_optimization_campaign_status(optimization_campaign, print_fn=emit)
    proposal_application = apply_notebook_optimization_proposal(
        notebook_parameters=resolved_parameters,
        campaign=optimization_campaign,
        telemetry=telemetry,
        print_fn=emit,
    )
    optimization_campaign = proposal_application.get("campaign", optimization_campaign)
    resolved_parameters = dict(proposal_application.get("notebook_parameters", resolved_parameters))
    if not proposal_application.get("applied"):
        if optimization_campaign.get("status") == "bootstrap_pending":
            emit("[OPT] No prior comparable run yet. Bootstrap run will use the visible notebook parameters.")
        elif optimization_campaign.get("status") == "stopped":
            emit("[OPT] Campaign is marked stopped. Notebook will keep the visible parameters as-is.")
        elif optimization_campaign.get("status") == "active" and not optimization_campaign.get("next_proposal"):
            emit("[OPT] No unseen proposal is available. Notebook will keep the visible parameters as-is.")

    continual_cfg = _build_notebook_continual_config(
        base_config=base_config,
        device=device,
        deterministic=deterministic,
        notebook_parameters=resolved_parameters,
        data_settings=data_settings,
        ood_settings=ood_settings,
        optimization_settings=optimization_settings,
        early_stopping_settings=early_stopping_settings,
    )
    emit(f"[ENGINE][OOD_CFG] {json.dumps(continual_cfg['ood'], sort_keys=True)}")
    optimization_cfg = continual_cfg.get("optimization", {})
    emit(
        f"[ENGINE][OPT_CFG] loss={optimization_cfg.get('loss_name')} tau={optimization_cfg.get('logitnorm_tau')} "
        f"label_smoothing={optimization_cfg.get('label_smoothing')} mixed_precision={optimization_cfg.get('mixed_precision')}"
    )

    adapter = IndependentCropAdapter(crop_name=resolved_crop_name, device=device)
    if hasattr(adapter, "part_name"):
        adapter.part_name = str(part_name or "unspecified").strip().lower() or "unspecified"
    emit("[ENGINE] Initializing adapter (may download backbone)...")
    adapter.initialize_engine(class_names=final_class_names, config={"training": {"continual": continual_cfg}})

    loader_kwargs: Dict[str, Any] = {}
    if int(loader_settings["NUM_WORKERS"]) > 0:
        loader_kwargs["prefetch_factor"] = int(loader_settings["PREFETCH"])
    loaders = create_training_loaders(
        data_dir=str(runtime_data_root),
        crop=dataset_key,
        class_names=final_class_names,
        batch_size=int(resolved_parameters["BATCH_SIZE"]),
        num_workers=int(loader_settings["NUM_WORKERS"]),
        use_cache=bool(loader_settings["USE_CACHE"]),
        cache_size=int(loader_settings["CACHE_SIZE"]),
        cache_train_split=bool(loader_settings["CACHE_TRAIN_SPLIT"]),
        target_size=int(loader_settings["TARGET_SIZE"]),
        error_policy=str(loader_settings["LOADER_ERROR_POLICY"]),
        sampler=str(loader_settings["DATA_SAMPLER"]),
        seed=int(loader_settings["SEED"]),
        validate_images_on_init=bool(loader_settings["VALIDATE_IMAGES_ON_INIT"]),
        augmentation_policy=str(data_settings["AUGMENTATION_POLICY"]),
        randaugment_num_ops=int(resolved_parameters.get("RANDAUGMENT_NUM_OPS", data_settings["RANDAUGMENT_NUM_OPS"])),
        randaugment_magnitude=int(resolved_parameters["RANDAUGMENT_MAGNITUDE"]),
        augmix_severity=int(resolved_parameters.get("AUGMIX_SEVERITY", data_settings.get("AUGMIX_SEVERITY", 3))),
        augmix_width=int(data_settings.get("AUGMIX_WIDTH", 3)),
        augmix_depth=int(data_settings.get("AUGMIX_DEPTH", -1)),
        augmix_alpha=float(data_settings.get("AUGMIX_ALPHA", 1.0)),
        ood_root=resolved_ood_root or None,
        oe_root=resolved_oe_root or None,
        pin_memory=bool(loader_settings["PIN_MEMORY"]),
        **loader_kwargs,
    )
    train_dataset_class_to_idx = dict(getattr(getattr(loaders.get("train"), "dataset", None), "class_to_idx", {}) or {})
    adapter_class_to_idx = dict(getattr(adapter, "class_to_idx", {}) or {})
    if train_dataset_class_to_idx and adapter_class_to_idx != train_dataset_class_to_idx:
        raise RuntimeError(
            "Adapter class_to_idx does not match the runtime dataset loader class_to_idx. "
            f"adapter={adapter_class_to_idx} loader={train_dataset_class_to_idx}"
        )

    verified_ood = verify_notebook_ood_config(
        continual_config=continual_cfg,
        threshold_factor=float(resolved_parameters["OOD_FACTOR"]),
        sure_semantic_percentile=float(ood_settings["sure_semantic_percentile"]),
        sure_confidence_percentile=float(ood_settings["sure_confidence_percentile"]),
        conformal_alpha=float(ood_settings["conformal_alpha"]),
        conformal_method=str(ood_settings["conformal_method"]),
        conformal_raps_lambda=float(ood_settings["conformal_raps_lambda"]),
        conformal_raps_k_reg=int(ood_settings["conformal_raps_k_reg"]),
    )
    emit("[VERIFY][OOD][expected] " + json.dumps(verified_ood["expected"], sort_keys=True))
    emit("[VERIFY][OOD][resolved] " + json.dumps(verified_ood["resolved"], sort_keys=True))
    emit("[VERIFY][OOD] OK: cozulmus OOD ayari istenen parametrelerle eslesiyor.")

    trainable = sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad)
    emit(f"[ENGINE] Hazir. trainable_params={trainable:,}  classes={len(final_class_names)}")

    state.update(
        {
            "runtime_dataset_root": runtime_data_root,
            "runtime_dataset_key": dataset_key,
            "crop_name": resolved_crop_name,
            "part_name": str(part_name or "unspecified").strip().lower() or "unspecified",
            "class_names": final_class_names,
            "class_resolution": class_resolution,
            "adapter": adapter,
            "loaders": loaders,
            "continual_config": continual_cfg,
            "loader_settings": dict(loader_settings),
            "notebook_parameters": dict(resolved_parameters),
            "optimization_campaign": optimization_campaign,
            "optimization_applied_params": proposal_application,
        }
    )
    _call_if_present(
        telemetry,
        "update_latest",
        {
            "phase": "engine_ready",
            "class_count": len(final_class_names),
            "runtime_dataset_root": str(runtime_data_root),
            "runtime_dataset_key": dataset_key,
            "selected_dataset_name": str(state.get("selected_dataset_name") or ""),
            "resolved_ood_root": str(state.get("resolved_ood_root") or ""),
            "resolved_oe_root": str(state.get("resolved_oe_root") or ""),
        },
    )
    return {
        "state": state,
        "notebook_parameters": resolved_parameters,
        "optimization_campaign": optimization_campaign,
        "proposal_application": proposal_application,
        "verified_ood": verified_ood,
        "trainable_params": trainable,
    }


def verify_notebook_ood_config(
    *,
    continual_config: Dict[str, Any],
    threshold_factor: float,
    sure_semantic_percentile: float,
    sure_confidence_percentile: float,
    conformal_alpha: float,
    conformal_method: str,
    conformal_raps_lambda: float,
    conformal_raps_k_reg: int,
) -> Dict[str, Any]:
    resolved_ood_cfg = dict(dict(continual_config or {}).get("ood", {}))
    expected_ood_cfg = {
        "threshold_factor": float(threshold_factor),
        "sure_semantic_percentile": float(sure_semantic_percentile),
        "sure_confidence_percentile": float(sure_confidence_percentile),
        "conformal_alpha": float(conformal_alpha),
        "conformal_method": str(conformal_method),
        "conformal_raps_lambda": float(conformal_raps_lambda),
        "conformal_raps_k_reg": int(conformal_raps_k_reg),
    }
    mismatches: List[str] = []
    for key, expected in expected_ood_cfg.items():
        actual = resolved_ood_cfg.get(key)
        if isinstance(expected, float):
            try:
                actual_float = float(actual)
            except Exception:
                import logging

                logging.exception("Unhandled exception")
                raise
                mismatches.append(f"{key}: expected={expected} actual={actual}")
                continue
            if abs(actual_float - expected) > 1e-12:
                mismatches.append(f"{key}: expected={expected} actual={actual_float}")
        elif actual != expected:
            mismatches.append(f"{key}: expected={expected} actual={actual}")
    if mismatches:
        raise RuntimeError("OOD config mismatch:\n - " + "\n - ".join(mismatches))
    return {
        "expected": expected_ood_cfg,
        "resolved": {key: resolved_ood_cfg.get(key) for key in expected_ood_cfg},
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
    from src.utils.training_helpers import loader_size
    from src.workflows.training_subroutines import (
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
    from src.utils.training_helpers import loader_size
    from src.workflows.training_readiness import record_primary_score_selection

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


def calibrate_and_save_notebook_adapter(
    *,
    root: Path,
    state: Dict[str, Any],
    telemetry: Any = None,
    rt_fn: Optional[Callable[[str], None]] = None,
    print_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    emit = print if print_fn is None else print_fn
    if state.get("adapter") is None or state.get("loaders") is None:
        raise RuntimeError("Once engine init hucresini calistirin.")

    adapter = state["adapter"]
    calibration_result = calibrate_notebook_ood_policy(
        adapter=adapter,
        loaders=state["loaders"],
        continual_config=dict(state.get("continual_config") or {}),
    )
    calibration = calibration_result["calibration"]
    policy = calibration_result["policy"]
    state["calibration"] = calibration
    state["ood_policy_selection"] = policy
    num_classes = calibration.get("ood_calibration", {}).get("num_classes", 0)
    version = calibration.get("ood_calibration", {}).get("version", 0)
    emit(f"[OOD] Kalibrasyon tamamlandi. classes={num_classes} version={version}")
    _call_if_present(
        telemetry,
        "update_latest",
        {
            "phase": "ood_calibrated",
            "ood_num_classes": num_classes,
            "ood_version": version,
            "ood_requested_primary_score_method": policy["requested_primary_score_method"],
            "ood_primary_score_method": policy["selected_primary_score_method"],
            "ood_primary_score_selection_source": policy["selection_source"],
        },
    )

    if rt_fn is not None:
        rt_fn("Cell 7: calibration and adapter save started", phase="export")

    notebook_output_root = Path(root) / "outputs" / "colab_notebook_training"
    crop_name = str(state.get("crop_name") or getattr(adapter, "crop_name", "") or "").strip().lower()
    part_name = str(state.get("part_name") or getattr(adapter, "part_name", "") or "unspecified").strip().lower()
    checkpoint_dir = build_adapter_bundle_root(notebook_output_root, crop_name, part_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter.save_adapter(str(checkpoint_dir))
    asset_dir = checkpoint_dir / "continual_sd_lora_adapter"
    state["adapter_export_dir"] = asset_dir

    emit("Kaydedilen adapter klasoru:")
    emit(str(asset_dir))
    if not asset_dir.exists():
        raise RuntimeError(f"Beklenen adapter klasoru bulunamadi: {asset_dir}")

    telemetry_adapter_root = (
        getattr(telemetry, "artifacts_dir", Path(root))
        / "adapter_export"
        / crop_name
        / part_name
        / "continual_sd_lora_adapter"
    )
    for path_in_adapter in sorted(asset_dir.rglob("*")):
        if path_in_adapter.is_file():
            relative_path = path_in_adapter.relative_to(notebook_output_root).as_posix()
            _call_if_present(telemetry, "copy_artifact_file", path_in_adapter, f"adapter_export/{relative_path}")
            try:
                emit(f" - {path_in_adapter.relative_to(Path(root))}")
            except Exception:
                import logging

                logging.exception("Unhandled exception")
                raise

    emit(f"Telemetry adapter klasoru: {telemetry_adapter_root}")
    _call_if_present(
        telemetry,
        "update_latest",
        {
            "phase": "adapter_saved",
            "adapter_export_dir": str(asset_dir),
        },
    )
    if rt_fn is not None:
        rt_fn("Cell 7: calibration and adapter save completed", phase="export")
    return {
        "calibration": calibration,
        "adapter_export_dir": asset_dir,
        "telemetry_adapter_root": telemetry_adapter_root,
    }


def _notebook_repo_extra_entries(
    *,
    repo_output_dir: Path,
    crop_name: str,
    part_name: str,
    repo_notebook_output_path: Path,
    repo_telemetry_dir: Path,
    repo_checkpoint_state_dir: Path,
) -> List[Dict[str, Any]]:
    adapter_dir = build_adapter_bundle_root(repo_output_dir, crop_name, part_name) / "continual_sd_lora_adapter"
    return [
        {
            "path": adapter_dir,
            "category": "adapter_export",
            "priority": "high",
            "title_tr": "Repo mirror adapter export klasoru",
            "description_tr": "Repo mirror icindeki adapter export klasoru.",
            "reader_goal": "Export edilen adapter klasorunu bulmak",
            "generated_by": "notebook_2",
            "decision_importance": "deploy_handoff",
            "read_order": 70,
        },
        {
            "path": repo_notebook_output_path,
            "category": "adapter_export",
            "priority": "medium",
            "title_tr": "Calistirilmis notebook exportu",
            "description_tr": "Bu kosuda calisan notebook'un kaydedilmis kopyasi.",
            "reader_goal": "Notebook'u ayni ciktiyla tekrar incelemek",
            "generated_by": "notebook_2",
            "decision_importance": "runtime_diagnostic",
            "read_order": 71,
        },
        {
            "path": repo_telemetry_dir / "events.jsonl",
            "category": "logs_and_checkpoints",
            "priority": "medium",
            "title_tr": "Telemetry event logu",
            "description_tr": "Notebook olayi bazli telemetry kaydi.",
            "reader_goal": "Notebook akisini olay bazinda incelemek",
            "generated_by": "notebook_2",
            "decision_importance": "runtime_diagnostic",
            "read_order": 80,
        },
        {
            "path": repo_telemetry_dir / "runtime.log",
            "category": "logs_and_checkpoints",
            "priority": "medium",
            "title_tr": "Runtime logu",
            "description_tr": "Notebook runtime boyunca yazilan metin logu.",
            "reader_goal": "Calisma sirasindaki log ciktilarini okumak",
            "generated_by": "notebook_2",
            "decision_importance": "runtime_diagnostic",
            "read_order": 81,
        },
        {
            "path": repo_telemetry_dir / "latest_status.json",
            "category": "logs_and_checkpoints",
            "priority": "low",
            "title_tr": "Son durum ozeti",
            "description_tr": "Notebook'un son durum snapshot'i.",
            "reader_goal": "Kosunun son durumunu hizli kontrol etmek",
            "generated_by": "notebook_2",
            "decision_importance": "runtime_diagnostic",
            "read_order": 82,
        },
        {
            "path": repo_telemetry_dir / "summary.json",
            "category": "logs_and_checkpoints",
            "priority": "high",
            "title_tr": "Telemetry ozeti",
            "description_tr": "Notebook final ozet dosyasi.",
            "reader_goal": "Notebook final ozetini okumak",
            "generated_by": "notebook_2",
            "decision_importance": "run_overview",
            "read_order": 83,
        },
        {
            "path": repo_checkpoint_state_dir / "best_checkpoint.json",
            "category": "logs_and_checkpoints",
            "priority": "medium",
            "title_tr": "Best checkpoint manifesti",
            "description_tr": "En iyi checkpoint'in repo mirror manifesti.",
            "reader_goal": "Hangi checkpoint secildigini gormek",
            "generated_by": "notebook_2",
            "decision_importance": "runtime_diagnostic",
            "read_order": 84,
        },
        {
            "path": repo_checkpoint_state_dir / "latest_checkpoint.json",
            "category": "logs_and_checkpoints",
            "priority": "medium",
            "title_tr": "Latest checkpoint manifesti",
            "description_tr": "Son checkpoint manifesti.",
            "reader_goal": "Checkpoint akisini gormek",
            "generated_by": "notebook_2",
            "decision_importance": "runtime_diagnostic",
            "read_order": 85,
        },
        {
            "path": repo_checkpoint_state_dir / "checkpoint_index.json",
            "category": "logs_and_checkpoints",
            "priority": "low",
            "title_tr": "Checkpoint indexi",
            "description_tr": "Mirror edilen checkpoint manifest listesi.",
            "reader_goal": "Checkpoint kayitlarini toplu gormek",
            "generated_by": "notebook_2",
            "decision_importance": "runtime_diagnostic",
            "read_order": 86,
        },
    ]


def _evaluate_notebook_split(
    *,
    root: Path,
    trainer: Any,
    loader: Any,
    ood_loader: Any,
    classes: List[str],
    crop_name: str,
    part_name: str,
    run_id: str,
    split_name: str,
    artifact_subdir: str,
    label: str,
    requested_primary_score_method: str = "auto",
    selected_primary_score_method: str = "ensemble",
    selection_source: str = "configured",
    telemetry: Any = None,
    print_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    from src.training.services.ood_score_selection import apply_primary_score_method_to_evaluation
    from src.training.validation import evaluate_model_with_artifact_metrics

    emit = print if print_fn is None else print_fn
    evaluation = evaluate_model_with_artifact_metrics(trainer, loader, ood_loader=ood_loader)
    if evaluation is None:
        raise RuntimeError("Degerlendirme ornegi bulunamadi.")
    evaluation = apply_primary_score_method_to_evaluation(
        evaluation,
        selected_primary_score_method,
        requested_primary_score_method=requested_primary_score_method,
        selection_source=selection_source,
    )
    artifacts = persist_validation_artifacts(
        root=root,
        y_true=evaluation.y_true,
        y_pred=evaluation.y_pred,
        classes=classes,
        telemetry=telemetry,
        artifact_subdir=artifact_subdir,
        ood_labels=evaluation.ood_labels,
        ood_scores=evaluation.ood_scores,
        ood_scores_by_method=getattr(evaluation, "ood_scores_by_method", None),
        sure_ds_f1=evaluation.sure_ds_f1,
        conformal_empirical_coverage=evaluation.conformal_empirical_coverage,
        conformal_avg_set_size=evaluation.conformal_avg_set_size,
        ood_type_breakdown=getattr(evaluation, "ood_type_breakdown", None),
        prediction_rows=getattr(evaluation, "prediction_rows", None),
        context={
            "crop_name": crop_name,
            "part_name": part_name,
            "run_id": run_id,
            "split_name": split_name,
            **evaluation.context,
        },
    )
    metrics = artifacts["metric_gate"]["metrics"]
    extras: List[str] = []
    if metrics.get("ood_auroc") is not None:
        extras.append(f"ood_auroc={float(metrics['ood_auroc']):.4f}")
    if metrics.get("sure_ds_f1") is not None:
        extras.append(f"sure_ds_f1={float(metrics['sure_ds_f1']):.4f}")
    if metrics.get("conformal_empirical_coverage") is not None:
        extras.append(f"conformal_cov={float(metrics['conformal_empirical_coverage']):.4f}")
    suffix = " " + " ".join(extras) if extras else ""
    accuracy = float(artifacts["report_dict"].get("accuracy", 0.0))
    emit(f"[{label}] ornek={len(evaluation.y_true)} sinif={len(classes)} accuracy={accuracy:.4f}{suffix}")
    return artifacts


def complete_notebook_training_run(
    *,
    root: Path,
    state: Dict[str, Any],
    base_config: Dict[str, Any],
    crop_name: str,
    part_name: str,
    run_id: str,
    device: str,
    epochs: int,
    runtime_dataset_root: str,
    repo_run_dir: Path,
    repo_output_dir: Path,
    repo_telemetry_dir: Path,
    repo_checkpoint_state_dir: Path,
    repo_notebook_output_path: Path,
    auto_push_to_github: bool,
    auto_push_remote_name: str,
    auto_push_branch: str,
    auto_disconnect_runtime: bool,
    auto_disconnect_grace_seconds: float,
    save_run_outputs_to_repo_fn: Callable[[], Dict[str, str]],
    export_current_colab_notebook_fn: Callable[[Path], Any],
    push_repo_run_to_github_fn: Callable[..., Dict[str, Any]],
    telemetry: Any = None,
    print_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    from datetime import datetime, timezone

    import matplotlib.pyplot as plt

    from src.adapter.independent_crop_adapter import IndependentCropAdapter
    from src.training.services.ood_benchmark import run_leave_one_class_out_benchmark

    emit = print if print_fn is None else print_fn
    if state.get("adapter") is None or state.get("loaders") is None:
        raise RuntimeError("Once engine init hucresi calistirilmali.")

    adapter = state["adapter"]
    loaders = state["loaders"]
    test_loader = loaders.get("test")
    if test_loader is None or len(test_loader.dataset) == 0:
        raise RuntimeError("Test loader bos. Final degerlendirme held-out test split ile yapilmali.")

    trainer = getattr(adapter, "_trainer", None)
    if trainer is None:
        raise RuntimeError("Trainer hazir degil.")

    notebook_config = clone_jsonable(base_config)
    notebook_config.setdefault("training", {})["continual"] = clone_jsonable(state["continual_config"])
    evaluation_cfg = notebook_config.get("training", {}).get("continual", {}).get("evaluation", {})
    artifact_root = notebook_artifact_root(root)

    trainer.adapter_model.eval()
    trainer.classifier.eval()
    trainer.fusion.eval()

    classes = [name for name, _ in sorted(adapter.class_to_idx.items(), key=lambda item: item[1])]
    ood_test_loader = loaders.get("ood")
    ood_dev_loader = loaders.get("ood_dev")
    policy = dict(state.get("ood_policy_selection") or {})
    if not policy:
        policy = select_notebook_ood_policy(
            trainer=trainer,
            loaders=loaders,
            continual_config=dict(state.get("continual_config") or {}),
        )
        state["ood_policy_selection"] = policy
    requested_primary_score_method = str(policy.get("requested_primary_score_method", "auto"))
    selected_primary_score_method = str(policy.get("selected_primary_score_method", "ensemble"))
    selection_source = str(policy.get("selection_source", "configured"))
    results: Dict[str, Any] = {}
    val_loader = loaders.get("val")
    if val_loader is not None and len(val_loader.dataset) > 0:
        results["validation"] = _evaluate_notebook_split(
            root=root,
            trainer=trainer,
            loader=val_loader,
            ood_loader=ood_dev_loader,
            classes=classes,
            crop_name=crop_name,
            part_name=part_name,
            run_id=run_id,
            split_name="val",
            artifact_subdir="validation",
            label="DOGRULAMA (referans)",
            requested_primary_score_method=requested_primary_score_method,
            selected_primary_score_method=selected_primary_score_method,
            selection_source=selection_source,
            telemetry=telemetry,
            print_fn=emit,
        )

    results["test"] = _evaluate_notebook_split(
        root=root,
        trainer=trainer,
        loader=test_loader,
        ood_loader=ood_test_loader,
        classes=classes,
        crop_name=crop_name,
        part_name=part_name,
        run_id=run_id,
        split_name="test",
        artifact_subdir="test",
        label="TEST (ayrilmis)",
        requested_primary_score_method=requested_primary_score_method,
        selected_primary_score_method=selected_primary_score_method,
        selection_source=selection_source,
        telemetry=telemetry,
        print_fn=emit,
    )
    selected_split = "test" if "test" in results else "validation"
    selected_artifacts = results[selected_split]
    real_ood_present = ood_test_loader is not None and len(ood_test_loader.dataset) > 0
    ood_evidence_source = "real_ood_split" if real_ood_present else "unavailable"
    ood_evidence_metrics = dict(selected_artifacts["metric_gate"]["metrics"]) if real_ood_present else {}
    benchmark_summary: Dict[str, Any] = {}
    if (
        not real_ood_present
        and str(evaluation_cfg.get("ood_fallback_strategy", "held_out_benchmark")) == "held_out_benchmark"
        and bool(evaluation_cfg.get("ood_benchmark_auto_run", True))
    ):
        emit("[OOD] Gercek OOD split bulunamadi; held-out benchmark fallback calisiyor...")
        benchmark_summary = run_leave_one_class_out_benchmark(
            crop_name=crop_name,
            class_names=classes,
            loaders=loaders,
            config=notebook_config,
            device=device,
            artifact_root=artifact_root,
            adapter_factory=IndependentCropAdapter,
            run_id=run_id,
            num_epochs=int(epochs),
            telemetry=telemetry,
            emit_event=lambda event_type, payload: telemetry.emit_event(event_type, payload, phase="evaluation"),
            min_classes=int(evaluation_cfg.get("ood_benchmark_min_classes", 3)),
        )
        ood_evidence_source = "held_out_benchmark"
        ood_evidence_metrics = dict(benchmark_summary.get("metrics", {}))

    readiness = persist_production_readiness_artifact(
        root=root,
        classification_metric_gate=selected_artifacts.get("metric_gate"),
        classification_split=selected_split,
        ood_evidence_source=ood_evidence_source,
        ood_metrics=ood_evidence_metrics,
        context={
            "run_id": run_id,
            "crop_name": crop_name,
            "part_name": part_name,
            "classification_split": selected_split,
            "ood_benchmark_status": benchmark_summary.get("status"),
            "ood_benchmark_passed": benchmark_summary.get("passed"),
            "ood_requested_primary_score_method": requested_primary_score_method,
            "ood_primary_score_method": selected_primary_score_method,
            "ood_primary_score_selection_source": selection_source,
            "ood_primary_score_selection": dict(policy.get("selection") or {}),
        },
        telemetry=telemetry,
    )
    behavioral = persist_behavioral_acceptance_artifact(
        root=root,
        classification_metrics=dict(selected_artifacts.get("metric_gate", {}).get("metrics", {})),
        prediction_rows=list(selected_artifacts.get("prediction_rows") or []),
        context={
            "run_id": run_id,
            "crop_name": crop_name,
            "part_name": part_name,
            "classification_split": selected_split,
            "ood_scope": "same_crop_same_part_unsupported_disease",
            "router_scope_excluded": ["wrong_part", "other_plant", "non_plant"],
            "production_readiness_status": readiness["payload"].get("status"),
        },
        telemetry=telemetry,
    )

    state["evaluation_artifacts"] = results
    state["ood_benchmark"] = benchmark_summary
    state["behavioral_acceptance"] = behavioral["payload"]
    state["production_readiness"] = readiness["payload"]
    plt.close("all")
    emit(
        f"[ACCEPTANCE] status={behavioral['payload'].get('status', 'fail')} pass={behavioral['payload'].get('pass', False)}"
    )
    emit(f"[READINESS-COMPAT] status={readiness['payload'].get('status', 'failed')}")
    _call_if_present(
        telemetry,
        "update_latest",
        {
            "phase": "evaluation_complete",
            "evaluation_splits": sorted(results.keys()),
            "ood_evidence_source": readiness["payload"].get("ood_evidence_source"),
            "adapter_behavioral_acceptance": behavioral["payload"].get("status"),
            "production_readiness": readiness["payload"].get("status"),
        },
    )
    emit("[DONE] Dogrulama ve held-out test artefaktlari kaydedildi.")

    repo_run_exports = save_run_outputs_to_repo_fn()
    notebook_export_result = export_current_colab_notebook_fn(repo_notebook_output_path)
    extra_entries = _notebook_repo_extra_entries(
        repo_output_dir=repo_output_dir,
        crop_name=crop_name,
        part_name=part_name,
        repo_notebook_output_path=repo_notebook_output_path,
        repo_telemetry_dir=repo_telemetry_dir,
        repo_checkpoint_state_dir=repo_checkpoint_state_dir,
    )
    summary_payload = merge_training_summary_fields(
        root=root,
        telemetry=telemetry,
        payload={
            "run_id": run_id,
            "run_label": run_id,
            "crop_name": crop_name,
            "part_name": part_name,
            "notebook_surface": "2_train_continual_sd_lora_adapter.ipynb",
            "dataset_roots": {
                "runtime_dataset_root": runtime_dataset_root,
                "runtime_dataset_key": str(state.get("runtime_dataset_key") or ""),
                "selected_dataset_name": str(state.get("selected_dataset_name") or ""),
                "selected_runtime_dataset_root": str(state.get("selected_dataset_root") or ""),
                "resolved_ood_root": str(state.get("resolved_ood_root") or ""),
                "resolved_runtime_dataset_root": str(state.get("runtime_dataset_root") or ""),
            },
            "notebook_parameters": {
                "epochs": int(epochs),
                "batch_size": state.get("continual_config", {}).get("batch_size"),
                "learning_rate": state.get("continual_config", {}).get("learning_rate"),
                "lora_r": state.get("continual_config", {}).get("adapter", {}).get("lora_r"),
                "ood_factor": state.get("continual_config", {}).get("ood", {}).get("threshold_factor"),
                "mixed_precision": state.get("continual_config", {}).get("optimization", {}).get("mixed_precision"),
                "num_workers": state.get("loader_settings", {}).get("NUM_WORKERS"),
                "checkpoint_every_n_steps": state.get("training_runtime", {}).get("checkpoint_every_n_steps"),
                "enable_bayesian_optimization": bool(
                    (state.get("notebook_parameters") or {}).get("ENABLE_BAYESIAN_OPTIMIZATION", True)
                ),
            },
            "export_paths": {
                "repo_run_dir": str(repo_run_dir),
                "repo_output_dir": str(repo_output_dir),
                "repo_telemetry_dir": str(repo_telemetry_dir),
                "repo_checkpoint_state_dir": str(repo_checkpoint_state_dir),
                "executed_notebook_path": str(notebook_export_result or repo_notebook_output_path),
                "adapter_export_dir": str(state.get("adapter_export_dir") or ""),
            },
            "access_check": state.get("access_report", {}),
            "behavioral_acceptance_summary": {
                "status": (state.get("behavioral_acceptance") or {}).get("status"),
                "pass": (state.get("behavioral_acceptance") or {}).get("pass"),
                "decision_authority": "adapter_behavioral_acceptance",
            },
            "production_readiness_compatibility": state.get("production_readiness") or {},
            "optimization_campaign": summarize_notebook_optimization_campaign(state.get("optimization_campaign")),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        extra_entries=extra_entries,
    )
    _call_if_present(
        telemetry,
        "merge_summary_metadata",
        {
            "access_check": state.get("access_report", {}),
            "repo_paths": {
                "repo_run_dir": str(repo_run_dir),
                "repo_output_dir": str(repo_output_dir),
                "repo_telemetry_dir": str(repo_telemetry_dir),
                "repo_checkpoint_state_dir": str(repo_checkpoint_state_dir),
            },
            "training_summary": summary_payload,
        },
    )
    _call_if_present(
        telemetry,
        "close",
        {
            "status": "ok",
            "evaluation_splits": sorted((state.get("evaluation_artifacts") or {}).keys()),
            "cell_outputs_dir": str(getattr(telemetry, "artifacts_dir", root) / "cell_outputs"),
            "repo_run_dir": str(repo_run_dir),
            "run_label": run_id,
        },
    )

    repo_run_exports = save_run_outputs_to_repo_fn()
    if state.get("optimization_campaign"):
        state["optimization_campaign"] = finalize_notebook_optimization_campaign(
            root=root,
            campaign=state.get("optimization_campaign"),
            run_id=run_id,
            telemetry=telemetry,
        )
        summary_payload = merge_training_summary_fields(
            root=root,
            telemetry=telemetry,
            payload={
                "optimization_campaign": summarize_notebook_optimization_campaign(state.get("optimization_campaign")),
            },
        )
        repo_run_exports = save_run_outputs_to_repo_fn()
        opt_snapshot = summarize_notebook_optimization_campaign(state.get("optimization_campaign"))
        emit(
            f"[OPT] finalized status={opt_snapshot.get('status', 'unknown')} frontier={opt_snapshot.get('frontier_count', 0)} "
            f"executed={opt_snapshot.get('executed_run_count', 0)} next_rank={opt_snapshot.get('next_proposal_rank')}"
        )

    for key in sorted(repo_run_exports):
        emit(f"[RUNS] {key} -> {repo_run_exports[key]}")
    emit(f"[RUNS] notebook -> {repo_notebook_output_path}")

    if auto_push_to_github:
        try:
            git_push_report = push_repo_run_to_github_fn(
                root,
                run_id,
                run_relative_dir=repo_run_dir.relative_to(root),
                remote_name=auto_push_remote_name,
                branch=auto_push_branch,
                print_fn=emit,
            )
        except RuntimeError as exc:
            emit(f"[GIT] Auto-push skipped: {exc}")
            git_push_report = {"enabled": True, "pushed": False, "run_dir": str(repo_run_dir), "error": str(exc)}
    else:
        git_push_report = {"enabled": False, "pushed": False, "run_dir": str(repo_run_dir)}
    state["git_push_report"] = git_push_report

    disconnect_report = build_notebook_completion_report(
        state=state,
        telemetry=telemetry,
        repo_run_exports=repo_run_exports,
        notebook_export_path=notebook_export_result or repo_notebook_output_path,
    )
    state["auto_disconnect_report"] = disconnect_report
    emit(f"[COLAB] completion checks -> {disconnect_report['checks']}")
    maybe_auto_disconnect_colab_runtime(
        enabled=auto_disconnect_runtime,
        grace_period_sec=auto_disconnect_grace_seconds,
        telemetry=telemetry,
        completion_report=disconnect_report,
    )
    return {
        "state": state,
        "evaluation_artifacts": results,
        "behavioral_acceptance": behavioral["payload"],
        "production_readiness": readiness["payload"],
        "repo_run_exports": repo_run_exports,
        "notebook_export_result": notebook_export_result,
        "summary_payload": summary_payload,
        "disconnect_report": disconnect_report,
        "git_push_report": git_push_report,
    }
