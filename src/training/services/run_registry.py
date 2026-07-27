"""Build a local training-run registry for Pareto preparation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.shared.json_utils import read_json, write_json
from src.training.services.optimization import build_bayesian_recommendations, build_pareto_frontiers
from src.training.services.traceability import (
    build_experiment_manifest,
    build_optimization_record,
    load_authoritative_artifacts_from_root,
)

JsonDict = Dict[str, Any]

REGISTRY_SCHEMA = "v1_training_run_registry"
TRIAL_SCHEMA = "v1_training_registry_trial"
AUTOMATIC_WINS_FILENAME = "automatic_wins.md"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        import logging
        logging.exception('Unhandled exception')
        raise
        return False


def discover_artifact_roots(runs_root: str | Path) -> List[Path]:
    root = Path(runs_root)
    candidates: set[Path] = set()
    if not root.exists():
        return []
    for summary_path in root.rglob("training/summary.json"):
        if "_index" in summary_path.parts:
            continue
        candidates.add(summary_path.parent.parent)
    for optimization_path in root.rglob("training/optimization_record.json"):
        if "_index" in optimization_path.parts:
            continue
        candidates.add(optimization_path.parent.parent)
    for manifest_path in root.rglob("training/experiment_manifest.json"):
        if "_index" in manifest_path.parts:
            continue
        candidates.add(manifest_path.parent.parent)
    for overview_path in root.rglob("guided/01_run_overview.json"):
        if "_index" in overview_path.parts:
            continue
        candidates.add(overview_path.parent.parent)
    return sorted(candidates, key=lambda item: str(item).lower())


def _find_run_dir(artifact_root: Path, runs_root: Path) -> Path:
    current = artifact_root.resolve()
    runs_root = runs_root.resolve()
    while current != current.parent:
        if current.parent == runs_root:
            return current
        if current.parent.is_relative_to(runs_root) and (
            (current / "outputs").is_dir()
            or (current / "telemetry").is_dir()
            or (current / "checkpoint_state").is_dir()
            or (current / "notebooks").is_dir()
        ):
            return current
        current = current.parent
    return artifact_root


def _artifact_root_priority(artifact_root: Path) -> int:
    normalized = artifact_root.as_posix().lower()
    if "/outputs/colab_notebook_training/artifacts" in normalized:
        return 0
    if normalized.endswith("/checkpoint_state/artifacts"):
        return 1
    if normalized.endswith("/telemetry/artifacts"):
        return 2
    if normalized.endswith("/training_metrics"):
        return 3
    return 4


def _read_json_if_exists(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return read_json(path, default=default)


def _collect_summary_payload(artifact_root: Path, run_dir: Path) -> JsonDict:
    summary = _read_json_if_exists(artifact_root / "training" / "summary.json", default={})
    if isinstance(summary, dict) and summary:
        return dict(summary)

    overview = _read_json_if_exists(artifact_root / "guided" / "01_run_overview.json", default={})
    telemetry_summary = _read_json_if_exists(run_dir / "telemetry" / "summary.json", default={})
    checkpoint_summary = _read_json_if_exists(run_dir / "checkpoint_state" / "summary.json", default={})
    merged: JsonDict = {}
    for payload in (overview, telemetry_summary.get("run_overview", {}) if isinstance(telemetry_summary, dict) else {}, checkpoint_summary):
        if isinstance(payload, dict):
            merged.update({key: value for key, value in payload.items() if value not in (None, "", [], {})})
    return merged


def _load_traceability_record(artifact_root: Path) -> tuple[JsonDict, JsonDict, str]:
    manifest_path = artifact_root / "training" / "experiment_manifest.json"
    optimization_path = artifact_root / "training" / "optimization_record.json"
    if manifest_path.exists() and optimization_path.exists():
        manifest = read_json(manifest_path, default={}, expect_type=dict)
        optimization = read_json(optimization_path, default={}, expect_type=dict)
        return dict(manifest), dict(optimization), "canonical"
    return {}, {}, "backfilled"


def _build_backfilled_records(artifact_root: Path, run_dir: Path) -> tuple[JsonDict, JsonDict]:
    summary = _collect_summary_payload(artifact_root, run_dir)
    run_context = _read_json_if_exists(artifact_root / "training" / "run_context.json", default={})
    production_readiness = _read_json_if_exists(artifact_root / "production_readiness.json", default={})
    classification_split = ""
    if isinstance(production_readiness, dict):
        classification_evidence = production_readiness.get("classification_evidence", {})
        if isinstance(classification_evidence, dict):
            classification_split = str(classification_evidence.get("split_name", "") or "")
    authoritative_artifacts = load_authoritative_artifacts_from_root(
        artifact_root,
        classification_split=classification_split,
    )
    manifest = build_experiment_manifest(
        summary_payload=summary,
        run_context_payload=run_context,
        artifact_root=artifact_root,
        explicit_surface=(str(summary.get("surface", "") or "").strip() or None),
        created_at=str(summary.get("created_at", "") or run_context.get("created_at", "") or ""),
        record_quality="backfilled",
    )
    optimization = build_optimization_record(
        summary_payload=summary,
        run_context_payload=run_context,
        production_readiness_payload=production_readiness,
        authoritative_artifacts=authoritative_artifacts,
        artifact_root=artifact_root,
        explicit_surface=(str(summary.get("surface", "") or "").strip() or None),
        created_at=str(summary.get("created_at", "") or run_context.get("created_at", "") or ""),
        record_quality="backfilled",
    )
    return manifest, optimization


def _build_trial_record(
    *,
    artifact_root: Path,
    run_dir: Path,
    experiment_manifest: JsonDict,
    optimization_record: JsonDict,
) -> JsonDict:
    record = dict(optimization_record)
    record["schema_version"] = TRIAL_SCHEMA
    record["registry_source"] = {
        "run_dir": str(run_dir),
        "artifact_root": str(artifact_root),
        "artifact_root_priority": _artifact_root_priority(artifact_root),
    }
    record["experiment_manifest"] = experiment_manifest
    return record


def _trial_group_key(trial: JsonDict, run_dir: Path) -> Tuple[str, str]:
    run_id = str(trial.get("run_id", "") or "")
    return (str(run_dir.name), run_id or str(trial.get("registry_source", {}).get("artifact_root", "")))


def _choose_preferred_trial(left: JsonDict, right: JsonDict) -> JsonDict:
    left_quality = str(left.get("record_quality", "") or "backfilled")
    right_quality = str(right.get("record_quality", "") or "backfilled")
    left_priority = int(left.get("registry_source", {}).get("artifact_root_priority", 99))
    right_priority = int(right.get("registry_source", {}).get("artifact_root_priority", 99))
    if left_quality != right_quality:
        return left if left_quality == "canonical" else right
    if left_priority != right_priority:
        return left if left_priority < right_priority else right
    return left


def collect_trial_records(runs_root: str | Path) -> List[JsonDict]:
    root = Path(runs_root)
    selected: Dict[Tuple[str, str], JsonDict] = {}
    for artifact_root in discover_artifact_roots(root):
        run_dir = _find_run_dir(artifact_root, root)
        manifest, optimization, quality = _load_traceability_record(artifact_root)
        if not manifest or not optimization:
            manifest, optimization = _build_backfilled_records(artifact_root, run_dir)
            quality = "backfilled"
        trial = _build_trial_record(
            artifact_root=artifact_root,
            run_dir=run_dir,
            experiment_manifest=manifest,
            optimization_record=optimization,
        )
        trial["record_quality"] = quality
        key = _trial_group_key(trial, run_dir)
        if key in selected:
            selected[key] = _choose_preferred_trial(selected[key], trial)
        else:
            selected[key] = trial
    return sorted(
        selected.values(),
        key=lambda item: (
            str(item.get("created_at", "") or ""),
            str(item.get("run_id", "") or ""),
        ),
    )


def build_pareto_inputs(trials: Iterable[JsonDict]) -> JsonDict:
    grouped: Dict[str, List[JsonDict]] = defaultdict(list)
    incomplete: List[JsonDict] = []
    for trial in trials:
        cohort_key = str(trial.get("comparability", {}).get("cohort_key", "") or "")
        if cohort_key:
            grouped[cohort_key].append(trial)
        else:
            incomplete.append(trial)
    cohorts = []
    for cohort_key, cohort_trials in sorted(grouped.items(), key=lambda item: item[0]):
        first = cohort_trials[0]
        cohorts.append(
            {
                "cohort_key": cohort_key,
                "comparability": dict(first.get("comparability", {})),
                "run_count": len(cohort_trials),
                "runs": [
                    {
                        "run_id": trial.get("run_id"),
                        "run_label": trial.get("run_label"),
                        "created_at": trial.get("created_at"),
                        "record_quality": trial.get("record_quality"),
                        "status": dict(trial.get("status", {})),
                        "objectives": dict(trial.get("objectives", {})),
                        "objective_directions": dict(trial.get("objective_directions", {})),
                        "parameters": dict(trial.get("parameters", {})),
                        "artifact_root": trial.get("registry_source", {}).get("artifact_root"),
                    }
                    for trial in sorted(cohort_trials, key=lambda item: str(item.get("created_at", "") or ""))
                ],
            }
        )
    return {
        "schema_version": REGISTRY_SCHEMA,
        "generated_at": _utc_now_iso(),
        "optimization_profile": "accuracy_plus_ood",
        "cohort_count": len(cohorts),
        "cohorts": cohorts,
        "incomplete_trials": [
            {
                "run_id": trial.get("run_id"),
                "run_label": trial.get("run_label"),
                "record_quality": trial.get("record_quality"),
                "artifact_root": trial.get("registry_source", {}).get("artifact_root"),
            }
            for trial in incomplete
        ],
    }


def _format_metric(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_parameter_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value or "n/a")


def _winner_parameter_highlights(parameters: Dict[str, Any]) -> List[Tuple[str, str]]:
    highlights: List[Tuple[str, str]] = []

    label_smoothing = parameters.get("training.optimization.label_smoothing")
    if label_smoothing not in (None, ""):
        highlights.append(("label smoothing", _format_parameter_value(label_smoothing)))

    energy_temperature_mode = str(parameters.get("training.ood.energy_temperature_mode", "") or "").strip()
    if energy_temperature_mode:
        energy_detail = energy_temperature_mode
        energy_temperature = parameters.get("training.ood.energy_temperature")
        if energy_temperature_mode == "fixed" and energy_temperature not in (None, ""):
            energy_detail = f"{energy_temperature_mode} ({_format_parameter_value(energy_temperature)})"
        highlights.append(("energy temperature", energy_detail))

    if "training.ood.react_enabled" in parameters:
        react_detail = "off"
        if bool(parameters.get("training.ood.react_enabled")):
            react_detail = "on"
            react_percentile = parameters.get("training.ood.react_percentile")
            if react_percentile not in (None, ""):
                react_detail = f"{react_detail} (percentile {_format_parameter_value(react_percentile)})"
        highlights.append(("ReAct", react_detail))

    augmentation_policy = str(parameters.get("training.data.augmentation_policy", "") or "").strip()
    if augmentation_policy:
        augmentation_detail = augmentation_policy
        if augmentation_policy == "randaugment":
            randaugment_parts = []
            if parameters.get("training.data.randaugment_num_ops") not in (None, ""):
                randaugment_parts.append(f"ops {_format_parameter_value(parameters.get('training.data.randaugment_num_ops'))}")
            if parameters.get("training.data.randaugment_magnitude") not in (None, ""):
                randaugment_parts.append(
                    f"magnitude {_format_parameter_value(parameters.get('training.data.randaugment_magnitude'))}"
                )
            if randaugment_parts:
                augmentation_detail = f"{augmentation_policy} ({', '.join(randaugment_parts)})"
        elif augmentation_policy == "augmix":
            augmix_parts = []
            for key, label in (
                ("training.data.augmix_severity", "severity"),
                ("training.data.augmix_width", "width"),
                ("training.data.augmix_depth", "depth"),
                ("training.data.augmix_alpha", "alpha"),
            ):
                value = parameters.get(key)
                if value not in (None, ""):
                    augmix_parts.append(f"{label} {_format_parameter_value(value)}")
            if augmix_parts:
                augmentation_detail = f"{augmentation_policy} ({', '.join(augmix_parts)})"
        highlights.append(("augmentation", augmentation_detail))

    if "training.classifier_rebalance.enabled" in parameters:
        rebalance_detail = "off"
        if bool(parameters.get("training.classifier_rebalance.enabled")):
            rebalance_parts = []
            objective = str(parameters.get("training.classifier_rebalance.objective", "") or "").strip()
            sampler = str(parameters.get("training.classifier_rebalance.sampler", "") or "").strip()
            epochs = parameters.get("training.classifier_rebalance.epochs")
            tau = parameters.get("training.classifier_rebalance.logit_adjustment_tau")
            if objective:
                rebalance_parts.append(objective)
            if sampler:
                rebalance_parts.append(f"sampler {sampler}")
            if epochs not in (None, ""):
                rebalance_parts.append(f"epochs {_format_parameter_value(epochs)}")
            if tau not in (None, ""):
                rebalance_parts.append(f"tau {_format_parameter_value(tau)}")
            rebalance_detail = "on"
            if rebalance_parts:
                rebalance_detail = f"{rebalance_detail} ({', '.join(rebalance_parts)})"
        highlights.append(("classifier rebalance", rebalance_detail))

    if "training.ood.oe_enabled" in parameters:
        oe_detail = "off"
        if bool(parameters.get("training.ood.oe_enabled")):
            oe_parts = []
            oe_target = str(parameters.get("training.ood.oe_target", "") or "").strip()
            oe_loss_weight = parameters.get("training.ood.oe_loss_weight")
            if oe_target:
                oe_parts.append(oe_target)
            if oe_loss_weight not in (None, ""):
                oe_parts.append(f"weight {_format_parameter_value(oe_loss_weight)}")
            oe_detail = "on"
            if oe_parts:
                oe_detail = f"{oe_detail} ({', '.join(oe_parts)})"
        highlights.append(("OE", oe_detail))

    return highlights


def _render_automatic_wins_markdown(
    *,
    latest_registry: JsonDict,
    pareto_frontiers: JsonDict,
) -> str:
    generated_at = str(latest_registry.get("generated_at", "") or "")
    lines: List[str] = [
        "# Automatic Wins",
        "",
        "Generated automatically from the local run registry. A cohort `win` means a Pareto-frontier run inside one comparable cohort.",
        "",
        "## Registry Snapshot",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Optimization profile: `{latest_registry.get('optimization_profile', '')}`",
        f"- Indexed trials: `{latest_registry.get('trial_count', 0)}`",
        f"- Comparable cohorts: `{latest_registry.get('cohort_count', 0)}`",
        f"- Canonical records: `{latest_registry.get('canonical_count', 0)}`",
        f"- Backfilled records: `{latest_registry.get('backfilled_count', 0)}`",
        "",
        "## Cohort Winners",
        "",
    ]

    cohorts = list(pareto_frontiers.get("cohorts", []) or [])
    if not cohorts:
        lines.append("No comparable cohorts were indexed.")
        lines.append("")
        return "\n".join(lines)

    for cohort in cohorts:
        if not isinstance(cohort, dict):
            continue
        comparability = dict(cohort.get("comparability", {})) if isinstance(cohort.get("comparability"), dict) else {}
        crop_name = str(comparability.get("crop_name", "") or "unknown")
        part_name = str(comparability.get("part_name", "") or "unspecified")
        cohort_key = str(cohort.get("cohort_key", "") or "")
        frontier_runs = list(cohort.get("frontier_runs", []) or [])
        excluded_runs = list(cohort.get("excluded_runs", []) or [])
        lines.extend(
            [
                f"### {crop_name} / {part_name}",
                "",
                f"- Cohort key: `{cohort_key}`",
                f"- Eligible runs: `{cohort.get('eligible_run_count', 0)}`",
                f"- Excluded runs: `{cohort.get('excluded_run_count', 0)}`",
                f"- Frontier wins: `{cohort.get('frontier_count', 0)}`",
                "",
            ]
        )
        if frontier_runs:
            lines.append("| Run ID | Macro F1 | OOD AUROC | OOD FPR | Readiness | Artifact Root |")
            lines.append("| --- | ---: | ---: | ---: | --- | --- |")
            for run in frontier_runs:
                if not isinstance(run, dict):
                    continue
                status = dict(run.get("status", {})) if isinstance(run.get("status"), dict) else {}
                objectives = dict(run.get("objectives", {})) if isinstance(run.get("objectives"), dict) else {}
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(run.get("run_id", "") or ""),
                            _format_metric(objectives.get("classification.macro_f1")),
                            _format_metric(objectives.get("ood.ood_auroc")),
                            _format_metric(objectives.get("ood.ood_false_positive_rate")),
                            str(status.get("readiness_status", "") or ""),
                            f"`{run.get('artifact_root', '') or ''}`",
                        ]
                    )
                    + " |"
                )
            lines.append("")
            highlight_lines = []
            for run in frontier_runs:
                if not isinstance(run, dict):
                    continue
                parameters = dict(run.get("parameters", {})) if isinstance(run.get("parameters"), dict) else {}
                highlights = _winner_parameter_highlights(parameters)
                if not highlights:
                    continue
                highlight_lines.append(
                    f"- `{str(run.get('run_id', '') or '')}`: "
                    + "; ".join(f"{label} `{value}`" for label, value in highlights)
                )
            if highlight_lines:
                lines.append("#### Winner Config Highlights")
                lines.append("")
                lines.extend(highlight_lines)
                lines.append("")
        else:
            lines.append("No eligible Pareto-frontier wins were found for this cohort.")
            if excluded_runs:
                lines.append("")
                lines.append("| Excluded Run ID | Reason |")
                lines.append("| --- | --- |")
                for run in excluded_runs:
                    if not isinstance(run, dict):
                        continue
                    lines.append(
                        f"| {str(run.get('run_id', '') or '')} | {str(run.get('reason', '') or '')} |"
                    )
                lines.append("")
            else:
                lines.append("")

    return "\n".join(lines)


def _build_lineage_report(trials: List[JsonDict]) -> JsonDict:
    adapters: List[JsonDict] = []
    for trial in trials:
        registry_source = dict(trial.get("registry_source", {})) if isinstance(trial.get("registry_source"), dict) else {}
        manifest = dict(trial.get("experiment_manifest", {})) if isinstance(trial.get("experiment_manifest"), dict) else {}
        comparability = dict(trial.get("comparability", {})) if isinstance(trial.get("comparability"), dict) else {}
        status = dict(trial.get("status", {})) if isinstance(trial.get("status"), dict) else {}
        artifacts = manifest.get("artifacts", {}) if isinstance(manifest.get("artifacts"), dict) else {}
        adapter_export_path = (
            artifacts.get("adapter_export_path")
            or artifacts.get("adapter_dir")
            or trial.get("adapter_export_path")
            or ""
        )
        adapters.append(
            {
                "run_id": trial.get("run_id"),
                "crop_name": comparability.get("crop_name"),
                "part_name": comparability.get("part_name"),
                "dataset_key": comparability.get("dataset_key") or manifest.get("dataset_key"),
                "record_quality": trial.get("record_quality"),
                "artifact_root": registry_source.get("artifact_root"),
                "adapter_export_path": adapter_export_path,
                "readiness_status": status.get("readiness_status"),
                "readiness_evidence_path": artifacts.get("production_readiness") or artifacts.get("production_readiness_json") or "",
            }
        )
    missing_lineage = [
        item
        for item in adapters
        if not item.get("artifact_root") or not item.get("readiness_status")
    ]
    return {
        "schema_version": "v1_artifact_lineage_report",
        "generated_at": _utc_now_iso(),
        "adapter_count": len(adapters),
        "missing_lineage_count": len(missing_lineage),
        "adapters": adapters,
    }


def _render_summary_dashboard(latest_registry: JsonDict, lineage_report: JsonDict) -> str:
    lines = [
        "# Run Registry Dashboard",
        "",
        f"- Generated at: `{latest_registry.get('generated_at', '')}`",
        f"- Indexed trials: `{latest_registry.get('trial_count', 0)}`",
        f"- Comparable cohorts: `{latest_registry.get('cohort_count', 0)}`",
        f"- Adapter lineage rows: `{lineage_report.get('adapter_count', 0)}`",
        f"- Missing lineage rows: `{lineage_report.get('missing_lineage_count', 0)}`",
        "",
        "## Adapters To Readiness",
        "",
        "| Run ID | Crop | Part | Readiness | Artifact Root |",
        "| --- | --- | --- | --- | --- |",
    ]
    adapters = list(lineage_report.get("adapters", []) or [])
    if not adapters:
        lines.append("| n/a | n/a | n/a | n/a | n/a |")
    for adapter in adapters:
        if not isinstance(adapter, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(adapter.get("run_id", "") or ""),
                    str(adapter.get("crop_name", "") or ""),
                    str(adapter.get("part_name", "") or ""),
                    str(adapter.get("readiness_status", "") or ""),
                    f"`{adapter.get('artifact_root', '') or ''}`",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_registry(
    *,
    trials: Iterable[JsonDict],
    output_root: str | Path,
    enable_bayesian_proposals: bool = True,
    proposal_count: int = 3,
    candidate_pool_size: int = 256,
    random_seed: int = 42,
    search_space_payload: Any = None,
) -> JsonDict:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    materialized_trials = list(trials)
    trials_jsonl = root / "trials.jsonl"
    with trials_jsonl.open("w", encoding="utf-8") as handle:
        for trial in materialized_trials:
            handle.write(json.dumps(trial, ensure_ascii=False))
            handle.write("\n")

    pareto_inputs = build_pareto_inputs(materialized_trials)
    pareto_inputs_path = write_json(root / "pareto_inputs.json", pareto_inputs, ensure_ascii=False)
    pareto_frontiers = build_pareto_frontiers(materialized_trials)
    pareto_frontiers_path = write_json(root / "pareto_frontiers.json", pareto_frontiers, ensure_ascii=False)
    bayesian_recommendations_path = root / "bayesian_recommendations.json"
    bayesian_recommendations = None
    if enable_bayesian_proposals:
        bayesian_recommendations = build_bayesian_recommendations(
            materialized_trials,
            proposal_count=int(proposal_count),
            candidate_pool_size=int(candidate_pool_size),
            random_seed=int(random_seed),
            search_space_payload=search_space_payload,
        )
        write_json(bayesian_recommendations_path, bayesian_recommendations, ensure_ascii=False)
    elif bayesian_recommendations_path.exists():
        bayesian_recommendations_path.unlink()
    latest_registry_payload = {
        "schema_version": REGISTRY_SCHEMA,
        "generated_at": pareto_inputs["generated_at"],
        "optimization_profile": "accuracy_plus_ood",
        "bayesian_optimization_enabled": bool(enable_bayesian_proposals),
        "trial_count": len(materialized_trials),
        "canonical_count": sum(1 for trial in materialized_trials if trial.get("record_quality") == "canonical"),
        "backfilled_count": sum(1 for trial in materialized_trials if trial.get("record_quality") == "backfilled"),
        "cohort_count": int(pareto_inputs.get("cohort_count", 0)),
        "paths": {
            "trials_jsonl": str(trials_jsonl),
            "pareto_inputs_json": str(pareto_inputs_path),
            "pareto_frontiers_json": str(pareto_frontiers_path),
        },
        "cohorts": [
            {
                "cohort_key": cohort["cohort_key"],
                "run_count": cohort["run_count"],
                "comparability": cohort["comparability"],
            }
            for cohort in pareto_inputs.get("cohorts", [])
        ],
    }
    if bayesian_recommendations is not None:
        latest_registry_payload["paths"]["bayesian_recommendations_json"] = str(bayesian_recommendations_path)
        latest_registry_payload["bayesian_recommendation_cohort_count"] = int(
            bayesian_recommendations.get("cohort_count", 0)
        )
    automatic_wins_markdown = _render_automatic_wins_markdown(
        latest_registry=latest_registry_payload,
        pareto_frontiers=pareto_frontiers,
    )
    automatic_wins_markdown_path = root / AUTOMATIC_WINS_FILENAME
    automatic_wins_markdown_path.write_text(automatic_wins_markdown, encoding="utf-8")
    latest_registry_payload["paths"]["automatic_wins_markdown"] = str(automatic_wins_markdown_path)
    lineage_report = _build_lineage_report(materialized_trials)
    lineage_report_path = write_json(root / "artifact_lineage_report.json", lineage_report, ensure_ascii=False)
    latest_registry_payload["paths"]["artifact_lineage_report_json"] = str(lineage_report_path)
    summary_dashboard = _render_summary_dashboard(latest_registry_payload, lineage_report)
    summary_dashboard_path = root / "summary_dashboard.md"
    summary_dashboard_path.write_text(summary_dashboard, encoding="utf-8")
    latest_registry_payload["paths"]["summary_dashboard_markdown"] = str(summary_dashboard_path)
    latest_registry_path = write_json(root / "latest_registry.json", latest_registry_payload, ensure_ascii=False)
    result = {
        "trials_jsonl": trials_jsonl,
        "pareto_inputs_json": pareto_inputs_path,
        "pareto_frontiers_json": pareto_frontiers_path,
        "automatic_wins_markdown": automatic_wins_markdown_path,
        "artifact_lineage_report_json": lineage_report_path,
        "summary_dashboard_markdown": summary_dashboard_path,
        "latest_registry_json": latest_registry_path,
        "latest_registry": latest_registry_payload,
    }
    if bayesian_recommendations is not None:
        result["bayesian_recommendations_json"] = bayesian_recommendations_path
        result["bayesian_recommendations"] = bayesian_recommendations
    return result


def build_run_registry(
    *,
    runs_root: str | Path,
    output_root: str | Path | None = None,
    enable_bayesian_proposals: bool = False,
    proposal_count: int = 3,
    candidate_pool_size: int = 256,
    random_seed: int = 42,
    search_space_payload: Any = None,
) -> JsonDict:
    resolved_runs_root = Path(runs_root)
    resolved_output_root = Path(output_root) if output_root is not None else (resolved_runs_root / "_index")
    trials = collect_trial_records(resolved_runs_root)
    return write_registry(
        trials=trials,
        output_root=resolved_output_root,
        enable_bayesian_proposals=enable_bayesian_proposals,
        proposal_count=proposal_count,
        candidate_pool_size=candidate_pool_size,
        random_seed=random_seed,
        search_space_payload=search_space_payload,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--enable-bayesian-proposals", action="store_true")
    parser.add_argument("--proposal-count", type=int, default=3)
    parser.add_argument("--candidate-pool-size", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.set_defaults(enable_bayesian_proposals=True)
    args = parser.parse_args()
    result = build_run_registry(
        runs_root=args.runs_root,
        output_root=args.output_root,
        enable_bayesian_proposals=bool(args.enable_bayesian_proposals),
        proposal_count=int(args.proposal_count),
        candidate_pool_size=int(args.candidate_pool_size),
        random_seed=int(args.random_seed),
    )
    print(json.dumps(result["latest_registry"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
