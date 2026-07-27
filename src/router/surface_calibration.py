"""Pure helpers for router surface calibration sweeps."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import statistics
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

JsonDict = Dict[str, Any]

REPLAY_SAFE_PARAMETERS = {
    "inference.router_min_confidence",
    "inference.router_min_margin",
    "inference.input_guard.enabled",
    "inference.input_guard.plant_min_score",
    "inference.input_guard.negative_margin",
}

SUPPORTED_PARAMETERS: Dict[str, str] = {
    "router_min_confidence": "inference.router_min_confidence",
    "router_min_margin": "inference.router_min_margin",
    "vlm_confidence_threshold": "router.vlm.confidence_threshold",
    "part_open_set_min_confidence": "router.vlm.policy_graph.part_resolution.part_open_set_min_confidence",
    "part_open_set_margin": "router.vlm.policy_graph.part_resolution.part_open_set_margin",
    "global_crop_context_weight": "router.vlm.policy_graph.crop_evidence.global_crop_context_weight",
    "sam3_mask_threshold": "router.vlm.policy_graph.roi_filter.sam3_mask_threshold",
    "sam3_prompt_limit": "router.vlm.sam3_prompt_limit",
    "crop_num_prompts": "router.vlm.policy_graph.crop_evidence.crop_num_prompts",
    "part_num_prompts": "router.vlm.policy_graph.part_evidence.part_num_prompts",
    "max_rois_for_classification": "router.vlm.policy_graph.roi_filter.max_rois_for_classification",
    "input_guard_enabled": "inference.input_guard.enabled",
    "input_guard_plant_min_score": "inference.input_guard.plant_min_score",
    "input_guard_negative_margin": "inference.input_guard.negative_margin",
}

INTEGER_PARAMETERS = {
    "router.vlm.sam3_prompt_limit",
    "router.vlm.policy_graph.crop_evidence.crop_num_prompts",
    "router.vlm.policy_graph.part_evidence.part_num_prompts",
    "router.vlm.policy_graph.roi_filter.max_rois_for_classification",
}
BOOL_PARAMETERS = {
    "inference.input_guard.enabled",
}

PRESET_SWEEPS: Dict[str, Dict[str, List[Any]]] = {
    "handoff": {
        "inference.router_min_confidence": [0.55, 0.65, 0.75],
        "inference.router_min_margin": [0.00, 0.10, 0.15],
    },
    "quick": {
        "inference.router_min_confidence": [0.55, 0.65, 0.75],
        "inference.router_min_margin": [0.00, 0.10, 0.15],
        "router.vlm.confidence_threshold": [0.20, 0.25, 0.35],
        "router.vlm.policy_graph.crop_evidence.global_crop_context_weight": [0.45, 0.65, 0.80],
    },
    "docs": {
        "inference.router_min_confidence": [0.55, 0.65, 0.75],
        "inference.router_min_margin": [0.00, 0.10, 0.15],
        "router.vlm.confidence_threshold": [0.20, 0.25, 0.35],
        "router.vlm.policy_graph.part_resolution.part_open_set_min_confidence": [0.30, 0.40, 0.50],
        "router.vlm.policy_graph.part_resolution.part_open_set_margin": [0.05, 0.10, 0.15],
        "router.vlm.policy_graph.crop_evidence.global_crop_context_weight": [0.45, 0.65, 0.80],
        "router.vlm.policy_graph.roi_filter.sam3_mask_threshold": [0.50, 0.60, 0.70],
        "router.vlm.sam3_prompt_limit": [4, 6],
        "router.vlm.policy_graph.crop_evidence.crop_num_prompts": [2, 4],
        "router.vlm.policy_graph.part_evidence.part_num_prompts": [2, 4],
        "router.vlm.policy_graph.roi_filter.max_rois_for_classification": [0, 16],
    },
}


def _canonical_parameter_name(raw_name: str) -> str:
    name = str(raw_name or "").strip()
    if name in SUPPORTED_PARAMETERS:
        return SUPPORTED_PARAMETERS[name]
    if name in SUPPORTED_PARAMETERS.values():
        return name
    supported = ", ".join(sorted(SUPPORTED_PARAMETERS))
    raise ValueError(f"Unsupported sweep parameter '{name}'. Supported aliases: {supported}")


def _coerce_parameter_value(parameter: str, raw_value: Any) -> Any:
    if parameter in BOOL_PARAMETERS:
        value = str(raw_value).strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Boolean sweep value for {parameter} must be true/false, got {raw_value!r}")
    if parameter in INTEGER_PARAMETERS:
        return int(raw_value)
    return float(raw_value)


def _get_nested_value(payload: JsonDict, dotted_path: str, default: Any = None) -> Any:
    cursor: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _set_nested_value(payload: JsonDict, dotted_path: str, value: Any) -> JsonDict:
    cloned = copy.deepcopy(payload)
    cursor: Any = cloned
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if not isinstance(cursor.get(part), dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = copy.deepcopy(value)
    return cloned


def parse_sweep_spec(raw_spec: str) -> tuple[str, List[Any]]:
    """Parse PARAM=v1,v2 syntax used by the CLI."""
    if "=" not in str(raw_spec):
        raise ValueError(f"Sweep spec must use PARAM=v1,v2 syntax, got {raw_spec!r}")
    raw_name, raw_values = str(raw_spec).split("=", 1)
    parameter = _canonical_parameter_name(raw_name)
    values = [_coerce_parameter_value(parameter, item.strip()) for item in raw_values.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Sweep spec for {parameter} did not include any values.")
    return parameter, values


def _dedupe_values(values: Iterable[Any]) -> List[Any]:
    deduped: List[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def resolve_sweep_grid(
    base_config: JsonDict,
    *,
    preset: str = "quick",
    sweep_specs: Sequence[str] | None = None,
    include_current: bool = True,
) -> Dict[str, List[Any]]:
    if preset not in PRESET_SWEEPS and preset != "none":
        raise ValueError(f"Unknown preset '{preset}'. Choose one of: none, {', '.join(sorted(PRESET_SWEEPS))}")

    grid: Dict[str, List[Any]] = {}
    if preset != "none":
        grid.update(copy.deepcopy(PRESET_SWEEPS[preset]))

    for raw_spec in list(sweep_specs or []):
        parameter, values = parse_sweep_spec(raw_spec)
        grid[parameter] = values

    if include_current:
        for parameter, values in list(grid.items()):
            current = _get_nested_value(base_config, parameter)
            if current is not None:
                current = _coerce_parameter_value(parameter, current)
                grid[parameter] = _dedupe_values([current, *values])

    return {parameter: _dedupe_values(values) for parameter, values in grid.items()}


def variant_count(grid: Dict[str, List[Any]]) -> int:
    total = 1
    for values in grid.values():
        total *= max(1, len(values))
    return total


def iter_sweep_overrides(grid: Dict[str, List[Any]]) -> Iterable[JsonDict]:
    parameters = list(grid.keys())
    value_lists = [grid[parameter] for parameter in parameters]
    for combo in itertools.product(*value_lists):
        yield {parameter: value for parameter, value in zip(parameters, combo)}


def apply_overrides(base_config: JsonDict, overrides: JsonDict) -> JsonDict:
    config = copy.deepcopy(base_config)
    for parameter, value in overrides.items():
        config = _set_nested_value(config, parameter, value)
        if parameter.startswith("router.vlm."):
            profile = _get_nested_value(config, "router.vlm.profile")
            if isinstance(profile, str) and profile.strip():
                relative_path = parameter.removeprefix("router.vlm.")
                profile_root = f"router.vlm.profiles.{profile.strip()}"
                if isinstance(_get_nested_value(config, profile_root), dict):
                    config = _set_nested_value(config, f"{profile_root}.{relative_path}", value)
    return config


def variant_id(overrides: JsonDict) -> str:
    body = json.dumps(overrides, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]


def is_replay_safe_overrides(overrides: JsonDict) -> bool:
    return set((overrides or {}).keys()).issubset(REPLAY_SAFE_PARAMETERS)


def _with_replayed_thresholds(samples: Sequence[JsonDict], overrides: JsonDict) -> List[JsonDict]:
    min_confidence = overrides.get("inference.router_min_confidence")
    min_margin = overrides.get("inference.router_min_margin")
    guard_enabled = overrides.get("inference.input_guard.enabled")
    guard_plant_min_score = overrides.get("inference.input_guard.plant_min_score")
    guard_negative_margin = overrides.get("inference.input_guard.negative_margin")
    replayed: List[JsonDict] = []
    for sample in samples:
        row = dict(sample)
        raw_handoff = bool(row.get("router_handoff_crop", row.get("handoff_crop", False)))
        gate_reasons: List[str] = []
        if raw_handoff and min_confidence is not None and float(row.get("crop_confidence", 0.0) or 0.0) < float(
            min_confidence
        ):
            gate_reasons.append("router_min_confidence")
        routing_margin = row.get("routing_margin")
        if raw_handoff and min_margin is not None and routing_margin is not None and float(routing_margin) < float(
            min_margin
        ):
            gate_reasons.append("router_min_margin")
        guard_payload = dict(row.get("input_guard") or {})
        if raw_handoff and bool(guard_enabled) and guard_payload:
            plant_score = float(guard_payload.get("plant_score", 0.0) or 0.0)
            non_plant_score = float(guard_payload.get("non_plant_score", 0.0) or 0.0)
            min_plant = 0.45 if guard_plant_min_score is None else float(guard_plant_min_score)
            negative_margin = 0.10 if guard_negative_margin is None else float(guard_negative_margin)
            if plant_score < min_plant:
                gate_reasons.append("input_guard_plant_min_score")
            elif non_plant_score - plant_score >= negative_margin:
                gate_reasons.append("input_guard_negative_margin")

        handoff = raw_handoff and not gate_reasons
        row["handoff_crop"] = bool(handoff)
        row["runtime_gate_rejected"] = bool(gate_reasons)
        row["runtime_gate_reasons"] = gate_reasons
        if not handoff:
            row["predicted_part"] = "unknown"
            row["part_abstained"] = True
            row["part_correct"] = False
            row["unsupported_part_emitted"] = False
        replayed.append(row)
    return replayed


def replay_variant(samples: Sequence[JsonDict], *, overrides: JsonDict, summarize_predictions: Any) -> JsonDict:
    started = time.perf_counter()
    replayed = _with_replayed_thresholds(samples, overrides)
    metrics = summarize_predictions(replayed)
    metrics["variant_wall_time_ms"] = round((time.perf_counter() - started) * 1000.0, 4)
    return {
        "variant_id": "baseline" if not overrides else variant_id(overrides),
        "overrides": copy.deepcopy(overrides),
        "metrics": metrics,
        "samples": replayed,
    }


def _eligibility_reasons(
    metrics: JsonDict,
    baseline_metrics: JsonDict,
    *,
    target_negative_false_accept_rate: float,
    max_crop_accuracy_drop: float,
    max_part_precision_drop: float,
    max_part_recall_drop: float,
    max_wrong_part_rejection_drop: float,
    max_p95_latency_regression: float,
) -> List[str]:
    reasons: List[str] = []
    if float(metrics.get("negative_false_accept_rate", 0.0)) > float(target_negative_false_accept_rate):
        reasons.append("negative_false_accept_rate_above_target")
    min_crop_accuracy = max(0.0, float(baseline_metrics.get("crop_accuracy", 0.0)) - float(max_crop_accuracy_drop))
    if float(metrics.get("crop_accuracy", 0.0)) < min_crop_accuracy:
        reasons.append("crop_accuracy_drop")
    min_part_precision = max(
        0.0,
        float(baseline_metrics.get("part_non_unknown_precision", 0.0)) - float(max_part_precision_drop),
    )
    if float(metrics.get("part_non_unknown_precision", 0.0)) < min_part_precision:
        reasons.append("part_precision_drop")
    min_part_recall = max(0.0, float(baseline_metrics.get("part_recall", 0.0)) - float(max_part_recall_drop))
    if float(metrics.get("part_recall", 0.0)) < min_part_recall:
        reasons.append("part_recall_drop")
    min_wrong_part_rejection = max(
        0.0,
        float(baseline_metrics.get("wrong_part_rejection_rate", 0.0)) - float(max_wrong_part_rejection_drop),
    )
    if float(metrics.get("wrong_part_rejection_rate", 0.0)) < min_wrong_part_rejection:
        reasons.append("wrong_part_rejection_drop")
    baseline_p95_latency = float(baseline_metrics.get("p95_latency_ms", 0.0) or 0.0)
    p95_latency = float(metrics.get("p95_latency_ms", 0.0) or 0.0)
    if baseline_p95_latency > 0.0 and p95_latency > baseline_p95_latency * (1.0 + float(max_p95_latency_regression)):
        reasons.append("p95_latency_regression")
    return reasons


def annotate_and_rank_variants(
    variants: Sequence[JsonDict],
    *,
    baseline: JsonDict,
    target_negative_false_accept_rate: float = 0.05,
    max_crop_accuracy_drop: float = 0.02,
    max_part_precision_drop: float = 0.02,
    max_part_recall_drop: float = 0.02,
    max_wrong_part_rejection_drop: float = 0.02,
    max_p95_latency_regression: float = 0.25,
) -> List[JsonDict]:
    baseline_metrics = baseline.get("metrics", {})
    annotated: List[JsonDict] = []
    for variant in variants:
        row = copy.deepcopy(variant)
        metrics = row.get("metrics", {})
        reasons = _eligibility_reasons(
            metrics,
            baseline_metrics,
            target_negative_false_accept_rate=target_negative_false_accept_rate,
            max_crop_accuracy_drop=max_crop_accuracy_drop,
            max_part_precision_drop=max_part_precision_drop,
            max_part_recall_drop=max_part_recall_drop,
            max_wrong_part_rejection_drop=max_wrong_part_rejection_drop,
            max_p95_latency_regression=max_p95_latency_regression,
        )
        row["eligible"] = not reasons
        row["eligibility_reasons"] = reasons
        annotated.append(row)

    def _rank_key(row: JsonDict) -> tuple[Any, ...]:
        metrics = row.get("metrics", {})
        return (
            not bool(row.get("eligible", False)),
            float(metrics.get("negative_false_accept_rate", 0.0)),
            int(metrics.get("unsupported_part_emissions", 0)),
            -float(metrics.get("wrong_part_rejection_rate", 0.0)),
            -float(metrics.get("crop_accuracy", 0.0)),
            -float(metrics.get("part_non_unknown_precision", 0.0)),
            -float(metrics.get("part_recall", 0.0)),
            float(metrics.get("abstention_rate", 0.0)),
            float(metrics.get("p95_latency_ms", 0.0)),
            float(metrics.get("mean_latency_ms", 0.0)),
            row.get("variant_id", ""),
        )

    return sorted(annotated, key=_rank_key)


def strip_samples(variant: JsonDict) -> JsonDict:
    slim = copy.deepcopy(variant)
    slim.pop("samples", None)
    return slim


def _sorted_numeric_summary(values: Iterable[Any]) -> JsonDict:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}

    ordered = sorted(cleaned)
    p95_index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95))))
    return {
        "count": len(ordered),
        "min": round(min(ordered), 4),
        "median": round(statistics.median(ordered), 4),
        "mean": round(statistics.fmean(ordered), 4),
        "p95": round(ordered[p95_index], 4),
        "max": round(max(ordered), 4),
    }


def _top_counts(values: Iterable[Any], *, limit: int = 5) -> List[JsonDict]:
    counts: Counter[str] = Counter()
    for value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized:
            continue
        counts[normalized] += 1
    return [{"value": value, "count": count} for value, count in counts.most_common(max(1, int(limit)))]


def _variant_failure_causes(sample: JsonDict) -> List[str]:
    causes: List[str] = []
    group = str(sample.get("group", "") or "").strip().lower()
    handoff = bool(sample.get("handoff_crop", False))
    gate_reasons = [str(reason) for reason in (sample.get("runtime_gate_reasons") or []) if str(reason).strip()]

    if group in {"off_crop", "non_plant", "ambiguous"} and handoff:
        guard = sample.get("input_guard") if isinstance(sample.get("input_guard"), dict) else {}
        plant_score = float(guard.get("plant_score", 0.0) or 0.0) if guard else 0.0
        non_plant_score = float(guard.get("non_plant_score", 0.0) or 0.0) if guard else 0.0
        if guard and non_plant_score >= plant_score:
            causes.append("input_guard_not_separating_negatives")
        if float(sample.get("routing_margin", 0.0) or 0.0) <= 0.10:
            causes.append("router_margin_too_small")
        if float(sample.get("crop_confidence", 0.0) or 0.0) <= 0.75:
            causes.append("router_min_confidence_too_low")

    if group == "wrong_part" and handoff:
        if bool(sample.get("unsupported_part_emitted", False)):
            causes.append("unsupported_part_emitted")
        if bool(sample.get("part_abstained", False)):
            causes.append("part_open_set_too_aggressive")
        else:
            causes.append("part_open_set_threshold_too_low")

    if not handoff and bool(sample.get("expected_handoff", False)):
        if any(reason.startswith("input_guard_") for reason in gate_reasons):
            causes.append("guard_too_aggressive")
        if any(reason.startswith("router_min_") for reason in gate_reasons):
            causes.append("router_thresholds_too_aggressive")

    return sorted(dict.fromkeys(causes))


def build_failure_analysis(samples: Sequence[JsonDict], *, limit: int = 5) -> JsonDict:
    rows = [dict(sample) for sample in samples]
    false_accepts = [
        row
        for row in rows
        if str(row.get("group", "")).strip().lower() in {"off_crop", "non_plant", "ambiguous", "wrong_part"}
        and bool(row.get("handoff_crop", False))
    ]
    false_rejects = [
        row for row in rows if bool(row.get("expected_handoff", False)) and not bool(row.get("handoff_crop", False))
    ]
    hardest_false_accepts = sorted(
        false_accepts,
        key=lambda row: (
            -float(row.get("crop_confidence", 0.0) or 0.0),
            -float(row.get("routing_margin", 0.0) or 0.0),
            str(row.get("image_path", "")),
        ),
    )[: max(1, int(limit))]

    false_accept_causes: Counter[str] = Counter()
    false_reject_causes: Counter[str] = Counter()
    for row in false_accepts:
        false_accept_causes.update(_variant_failure_causes(row))
    for row in false_rejects:
        false_reject_causes.update(_variant_failure_causes(row))

    input_guard_rows = [row for row in false_accepts if isinstance(row.get("input_guard"), dict) and row.get("input_guard")]
    return {
        "false_accept_count": len(false_accepts),
        "false_accept_counts_by_group": {
            group: sum(1 for row in false_accepts if str(row.get("group", "")).strip().lower() == group)
            for group in ("off_crop", "non_plant", "ambiguous", "wrong_part")
        },
        "false_accept_top_predicted_crops": _top_counts(row.get("predicted_crop") for row in false_accepts),
        "false_accept_top_predicted_parts": _top_counts(row.get("predicted_part") for row in false_accepts),
        "false_accept_confidence_distribution": _sorted_numeric_summary(row.get("crop_confidence") for row in false_accepts),
        "false_accept_margin_distribution": _sorted_numeric_summary(row.get("routing_margin") for row in false_accepts),
        "false_accept_crop_confidence_margin_distribution": _sorted_numeric_summary(
            row.get("crop_confidence_margin") for row in false_accepts
        ),
        "false_accept_input_guard_plant_score_distribution": _sorted_numeric_summary(
            (row.get("input_guard") or {}).get("plant_score") for row in input_guard_rows
        ),
        "false_accept_input_guard_non_plant_score_distribution": _sorted_numeric_summary(
            (row.get("input_guard") or {}).get("non_plant_score") for row in input_guard_rows
        ),
        "hardest_false_accept_examples": [
            {
                "image_path": row.get("image_path"),
                "group": row.get("group"),
                "predicted_crop": row.get("predicted_crop"),
                "predicted_part": row.get("predicted_part"),
                "crop_confidence": row.get("crop_confidence"),
                "routing_margin": row.get("routing_margin"),
                "crop_confidence_margin": row.get("crop_confidence_margin"),
                "rejection_reason": row.get("rejection_reason"),
                "runtime_gate_reasons": row.get("runtime_gate_reasons"),
            }
            for row in hardest_false_accepts
        ],
        "false_accept_failure_causes": [
            {"cause": cause, "count": count} for cause, count in false_accept_causes.most_common()
        ],
        "false_reject_count": len(false_rejects),
        "false_reject_failure_causes": [
            {"cause": cause, "count": count} for cause, count in false_reject_causes.most_common()
        ],
    }


def select_recommendation(ranked_variants: Sequence[JsonDict]) -> JsonDict:
    eligible = [row for row in ranked_variants if row.get("variant_id") != "baseline" and bool(row.get("eligible", False))]
    rejected = [row for row in ranked_variants if row.get("variant_id") != "baseline" and not bool(row.get("eligible", False))]
    recommended = eligible[0] if eligible else {}
    best_rejected = rejected[0] if rejected else {}
    return {
        "eligible_variants": eligible,
        "rejected_variants": rejected,
        "recommended": recommended,
        "best_rejected": best_rejected,
        "selection_summary": {
            "eligible_variant_count": len(eligible),
            "rejected_variant_count": len(rejected),
            "has_eligible_recommendation": bool(eligible),
            "recommended_variant_id": recommended.get("variant_id") if recommended else None,
            "best_rejected_variant_id": best_rejected.get("variant_id") if best_rejected else None,
        },
    }
