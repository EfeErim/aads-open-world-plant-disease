#!/usr/bin/env python3
"""Run a multi-parameter calibration sweep for the router handoff surface."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from typing import List, Sequence

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_router_surface import (  # noqa: E402, I001
    discover_eval_samples,
    sample_from_analysis,
    summarize_predictions,
)
from src.core.config_manager import get_config  # noqa: E402
from src.pipeline.input_guard import evaluate_plantness_input_guard  # noqa: E402
from src.router.policy_taxonomy_utils import resolve_requested_profile  # noqa: E402
from src.router.router_pipeline import RouterPipeline  # noqa: E402
from src.router.surface_calibration import (  # noqa: E402
    PRESET_SWEEPS,
    REPLAY_SAFE_PARAMETERS,
    JsonDict,
    _set_nested_value,
    annotate_and_rank_variants,
    apply_overrides,
    build_failure_analysis,
    is_replay_safe_overrides,
    iter_sweep_overrides,
    parse_sweep_spec,  # noqa: F401 - compatibility export
    replay_variant as _replay_variant,
    resolve_sweep_grid,
    select_recommendation,
    strip_samples,
    variant_count,
    variant_id,
)
from src.router.vlm_stages import build_pipeline_surface_config  # noqa: E402


def replay_variant(samples: Sequence[JsonDict], *, overrides: JsonDict) -> JsonDict:
    """Preserve the script-level API while delegating pure replay logic to ``src``."""
    return _replay_variant(samples, overrides=overrides, summarize_predictions=summarize_predictions)


def _apply_config_to_loaded_router(router: RouterPipeline, config: JsonDict) -> None:
    surface = build_pipeline_surface_config(config)
    router.config = copy.deepcopy(config)
    router.vlm_config = surface.vlm_config
    router._base_vlm_config = surface.base_vlm_config
    router.set_runtime_profile(resolve_requested_profile(router.vlm_config), suppress_warning=True)


def _print_progress(message: str) -> None:
    print(message, flush=True)


def evaluate_variant(
    router: RouterPipeline,
    dataset: Sequence[JsonDict],
    *,
    config: JsonDict,
    overrides: JsonDict,
    progress_label: str = "",
    progress_every: int = 25,
    collect_input_guard_scores: bool = False,
) -> JsonDict:
    _apply_config_to_loaded_router(router, config)
    guard_score_config = copy.deepcopy(config)
    guard_score_config = _set_nested_value(guard_score_config, "inference.input_guard.enabled", True)
    guard_score_config = _set_nested_value(guard_score_config, "inference.input_guard.debug_scores", False)

    samples: List[JsonDict] = []
    started_variant = time.perf_counter()
    total = len(dataset)
    for index, item in enumerate(dataset, start=1):
        image = Image.open(item["image_path"]).convert("RGB")
        started = time.perf_counter()
        analysis = router.analyze_image_result(image)
        latency_ms = (time.perf_counter() - started) * 1000.0
        sample = sample_from_analysis(
            item=item,
            analysis=analysis,
            latency_ms=latency_ms,
            config=config,
        )
        if collect_input_guard_scores:
            guard = evaluate_plantness_input_guard(
                router,
                image,
                guard_score_config,
                requested_part=str(item.get("expected_part") or ""),
            )
            sample["input_guard"] = guard.to_dict()
        samples.append(sample)
        if progress_label and (index == 1 or index == total or index % max(1, int(progress_every)) == 0):
            elapsed = time.perf_counter() - started_variant
            _print_progress(f"[{progress_label}] {index}/{total} samples elapsed={elapsed:.1f}s")

    metrics = summarize_predictions(samples)
    metrics["variant_wall_time_ms"] = round((time.perf_counter() - started_variant) * 1000.0, 4)
    return {
        "variant_id": "baseline" if not overrides else variant_id(overrides),
        "overrides": copy.deepcopy(overrides),
        "metrics": metrics,
        "samples": samples,
    }


def calibrate_router_surface(
    root: Path,
    *,
    config_env: str | None = "colab",
    device: str = "cuda",
    preset: str = "quick",
    sweep_specs: Sequence[str] | None = None,
    include_current: bool = True,
    max_variants: int = 128,
    target_negative_false_accept_rate: float = 0.05,
    max_crop_accuracy_drop: float = 0.02,
    max_part_precision_drop: float = 0.02,
    max_part_recall_drop: float = 0.02,
    max_wrong_part_rejection_drop: float = 0.02,
    max_p95_latency_regression: float = 0.25,
    include_samples: bool = False,
    strategy: str = "grid",
    progress_every: int = 25,
    collect_input_guard_scores: bool = False,
    adaptive_top_k: int = 10,
    adaptive_n_per_group: int = 5,
) -> JsonDict:
    dataset = discover_eval_samples(root)
    if not dataset:
        raise RuntimeError(f"No router eval images found under {root}")

    base_config = get_config(environment=config_env)
    grid = resolve_sweep_grid(
        base_config,
        preset=preset,
        sweep_specs=sweep_specs,
        include_current=include_current,
    )
    total_variants = variant_count(grid)
    if total_variants > int(max_variants):
        raise RuntimeError(
            f"Sweep expands to {total_variants} variants, above --max-variants={max_variants}. "
            "Use a smaller preset, fewer --sweep values, or raise --max-variants intentionally."
        )
    resolved_strategy = str(strategy or "grid").strip().lower()
    if resolved_strategy not in {"grid", "replay-thresholds", "adaptive"}:
        raise ValueError("strategy must be one of: grid, replay-thresholds, adaptive")
    if resolved_strategy == "replay-thresholds":
        unsafe = sorted(parameter for parameter in grid if parameter not in REPLAY_SAFE_PARAMETERS)
        if unsafe:
            raise RuntimeError(
                "replay-thresholds can only replay cached runtime threshold gates. "
                f"Move these parameters to grid mode or remove them: {', '.join(unsafe)}"
            )

    router = RouterPipeline(config=base_config, device=device)
    router.load_models()
    if not router.is_ready():
        raise RuntimeError(
            "Router models failed to become ready for inference. "
            "Check router.vlm.enabled, model availability, and router dependency installation."
        )

    baseline = evaluate_variant(
        router,
        dataset,
        config=base_config,
        overrides={},
        progress_label="BASELINE",
        progress_every=progress_every,
        collect_input_guard_scores=collect_input_guard_scores,
    )
    variants: List[JsonDict] = []
    seen_variants = {json.dumps({}, sort_keys=True)}
    seen_configs = {json.dumps(base_config, sort_keys=True, default=str)}
    baseline_samples = list(baseline.get("samples") or [])
    for index, overrides in enumerate(iter_sweep_overrides(grid), start=1):
        key = json.dumps(overrides, sort_keys=True)
        if key in seen_variants:
            continue
        seen_variants.add(key)
        variant_config = apply_overrides(base_config, overrides)
        config_key = json.dumps(variant_config, sort_keys=True, default=str)
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        if resolved_strategy == "replay-thresholds" and is_replay_safe_overrides(overrides):
            variants.append(replay_variant(baseline_samples, overrides=overrides))
            continue
        variants.append(
            evaluate_variant(
                router,
                dataset,
                config=variant_config,
                overrides=overrides,
                progress_label=f"VARIANT {index}",
                progress_every=progress_every,
                collect_input_guard_scores=collect_input_guard_scores,
            )
        )

    # Adaptive successive-halving strategy: evaluate many candidates on a small
    # balanced subset first, promote top candidates to full eval, then rank.
    if resolved_strategy == "adaptive":
        # build full candidate list but do not evaluate yet
        grid_variants = list(iter_sweep_overrides(grid))
        # small balanced subset: sample up to N_PER_GROUP images per group
        N_PER_GROUP = int(adaptive_n_per_group)
        groups = {}
        for row in baseline_samples:
            g = str(row.get("group") or "")
            groups.setdefault(g, []).append(row)
        small_subset = []
        for g, rows in groups.items():
            small_subset.extend(rows[:N_PER_GROUP])

        quick_results: List[JsonDict] = []
        for overrides in grid_variants:
            variant_config = apply_overrides(base_config, overrides)
            # reuse router without reloading models
            quick = evaluate_variant(
                router,
                small_subset,
                config=variant_config,
                overrides=overrides,
                progress_label="ADAPTIVE_QUICK",
                progress_every=max(1, progress_every),
                collect_input_guard_scores=collect_input_guard_scores,
            )
            quick_results.append(quick)

        # Promote top-K by negative FAR then abstention, keep reasonable cap
        top_k = int(adaptive_top_k)
        promoted = sorted(
            quick_results,
            key=lambda r: (float(r.get("metrics", {}).get("negative_false_accept_rate", 1.0)), float(r.get("metrics", {}).get("abstention_rate", 1.0)))
        )[: min(top_k, max(1, len(quick_results)))]

        # Full evaluation for promoted
        variants = []
        for promo in promoted:
            overrides = promo.get("overrides") or {}
            variant_config = apply_overrides(base_config, overrides)
            full = evaluate_variant(
                router,
                dataset,
                config=variant_config,
                overrides=overrides,
                progress_label="ADAPTIVE_FULL",
                progress_every=progress_every,
                collect_input_guard_scores=collect_input_guard_scores,
            )
            variants.append(full)

    ranked = annotate_and_rank_variants(
        [baseline, *variants],
        baseline=baseline,
        target_negative_false_accept_rate=target_negative_false_accept_rate,
        max_crop_accuracy_drop=max_crop_accuracy_drop,
        max_part_precision_drop=max_part_precision_drop,
        max_part_recall_drop=max_part_recall_drop,
        max_wrong_part_rejection_drop=max_wrong_part_rejection_drop,
        max_p95_latency_regression=max_p95_latency_regression,
    )
    selection = select_recommendation(ranked)
    recommended = selection["recommended"]
    best_rejected = selection["best_rejected"]
    failure_analysis_source = recommended or best_rejected or baseline
    variant_times = [float(row.get("metrics", {}).get("variant_wall_time_ms", 0.0)) for row in variants]

    result = {
        "dataset_root": str(root),
        "sample_count": len(dataset),
        "config_env": config_env,
        "device": device,
        "strategy": resolved_strategy,
        "input_guard_scores_cached": bool(collect_input_guard_scores),
        "preset": preset,
        "sweep_grid": grid,
        "variant_count": len(variants),
        "baseline": baseline if include_samples else strip_samples(baseline),
        "recommended": recommended if include_samples else strip_samples(recommended),
        "best_rejected": best_rejected if include_samples else strip_samples(best_rejected),
        "selection_summary": selection["selection_summary"],
        "failure_analysis": build_failure_analysis(failure_analysis_source.get("samples") or []),
        "variants": ranked if include_samples else [strip_samples(row) for row in ranked],
        "eligible_variants": selection["eligible_variants"] if include_samples else [strip_samples(row) for row in selection["eligible_variants"]],
        "rejected_variants": selection["rejected_variants"] if include_samples else [strip_samples(row) for row in selection["rejected_variants"]],
        "runtime_summary": {
            "mean_variant_wall_time_ms": 0.0 if not variant_times else round(statistics.fmean(variant_times), 4),
            "max_variant_wall_time_ms": 0.0 if not variant_times else round(max(variant_times), 4),
        },
    }
    return result


def validate_router_candidate_overrides(
    root: Path,
    *,
    candidate_overrides: Sequence[JsonDict],
    config_env: str | None = "colab",
    device: str = "cuda",
    target_negative_false_accept_rate: float = 0.05,
    max_crop_accuracy_drop: float = 0.02,
    max_part_precision_drop: float = 0.02,
    max_part_recall_drop: float = 0.02,
    max_wrong_part_rejection_drop: float = 0.02,
    max_p95_latency_regression: float = 0.25,
    include_samples: bool = False,
    strategy: str = "grid",
    progress_every: int = 25,
    collect_input_guard_scores: bool = False,
) -> JsonDict:
    """Evaluate selected dev-set calibration candidates on an independent root."""
    dataset = discover_eval_samples(root)
    if not dataset:
        raise RuntimeError(f"No router eval images found under {root}")

    base_config = get_config(environment=config_env)
    router = RouterPipeline(config=base_config, device=device)
    router.load_models()
    if not router.is_ready():
        raise RuntimeError(
            "Router models failed to become ready for inference. "
            "Check router.vlm.enabled, model availability, and router dependency installation."
        )

    baseline = evaluate_variant(
        router,
        dataset,
        config=base_config,
        overrides={},
        progress_label="HOLDOUT BASELINE",
        progress_every=progress_every,
        collect_input_guard_scores=collect_input_guard_scores,
    )
    variants: List[JsonDict] = []
    seen: set[str] = set()
    resolved_strategy = str(strategy or "grid").strip().lower()
    baseline_samples = list(baseline.get("samples") or [])
    for overrides in candidate_overrides:
        normalized = dict(overrides or {})
        if not normalized:
            continue
        key = json.dumps(normalized, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        variant_config = apply_overrides(base_config, normalized)
        if resolved_strategy == "replay-thresholds" and is_replay_safe_overrides(normalized):
            variants.append(
                replay_variant(baseline_samples, overrides=normalized)
            )
        else:
            variants.append(
                evaluate_variant(
                    router,
                    dataset,
                    config=variant_config,
                    overrides=normalized,
                    progress_label="HOLDOUT VARIANT",
                    progress_every=progress_every,
                    collect_input_guard_scores=collect_input_guard_scores,
                )
            )

    ranked = annotate_and_rank_variants(
        [baseline, *variants],
        baseline=baseline,
        target_negative_false_accept_rate=target_negative_false_accept_rate,
        max_crop_accuracy_drop=max_crop_accuracy_drop,
        max_part_precision_drop=max_part_precision_drop,
        max_part_recall_drop=max_part_recall_drop,
        max_wrong_part_rejection_drop=max_wrong_part_rejection_drop,
        max_p95_latency_regression=max_p95_latency_regression,
    )
    selection = select_recommendation(ranked)
    accepted = selection["eligible_variants"]

    return {
        "dataset_root": str(root),
        "sample_count": len(dataset),
        "config_env": config_env,
        "device": device,
        "strategy": resolved_strategy,
        "input_guard_scores_cached": bool(collect_input_guard_scores),
        "candidate_count": len(variants),
        "baseline": baseline if include_samples else strip_samples(baseline),
        "variants": ranked if include_samples else [strip_samples(row) for row in ranked],
        "accepted": accepted if include_samples else [strip_samples(row) for row in accepted],
        "recommended": selection["recommended"] if include_samples else strip_samples(selection["recommended"]),
        "best_rejected": selection["best_rejected"] if include_samples else strip_samples(selection["best_rejected"]),
        "selection_summary": selection["selection_summary"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Eval root: data/router_eval/{id,negatives,ambiguous,wrong_part}/...",
    )
    parser.add_argument("--config-env", default="colab", help="Config environment override (default: colab)")
    parser.add_argument("--device", default="cuda", help="Torch device preference")
    parser.add_argument(
        "--preset",
        choices=["none", *sorted(PRESET_SWEEPS.keys())],
        default="quick",
        help="Built-in sweep grid. Use 'none' with explicit --sweep entries.",
    )
    parser.add_argument(
        "--strategy",
        choices=["grid", "replay-thresholds"],
        default="grid",
        help=(
            "grid re-runs router inference for every variant. replay-thresholds runs router inference once "
            "and replays router_min_confidence/router_min_margin gates from cached evidence."
        ),
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        help=(
            "Override or add one grid dimension as PARAM=v1,v2. "
            "Aliases include router_min_confidence, router_min_margin, "
            "vlm_confidence_threshold, global_crop_context_weight, sam3_mask_threshold."
        ),
    )
    parser.add_argument(
        "--exclude-current",
        action="store_true",
        help="Do not automatically include the current config value for every swept parameter.",
    )
    parser.add_argument("--max-variants", type=int, default=128, help="Refuse sweeps larger than this count.")
    parser.add_argument(
        "--target-negative-far",
        type=float,
        default=0.05,
        help="Maximum negative false-accept rate for an eligible recommendation.",
    )
    parser.add_argument("--max-crop-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--max-part-precision-drop", type=float, default=0.02)
    parser.add_argument("--max-part-recall-drop", type=float, default=0.02)
    parser.add_argument("--max-wrong-part-rejection-drop", type=float, default=0.02)
    parser.add_argument(
        "--max-p95-latency-regression",
        type=float,
        default=0.25,
        help="Maximum allowed p95 latency increase vs baseline as a fraction (default: 0.25).",
    )
    parser.add_argument("--include-samples", action="store_true", help="Include per-sample rows for every variant.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print router progress every N samples.")
    parser.add_argument(
        "--collect-input-guard-scores",
        action="store_true",
        help="Cache plant/non-plant input-guard scores once so input guard thresholds can be replayed.",
    )
    parser.add_argument(
        "--adaptive-top-k",
        type=int,
        default=10,
        help="Number of top quick-eval candidates to promote to full evaluation in adaptive strategy.",
    )
    parser.add_argument(
        "--adaptive-n-per-group",
        type=int,
        default=5,
        help="Number of per-group samples to use for quick adaptive evaluation.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = calibrate_router_surface(
        args.root,
        config_env=args.config_env,
        device=args.device,
        preset=args.preset,
        sweep_specs=args.sweep,
        include_current=not args.exclude_current,
        max_variants=args.max_variants,
        target_negative_false_accept_rate=args.target_negative_far,
        max_crop_accuracy_drop=args.max_crop_accuracy_drop,
        max_part_precision_drop=args.max_part_precision_drop,
        max_part_recall_drop=args.max_part_recall_drop,
        max_wrong_part_rejection_drop=args.max_wrong_part_rejection_drop,
        max_p95_latency_regression=args.max_p95_latency_regression,
        include_samples=args.include_samples,
        strategy=args.strategy,
        progress_every=args.progress_every,
        collect_input_guard_scores=args.collect_input_guard_scores,
        adaptive_top_k=args.adaptive_top_k,
        adaptive_n_per_group=args.adaptive_n_per_group,
    )
    body = json.dumps(result, indent=2)
    # Try to write the requested output path; if that fails, write a fallback file
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(body, encoding="utf-8")
        except Exception as e:
            print(f"FAILED_TO_WRITE_REQUESTED_OUTPUT: {e}", file=sys.stderr)
    try:
        fallback = Path(".runtime_tmp/router_calibration_fallback.json")
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(body, encoding="utf-8")
        print(f"FALLBACK_WRITTEN {fallback}")
    except Exception as e:
        print(f"FAILED_TO_WRITE_FALLBACK_OUTPUT: {e}", file=sys.stderr)
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
