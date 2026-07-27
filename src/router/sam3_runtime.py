"""SAM3 orchestration helpers for the VLM router runtime."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from src.router import clip_runtime
from src.router.analysis_results import (
    build_sam3_analysis_result,
    init_sam3_stage_timings,
    summarize_sam3_stage_timings,
)
from src.router.compatibility_utils import compatible_parts_for_crop
from src.router.confidence_utils import (
    passes_open_set_gate,
    resolve_effective_confidence_threshold,
)
from src.router.heuristics import (
    apply_generic_part_penalty,
    apply_leaf_like_override,
    compute_leaf_likeness,
    rebalance_part_scores_for_leaf_like_roi,
    select_best_crop_with_fallback,
    select_part_label_with_specificity,
)
from src.router.label_normalization import normalize_part_label
from src.router.pipeline_flow_utils import resolve_effective_max_detections
from src.router.roi_helpers import (
    bbox_area_ratio,
    coerce_image_input,
    extract_roi,
    sanitize_bbox,
    suppress_overlapping_detections,
)
from src.router.roi_pipeline import (
    RoiClassificationHooks,
    classify_sam3_roi_candidate,
    collect_sam3_roi_candidates,
    filter_classified_sam3_detections,
    finalize_sam3_roi_candidate,
    run_sam3_roi_classification_stage,
)
from src.router.runtime_settings import build_sam3_runtime_settings, resolve_sam3_stage_order
from src.router.runtime_surface import coerce_bool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sam3RequestContext:
    pil_image: Image.Image
    image_size: Tuple[int, int, int]
    image_width: int
    image_height: int
    effective_threshold: float
    effective_max_detections: Optional[int]
    stage_order: List[str]
    settings: Dict[str, Any]
    sam3_prompts: Tuple[str, ...]
    sam3_threshold: float
    timing_logs_enabled: bool
    preprocess_ms: float


def _normalize_sam3_prompt_label(label: Any) -> str:
    return normalize_part_label(label)


def build_request_context(
    runtime: Any,
    *,
    pil_image: Image.Image,
    image_size: Tuple[int, int, int],
    confidence_threshold: float,
    max_detections: Optional[int],
) -> Sam3RequestContext:
    """Build request-local state once per analyze_image call."""
    started_at = time.perf_counter()
    effective_threshold = resolve_effective_confidence_threshold(confidence_threshold, runtime._policy_value)
    effective_max_detections = resolve_effective_max_detections(max_detections)
    settings = build_sam3_runtime_settings(runtime._policy_value, runtime.vlm_config, effective_threshold)
    stage_order = resolve_sam3_stage_order(runtime._policy_value)
    image_width, image_height = pil_image.size
    preprocess_ms = (time.perf_counter() - started_at) * 1000.0
    return Sam3RequestContext(
        pil_image=pil_image,
        image_size=image_size,
        image_width=image_width,
        image_height=image_height,
        effective_threshold=effective_threshold,
        effective_max_detections=effective_max_detections,
        stage_order=stage_order,
        settings=settings,
        sam3_prompts=resolve_sam3_text_prompts(runtime),
        sam3_threshold=settings["sam3_threshold"],
        timing_logs_enabled=coerce_bool(runtime.vlm_config.get("timing_logs_enabled", True), default=True),
        preprocess_ms=preprocess_ms,
    )


def postprocess_sam3_detections(
    runtime: Any,
    detections: List[Dict[str, Any]],
    *,
    context: Sam3RequestContext,
) -> List[Dict[str, Any]]:
    """Finalize SAM3 detections with optional dedupe and cap."""
    ordered = sorted(detections, key=lambda d: float(d.get("_quality_score", 0.0)), reverse=True)
    if runtime._policy_enabled("dedupe", True) and "postprocess" in context.stage_order:
        ordered = suppress_overlapping_detections(
            ordered,
            iou_threshold=context.settings["detection_nms_iou_threshold"],
            same_crop_only=context.settings["detection_nms_same_crop_only"],
        )
    if context.effective_max_detections is not None:
        ordered = ordered[: context.effective_max_detections]
    for det in ordered:
        quality_score = det.pop("_quality_score", None)
        if quality_score is not None:
            det["quality_score"] = float(quality_score)
    return ordered


def run_sam3_roi_filter_stage(
    runtime: Any,
    boxes: Any,
    scores: Any,
    *,
    context: Sam3RequestContext,
) -> Tuple[List[Dict[str, Any]], int]:
    """Execute ROI candidate filtering stage."""
    apply_roi_filters = runtime._policy_enabled("roi_filter", True) and "roi_filter" in context.stage_order
    return collect_sam3_roi_candidates(
        boxes=boxes,
        scores=scores,
        image_width=context.image_width,
        image_height=context.image_height,
        settings=context.settings,
        apply_roi_filters=apply_roi_filters,
        sanitize_bbox_fn=sanitize_bbox,
        bbox_area_ratio_fn=bbox_area_ratio,
    )


def _mask_count(masks: Any) -> int:
    if torch.is_tensor(masks):
        return int(masks.shape[0]) if masks.ndim > 0 else int(masks.numel() > 0)
    if isinstance(masks, (list, tuple)):
        return len(masks)
    return 0


def _resolve_positive_int(value: Any, default: int) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError):
        resolved = int(default)
    return max(1, resolved)


def resolve_sam3_text_prompts(runtime: Any) -> Tuple[str, ...]:
    """Resolve ordered SAM3 prompts from config and the supported crop-part surface."""
    prompt_candidates: List[str] = []

    raw_prompts = runtime.vlm_config.get("sam3_text_prompts")
    if isinstance(raw_prompts, list):
        prompt_candidates.extend(str(prompt).strip() for prompt in raw_prompts if str(prompt).strip())

    base_prompt = str(runtime.vlm_config.get("sam3_text_prompt", "plant")).strip() or "plant"
    if not prompt_candidates:
        prompt_candidates.append(base_prompt)

    if coerce_bool(runtime.vlm_config.get("sam3_expand_with_supported_parts", True), default=True):
        raw_part_prompts = runtime.vlm_config.get("sam3_part_prompts")
        if isinstance(raw_part_prompts, list):
            prompt_candidates.extend(str(prompt).strip() for prompt in raw_part_prompts if str(prompt).strip())
        else:
            router_cfg = runtime.config.get("router", {}) if isinstance(runtime.config, dict) else {}
            crop_mapping = router_cfg.get("crop_mapping", {}) if isinstance(router_cfg, dict) else {}
            if isinstance(crop_mapping, dict) and crop_mapping:
                for crop_data in crop_mapping.values():
                    if not isinstance(crop_data, dict):
                        continue
                    for part_name in crop_data.get("parts", []):
                        normalized = _normalize_sam3_prompt_label(part_name)
                        if normalized:
                            prompt_candidates.append(normalized)

    deduped: List[str] = []
    seen: set[str] = set()
    for prompt in prompt_candidates:
        normalized = str(prompt).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)

    raw_limit = runtime.vlm_config.get("sam3_prompt_limit")
    try:
        prompt_limit = int(raw_limit)
    except (TypeError, ValueError, OverflowError):
        prompt_limit = 0
    if prompt_limit > 0:
        deduped = deduped[:prompt_limit]

    return tuple(deduped or ["plant"])


def _empty_classification_result() -> Dict[str, Any]:
    return {
        "detections": [],
        "roi_kept": 0,
        "roi_classification_calls": 0,
        "roi_classification_ms": 0.0,
    }


def _build_sam3_analysis_payload(
    *,
    context: Sam3RequestContext,
    detections: List[Dict[str, Any]],
    stage_timings_ms: Dict[str, float],
    elapsed_ms: float,
    roi_seen: int,
    roi_kept: int,
    roi_classification_calls: int,
    mask_count: int,
    message: str = "",
) -> Dict[str, Any]:
    stage_summary = summarize_sam3_stage_timings(stage_timings_ms, roi_seen, roi_classification_calls)
    result = build_sam3_analysis_result(
        detections=detections,
        image_size=context.image_size,
        elapsed_ms=elapsed_ms,
        stage_summary=stage_summary,
        roi_seen=roi_seen,
        roi_kept=roi_kept,
        roi_classification_calls=roi_classification_calls,
        mask_count=mask_count,
        sam3_threshold=context.sam3_threshold,
    )
    if message:
        result["message"] = str(message)
    return result


def _build_empty_detection_message(
    *,
    context: Sam3RequestContext,
    mask_count: int,
    roi_seen: int,
    roi_kept: int,
) -> str:
    prompt_text = ",".join(context.sam3_prompts)
    if mask_count <= 0:
        return f"No SAM3 instances for prompts={prompt_text} threshold={context.sam3_threshold:.2f}."
    return (
        f"SAM3 produced {mask_count} instances for prompts={prompt_text} "
        f"but retained 0 detections after ROI filtering/classification "
        f"(roi_seen={roi_seen}, roi_kept={roi_kept}, "
        f"classification_min_confidence={float(context.settings['classification_min_confidence']):.2f})."
    )


def _build_part_abstention_message(detections: List[Dict[str, Any]]) -> str:
    if not detections:
        return ""
    primary = detections[0]
    if str(primary.get("part", "")).strip().lower() != "unknown":
        return ""
    reason = str(primary.get("part_rejection_reason", "")).strip()
    if not reason:
        return ""
    crop_name = str(primary.get("crop", "unknown") or "unknown")
    return f"Part abstained for crop={crop_name}: {reason}"


def _build_batch_contexts(
    runtime: Any,
    batch: torch.Tensor,
    *,
    confidence_threshold: float,
    max_detections: Optional[int],
) -> List[Sam3RequestContext]:
    batch_size = int(batch.shape[0]) if hasattr(batch, "shape") else 0
    contexts: List[Sam3RequestContext] = []
    for index in range(batch_size):
        pil_image, image_size = coerce_image_input(batch[index])
        contexts.append(
            build_request_context(
                runtime,
                pil_image=pil_image,
                image_size=image_size,
                confidence_threshold=confidence_threshold,
                max_detections=max_detections,
            )
        )
    return contexts


def _build_image_contexts(
    runtime: Any,
    images: List[Any],
    *,
    confidence_threshold: float,
    max_detections: Optional[int],
) -> List[Sam3RequestContext]:
    contexts: List[Sam3RequestContext] = []
    for image in images:
        pil_image, image_size = coerce_image_input(image)
        contexts.append(
            build_request_context(
                runtime,
                pil_image=pil_image,
                image_size=image_size,
                confidence_threshold=confidence_threshold,
                max_detections=max_detections,
            )
        )
    return contexts


def _chunk_supports_batched_sam3(contexts: List[Sam3RequestContext]) -> bool:
    prompts = {context.sam3_prompts for context in contexts}
    thresholds = {context.sam3_threshold for context in contexts}
    return len(prompts) == 1 and len(thresholds) == 1


def _collect_chunk_roi_candidates(
    runtime: Any,
    *,
    contexts: List[Sam3RequestContext],
    batched_results: List[Dict[str, Any]],
) -> tuple[List[List[Dict[str, Any]]], List[int], List[float], List[int]]:
    candidates_per_image: List[List[Dict[str, Any]]] = []
    roi_seen_per_image: List[int] = []
    roi_filter_ms_per_image: List[float] = []
    mask_count_per_image: List[int] = []
    for context, sam3_results in zip(contexts, batched_results):
        boxes = sam3_results.get("boxes", [])
        scores = sam3_results.get("scores", [])
        masks = sam3_results.get("masks", [])
        mask_count_per_image.append(_mask_count(masks))
        roi_started_at = time.perf_counter()
        candidates, roi_seen = run_sam3_roi_filter_stage(runtime, boxes, scores, context=context)
        roi_filter_ms_per_image.append((time.perf_counter() - roi_started_at) * 1000.0)
        candidates_per_image.append(candidates)
        roi_seen_per_image.append(roi_seen)
    return candidates_per_image, roi_seen_per_image, roi_filter_ms_per_image, mask_count_per_image


def _build_roi_classification_hooks(runtime: Any) -> RoiClassificationHooks:
    def _clip_score(
        roi_image: Image.Image,
        labels: List[str],
        *,
        label_type: str = "generic",
        num_prompts: Optional[int] = None,
    ) -> Tuple[str, float, Dict[str, float]]:
        return clip_runtime.clip_score_labels_ensemble(
            runtime,
            roi_image,
            labels,
            label_type=label_type,
            num_prompts=num_prompts,
        )

    def _select_best_crop(
        crop_scores: Dict[str, float],
        part_scores: Dict[str, float],
        **kwargs: Any,
    ) -> Tuple[str, float]:
        return select_best_crop_with_fallback(
            crop_scores,
            part_scores,
            runtime.crop_part_compatibility,
            runtime.part_labels,
            **kwargs,
        )

    def _compatible_parts(crop_label: str) -> List[str]:
        return compatible_parts_for_crop(
            crop_label,
            runtime.crop_part_compatibility,
            runtime.part_labels,
        )

    def _score_label_candidates(
        roi_image: Image.Image,
        labels: List[str],
        *,
        label_type: str = "generic",
        num_prompts: Optional[int] = None,
        open_set_enabled: bool = False,
        open_set_min_confidence: Optional[float] = None,
        open_set_margin: Optional[float] = None,
        unknown_label: str = "unknown",
    ) -> Dict[str, Any]:
        return clip_runtime.clip_score_labels_open_set(
            runtime,
            roi_image,
            labels,
            label_type=label_type,
            num_prompts=num_prompts,
            open_set_enabled=open_set_enabled,
            open_set_min_confidence=open_set_min_confidence,
            open_set_margin=open_set_margin,
            unknown_label=unknown_label,
        )

    return RoiClassificationHooks(
        policy_enabled=runtime._policy_enabled,
        extract_roi=extract_roi,
        clip_score_labels_ensemble=_clip_score,
        compute_leaf_likeness=compute_leaf_likeness,
        rebalance_part_scores_for_leaf_like_roi=rebalance_part_scores_for_leaf_like_roi,
        select_best_crop_with_fallback=_select_best_crop,
        compatible_parts_for_crop=_compatible_parts,
        score_parts_conditioned_on_crop=runtime._score_parts_conditioned_on_crop,
        score_label_candidates=_score_label_candidates,
        apply_generic_part_penalty=apply_generic_part_penalty,
        select_part_label_with_specificity=select_part_label_with_specificity,
        apply_leaf_like_override=apply_leaf_like_override,
    )


def _score_global_crop_context(
    runtime: Any,
    image: Image.Image,
    *,
    num_prompts: Optional[int] = None,
) -> Dict[str, float]:
    if not runtime.crop_labels:
        return {}
    _label, _score, score_map = clip_runtime.clip_score_labels_ensemble(
        runtime,
        image,
        runtime.crop_labels,
        label_type="crop",
        num_prompts=num_prompts,
    )
    return score_map


def _classify_chunk_candidates_batched(
    runtime: Any,
    *,
    contexts: List[Sam3RequestContext],
    candidates_per_image: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Batch-score ROI candidates, then finalize/filter detections per image."""
    records: List[Dict[str, Any]] = []
    hooks = _build_roi_classification_hooks(runtime)
    roi_score_batch_size = _resolve_positive_int(runtime.vlm_config.get("roi_score_batch_size", 32), 32)
    for image_index, (context, candidates) in enumerate(zip(contexts, candidates_per_image)):
        for candidate_index, candidate in enumerate(candidates):
            roi_image = extract_roi(context.pil_image, candidate["bbox"], pad_ratio=0.08)
            records.append(
                {
                    "image_index": image_index,
                    "candidate_index": candidate_index,
                    "candidate": candidate,
                    "context": context,
                    "roi_image": roi_image,
                }
            )

    if not records:
        return [_empty_classification_result() for _ in contexts]

    roi_images = [record["roi_image"] for record in records]
    if contexts and contexts[0].settings.get("global_crop_context_enabled", True) and runtime.crop_labels:
        global_crop_results = clip_runtime.clip_score_labels_ensemble_batch(
            runtime,
            [context.pil_image for context in contexts],
            runtime.crop_labels,
            label_type="crop",
            num_prompts=contexts[0].settings.get("crop_num_prompts"),
            image_batch_size=roi_score_batch_size,
        )
        global_crop_scores_by_image: List[Dict[str, float]] = [result[2] for result in global_crop_results]
    else:
        global_crop_scores_by_image = [{} for _ in contexts]

    scoring_started_at = time.perf_counter()
    part_results = clip_runtime.clip_score_labels_ensemble_batch(
        runtime,
        roi_images,
        runtime.part_labels,
        label_type="part",
        num_prompts=contexts[0].settings.get("part_num_prompts"),
        image_batch_size=roi_score_batch_size,
    )
    crop_results = clip_runtime.clip_score_labels_ensemble_batch(
        runtime,
        roi_images,
        runtime.crop_labels,
        label_type="crop",
        num_prompts=contexts[0].settings.get("crop_num_prompts"),
        image_batch_size=roi_score_batch_size,
    )
    scoring_ms = (time.perf_counter() - scoring_started_at) * 1000.0

    per_image_all_detections: List[List[Dict[str, Any]]] = [[] for _ in contexts]
    per_image_calls: List[int] = [0 for _ in contexts]
    finalize_started_at = time.perf_counter()
    for record, part_result, crop_result in zip(records, part_results, crop_results):
        image_index = int(record["image_index"])
        context = record["context"]
        detection = finalize_sam3_roi_candidate(
            roi_image=record["roi_image"],
            candidate=record["candidate"],
            image_width=context.image_width,
            image_height=context.image_height,
            settings=context.settings,
            hooks=hooks,
            part_label=part_result[0],
            part_conf=part_result[1],
            part_scores=part_result[2],
            crop_label=crop_result[0],
            crop_conf=crop_result[1],
            crop_scores=crop_result[2],
            global_crop_scores=global_crop_scores_by_image[image_index],
        )
        per_image_all_detections[image_index].append(detection)
        per_image_calls[image_index] += 2
    finalize_ms = (time.perf_counter() - finalize_started_at) * 1000.0

    total_records = max(1, len(records))
    results: List[Dict[str, Any]] = []
    for image_index, all_detections in enumerate(per_image_all_detections):
        detections, roi_kept = filter_classified_sam3_detections(
            all_detections=all_detections,
            settings=contexts[image_index].settings,
            stage_order=contexts[image_index].stage_order,
            policy_enabled_fn=runtime._policy_enabled,
            passes_open_set_gate_fn=passes_open_set_gate,
        )
        record_count = max(0, len(all_detections))
        timing_share = (scoring_ms + finalize_ms) * (float(record_count) / float(total_records))
        results.append(
            {
                "detections": detections,
                "roi_kept": roi_kept,
                "roi_classification_calls": per_image_calls[image_index],
                "roi_classification_ms": timing_share,
            }
        )
    return results


def analyze_sam3_image(runtime: Any, context: Sam3RequestContext) -> Dict[str, Any]:
    """Analyze an image using the SAM3 + BioCLIP runtime."""
    start_time = time.perf_counter()
    stage_timings_ms = init_sam3_stage_timings()
    stage_timings_ms["preprocess"] = context.preprocess_ms
    hooks = _build_roi_classification_hooks(runtime)

    stage_start = time.perf_counter()
    sam3_results = runtime._run_sam3(
        context.pil_image,
        prompt=context.sam3_prompts,
        threshold=context.sam3_threshold,
    )
    stage_timings_ms["sam3_inference"] = (time.perf_counter() - stage_start) * 1000.0
    masks = sam3_results.get("masks", [])
    boxes = sam3_results.get("boxes", [])
    scores = sam3_results.get("scores", [])

    mask_count = _mask_count(masks)

    detections: List[Dict[str, Any]] = []
    roi_seen = 0
    roi_kept = 0
    roi_classification_calls = 0
    if mask_count > 0:
        global_crop_scores = _score_global_crop_context(
            runtime,
            context.pil_image,
            num_prompts=context.settings.get("crop_num_prompts"),
        )
        roi_stage_start = time.perf_counter()
        candidates, roi_seen = run_sam3_roi_filter_stage(runtime, boxes, scores, context=context)

        detections, roi_kept, roi_classification_calls, roi_classification_ms = run_sam3_roi_classification_stage(
            candidates=candidates,
            settings=context.settings,
            stage_order=context.stage_order,
            policy_enabled_fn=runtime._policy_enabled,
            classify_candidate_fn=lambda candidate: classify_sam3_roi_candidate(
                pil_image=context.pil_image,
                candidate=candidate,
                image_width=context.image_width,
                image_height=context.image_height,
                settings=context.settings,
                part_labels=runtime.part_labels,
                crop_labels=runtime.crop_labels,
                hooks=hooks,
                global_crop_scores=global_crop_scores,
            ),
            passes_open_set_gate_fn=passes_open_set_gate,
        )
        stage_timings_ms["roi_classification"] += roi_classification_ms
        stage_timings_ms["roi_total"] = (time.perf_counter() - roi_stage_start) * 1000.0

    stage_start = time.perf_counter()
    detections = postprocess_sam3_detections(runtime, detections, context=context)
    stage_timings_ms["postprocess"] = (time.perf_counter() - stage_start) * 1000.0
    message = ""
    if not detections:
        message = _build_empty_detection_message(
            context=context,
            mask_count=mask_count,
            roi_seen=roi_seen,
            roi_kept=roi_kept,
        )
    else:
        message = _build_part_abstention_message(detections)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    if context.timing_logs_enabled:
        logger.info(
            (
                "[TIMING] SAM3 pipeline | total=%.2fms | sam3=%.2fms | roi_total=%.2fms "
                "| roi_class=%.2fms | rois=%d | kept=%d"
            ),
            elapsed_ms,
            stage_timings_ms["sam3_inference"],
            stage_timings_ms["roi_total"],
            stage_timings_ms["roi_classification"],
            roi_seen,
            roi_kept,
        )

    return _build_sam3_analysis_payload(
        context=context,
        detections=detections,
        elapsed_ms=elapsed_ms,
        stage_timings_ms=stage_timings_ms,
        roi_seen=roi_seen,
        roi_kept=roi_kept,
        roi_classification_calls=roi_classification_calls,
        mask_count=mask_count,
        message=message,
    )


def analyze_sam3_batch(
    runtime: Any,
    batch: torch.Tensor,
    confidence_threshold: float = 0.8,
    max_detections: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Analyze a tensor batch with chunked SAM3 inference and batched CLIP ROI scoring."""
    return analyze_sam3_contexts(
        runtime,
        _build_batch_contexts(
            runtime,
            batch,
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
        ),
    )


def analyze_sam3_images(
    runtime: Any,
    images: List[Any],
    confidence_threshold: float = 0.8,
    max_detections: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Analyze image objects with chunked SAM3 inference and batched CLIP ROI scoring."""
    return analyze_sam3_contexts(
        runtime,
        _build_image_contexts(
            runtime,
            images,
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
        ),
    )


def analyze_sam3_contexts(
    runtime: Any,
    contexts: List[Sam3RequestContext],
) -> List[Dict[str, Any]]:
    """Analyze prebuilt SAM3 contexts with chunked SAM3 and batched CLIP ROI scoring."""
    if not contexts:
        return []

    chunk_size = _resolve_positive_int(runtime.vlm_config.get("batch_chunk_size", 8), 8)
    analyses: List[Dict[str, Any]] = []
    for start in range(0, len(contexts), chunk_size):
        chunk_contexts = contexts[start : start + chunk_size]
        if not chunk_contexts:
            continue

        if not _chunk_supports_batched_sam3(chunk_contexts):
            analyses.extend(analyze_sam3_image(runtime, context) for context in chunk_contexts)
            continue

        try:
            sam3_started_at = time.perf_counter()
            batched_results = runtime._run_sam3_batch(
                [context.pil_image for context in chunk_contexts],
                prompt=chunk_contexts[0].sam3_prompts,
                threshold=chunk_contexts[0].sam3_threshold,
            )
            sam3_total_ms = (time.perf_counter() - sam3_started_at) * 1000.0
        except Exception as exc:
            logger.info("Retrying chunk with per-image SAM3 analysis: %s", exc)
            analyses.extend(analyze_sam3_image(runtime, context) for context in chunk_contexts)
            continue

        candidates_per_image, roi_seen_per_image, roi_filter_ms_per_image, mask_count_per_image = (
            _collect_chunk_roi_candidates(
                runtime,
                contexts=chunk_contexts,
                batched_results=batched_results,
            )
        )

        classification_results = _classify_chunk_candidates_batched(
            runtime,
            contexts=chunk_contexts,
            candidates_per_image=candidates_per_image,
        )
        sam3_share_ms = sam3_total_ms / float(max(1, len(chunk_contexts)))
        for context, classification_result, mask_count, roi_seen, roi_filter_ms in zip(
            chunk_contexts,
            classification_results,
            mask_count_per_image,
            roi_seen_per_image,
            roi_filter_ms_per_image,
        ):
            stage_timings_ms = init_sam3_stage_timings()
            stage_timings_ms["preprocess"] = context.preprocess_ms
            stage_timings_ms["sam3_inference"] = sam3_share_ms
            stage_timings_ms["roi_classification"] = float(classification_result["roi_classification_ms"])
            stage_timings_ms["roi_total"] = roi_filter_ms + stage_timings_ms["roi_classification"]

            postprocess_started_at = time.perf_counter()
            detections = postprocess_sam3_detections(
                runtime,
                classification_result["detections"],
                context=context,
            )
            stage_timings_ms["postprocess"] = (time.perf_counter() - postprocess_started_at) * 1000.0
            elapsed_ms = sum(float(value) for value in stage_timings_ms.values())
            message = ""
            if not detections:
                message = _build_empty_detection_message(
                    context=context,
                    mask_count=mask_count,
                    roi_seen=roi_seen,
                    roi_kept=int(classification_result["roi_kept"]),
                )
            else:
                message = _build_part_abstention_message(detections)
            analyses.append(
                _build_sam3_analysis_payload(
                    context=context,
                    detections=detections,
                    elapsed_ms=elapsed_ms,
                    stage_timings_ms=stage_timings_ms,
                    roi_seen=roi_seen,
                    roi_kept=int(classification_result["roi_kept"]),
                    roi_classification_calls=int(classification_result["roi_classification_calls"]),
                    mask_count=mask_count,
                    message=message,
                )
            )
    return analyses
