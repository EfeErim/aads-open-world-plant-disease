#!/usr/bin/env python3
"""SAM3 ROI classification orchestration helpers for VLM router."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image

from src.router.confidence_utils import build_open_set_rejection_reasons
from src.router.label_normalization import normalize_part_label

logger = logging.getLogger(__name__)

BoundingBox = List[float]
Detection = Dict[str, Any]
Candidate = Dict[str, Any]

@dataclass(frozen=True)
class RoiClassificationHooks:
    policy_enabled: Callable[[str, bool], bool]
    extract_roi: Callable[..., Image.Image]
    clip_score_labels_ensemble: Callable[..., Tuple[str, float, Dict[str, float]]]
    compute_leaf_likeness: Callable[..., float]
    rebalance_part_scores_for_leaf_like_roi: Callable[..., Dict[str, float]]
    select_best_crop_with_fallback: Callable[..., Tuple[str, float]]
    compatible_parts_for_crop: Callable[[str], List[str]]
    score_parts_conditioned_on_crop: Callable[..., Dict[str, float]]
    score_label_candidates: Callable[..., Dict[str, Any]]
    apply_generic_part_penalty: Callable[[Dict[str, float], List[str], float], Dict[str, float]]
    select_part_label_with_specificity: Callable[..., Tuple[Optional[str], float]]
    apply_leaf_like_override: Callable[..., Tuple[str, float]]

LEAF_VISUAL_GENERIC_LABELS = {
    'whole plant',
    'whole',
    'plant',
    'entire plant',
}

__all__ = [
    'collect_sam3_roi_candidates',
    'classify_sam3_roi_candidate',
    'finalize_sam3_roi_candidate',
    'filter_classified_sam3_detections',
    'RoiClassificationHooks',
    'run_sam3_roi_classification_stage',
]


def _matches_focus_part(part_label: str, focus_parts_lower: List[str]) -> bool:
    return any(focus_part in part_label for focus_part in focus_parts_lower)


def _resolve_focus_settings(settings: Dict[str, Any]) -> tuple[bool, bool, float, List[str], List[str]]:
    focus_mode_enabled = bool(settings.get('focus_part_mode_enabled'))
    focus_fallback_enabled = bool(settings.get('focus_fallback_enabled'))
    focus_min_confidence = float(settings.get('focus_min_confidence_fallback', 0.50))
    focus_parts_raw = settings.get('focus_parts', ['leaf'])
    focus_parts = focus_parts_raw if isinstance(focus_parts_raw, list) else ['leaf']
    focus_parts_lower = [str(part).strip().lower() for part in focus_parts if str(part).strip()]
    return focus_mode_enabled, focus_fallback_enabled, focus_min_confidence, focus_parts, focus_parts_lower


def _collect_focused_detections(
    all_detections: List[Detection],
    *,
    focus_parts_lower: List[str],
    focus_min_confidence: float,
) -> List[Detection]:
    focused_detections: List[Detection] = []
    for detection in all_detections:
        part_label = str(detection.get('part', 'unknown')).strip().lower()
        part_confidence = float(detection.get('part_confidence', 0.0))
        if _matches_focus_part(part_label, focus_parts_lower) and part_confidence >= focus_min_confidence:
            focused_detections.append(detection)
    return focused_detections


def _resolve_focus_result_detections(
    *,
    all_detections: List[Detection],
    focused_detections: List[Detection],
    focus_mode_enabled: bool,
    focus_fallback_enabled: bool,
    focus_parts: List[str],
    focus_min_confidence: float,
) -> List[Detection]:
    if not (focus_mode_enabled and focus_fallback_enabled):
        return all_detections
    if focused_detections:
        max_focused_conf = max(float(d.get('part_confidence', 0.0)) for d in focused_detections)
        if max_focused_conf >= focus_min_confidence:
            logger.debug(
                "Using %d focused ROIs (parts=%s, max_conf=%.2f)",
                len(focused_detections),
                focus_parts,
                max_focused_conf,
            )
            return focused_detections
        logger.debug(
            "Focus mode confidence threshold not met (%.2f < %.2f); reverting to full ROI set",
            max_focused_conf,
            focus_min_confidence,
        )
        return all_detections
    logger.debug(
        "No ROIs with focus_parts=%s; reverting to full ROI set (%d total)",
        focus_parts,
        len(all_detections),
    )
    return all_detections


def _passes_detection_gate(
    detection: Detection,
    *,
    run_open_set_gate: bool,
    classification_min_confidence: float,
    passes_open_set_gate_fn: Callable[..., bool],
) -> bool:
    if not run_open_set_gate:
        return True
    return bool(
        passes_open_set_gate_fn(
            crop_label=str(detection.get('crop', 'unknown')),
            crop_confidence=float(detection.get('crop_confidence', 0.0)),
            min_confidence=classification_min_confidence,
        )
    )


def _classify_candidates(
    candidates: List[Candidate],
    *,
    classify_candidate_fn: Callable[[Candidate], Tuple[Optional[Detection], int]],
) -> tuple[List[Detection], int, float]:
    all_detections: List[Detection] = []
    roi_classification_calls = 0
    roi_classification_ms = 0.0
    for candidate in candidates:
        classify_start = time.perf_counter()
        detection, classification_calls = classify_candidate_fn(candidate)
        roi_classification_calls += classification_calls
        roi_classification_ms += (time.perf_counter() - classify_start) * 1000.0
        if detection is not None:
            all_detections.append(detection)
    return all_detections, roi_classification_calls, roi_classification_ms


def _resolve_top_part_metrics(
    part_scores: Dict[str, float],
    *,
    selected_label: str,
    selected_score: float,
) -> tuple[str, float, float, float]:
    """Return normalized top-label metrics from a part score map."""
    raw_label = str(selected_label or "unknown")
    raw_confidence = float(selected_score)
    label_key = str(selected_label).strip().lower()
    normalized_scores = [
        float(score)
        for part_name, score in part_scores.items()
        if str(part_name).strip().lower() != label_key
    ]
    second_confidence = max(normalized_scores, default=0.0)
    return raw_label, raw_confidence, float(second_confidence), float(raw_confidence - second_confidence)


def _resolve_surface_part_metrics(
    part_scores: Dict[str, float],
    *,
    generic_part_labels: List[str],
    generic_part_penalty: float,
    specific_part_override_ratio: float,
    specific_part_min_confidence: float,
    preferred_part_labels: List[str],
    preferred_part_override_ratio: float,
    apply_generic_part_penalty_fn: Callable[[Dict[str, float], List[str], float], Dict[str, float]],
    select_part_label_with_specificity_fn: Callable[..., Tuple[Optional[str], float]],
) -> tuple[str, float, float, float]:
    if not part_scores:
        return "unknown", 0.0, 0.0, 0.0

    adjusted_scores = apply_generic_part_penalty_fn(
        part_scores,
        generic_part_labels,
        generic_part_penalty,
    )
    if not adjusted_scores:
        return "unknown", 0.0, 0.0, 0.0

    selected_label, selected_score = select_part_label_with_specificity_fn(
        adjusted_scores,
        generic_part_labels,
        specific_override_ratio=specific_part_override_ratio,
        specific_min_confidence=specific_part_min_confidence,
        preferred_part_labels=preferred_part_labels,
        preferred_override_ratio=preferred_part_override_ratio,
    )
    if not selected_label:
        return "unknown", 0.0, 0.0, 0.0

    return _resolve_top_part_metrics(
        adjusted_scores,
        selected_label=selected_label,
        selected_score=selected_score,
    )


def _can_restore_part_after_unknown_open_set_rejection(
    *,
    open_set_rejection_reasons: List[str],
    raw_part_label: str,
    raw_part_confidence: float,
    raw_part_margin: float,
    generic_part_scores: Dict[str, float],
    conditioned_part_scores: Dict[str, float],
    generic_part_labels: List[str],
    generic_part_penalty: float,
    specific_part_override_ratio: float,
    specific_part_min_confidence: float,
    preferred_part_labels: List[str],
    preferred_part_override_ratio: float,
    min_confidence: float,
    margin_threshold: float,
    unknown_label: str,
    apply_generic_part_penalty_fn: Callable[[Dict[str, float], List[str], float], Dict[str, float]],
    select_part_label_with_specificity_fn: Callable[..., Tuple[Optional[str], float]],
) -> bool:
    normalized_reasons = [str(reason).strip() for reason in open_set_rejection_reasons if str(reason).strip()]
    if not normalized_reasons:
        return False
    if any(not reason.startswith("unknown_confidence") for reason in normalized_reasons):
        return False

    normalized_raw_label = normalize_part_label(raw_part_label)
    normalized_unknown = normalize_part_label(unknown_label) or "unknown"
    if not normalized_raw_label or normalized_raw_label == normalized_unknown:
        return False

    if float(raw_part_confidence) < float(min_confidence):
        return False

    strong_margin_floor = max(float(margin_threshold) * 2.0, 0.20)
    if float(raw_part_margin) < strong_margin_floor:
        return False

    generic_label, _generic_conf, _generic_second_conf, _generic_margin = _resolve_surface_part_metrics(
        generic_part_scores,
        generic_part_labels=generic_part_labels,
        generic_part_penalty=generic_part_penalty,
        specific_part_override_ratio=specific_part_override_ratio,
        specific_part_min_confidence=specific_part_min_confidence,
        preferred_part_labels=preferred_part_labels,
        preferred_part_override_ratio=preferred_part_override_ratio,
        apply_generic_part_penalty_fn=apply_generic_part_penalty_fn,
        select_part_label_with_specificity_fn=select_part_label_with_specificity_fn,
    )
    conditioned_label, _conditioned_conf, _conditioned_second_conf, _conditioned_margin = _resolve_surface_part_metrics(
        conditioned_part_scores,
        generic_part_labels=generic_part_labels,
        generic_part_penalty=generic_part_penalty,
        specific_part_override_ratio=specific_part_override_ratio,
        specific_part_min_confidence=specific_part_min_confidence,
        preferred_part_labels=preferred_part_labels,
        preferred_part_override_ratio=preferred_part_override_ratio,
        apply_generic_part_penalty_fn=apply_generic_part_penalty_fn,
        select_part_label_with_specificity_fn=select_part_label_with_specificity_fn,
    )

    return (
        normalize_part_label(generic_label) == normalized_raw_label
        and normalize_part_label(conditioned_label) == normalized_raw_label
    )


def collect_sam3_roi_candidates(
    boxes: Any,
    scores: Any,
    image_width: int,
    image_height: int,
    settings: Dict[str, Any],
    apply_roi_filters: bool,
    sanitize_bbox_fn: Callable[[Optional[BoundingBox], int, int], Optional[BoundingBox]],
    bbox_area_ratio_fn: Callable[[Optional[BoundingBox], int, int], float],
) -> Tuple[List[Candidate], int]:
    """Collect and filter SAM3 ROI candidates from raw boxes/scores."""
    candidates: List[Candidate] = []
    roi_seen = 0

    roi_pairs = list(zip(boxes, scores))

    if settings.get('focus_part_mode_enabled'):
        focus_parts = settings.get('focus_parts', ['leaf'])
        logger.debug(
            "Focus part mode enabled: focusing on %s; strong filtering applied during classification stage",
            focus_parts,
        )

    max_rois = settings.get('max_rois_for_classification')
    if max_rois is not None and len(roi_pairs) > max_rois:
        roi_pairs.sort(
            key=lambda pair: float(pair[1].item()) if torch.is_tensor(pair[1]) else float(pair[1]),
            reverse=True,
        )
        roi_pairs = roi_pairs[:max_rois]

    for box, score in roi_pairs:
        roi_seen += 1
        sam3_score = float(score)

        raw_bbox = box.tolist() if torch.is_tensor(box) else box
        bbox = sanitize_bbox_fn(raw_bbox, image_width, image_height)
        if bbox is None:
            continue

        if apply_roi_filters:
            if sam3_score < settings['sam3_threshold']:
                continue
            box_w = bbox[2] - bbox[0]
            box_h = bbox[3] - bbox[1]
            if box_w < settings['min_box_side_px'] or box_h < settings['min_box_side_px']:
                continue
            area_ratio = bbox_area_ratio_fn(bbox, image_width, image_height)
            if area_ratio < settings['min_box_area_ratio']:
                continue

        candidates.append({'bbox': bbox, 'sam3_score': sam3_score})

    return candidates, roi_seen


def classify_sam3_roi_candidate(
    pil_image: Image.Image,
    candidate: Candidate,
    image_width: int,
    image_height: int,
    settings: Dict[str, Any],
    part_labels: List[str],
    crop_labels: List[str],
    hooks: RoiClassificationHooks,
    global_crop_scores: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[Detection], int]:
    """Classify a single ROI candidate and return detection payload + call count."""
    part_num_prompts = settings.get('part_num_prompts')
    crop_num_prompts = settings.get('crop_num_prompts')
    extract_roi_fn = hooks.extract_roi
    clip_score_labels_ensemble_fn = hooks.clip_score_labels_ensemble

    bbox = candidate['bbox']
    roi_image = extract_roi_fn(pil_image, bbox, pad_ratio=0.08)

    part_label, part_conf, part_scores = clip_score_labels_ensemble_fn(
        roi_image, part_labels, label_type='part', num_prompts=part_num_prompts
    )
    classification_calls = 1

    crop_label, crop_conf, crop_scores = clip_score_labels_ensemble_fn(
        roi_image, crop_labels, label_type='crop', num_prompts=crop_num_prompts
    )
    classification_calls += 1

    detection = finalize_sam3_roi_candidate(
        roi_image=roi_image,
        candidate=candidate,
        image_width=image_width,
        image_height=image_height,
        settings=settings,
        hooks=hooks,
        part_label=part_label,
        part_conf=part_conf,
        part_scores=part_scores,
        crop_label=crop_label,
        crop_conf=crop_conf,
        crop_scores=crop_scores,
        global_crop_scores=global_crop_scores,
    )
    return detection, classification_calls


def finalize_sam3_roi_candidate(
    *,
    roi_image: Image.Image,
    candidate: Candidate,
    image_width: int,
    image_height: int,
    settings: Dict[str, Any],
    hooks: RoiClassificationHooks,
    part_label: str,
    part_conf: float,
    part_scores: Dict[str, float],
    crop_label: str,
    crop_conf: float,
    crop_scores: Dict[str, float],
    global_crop_scores: Optional[Dict[str, float]] = None,
) -> Detection:
    """Finalize ROI detection from precomputed crop/part scores."""
    part_num_prompts = settings.get('part_num_prompts')
    policy_enabled_fn = hooks.policy_enabled
    compute_leaf_likeness_fn = hooks.compute_leaf_likeness
    rebalance_part_scores_for_leaf_like_roi_fn = hooks.rebalance_part_scores_for_leaf_like_roi
    select_best_crop_with_fallback_fn = hooks.select_best_crop_with_fallback
    compatible_parts_for_crop_fn = hooks.compatible_parts_for_crop
    score_parts_conditioned_on_crop_fn = hooks.score_parts_conditioned_on_crop
    score_label_candidates_fn = hooks.score_label_candidates
    apply_generic_part_penalty_fn = hooks.apply_generic_part_penalty
    select_part_label_with_specificity_fn = hooks.select_part_label_with_specificity
    apply_leaf_like_override_fn = hooks.apply_leaf_like_override

    compatibility_fusion_enabled = policy_enabled_fn('compatibility_fusion', True)
    conditioned_part_weight = float(settings['conditioned_part_weight'])
    one_minus_conditioned_weight = 1.0 - conditioned_part_weight
    generic_part_labels = settings['generic_part_labels']
    generic_part_penalty = float(settings['generic_part_penalty'])

    specific_part_override_ratio = float(settings['specific_part_override_ratio'])
    specific_part_min_confidence = float(settings['specific_part_min_confidence'])
    preferred_part_labels = settings['preferred_part_labels']
    preferred_part_override_ratio = float(settings['preferred_part_override_ratio'])

    leaf_override_label = settings['leaf_override_label']
    leaf_override_enabled = bool(settings.get('leaf_override_enabled', True))
    leaf_visual_override_enabled = bool(settings.get('leaf_visual_override_enabled', True))

    weight_crop = float(settings['weight_crop'])
    weight_part = float(settings['weight_part'])
    weight_sam3 = float(settings['weight_sam3'])

    bbox = candidate['bbox']
    sam3_score = float(candidate['sam3_score'])

    leaf_likeness = compute_leaf_likeness_fn(
        roi_image=roi_image,
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
    )
    if settings.get('leaf_part_rebalance_enabled', True):
        part_scores = rebalance_part_scores_for_leaf_like_roi_fn(
            part_scores=part_scores,
            leaf_likeness=leaf_likeness,
            leaf_label=leaf_override_label,
            non_foliar_part_labels=settings['leaf_non_foliar_part_labels'],
            activation_threshold=settings['leaf_part_rebalance_threshold'],
            non_foliar_penalty=settings['leaf_part_rebalance_penalty'],
            leaf_boost=settings['leaf_part_rebalance_boost'],
            leaf_min_confidence=settings.get('leaf_part_rebalance_min_confidence', 0.18),
            leaf_support_ratio=settings.get('leaf_part_rebalance_support_ratio', 0.75),
        )
        if part_scores:
            part_label = max(part_scores, key=lambda label: float(part_scores.get(label, 0.0)))
            part_conf = float(part_scores.get(part_label, 0.0))

    crop_label, crop_conf = select_best_crop_with_fallback_fn(
        crop_scores,
        part_scores,
        global_crop_scores=global_crop_scores if settings.get('global_crop_context_enabled', True) else None,
        global_crop_context_weight=float(settings.get('global_crop_context_weight', 0.65)),
    )

    compatible_parts = compatible_parts_for_crop_fn(crop_label)
    part_unknown_label = str(settings.get('part_unknown_label', 'unknown') or 'unknown')
    part_open_set_enabled = bool(settings.get('part_open_set_enabled', True))
    resolved_part_scores: Dict[str, float] = {}
    compatible_part_scores: Dict[str, float] = {}
    conditioned_part_scores: Dict[str, float] = {}
    part_unknown_confidence = 0.0
    part_rejection_reasons: List[str] = []
    if not crop_label or str(crop_label).strip().lower() == part_unknown_label:
        part_rejection_reasons.append("crop unresolved for part surface")
    elif not compatible_parts:
        part_rejection_reasons.append(f"no compatible parts configured for crop ({crop_label})")
    else:
        compatible_part_scores = {
            part_name: float(part_scores.get(part_name, 0.0))
            for part_name in compatible_parts
        }
        conditioned_terms = [f"{crop_label} {part_name}".strip() for part_name in compatible_parts]
        term_to_part = {
            term: part_name
            for term, part_name in zip(conditioned_terms, compatible_parts)
        }
        conditioned_result = score_label_candidates_fn(
            roi_image,
            conditioned_terms,
            label_type='part',
            num_prompts=part_num_prompts,
            open_set_enabled=part_open_set_enabled,
            open_set_min_confidence=settings.get('part_open_set_min_confidence', 0.40),
            open_set_margin=settings.get('part_open_set_margin', 0.10),
            unknown_label=part_unknown_label,
        )
        part_unknown_confidence = float(conditioned_result.get('unknown_confidence', 0.0))
        for term, score in conditioned_result.get('label_scores', {}).items():
            part_name = term_to_part.get(str(term).strip())
            if part_name:
                conditioned_part_scores[part_name] = float(score)
        if not conditioned_part_scores:
            conditioned_part_scores = score_parts_conditioned_on_crop_fn(
                roi_image,
                crop_label,
                compatible_parts,
                num_prompts=part_num_prompts,
            )
        if conditioned_part_scores:
            if compatibility_fusion_enabled and compatible_part_scores:
                for part_name in compatible_parts:
                    generic_score = float(compatible_part_scores.get(part_name, 0.0))
                    conditioned_score = float(conditioned_part_scores.get(part_name, 0.0))
                    resolved_part_scores[part_name] = (
                        one_minus_conditioned_weight * generic_score
                        + conditioned_part_weight * conditioned_score
                    )
            else:
                resolved_part_scores = {
                    part_name: float(conditioned_part_scores.get(part_name, 0.0))
                    for part_name in compatible_parts
                }
        else:
            resolved_part_scores = compatible_part_scores

        resolved_part_scores = apply_generic_part_penalty_fn(
            resolved_part_scores,
            generic_part_labels,
            generic_part_penalty,
        )
        if resolved_part_scores:
            refined_part_label, refined_part_conf = select_part_label_with_specificity_fn(
                resolved_part_scores,
                generic_part_labels,
                specific_override_ratio=specific_part_override_ratio,
                specific_min_confidence=specific_part_min_confidence,
                preferred_part_labels=preferred_part_labels,
                preferred_override_ratio=preferred_part_override_ratio,
            )
            if refined_part_label:
                part_label = refined_part_label
                part_conf = refined_part_conf
        else:
            part_label = part_unknown_label
            part_conf = 0.0

    if leaf_override_enabled:
        part_label, part_conf = apply_leaf_like_override_fn(
            selected_label=part_label,
            selected_score=part_conf,
            part_scores=resolved_part_scores,
            bbox=bbox,
            image_width=image_width,
            image_height=image_height,
            leaf_label=leaf_override_label,
            override_target_labels=settings['leaf_override_target_labels'],
            leaf_score_ratio=settings['leaf_override_ratio'],
            leaf_min_confidence=settings['leaf_override_min_confidence'],
            leaf_min_margin=settings.get('leaf_override_min_margin', 0.04),
            leaf_min_area_ratio=settings['leaf_override_min_area_ratio'],
            leaf_aspect_min=settings['leaf_override_aspect_min'],
            leaf_aspect_max=settings['leaf_override_aspect_max'],
        )

    part_label_key = str(part_label).strip().lower()
    if leaf_visual_override_enabled and part_label_key in LEAF_VISUAL_GENERIC_LABELS:
        leaf_key = str(leaf_override_label).strip().lower()

        leaf_score = max(
            float(resolved_part_scores.get(leaf_override_label, 0.0)),
            float(resolved_part_scores.get(leaf_key, 0.0)),
        )
        if leaf_score <= 0.0:
            for score_label, score_value in resolved_part_scores.items():
                if str(score_label).strip().lower() == leaf_key:
                    leaf_score = max(leaf_score, float(score_value))

        threshold = max(0.0, min(1.0, settings['leaf_visual_likeness_threshold']))
        min_leaf_score = max(0.0, min(1.0, settings['leaf_visual_green_min']))
        min_leaf_margin = max(0.0, float(settings.get('leaf_visual_min_margin', 0.05)))
        strongest_non_leaf = max(
            (
                float(score_value)
                for score_label, score_value in resolved_part_scores.items()
                if str(score_label).strip().lower() != leaf_key
            ),
            default=0.0,
        )
        leaf_margin_ok = leaf_score >= strongest_non_leaf + min_leaf_margin
        if leaf_likeness >= threshold:
            leaf_score_ok = leaf_score >= min_leaf_score
            if (leaf_score_ok and leaf_margin_ok) or (
                settings.get('leaf_visual_force_without_leaf_score', False) and leaf_margin_ok
            ):
                if settings.get('leaf_visual_force_generic', True):
                    floor_conf = max(0.0, min(1.0, settings['leaf_visual_force_conf_floor']))
                    part_factor = max(0.0, min(1.0, settings['leaf_visual_force_part_factor']))
                    forced_conf = max(leaf_score, float(part_conf) * part_factor, floor_conf)
                    part_label = leaf_override_label
                    part_conf = forced_conf
                elif leaf_score_ok and leaf_margin_ok:
                    part_label = leaf_override_label
                    part_conf = max(part_conf, leaf_score)

    if part_label:
        part_label = normalize_part_label(part_label)
    compatible_part_surface = {normalize_part_label(part_name) for part_name in compatible_parts}
    if compatible_part_surface and part_label not in compatible_part_surface:
        part_rejection_reasons.append(f"part '{part_label}' not allowed for crop ({crop_label})")

    raw_part_label, raw_part_conf, raw_part_second_conf, raw_part_margin = _resolve_top_part_metrics(
        resolved_part_scores,
        selected_label=part_label,
        selected_score=part_conf,
    )
    if compatible_part_surface and str(raw_part_label).strip().lower() not in compatible_part_surface:
        part_rejection_reasons.append(f"raw part '{raw_part_label}' not compatible with crop ({crop_label})")
    if not resolved_part_scores and not part_rejection_reasons:
        part_rejection_reasons.append("no compatible part evidence survived conditioning")
    open_set_rejection_reasons: List[str] = []
    part_recovery_reason = ""
    if part_open_set_enabled and not part_rejection_reasons:
        open_set_rejection_reasons = build_open_set_rejection_reasons(
            label=raw_part_label,
            confidence=raw_part_conf,
            second_confidence=raw_part_second_conf,
            unknown_confidence=part_unknown_confidence,
            min_confidence=settings.get('part_open_set_min_confidence', 0.40),
            margin_threshold=settings.get('part_open_set_margin', 0.10),
            unknown_label=part_unknown_label,
        )
        if open_set_rejection_reasons and not _can_restore_part_after_unknown_open_set_rejection(
            open_set_rejection_reasons=open_set_rejection_reasons,
            raw_part_label=raw_part_label,
            raw_part_confidence=raw_part_conf,
            raw_part_margin=raw_part_margin,
            generic_part_scores=compatible_part_scores,
            conditioned_part_scores=conditioned_part_scores,
            generic_part_labels=generic_part_labels,
            generic_part_penalty=generic_part_penalty,
            specific_part_override_ratio=specific_part_override_ratio,
            specific_part_min_confidence=specific_part_min_confidence,
            preferred_part_labels=preferred_part_labels,
            preferred_part_override_ratio=preferred_part_override_ratio,
            min_confidence=settings.get('part_open_set_min_confidence', 0.40),
            margin_threshold=settings.get('part_open_set_margin', 0.10),
            unknown_label=part_unknown_label,
            apply_generic_part_penalty_fn=apply_generic_part_penalty_fn,
            select_part_label_with_specificity_fn=select_part_label_with_specificity_fn,
        ):
            part_rejection_reasons.extend(open_set_rejection_reasons)
        elif open_set_rejection_reasons:
            # Keep the supported organ when both compatible-part scorers agree and
            # only the unknown proxy would have forced abstention.
            part_recovery_reason = (
                "retained compatible part because generic and crop-conditioned part "
                "surfaces agreed despite the unknown proxy"
            )

    exposed_part_label = raw_part_label
    exposed_part_conf = raw_part_conf
    if part_rejection_reasons:
        exposed_part_label = part_unknown_label
        exposed_part_conf = 0.0

    quality_score = (
        weight_crop * float(crop_conf)
        + weight_part * float(raw_part_conf)
        + weight_sam3 * sam3_score
    )

    detection: Detection = {
        'crop': crop_label,
        'part': exposed_part_label,
        'crop_confidence': crop_conf,
        'part_confidence': exposed_part_conf,
        'disease': None,
        'disease_confidence': 0.0,
        'bbox': bbox,
        'mask': None,
        'sam3_score': sam3_score,
        '_quality_score': quality_score,
    }
    if settings.get('part_rejection_metadata_enabled', True):
        detection['raw_part_label'] = raw_part_label
        detection['raw_part_confidence'] = raw_part_conf
        detection['raw_part_second_confidence'] = raw_part_second_conf
        detection['part_unknown_confidence'] = part_unknown_confidence
        detection['raw_part_margin'] = raw_part_margin
        if part_rejection_reasons:
            detection['part_rejection_reason'] = "; ".join(part_rejection_reasons)
        elif part_recovery_reason:
            detection['part_recovery_reason'] = part_recovery_reason
            detection['part_open_set_rejection_reason'] = "; ".join(open_set_rejection_reasons)
    return detection


def filter_classified_sam3_detections(
    *,
    all_detections: List[Detection],
    settings: Dict[str, Any],
    stage_order: List[str],
    policy_enabled_fn: Callable[[str, bool], bool],
    passes_open_set_gate_fn: Callable[..., bool],
) -> Tuple[List[Detection], int]:
    """Apply focus fallback and open-set gate to already-classified detections."""
    detections: List[Detection] = []
    roi_kept = 0

    run_open_set_gate = policy_enabled_fn('open_set_gate', True) and 'open_set_gate' in stage_order
    (
        focus_mode_enabled,
        focus_fallback_enabled,
        focus_min_confidence,
        focus_parts,
        focus_parts_lower,
    ) = _resolve_focus_settings(settings)
    classification_min_confidence = float(settings['classification_min_confidence'])

    focused_detections = (
        _collect_focused_detections(
            all_detections,
            focus_parts_lower=focus_parts_lower,
            focus_min_confidence=focus_min_confidence,
        )
        if focus_mode_enabled
        else []
    )
    result_detections = _resolve_focus_result_detections(
        all_detections=all_detections,
        focused_detections=focused_detections,
        focus_mode_enabled=focus_mode_enabled,
        focus_fallback_enabled=focus_fallback_enabled,
        focus_parts=focus_parts,
        focus_min_confidence=focus_min_confidence,
    )

    for detection in result_detections:
        if not _passes_detection_gate(
            detection,
            run_open_set_gate=run_open_set_gate,
            classification_min_confidence=classification_min_confidence,
            passes_open_set_gate_fn=passes_open_set_gate_fn,
        ):
            continue
        detections.append(detection)
        roi_kept += 1

    return detections, roi_kept


def run_sam3_roi_classification_stage(
    candidates: List[Candidate],
    settings: Dict[str, Any],
    stage_order: List[str],
    policy_enabled_fn: Callable[[str, bool], bool],
    classify_candidate_fn: Callable[[Candidate], Tuple[Optional[Detection], int]],
    passes_open_set_gate_fn: Callable[..., bool],
) -> Tuple[List[Detection], int, int, float]:
    """Execute ROI classification stage with focus fallback and optional open-set gate."""
    run_classification = policy_enabled_fn('crop_evidence', True) and 'roi_classification' in stage_order
    if not run_classification:
        return [], 0, 0, 0.0

    all_detections, roi_classification_calls, roi_classification_ms = _classify_candidates(
        candidates,
        classify_candidate_fn=classify_candidate_fn,
    )

    detections, roi_kept = filter_classified_sam3_detections(
        all_detections=all_detections,
        settings=settings,
        stage_order=stage_order,
        policy_enabled_fn=policy_enabled_fn,
        passes_open_set_gate_fn=passes_open_set_gate_fn,
    )

    return detections, roi_kept, roi_classification_calls, roi_classification_ms
