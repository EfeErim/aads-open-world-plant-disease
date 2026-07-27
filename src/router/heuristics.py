"""Pure heuristic helpers for SAM3 ROI post-classification decisions."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.router.compatibility_utils import compatible_parts_for_crop


def _normalized_label_set(labels: List[str] | None) -> set[str]:
    return {
        str(label).strip().lower()
        for label in (labels or [])
        if str(label).strip()
    }


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bbox_geometry(
    bbox: Optional[List[float]],
    *,
    image_width: int,
    image_height: int,
) -> Optional[tuple[float, float, float, float]]:
    if bbox is None or len(bbox) != 4 or image_width <= 0 or image_height <= 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if box_w <= 0.0 or box_h <= 0.0:
        return None
    area_ratio = (box_w * box_h) / float(image_width * image_height)
    aspect = box_w / max(1e-6, box_h)
    return box_w, box_h, area_ratio, aspect


def apply_generic_part_penalty(
    part_scores: Dict[str, float],
    generic_part_labels: List[str],
    generic_penalty: float,
) -> Dict[str, float]:
    """Down-weight overly generic part labels to prefer specific parts when close."""
    if not part_scores:
        return {}

    generic_set = _normalized_label_set(generic_part_labels)
    if not generic_set:
        return dict(part_scores)

    penalty = _clamp_unit_interval(generic_penalty)
    adjusted: Dict[str, float] = {}
    for part_name, score in part_scores.items():
        normalized = str(part_name).strip().lower()
        base_score = float(score)
        if normalized in generic_set:
            adjusted[part_name] = base_score * penalty
        else:
            adjusted[part_name] = base_score
    return adjusted


def select_part_label_with_specificity(
    part_scores: Dict[str, float],
    generic_part_labels: List[str],
    specific_override_ratio: float,
    specific_min_confidence: float,
    preferred_part_labels: Optional[List[str]] = None,
    preferred_override_ratio: float = 0.50,
) -> Tuple[str, float]:
    """Select part label while preferring specific parts over generic labels when close."""
    if not part_scores:
        return "unknown", 0.0

    best_label = max(part_scores, key=lambda label: part_scores[label])
    best_score = float(part_scores.get(best_label, 0.0))

    generic_set = _normalized_label_set(generic_part_labels)

    specific_override_ratio = _clamp_unit_interval(specific_override_ratio)
    specific_min_confidence = _clamp_unit_interval(specific_min_confidence)
    preferred_override_ratio = _clamp_unit_interval(preferred_override_ratio)
    preferred_set = _normalized_label_set(preferred_part_labels)

    if preferred_set:
        preferred_candidates = [
            (label, float(score))
            for label, score in part_scores.items()
            if str(label).strip().lower() in preferred_set
        ]
        if preferred_candidates:
            preferred_label, preferred_score = max(preferred_candidates, key=lambda item: item[1])
            if (
                preferred_score >= specific_min_confidence
                and preferred_score >= best_score * preferred_override_ratio
            ):
                return preferred_label, preferred_score

    if str(best_label).strip().lower() not in generic_set:
        return best_label, best_score

    specific_candidates = [
        (label, float(score))
        for label, score in part_scores.items()
        if str(label).strip().lower() not in generic_set
    ]
    if not specific_candidates:
        return best_label, best_score

    specific_label, specific_score = max(specific_candidates, key=lambda item: item[1])
    if specific_score >= specific_min_confidence and specific_score >= best_score * specific_override_ratio:
        return specific_label, specific_score

    return best_label, best_score


def apply_leaf_like_override(
    selected_label: str,
    selected_score: float,
    part_scores: Dict[str, float],
    bbox: Optional[List[float]],
    image_width: int,
    image_height: int,
    leaf_label: str = "leaf",
    override_target_labels: Optional[List[str]] = None,
    leaf_score_ratio: float = 0.35,
    leaf_min_confidence: float = 0.10,
    leaf_min_margin: float = 0.0,
    leaf_min_area_ratio: float = 0.02,
    leaf_aspect_min: float = 0.30,
    leaf_aspect_max: float = 3.20,
) -> Tuple[str, float]:
    """Prefer leaf part only when leaf evidence dominates broad/generic alternatives."""
    if not part_scores:
        return selected_label, float(selected_score)

    leaf_key = str(leaf_label).strip().lower()
    if not leaf_key:
        return selected_label, float(selected_score)

    leaf_candidates = {
        label: float(score)
        for label, score in part_scores.items()
        if str(label).strip().lower() == leaf_key
    }
    if not leaf_candidates:
        return selected_label, float(selected_score)

    leaf_name, leaf_score = max(leaf_candidates.items(), key=lambda item: item[1])

    target_labels = override_target_labels or ["whole plant", "whole", "plant", "entire plant"]
    target_set = _normalized_label_set(target_labels)
    current_key = str(selected_label).strip().lower()
    if current_key not in target_set:
        return selected_label, float(selected_score)

    geometry = _bbox_geometry(bbox, image_width=image_width, image_height=image_height)
    if geometry is None:
        return selected_label, float(selected_score)
    _box_w, _box_h, area_ratio, aspect = geometry
    if area_ratio < max(0.0, float(leaf_min_area_ratio)):
        return selected_label, float(selected_score)

    aspect_min = max(0.05, float(leaf_aspect_min))
    aspect_max = max(aspect_min, float(leaf_aspect_max))
    if aspect < aspect_min or aspect > aspect_max:
        return selected_label, float(selected_score)

    ratio = _clamp_unit_interval(leaf_score_ratio)
    min_conf = _clamp_unit_interval(leaf_min_confidence)
    min_margin = max(0.0, float(leaf_min_margin))
    strongest_alternative = max(
        (
            float(score)
            for label, score in part_scores.items()
            if str(label).strip().lower() != leaf_key
        ),
        default=0.0,
    )
    if (
        leaf_score >= min_conf
        and leaf_score >= float(selected_score) * ratio
        and leaf_score >= strongest_alternative + min_margin
    ):
        return leaf_name, leaf_score

    return selected_label, float(selected_score)


def compute_leaf_likeness(
    roi_image: Image.Image,
    bbox: Optional[List[float]],
    image_width: int,
    image_height: int,
) -> float:
    """Estimate how leaf-like an ROI is using color + geometry cues in [0,1]."""
    if roi_image is None or bbox is None or len(bbox) != 4 or image_width <= 0 or image_height <= 0:
        return 0.0

    try:
        roi_np = np.asarray(roi_image.convert("RGB"), dtype=np.float32)
    except (TypeError, ValueError, OSError):
        return 0.0

    if roi_np.size == 0:
        return 0.0

    h, w = roi_np.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0

    y1 = int(max(0, round(h * 0.15)))
    y2 = int(min(h, round(h * 0.85)))
    x1 = int(max(0, round(w * 0.15)))
    x2 = int(min(w, round(w * 0.85)))
    center = roi_np[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else roi_np

    r = center[..., 0]
    g = center[..., 1]
    b = center[..., 2]
    green_mask = (g > (r * 0.95)) & (g > (b * 1.05)) & (g > 45.0)
    green_ratio = float(np.mean(green_mask)) if green_mask.size > 0 else 0.0

    bx1, by1, bx2, by2 = [float(v) for v in bbox]
    box_w = max(0.0, bx2 - bx1)
    box_h = max(0.0, by2 - by1)
    area_ratio = (box_w * box_h) / float(image_width * image_height)
    aspect = box_w / max(1e-6, box_h)

    size_score = max(0.0, min(1.0, area_ratio / 0.15))
    if 0.25 <= aspect <= 4.2:
        shape_score = 1.0
    elif 0.15 <= aspect <= 5.5:
        shape_score = 0.6
    else:
        shape_score = 0.2

    return float(max(0.0, min(1.0, (0.60 * green_ratio) + (0.25 * size_score) + (0.15 * shape_score))))


def rebalance_part_scores_for_leaf_like_roi(
    part_scores: Dict[str, float],
    leaf_likeness: float,
    leaf_label: str = "leaf",
    non_foliar_part_labels: Optional[List[str]] = None,
    activation_threshold: float = 0.34,
    non_foliar_penalty: float = 0.55,
    leaf_boost: float = 1.35,
    leaf_min_confidence: float = 0.10,
    leaf_support_ratio: float = 0.0,
) -> Dict[str, float]:
    """Rebalance part scores only when visual and classifier evidence agree on leaf."""
    if not part_scores:
        return {}

    threshold = _clamp_unit_interval(activation_threshold)
    if float(leaf_likeness) < threshold:
        return dict(part_scores)

    leaf_key = str(leaf_label).strip().lower()
    if not leaf_key:
        return dict(part_scores)

    leaf_name = None
    leaf_score = 0.0
    strongest_other = 0.0
    for part_name, score in part_scores.items():
        key = str(part_name).strip().lower()
        value = float(score)
        if key == leaf_key and value >= leaf_score:
            leaf_name = part_name
            leaf_score = value
        elif value > strongest_other:
            strongest_other = value

    if leaf_name is None:
        return dict(part_scores)

    min_leaf_conf = _clamp_unit_interval(leaf_min_confidence)
    support_ratio = _clamp_unit_interval(leaf_support_ratio)
    if leaf_score < max(min_leaf_conf, strongest_other * support_ratio):
        return dict(part_scores)

    default_non_foliar = [
        "husk",
        "shell",
        "pod",
        "seed",
        "grain",
        "ear",
        "tuber",
        "bulb",
        "fruit",
        "berry",
        "bark",
        "peel",
        "whole plant",
        "whole",
        "plant",
        "entire plant",
    ]
    non_foliar_set = _normalized_label_set(
        non_foliar_part_labels if isinstance(non_foliar_part_labels, list) else default_non_foliar
    )

    evidence = (float(leaf_likeness) - threshold) / max(1e-6, 1.0 - threshold)
    evidence = _clamp_unit_interval(evidence)
    penalty_floor = _clamp_unit_interval(non_foliar_penalty)
    penalty = 1.0 - ((1.0 - penalty_floor) * evidence)
    boost_ceiling = max(1.0, float(leaf_boost))
    boost = 1.0 + ((boost_ceiling - 1.0) * evidence)

    adjusted: Dict[str, float] = {}
    for part_name, score in part_scores.items():
        key = str(part_name).strip().lower()
        value = float(score)
        if key == leaf_key:
            adjusted[part_name] = value * boost
        elif key in non_foliar_set:
            adjusted[part_name] = value * penalty
        else:
            adjusted[part_name] = value
    return adjusted


def select_best_crop_with_fallback(
    crop_scores: Dict[str, float],
    part_scores: Dict[str, float],
    crop_part_compatibility: Dict[str, List[str]],
    part_labels: List[str],
    *,
    global_crop_scores: Optional[Dict[str, float]] = None,
    global_crop_context_weight: float = 0.0,
) -> Tuple[str, float]:
    """Rerank crop scores by part compatibility and whole-image crop context."""
    if not crop_scores:
        return "unknown", 0.0

    reranked_scores: Dict[str, float] = {crop_name: float(score) for crop_name, score in crop_scores.items()}
    best_crop_raw, best_score_raw = max(reranked_scores.items(), key=lambda item: item[1])

    if part_scores:
        sorted_crop_scores = sorted(float(v) for v in reranked_scores.values())
        second_best = sorted_crop_scores[-2] if len(sorted_crop_scores) > 1 else 0.0
        uncertainty = max(0.0, 1.0 - max(0.0, min(1.0, best_score_raw - second_best)))

        reranked_with_parts: Dict[str, float] = {}
        for crop_name, base_score in reranked_scores.items():
            compatible_parts = compatible_parts_for_crop(crop_name, crop_part_compatibility, part_labels)
            compatible_part_score = 0.0
            if compatible_parts:
                compatible_part_score = max(float(part_scores.get(part, 0.0)) for part in compatible_parts)
            reranked_with_parts[crop_name] = (
                float(base_score) + (1.0 - float(base_score)) * compatible_part_score * uncertainty
            )
        reranked_scores = reranked_with_parts

    context_weight = _clamp_unit_interval(global_crop_context_weight)
    if global_crop_scores and context_weight > 0.0:
        normalized_context_scores = {
            str(crop_name).strip().lower(): float(score)
            for crop_name, score in global_crop_scores.items()
        }
        reranked_scores = {
            crop_name: ((1.0 - context_weight) * float(score))
            + (context_weight * float(normalized_context_scores.get(str(crop_name).strip().lower(), 0.0)))
            for crop_name, score in reranked_scores.items()
        }

    best_crop, best_score = max(reranked_scores.items(), key=lambda item: item[1])
    return best_crop, best_score
