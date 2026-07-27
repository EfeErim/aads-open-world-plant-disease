from typing import Any, Callable, Dict, List

from src.router.runtime_surface import coerce_bool, coerce_float


def resolve_sam3_stage_order(policy_value_fn: Callable[[str, str, Any], Any]) -> List[str]:
    """Resolve SAM3 stage execution order from policy graph."""
    default_order = ["roi_filter", "roi_classification", "open_set_gate", "postprocess"]
    configured = policy_value_fn("execution", "sam3_stage_order", default_order)
    if not isinstance(configured, list):
        return default_order

    allowed = {"roi_filter", "roi_classification", "open_set_gate", "postprocess"}
    ordered: List[str] = []
    for stage_name in configured:
        normalized = str(stage_name).strip()
        if normalized in allowed and normalized not in ordered:
            ordered.append(normalized)
    return ordered or default_order


def build_sam3_runtime_settings(
    policy_value_fn: Callable[[str, str, Any], Any],
    vlm_config: Dict[str, Any],
    effective_threshold: float,
) -> Dict[str, Any]:
    """Collect policy-controlled runtime settings for SAM3 analysis."""
    settings: Dict[str, Any] = {}
    settings["sam3_threshold"] = coerce_float(policy_value_fn("roi_filter", "sam3_mask_threshold", 0.60), 0.60)
    settings["min_box_area_ratio"] = coerce_float(policy_value_fn("roi_filter", "min_box_area_ratio", 0.001), 0.001)
    settings["min_box_side_px"] = coerce_float(policy_value_fn("roi_filter", "min_box_side_px", 10), 10.0)
    settings["classification_min_confidence"] = max(
        coerce_float(policy_value_fn("crop_evidence", "classification_min_confidence", 0.20), 0.20),
        coerce_float(effective_threshold, 0.20),
    )
    settings["detection_nms_iou_threshold"] = coerce_float(
        policy_value_fn("dedupe", "detection_nms_iou_threshold", 0.75),
        0.75,
    )
    settings["detection_nms_same_crop_only"] = coerce_bool(
        policy_value_fn("dedupe", "detection_nms_same_crop_only", True),
        default=True,
    )
    settings["conditioned_part_weight"] = max(
        0.0,
        min(1.0, float(policy_value_fn("compatibility_fusion", "conditioned_part_weight", 0.45))),
    )
    settings["generic_part_penalty"] = float(policy_value_fn("part_resolution", "generic_part_penalty", 0.78))

    generic_part_labels_raw = policy_value_fn(
        "part_resolution",
        "generic_part_labels",
        ["whole plant", "whole", "plant", "entire plant"],
    )
    settings["generic_part_labels"] = (
        [str(label) for label in generic_part_labels_raw]
        if isinstance(generic_part_labels_raw, list)
        else ["whole plant", "whole", "plant", "entire plant"]
    )

    settings["specific_part_override_ratio"] = coerce_float(
        policy_value_fn("part_resolution", "specific_part_override_ratio", 0.45),
        0.45,
    )
    settings["specific_part_min_confidence"] = coerce_float(
        policy_value_fn("part_resolution", "specific_part_min_confidence", 0.12),
        0.12,
    )

    preferred_part_labels_raw = policy_value_fn("part_resolution", "preferred_part_labels", [])
    settings["preferred_part_labels"] = (
        [str(label) for label in preferred_part_labels_raw] if isinstance(preferred_part_labels_raw, list) else []
    )
    settings["preferred_part_override_ratio"] = float(
        policy_value_fn("part_resolution", "preferred_part_override_ratio", 0.50)
    )
    settings["part_open_set_enabled"] = coerce_bool(
        policy_value_fn("part_resolution", "part_open_set_enabled", True),
        default=True,
    )
    settings["part_open_set_min_confidence"] = coerce_float(
        policy_value_fn("part_resolution", "part_open_set_min_confidence", 0.40),
        0.40,
    )
    settings["part_open_set_margin"] = coerce_float(
        policy_value_fn("part_resolution", "part_open_set_margin", 0.10),
        0.10,
    )
    settings["part_unknown_label"] = str(policy_value_fn("part_resolution", "part_unknown_label", "unknown"))
    settings["part_rejection_metadata_enabled"] = coerce_bool(
        policy_value_fn("part_resolution", "part_rejection_metadata_enabled", True),
        default=True,
    )

    settings["leaf_override_enabled"] = coerce_bool(
        policy_value_fn("part_resolution", "leaf_override_enabled", True),
        default=True,
    )
    settings["leaf_override_label"] = str(policy_value_fn("part_resolution", "leaf_override_label", "leaf"))

    leaf_override_target_raw = policy_value_fn(
        "part_resolution",
        "leaf_override_target_labels",
        ["whole plant", "whole", "plant", "entire plant"],
    )
    settings["leaf_override_target_labels"] = (
        [str(label) for label in leaf_override_target_raw]
        if isinstance(leaf_override_target_raw, list)
        else ["whole plant", "whole", "plant", "entire plant"]
    )

    settings["leaf_override_ratio"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_override_ratio", 0.90),
        0.90,
    )
    settings["leaf_override_min_confidence"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_override_min_confidence", 0.16),
        0.16,
    )
    settings["leaf_override_min_margin"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_override_min_margin", 0.04),
        0.04,
    )
    settings["leaf_override_min_area_ratio"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_override_min_area_ratio", 0.02),
        0.02,
    )
    settings["leaf_override_aspect_min"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_override_aspect_min", 0.30),
        0.30,
    )
    settings["leaf_override_aspect_max"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_override_aspect_max", 3.20),
        3.20,
    )

    settings["leaf_visual_override_enabled"] = coerce_bool(
        policy_value_fn("part_resolution", "leaf_visual_override_enabled", True),
        default=True,
    )
    settings["leaf_visual_likeness_threshold"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_visual_likeness_threshold", 0.58),
        0.58,
    )
    settings["leaf_visual_green_min"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_visual_green_min", 0.18),
        0.18,
    )
    settings["leaf_visual_min_margin"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_visual_min_margin", 0.05),
        0.05,
    )
    settings["leaf_visual_force_generic"] = coerce_bool(
        policy_value_fn("part_resolution", "leaf_visual_force_generic", True),
        default=True,
    )
    settings["leaf_visual_force_without_leaf_score"] = coerce_bool(
        policy_value_fn("part_resolution", "leaf_visual_force_without_leaf_score", False),
        default=False,
    )
    settings["leaf_visual_force_conf_floor"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_visual_force_conf_floor", 0.16),
        0.16,
    )
    settings["leaf_visual_force_part_factor"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_visual_force_part_factor", 0.65),
        0.65,
    )

    leaf_non_foliar_labels_raw = policy_value_fn(
        "part_resolution",
        "leaf_non_foliar_part_labels",
        ["husk", "shell", "pod", "seed", "grain", "ear", "tuber", "bulb", "fruit", "berry", "bark", "peel"],
    )
    settings["leaf_non_foliar_part_labels"] = (
        [str(label) for label in leaf_non_foliar_labels_raw]
        if isinstance(leaf_non_foliar_labels_raw, list)
        else ["husk", "shell", "pod", "seed", "grain", "ear", "tuber", "bulb", "fruit", "berry", "bark", "peel"]
    )

    settings["leaf_part_rebalance_enabled"] = coerce_bool(
        policy_value_fn("part_resolution", "leaf_part_rebalance_enabled", True),
        default=True,
    )
    settings["leaf_part_rebalance_threshold"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_part_rebalance_threshold", 0.52),
        0.52,
    )
    settings["leaf_part_rebalance_penalty"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_part_rebalance_penalty", 0.80),
        0.80,
    )
    settings["leaf_part_rebalance_boost"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_part_rebalance_boost", 1.15),
        1.15,
    )
    settings["leaf_part_rebalance_min_confidence"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_part_rebalance_min_confidence", 0.18),
        0.18,
    )
    settings["leaf_part_rebalance_support_ratio"] = coerce_float(
        policy_value_fn("part_resolution", "leaf_part_rebalance_support_ratio", 0.75),
        0.75,
    )

    max_rois_raw = policy_value_fn("roi_filter", "max_rois_for_classification", 0)
    try:
        max_rois = int(max_rois_raw)
    except (TypeError, ValueError, OverflowError):
        max_rois = 0
    settings["max_rois_for_classification"] = None if max_rois <= 0 else max_rois

    ensemble_config = vlm_config.get("ensemble_config", {})
    crop_num_prompts_raw = policy_value_fn(
        "crop_evidence", "crop_num_prompts", ensemble_config.get("crop_num_prompts", None)
    )
    part_num_prompts_raw = policy_value_fn(
        "part_evidence", "part_num_prompts", ensemble_config.get("part_num_prompts", None)
    )

    try:
        settings["crop_num_prompts"] = int(crop_num_prompts_raw) if crop_num_prompts_raw is not None else None
    except (TypeError, ValueError, OverflowError):
        settings["crop_num_prompts"] = None
    try:
        settings["part_num_prompts"] = int(part_num_prompts_raw) if part_num_prompts_raw is not None else None
    except (TypeError, ValueError, OverflowError):
        settings["part_num_prompts"] = None

    quality_weights = vlm_config.get("quality_score_weights", {})
    settings["weight_crop"] = coerce_float(quality_weights.get("crop_confidence", 0.65), 0.65)
    settings["weight_part"] = coerce_float(quality_weights.get("part_confidence", 0.20), 0.20)
    settings["weight_sam3"] = coerce_float(quality_weights.get("sam3_score", 0.15), 0.15)
    settings["global_crop_context_enabled"] = coerce_bool(
        policy_value_fn("crop_evidence", "global_crop_context_enabled", True),
        default=True,
    )
    settings["global_crop_context_weight"] = coerce_float(
        policy_value_fn("crop_evidence", "global_crop_context_weight", 0.65),
        0.65,
    )

    settings["focus_part_mode_enabled"] = coerce_bool(
        policy_value_fn("focus_mode", "focus_part_mode_enabled", False),
        default=False,
    )
    focus_parts_raw = policy_value_fn("focus_mode", "focus_parts", ["leaf"])
    settings["focus_parts"] = (
        [str(label) for label in focus_parts_raw] if isinstance(focus_parts_raw, list) else ["leaf"]
    )
    settings["focus_min_confidence_fallback"] = coerce_float(
        policy_value_fn("focus_mode", "focus_min_confidence_fallback", 0.50),
        0.50,
    )
    settings["focus_fallback_enabled"] = coerce_bool(
        policy_value_fn("focus_mode", "focus_fallback_enabled", True),
        default=True,
    )
    return settings
