"""Plantness input guard for router-driven inference."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from PIL import Image

from src.router import clip_runtime
from src.router.prompt_clip_utils import (
    TURKEY_PRIORITY_CROP_PARTS,
    TURKEY_PRIORITY_TOP_10_CROPS,
    canonicalize_crop_prompt_label,
    crop_prompt_aliases,
)
from src.router.runtime_surface import coerce_bool, coerce_float
from src.shared.contracts import InputGuardAnalysis

METHOD_NAME = "bioclip_prompt_plantness"

SUPPORTED_CROP_PROMPT_CROPS = (
    *TURKEY_PRIORITY_TOP_10_CROPS,
    "hazelnut",
    "apricot",
    "strawberry",
)

POSITIVE_PROMPT_GROUPS: Dict[str, List[str]] = {
    "generic_plant": [
        "a plant",
        "a green plant",
        "a living plant",
        "a crop plant",
        "an agricultural plant",
        "a cultivated plant",
        "a plant growing in soil",
        "a close-up photo of a plant",
        "a photo of vegetation",
        "plant material",
        "botanical subject",
    ],
    "plant_part": [
        "a leaf",
        "plant leaves",
        "a crop leaf",
        "a diseased leaf",
        "a healthy leaf",
        "a fruit on a plant",
        "a crop fruit",
        "a plant stem",
        "a crop stem",
        "a plant shoot",
        "a plant branch",
        "a flower on a plant",
        "a root or tuber from a plant",
    ],
    "crop_context": [
        "a crop in a field",
        "crop rows with visible plants",
        "an agricultural crop",
        "a farm crop plant",
        "vegetable crop plants",
        "fruit crop plants",
        "green crop canopy",
        "plant seedlings",
        "plants in a greenhouse",
        "plants in an orchard",
    ],
    "disease_inspection": [
        "a close-up of a crop leaf",
        "a close-up of plant disease symptoms",
        "a plant disease inspection image",
        "a diseased crop plant",
        "a healthy crop plant for disease inspection",
        "a leaf with spots",
        "a leaf with lesions",
        "a fruit with plant disease symptoms",
        "a plant part held for inspection",
    ],
}

NEGATIVE_PROMPT_GROUPS: Dict[str, List[str]] = {
    "animals_people": [
        "an animal",
        "a dog",
        "a cat",
        "a bird",
        "livestock",
        "a person",
        "a human hand without a plant",
        "a face",
        "an insect without a plant",
    ],
    "vehicles_machinery": [
        "a vehicle",
        "a tractor",
        "a truck",
        "a car",
        "a motorcycle",
        "farm machinery",
        "agricultural equipment",
        "a harvesting machine",
        "a plow",
        "a trailer",
    ],
    "buildings_scenes": [
        "a building",
        "a house",
        "a room",
        "an indoor scene",
        "a wall",
        "a road",
        "a street",
        "a fence",
        "a greenhouse structure without visible plants",
        "farm infrastructure without visible plants",
    ],
    "bare_field_background": [
        "bare soil",
        "a field without visible plants",
        "a distant field landscape",
        "dry ground",
        "mud",
        "rocks",
        "mulch",
        "background scenery",
        "sky and landscape",
        "an empty farm field",
    ],
    "tools_objects": [
        "a tool",
        "a gardening tool",
        "a machine part",
        "a plastic container",
        "a bottle",
        "a phone",
        "a computer screen",
        "a document",
        "a label",
        "a table",
        "a plate",
        "a bag",
    ],
    "food_harvested": [
        "cooked food",
        "prepared meal",
        "processed food",
        "a grocery item",
        "a fruit on a plate",
        "a vegetable on a table",
        "packaged produce",
    ],
    "abstract_bad_input": [
        "a drawing",
        "a diagram",
        "a screenshot",
        "a chart",
        "text on a page",
        "a logo",
        "a blurry non-plant image",
        "a blank image",
        "a corrupted image",
    ],
}


def input_guard_enabled(config: Dict[str, Any]) -> bool:
    return coerce_bool(_input_guard_config(config).get("enabled", False), default=False)


def evaluate_plantness_input_guard(
    runtime: Any,
    image: Image.Image,
    config: Dict[str, Any],
    *,
    requested_part: Optional[str] = None,
) -> InputGuardAnalysis:
    guard_cfg = _input_guard_config(config)
    if not coerce_bool(guard_cfg.get("enabled", False), default=False):
        return InputGuardAnalysis()

    positive_groups = {key: list(value) for key, value in POSITIVE_PROMPT_GROUPS.items()}
    supported_crop_prompts = _build_supported_crop_prompts(runtime, config)
    if supported_crop_prompts:
        positive_groups["supported_crop"] = supported_crop_prompts

    negative_groups = {key: list(value) for key, value in NEGATIVE_PROMPT_GROUPS.items()}
    num_prompts = _coerce_optional_positive_int(guard_cfg.get("num_prompts"))
    debug_enabled = coerce_bool(guard_cfg.get("debug_scores", False), default=False)

    group_scores: Dict[str, float] = {}
    for group_name, labels in positive_groups.items():
        group_scores[group_name] = _score_prompt_group(
            runtime,
            image,
            labels,
            num_prompts=num_prompts,
        )

    food_weight = (
        coerce_float(guard_cfg.get("fruit_food_weight", 0.4), 0.4)
        if str(requested_part or "").strip().lower() == "fruit"
        else coerce_float(guard_cfg.get("default_food_weight", 0.8), 0.8)
    )
    for group_name, labels in negative_groups.items():
        score = _score_prompt_group(runtime, image, labels, num_prompts=num_prompts)
        if group_name == "food_harvested":
            score *= food_weight
        group_scores[group_name] = score

    positive_keys = set(positive_groups)
    negative_keys = set(negative_groups)
    plant_score = max((group_scores.get(key, 0.0) for key in positive_keys), default=0.0)
    non_plant_score = max((group_scores.get(key, 0.0) for key in negative_keys), default=0.0)
    margin = float(plant_score) - float(non_plant_score)

    min_plant_score = coerce_float(guard_cfg.get("plant_min_score", 0.45), 0.45)
    negative_margin = coerce_float(guard_cfg.get("negative_margin", 0.10), 0.10)
    decision = "pass"
    is_plant_like = True
    reason = ""
    if plant_score < min_plant_score:
        decision = "non_plant_rejected"
        is_plant_like = False
        reason = "plant_score fell below configured minimum"
    elif non_plant_score - plant_score >= negative_margin:
        decision = "non_plant_rejected"
        is_plant_like = False
        reason = "non_plant_score exceeded plant_score by configured margin"

    return InputGuardAnalysis(
        enabled=True,
        decision=decision,
        is_plant_like=is_plant_like,
        method=METHOD_NAME,
        plant_score=plant_score,
        non_plant_score=non_plant_score,
        margin=margin,
        reason=reason,
        debug_scores=(group_scores if debug_enabled else None),
    )


def _input_guard_config(config: Dict[str, Any]) -> Dict[str, Any]:
    inference_cfg = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}
    guard_cfg = inference_cfg.get("input_guard", {}) if isinstance(inference_cfg, dict) else {}
    return dict(guard_cfg) if isinstance(guard_cfg, dict) else {}


def _score_prompt_group(
    runtime: Any,
    image: Image.Image,
    labels: List[str],
    *,
    num_prompts: Optional[int],
) -> float:
    if not labels:
        return 0.0
    _label, best_score, scores = clip_runtime.clip_score_labels_ensemble(
        runtime,
        image,
        labels,
        label_type="generic",
        num_prompts=num_prompts,
    )
    if scores:
        return max(float(value) for value in scores.values())
    return float(best_score)


def _build_supported_crop_prompts(runtime: Any, config: Dict[str, Any]) -> List[str]:
    crop_names = _unique_terms(
        [
            *list(getattr(runtime, "crop_labels", []) or []),
            *_configured_crop_names(config),
            *list(SUPPORTED_CROP_PROMPT_CROPS),
        ]
    )
    aliases = crop_prompt_aliases()
    prompts: List[str] = []
    for crop_name in crop_names:
        canonical_crop = canonicalize_crop_prompt_label(crop_name)
        alias_terms = aliases.get(canonical_crop, aliases.get(str(crop_name).strip().lower(), [crop_name]))
        crop_terms = _unique_terms([crop_name, canonical_crop, *alias_terms])
        parts = _unique_terms(
            [
                "plant",
                *list(TURKEY_PRIORITY_CROP_PARTS.get(canonical_crop or str(crop_name).strip().lower(), [])),
                "leaf",
                "fruit",
                "flower",
                "stem",
                "whole plant",
            ]
        )
        for term in crop_terms:
            for part in parts:
                prompts.append(f"a {term} {part}")
    return _unique_terms(prompts)


def _configured_crop_names(config: Dict[str, Any]) -> Iterable[str]:
    router_cfg = config.get("router", {}) if isinstance(config.get("router"), dict) else {}
    crop_mapping = router_cfg.get("crop_mapping", {}) if isinstance(router_cfg, dict) else {}
    if not isinstance(crop_mapping, dict):
        return []
    return [str(key) for key in crop_mapping.keys()]


def _unique_terms(values: Iterable[Any]) -> List[str]:
    terms: List[str] = []
    seen: set[str] = set()
    for value in values:
        term = str(value or "").strip().lower()
        if not term or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _coerce_optional_positive_int(value: Any) -> Optional[int]:
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if resolved > 0 else None
