import pytest
from PIL import Image

from src.router.roi_helpers import bbox_area_ratio, sanitize_bbox
from src.router.roi_pipeline import (
    RoiClassificationHooks,
    collect_sam3_roi_candidates,
    finalize_sam3_roi_candidate,
    run_sam3_roi_classification_stage,
)


def _build_hooks(**overrides):
    hooks = {
        "policy_enabled": lambda _stage, default=True: default,
        "extract_roi": lambda image, _bbox, pad_ratio=0.08: image,
        "clip_score_labels_ensemble": lambda *_args, **_kwargs: ("unknown", 0.0, {}),
        "compute_leaf_likeness": lambda **_: 0.0,
        "rebalance_part_scores_for_leaf_like_roi": lambda **kwargs: kwargs["part_scores"],
        "select_best_crop_with_fallback": lambda crop_scores, _part_scores, **_: (
            max(crop_scores.items(), key=lambda item: item[1]) if crop_scores else ("unknown", 0.0)
        ),
        "compatible_parts_for_crop": lambda _crop: [],
        "score_parts_conditioned_on_crop": lambda *_args, **_kwargs: {},
        "score_label_candidates": lambda *_args, **_kwargs: {},
        "apply_generic_part_penalty": lambda scores, *_args, **_kwargs: scores,
        "select_part_label_with_specificity": lambda scores, *_args, **_kwargs: max(
            scores.items(), key=lambda item: item[1]
        ),
        "apply_leaf_like_override": lambda **kwargs: (kwargs["selected_label"], kwargs["selected_score"]),
    }
    hooks.update(overrides)
    return RoiClassificationHooks(**hooks)


def test_collect_sam3_roi_candidates_applies_filters():
    boxes = [[0, 0, 20, 20], [0, 0, 2, 2]]
    scores = [0.9, 0.95]
    settings = {
        "sam3_threshold": 0.85,
        "min_box_side_px": 5,
        "min_box_area_ratio": 0.01,
        "max_rois_for_classification": None,
    }

    candidates, seen = collect_sam3_roi_candidates(
        boxes=boxes,
        scores=scores,
        image_width=100,
        image_height=100,
        settings=settings,
        apply_roi_filters=True,
        sanitize_bbox_fn=sanitize_bbox,
        bbox_area_ratio_fn=bbox_area_ratio,
    )

    assert seen == 2
    assert len(candidates) == 1
    assert candidates[0]["bbox"] == [0.0, 0.0, 20.0, 20.0]


def test_run_sam3_roi_classification_stage_filters_with_open_set_gate():
    candidates = [{"bbox": [0.0, 0.0, 10.0, 10.0], "sam3_score": 0.9}]
    settings = {
        "classification_min_confidence": 0.4,
        "focus_part_mode_enabled": False,
        "focus_fallback_enabled": False,
    }
    stage_order = ["roi_filter", "roi_classification", "open_set_gate", "postprocess"]

    def policy_enabled(_stage, default=True):
        return default

    def classify_candidate(_candidate):
        return (
            {
                "crop": "tomato",
                "part": "leaf",
                "crop_confidence": 0.75,
                "part_confidence": 0.8,
                "bbox": [0.0, 0.0, 10.0, 10.0],
            },
            2,
        )

    def open_set_gate(crop_label, crop_confidence, min_confidence):
        return crop_label == "tomato" and crop_confidence >= min_confidence

    detections, kept, calls, elapsed_ms = run_sam3_roi_classification_stage(
        candidates=candidates,
        settings=settings,
        stage_order=stage_order,
        policy_enabled_fn=policy_enabled,
        classify_candidate_fn=classify_candidate,
        passes_open_set_gate_fn=open_set_gate,
    )

    assert len(detections) == 1
    assert kept == 1
    assert calls == 2
    assert elapsed_ms >= 0.0


def test_run_sam3_roi_classification_stage_returns_early_when_disabled():
    detections, kept, calls, elapsed_ms = run_sam3_roi_classification_stage(
        candidates=[{"bbox": [0.0, 0.0, 1.0, 1.0], "sam3_score": 0.9}],
        settings={"classification_min_confidence": 0.2},
        stage_order=["roi_filter", "postprocess"],
        policy_enabled_fn=lambda *_: True,
        classify_candidate_fn=lambda *_: (None, 0),
        passes_open_set_gate_fn=lambda *_: True,
    )

    assert detections == []
    assert kept == 0
    assert calls == 0
    assert elapsed_ms == 0.0


def test_finalize_sam3_roi_candidate_does_not_force_leaf_from_visual_bias_alone():
    settings = {
        "conditioned_part_weight": 0.45,
        "generic_part_labels": ["whole plant", "whole", "plant", "entire plant"],
        "generic_part_penalty": 0.78,
        "specific_part_override_ratio": 0.45,
        "specific_part_min_confidence": 0.12,
        "preferred_part_labels": [],
        "preferred_part_override_ratio": 0.50,
        "part_open_set_enabled": False,
        "part_open_set_min_confidence": 0.40,
        "part_open_set_margin": 0.10,
        "part_unknown_label": "unknown",
        "part_rejection_metadata_enabled": True,
        "leaf_override_label": "leaf",
        "leaf_override_enabled": False,
        "leaf_visual_override_enabled": True,
        "leaf_visual_likeness_threshold": 0.58,
        "leaf_visual_green_min": 0.18,
        "leaf_visual_min_margin": 0.05,
        "leaf_visual_force_generic": True,
        "leaf_visual_force_without_leaf_score": False,
        "leaf_visual_force_conf_floor": 0.16,
        "leaf_visual_force_part_factor": 0.65,
        "leaf_part_rebalance_enabled": False,
        "weight_crop": 0.65,
        "weight_part": 0.20,
        "weight_sam3": 0.15,
    }

    detection = finalize_sam3_roi_candidate(
        roi_image=Image.new("RGB", (80, 80), color="green"),
        candidate={"bbox": [10.0, 10.0, 90.0, 90.0], "sam3_score": 0.91},
        image_width=100,
        image_height=100,
        settings=settings,
        hooks=_build_hooks(
            policy_enabled=lambda stage, default=True: False if stage == "compatibility_fusion" else default,
            compute_leaf_likeness=lambda **_: 0.92,
            rebalance_part_scores_for_leaf_like_roi=lambda **kwargs: kwargs["part_scores"],
            select_best_crop_with_fallback=lambda crop_scores, part_scores, **_: ("tomato", crop_scores["tomato"]),
            compatible_parts_for_crop=lambda _crop: ["leaf", "fruit", "whole plant"],
            score_parts_conditioned_on_crop=lambda *_args, **_kwargs: {},
            score_label_candidates=lambda *_args, **_kwargs: {
                "label": "tomato whole plant",
                "confidence": 0.42,
                "second_confidence": 0.35,
                "unknown_confidence": 0.08,
                "margin": 0.07,
                "label_scores": {
                    "tomato leaf": 0.16,
                    "tomato fruit": 0.35,
                    "tomato whole plant": 0.42,
                },
                "rejection_reasons": [],
            },
            apply_generic_part_penalty=lambda scores, *_args, **_kwargs: scores,
            select_part_label_with_specificity=lambda scores, *_args, **_kwargs: max(
                scores.items(), key=lambda item: item[1]
            ),
            apply_leaf_like_override=lambda **kwargs: (kwargs["selected_label"], kwargs["selected_score"]),
        ),
        part_label="whole plant",
        part_conf=0.42,
        part_scores={"whole plant": 0.42, "fruit": 0.35, "leaf": 0.16},
        crop_label="tomato",
        crop_conf=0.94,
        crop_scores={"tomato": 0.94, "wheat": 0.08},
        global_crop_scores={"tomato": 0.95},
    )

    assert detection["part"] == "whole plant"
    assert detection["part_confidence"] == 0.42


def test_finalize_sam3_roi_candidate_abstains_part_without_losing_quality_score():
    settings = {
        "conditioned_part_weight": 0.45,
        "generic_part_labels": ["whole plant", "whole", "plant", "entire plant"],
        "generic_part_penalty": 0.78,
        "specific_part_override_ratio": 0.45,
        "specific_part_min_confidence": 0.12,
        "preferred_part_labels": [],
        "preferred_part_override_ratio": 0.50,
        "part_open_set_enabled": True,
        "part_open_set_min_confidence": 0.40,
        "part_open_set_margin": 0.10,
        "part_unknown_label": "unknown",
        "part_rejection_metadata_enabled": True,
        "leaf_override_label": "leaf",
        "leaf_override_enabled": False,
        "leaf_visual_override_enabled": False,
        "leaf_part_rebalance_enabled": False,
        "weight_crop": 0.65,
        "weight_part": 0.20,
        "weight_sam3": 0.15,
    }

    detection = finalize_sam3_roi_candidate(
        roi_image=Image.new("RGB", (80, 80), color="orange"),
        candidate={"bbox": [10.0, 10.0, 90.0, 90.0], "sam3_score": 0.91},
        image_width=100,
        image_height=100,
        settings=settings,
        hooks=_build_hooks(
            policy_enabled=lambda *_args, **_kwargs: True,
            compute_leaf_likeness=lambda **_: 0.10,
            rebalance_part_scores_for_leaf_like_roi=lambda **kwargs: kwargs["part_scores"],
            select_best_crop_with_fallback=lambda crop_scores, part_scores, **_: ("tomato", crop_scores["tomato"]),
            compatible_parts_for_crop=lambda _crop: ["leaf", "fruit", "whole plant"],
            score_parts_conditioned_on_crop=lambda *_args, **_kwargs: {
                "leaf": 0.05,
                "fruit": 0.41,
                "whole plant": 0.35,
            },
            score_label_candidates=lambda *_args, **_kwargs: {
                "label": "tomato fruit",
                "confidence": 0.41,
                "second_confidence": 0.35,
                "unknown_confidence": 0.46,
                "margin": 0.06,
                "label_scores": {
                    "tomato leaf": 0.05,
                    "tomato fruit": 0.41,
                    "tomato whole plant": 0.35,
                },
                "rejection_reasons": [
                    "unknown_confidence (0.4600) >= confidence (0.4100)",
                    "margin (0.0600) < threshold (0.1000)",
                ],
            },
            apply_generic_part_penalty=lambda scores, *_args, **_kwargs: scores,
            select_part_label_with_specificity=lambda scores, *_args, **_kwargs: max(
                scores.items(), key=lambda item: item[1]
            ),
            apply_leaf_like_override=lambda **kwargs: (kwargs["selected_label"], kwargs["selected_score"]),
        ),
        part_label="flower",
        part_conf=0.92,
        part_scores={"flower": 0.92, "fruit": 0.36, "leaf": 0.08},
        crop_label="tomato",
        crop_conf=0.94,
        crop_scores={"tomato": 0.94, "wheat": 0.08},
        global_crop_scores={"tomato": 0.95},
    )

    assert detection["part"] == "unknown"
    assert detection["part_confidence"] == 0.0
    assert detection["raw_part_label"] == "fruit"
    assert detection["raw_part_confidence"] == 0.3825
    assert detection["part_unknown_confidence"] == 0.46
    assert detection["part_rejection_reason"] == (
        "unknown_confidence (0.4600) >= confidence (0.3825); "
        "confidence (0.3825) < threshold (0.4000)"
    )
    assert detection["_quality_score"] == ((0.65 * 0.94) + (0.20 * 0.3825) + (0.15 * 0.91))


def test_finalize_sam3_roi_candidate_keeps_supported_part_when_surfaces_agree_despite_unknown_proxy():
    settings = {
        "conditioned_part_weight": 0.45,
        "generic_part_labels": ["whole plant", "whole", "plant", "entire plant"],
        "generic_part_penalty": 0.78,
        "specific_part_override_ratio": 0.45,
        "specific_part_min_confidence": 0.12,
        "preferred_part_labels": [],
        "preferred_part_override_ratio": 0.50,
        "part_open_set_enabled": True,
        "part_open_set_min_confidence": 0.40,
        "part_open_set_margin": 0.10,
        "part_unknown_label": "unknown",
        "part_rejection_metadata_enabled": True,
        "leaf_override_label": "leaf",
        "leaf_override_enabled": False,
        "leaf_visual_override_enabled": False,
        "leaf_part_rebalance_enabled": False,
        "weight_crop": 0.65,
        "weight_part": 0.20,
        "weight_sam3": 0.15,
    }

    detection = finalize_sam3_roi_candidate(
        roi_image=Image.new("RGB", (80, 80), color="orange"),
        candidate={"bbox": [10.0, 10.0, 90.0, 90.0], "sam3_score": 0.91},
        image_width=100,
        image_height=100,
        settings=settings,
        hooks=_build_hooks(
            policy_enabled=lambda *_args, **_kwargs: True,
            compute_leaf_likeness=lambda **_: 0.10,
            rebalance_part_scores_for_leaf_like_roi=lambda **kwargs: kwargs["part_scores"],
            select_best_crop_with_fallback=lambda crop_scores, part_scores, **_: ("tomato", crop_scores["tomato"]),
            compatible_parts_for_crop=lambda _crop: ["leaf", "fruit", "whole plant"],
            score_parts_conditioned_on_crop=lambda *_args, **_kwargs: {
                "leaf": 0.05,
                "fruit": 0.56,
                "whole plant": 0.16,
            },
            score_label_candidates=lambda *_args, **_kwargs: {
                "label": "tomato fruit",
                "confidence": 0.56,
                "second_confidence": 0.16,
                "unknown_confidence": 0.82,
                "margin": 0.40,
                "label_scores": {
                    "tomato leaf": 0.05,
                    "tomato fruit": 0.56,
                    "tomato whole plant": 0.16,
                },
                "rejection_reasons": [
                    "unknown_confidence (0.8200) >= confidence (0.5600)",
                ],
            },
            apply_generic_part_penalty=lambda scores, generic_labels, penalty: {
                label: (score * penalty if label in set(generic_labels) else score)
                for label, score in scores.items()
            },
            select_part_label_with_specificity=lambda scores, *_args, **_kwargs: max(
                scores.items(), key=lambda item: item[1]
            ),
            apply_leaf_like_override=lambda **kwargs: (kwargs["selected_label"], kwargs["selected_score"]),
        ),
        part_label="fruit",
        part_conf=0.62,
        part_scores={"fruit": 0.62, "leaf": 0.07, "whole plant": 0.24},
        crop_label="tomato",
        crop_conf=0.94,
        crop_scores={"tomato": 0.94, "wheat": 0.08},
        global_crop_scores={"tomato": 0.95},
    )

    assert detection["part"] == "fruit"
    assert detection["part_confidence"] == pytest.approx(0.593)
    assert detection["raw_part_label"] == "fruit"
    assert detection["part_unknown_confidence"] == 0.82
    assert "part_rejection_reason" not in detection
    assert detection["part_open_set_rejection_reason"] == "unknown_confidence (0.8200) >= confidence (0.5930)"
    assert detection["part_recovery_reason"] == (
        "retained compatible part because generic and crop-conditioned part surfaces agreed despite the unknown proxy"
    )
