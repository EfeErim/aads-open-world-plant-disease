from PIL import Image

from src.router.heuristics import (
    apply_leaf_like_override,
    compute_leaf_likeness,
    rebalance_part_scores_for_leaf_like_roi,
    select_best_crop_with_fallback,
    select_part_label_with_specificity,
)


def test_compute_leaf_likeness_scores_green_roi_high():
    likeness = compute_leaf_likeness(
        Image.new("RGB", (80, 80), color="green"),
        bbox=[0.0, 0.0, 80.0, 80.0],
        image_width=100,
        image_height=100,
    )

    assert likeness > 0.8


def test_rebalance_part_scores_for_leaf_like_roi_boosts_leaf_and_penalizes_non_foliar():
    adjusted = rebalance_part_scores_for_leaf_like_roi(
        {"leaf": 0.30, "fruit": 0.40, "stem": 0.20},
        leaf_likeness=0.9,
        leaf_min_confidence=0.20,
        leaf_support_ratio=0.70,
    )

    assert adjusted["leaf"] > 0.30
    assert adjusted["fruit"] < 0.40
    assert adjusted["stem"] == 0.20


def test_rebalance_part_scores_for_leaf_like_roi_requires_leaf_score_support():
    original = {"leaf": 0.08, "fruit": 0.62, "stem": 0.14}

    adjusted = rebalance_part_scores_for_leaf_like_roi(
        original,
        leaf_likeness=0.95,
        leaf_min_confidence=0.18,
        leaf_support_ratio=0.75,
    )

    assert adjusted == original


def test_apply_leaf_like_override_prefers_leaf_for_generic_selection():
    label, score = apply_leaf_like_override(
        selected_label="whole plant",
        selected_score=0.20,
        part_scores={"whole plant": 0.20, "leaf": 0.28},
        bbox=[10.0, 10.0, 90.0, 60.0],
        image_width=100,
        image_height=100,
        leaf_min_margin=0.04,
    )

    assert label == "leaf"
    assert score == 0.28


def test_apply_leaf_like_override_does_not_clobber_specific_fruit_label():
    label, score = apply_leaf_like_override(
        selected_label="fruit",
        selected_score=0.42,
        part_scores={"fruit": 0.42, "leaf": 0.33, "whole plant": 0.12},
        bbox=[10.0, 10.0, 90.0, 60.0],
        image_width=100,
        image_height=100,
        leaf_min_margin=0.04,
    )

    assert label == "fruit"
    assert score == 0.42


def test_select_part_label_with_specificity_prefers_specific_part_when_close():
    label, score = select_part_label_with_specificity(
        {"whole plant": 0.50, "leaf": 0.35},
        generic_part_labels=["whole plant"],
        specific_override_ratio=0.60,
        specific_min_confidence=0.20,
        preferred_part_labels=["leaf"],
        preferred_override_ratio=0.60,
    )

    assert label == "leaf"
    assert score == 0.35


def test_select_best_crop_with_fallback_uses_compatible_part_signal():
    crop, score = select_best_crop_with_fallback(
        {"tomato": 0.49, "potato": 0.50},
        {"leaf": 0.95, "tuber": 0.10},
        {"tomato": ["leaf"], "potato": ["tuber"]},
        ["leaf", "tuber"],
    )

    assert crop == "tomato"
    assert score > 0.50


def test_select_best_crop_with_fallback_uses_global_crop_context_to_override_misleading_roi():
    crop, score = select_best_crop_with_fallback(
        {"tomato": 0.20, "wheat": 0.94},
        {"leaf": 0.90, "fruit": 0.10},
        {"tomato": ["leaf", "fruit"], "wheat": ["leaf"]},
        ["leaf", "fruit"],
        global_crop_scores={"tomato": 0.95, "wheat": 0.10},
        global_crop_context_weight=0.65,
    )

    assert crop == "tomato"
    assert score > 0.70
