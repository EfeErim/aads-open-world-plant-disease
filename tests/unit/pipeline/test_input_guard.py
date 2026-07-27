import pytest
from PIL import Image

from src.pipeline import input_guard
from src.pipeline.input_guard import SUPPORTED_CROP_PROMPT_CROPS


class FakeRuntime:
    crop_labels = ["pepper"]


def test_supported_crop_prompts_include_turkey_priority_top_10(monkeypatch):
    def _fake_score(runtime, image, labels, *, label_type="generic", num_prompts=None):
        del runtime, image, label_type, num_prompts
        return labels[0], 0.05, {labels[0]: 0.05}

    monkeypatch.setattr(input_guard.clip_runtime, "clip_score_labels_ensemble", _fake_score)

    prompts = input_guard._build_supported_crop_prompts(
        FakeRuntime(),
        {"router": {"crop_mapping": {"pepper": {"parts": ["leaf"]}}}},
    )

    for crop_name in SUPPORTED_CROP_PROMPT_CROPS:
        assert f"a {crop_name} plant" in prompts
        assert f"a {crop_name} leaf" in prompts
        assert f"a {crop_name} fruit" in prompts
    assert "a hazelnut fruit cluster" in prompts
    assert "a apricot fruit" in prompts
    assert "a strawberry fruit" in prompts


def test_plantness_guard_rejects_when_negative_prompt_margin_dominates(monkeypatch):
    def _fake_score(runtime, image, labels, *, label_type="generic", num_prompts=None):
        del runtime, image, label_type, num_prompts
        if "a dog" in labels:
            return "a dog", 0.66, {"a dog": 0.66}
        if "a plant" in labels:
            return "a plant", 0.21, {"a plant": 0.21}
        return labels[0], 0.05, {labels[0]: 0.05}

    monkeypatch.setattr(input_guard.clip_runtime, "clip_score_labels_ensemble", _fake_score)

    result = input_guard.evaluate_plantness_input_guard(
        FakeRuntime(),
        Image.new("RGB", (8, 8)),
        {
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}},
            "inference": {"input_guard": {"enabled": True, "debug_scores": True}},
        },
    )

    assert result.decision == "non_plant_rejected"
    assert result.is_plant_like is False
    assert result.plant_score == 0.21
    assert result.non_plant_score == 0.66
    assert result.debug_scores is not None
    assert result.debug_scores["supported_crop"] == 0.05


def test_plantness_guard_downweights_harvested_food_for_fruit_part(monkeypatch):
    def _fake_score(runtime, image, labels, *, label_type="generic", num_prompts=None):
        del runtime, image, label_type, num_prompts
        if "cooked food" in labels:
            return "cooked food", 0.8, {"cooked food": 0.8}
        if "a plant" in labels:
            return "a plant", 0.5, {"a plant": 0.5}
        return labels[0], 0.1, {labels[0]: 0.1}

    monkeypatch.setattr(input_guard.clip_runtime, "clip_score_labels_ensemble", _fake_score)

    result = input_guard.evaluate_plantness_input_guard(
        FakeRuntime(),
        Image.new("RGB", (8, 8)),
        {"inference": {"input_guard": {"enabled": True, "fruit_food_weight": 0.4}}},
        requested_part="fruit",
    )

    assert result.decision == "pass"
    assert result.plant_score == 0.5
    assert result.non_plant_score == pytest.approx(0.32)
    assert result.margin == pytest.approx(0.18)
