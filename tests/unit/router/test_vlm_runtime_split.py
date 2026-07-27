import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from src.router import clip_runtime, sam3_runtime
from src.router.router_pipeline import RouterPipeline


class FakeTokenizer:
    def __call__(self, prompts):
        return torch.arange(len(prompts), dtype=torch.float32).unsqueeze(1)


class FakeOpenClipModel:
    def __init__(self):
        self.encode_image_calls = 0
        self.logit_scale = torch.tensor(0.0)

    def encode_image(self, image_tensor):
        self.encode_image_calls += 1
        batch = int(image_tensor.shape[0])
        embeds = torch.zeros(batch, 2, dtype=torch.float32, device=image_tensor.device)
        embeds[:, 0] = 1.0
        return embeds

    def encode_text(self, text_tokens):
        prompt_count = int(text_tokens.shape[0])
        embeds = torch.zeros(prompt_count, 2, dtype=torch.float32, device=text_tokens.device)
        if prompt_count > 0:
            embeds[0] = torch.tensor([1.0, 0.0], dtype=torch.float32, device=text_tokens.device)
        if prompt_count > 1:
            embeds[1:] = torch.tensor([0.0, 1.0], dtype=torch.float32, device=text_tokens.device)
        return embeds


def _build_pipeline() -> RouterPipeline:
    return RouterPipeline(
        config={
            "router": {
                "crop_mapping": {"tomato": {"parts": ["leaf"]}},
                "vlm": {
                    "enabled": True,
                    "confidence_threshold": 0.25,
                    "max_detections": 7,
                    "crop_labels": ["tomato", "potato"],
                    "part_labels": ["leaf"],
                    "prompt_templates": {
                        "crop": ["{term}", "close {term}"],
                        "part": ["{term}"],
                    },
                },
            },
        },
        device="cpu",
    )


def _attach_open_clip_runtime(pipeline: RouterPipeline) -> FakeOpenClipModel:
    model = FakeOpenClipModel()
    pipeline.bioclip_backend = "open_clip"
    pipeline.bioclip = model
    pipeline.bioclip_processor = {
        "preprocess": lambda _image: torch.ones(3, 4, 4, dtype=torch.float32),
        "tokenizer": FakeTokenizer(),
    }
    return model


def test_pipeline_parses_string_boolean_taxonomy_flags():
    pipeline = RouterPipeline(
        config={
            "router": {
                "crop_mapping": {"tomato": {"parts": ["leaf"]}},
                "vlm": {
                    "enabled": True,
                    "use_dynamic_taxonomy": "false",
                    "strict_taxonomy_loading": "false",
                    "crop_labels": ["tomato"],
                    "part_labels": ["leaf"],
                },
            },
        },
        device="cpu",
    )

    assert pipeline.use_dynamic_taxonomy is False
    assert pipeline.strict_taxonomy_loading is False


def test_pipeline_keeps_configured_crop_part_surface_when_dynamic_taxonomy_is_enabled(tmp_path):
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(
            {
                "crops": ["tomato"],
                "parts": {"core": ["leaf", "fruit", "flower", "stem", "whole plant"]},
                "crop_part_compatibility": {
                    "tomato": ["leaf", "fruit", "flower", "stem", "whole plant"],
                },
            }
        ),
        encoding="utf-8",
    )

    pipeline = RouterPipeline(
        config={
            "router": {
                "crop_mapping": {"tomato": {"parts": ["leaf", "fruit", "stem", "whole"]}},
                "vlm": {
                    "enabled": True,
                    "use_dynamic_taxonomy": True,
                    "taxonomy_path": str(taxonomy_path),
                },
            },
        },
        device="cpu",
    )

    assert pipeline.crop_part_compatibility["tomato"] == ["leaf", "fruit", "stem", "whole plant"]
    assert "flower" not in pipeline.crop_part_compatibility["tomato"]
    assert "whole plant" in pipeline.part_labels


def test_pipeline_falls_back_to_configured_labels_when_taxonomy_file_is_missing():
    pipeline = RouterPipeline(
        config={
            "router": {
                "crop_mapping": {
                    "tomato": {"parts": ["leaf", "fruit"]},
                    "potato": {"parts": ["leaf"]},
                },
                "vlm": {
                    "enabled": False,
                    "use_dynamic_taxonomy": True,
                    "strict_taxonomy_loading": False,
                    "taxonomy_path": "config/definitely_missing_taxonomy.json",
                },
            },
        },
        device="cpu",
    )

    assert pipeline.crop_labels == ["tomato", "potato"]
    assert "leaf" in pipeline.part_labels
    assert pipeline.crop_part_compatibility["tomato"] == ["leaf", "fruit"]


def test_pipeline_raises_when_dynamic_taxonomy_missing_in_strict_mode():
    with pytest.raises(RuntimeError, match="Failed to load taxonomy"):
        RouterPipeline(
            config={
                "router": {
                    "crop_mapping": {
                        "tomato": {"parts": ["leaf", "fruit"]},
                        "potato": {"parts": ["leaf"]},
                    },
                    "vlm": {
                        "enabled": False,
                        "use_dynamic_taxonomy": True,
                        "taxonomy_path": "config/definitely_missing_taxonomy.json",
                    },
                },
            },
            device="cpu",
        )


def test_analyze_image_result_includes_taxonomy_runtime_warnings_when_non_strict():
    pipeline = RouterPipeline(
        config={
            "router": {
                "crop_mapping": {
                    "tomato": {"parts": ["leaf", "fruit"]},
                },
                "vlm": {
                    "enabled": False,
                    "use_dynamic_taxonomy": True,
                    "strict_taxonomy_loading": False,
                    "taxonomy_path": "config/definitely_missing_taxonomy.json",
                },
            },
        },
        device="cpu",
    )

    result = pipeline.analyze_image_result(Image.new("RGB", (8, 8), color="green"))

    assert result.status == "unavailable"
    assert "runtime_warnings" in result.metadata
    assert any("Failed to load taxonomy" in warning for warning in result.metadata["runtime_warnings"])


def test_clip_score_labels_ensemble_reuses_open_clip_image_embedding():
    pipeline = _build_pipeline()
    model = _attach_open_clip_runtime(pipeline)

    best_label, best_score, scores = clip_runtime.clip_score_labels_ensemble(
        pipeline,
        Image.new("RGB", (8, 8), color="green"),
        ["tomato", "potato"],
        label_type="crop",
        num_prompts=2,
    )

    assert best_label in {"tomato", "potato"}
    assert best_score >= 0.0
    assert set(scores.keys()) == {"tomato", "potato"}
    assert model.encode_image_calls == 1


def test_clip_score_labels_ensemble_uses_crop_alias_prompts(monkeypatch):
    pipeline = _build_pipeline()
    pipeline.vlm_config.pop("prompt_templates", None)
    captured_prompts = []

    def _fake_score(_runtime, prompts, *, request=None, **_kwargs):
        del request
        captured_prompts.extend(prompts)
        scores = torch.zeros((1, len(prompts)), dtype=torch.float32)
        alias_index = next(index for index, prompt in enumerate(prompts) if "Solanum lycopersicum" in prompt)
        scores[0, alias_index] = 1.0
        return scores

    monkeypatch.setattr(clip_runtime, "_score_prompt_probabilities", _fake_score)

    best_label, best_score, scores = clip_runtime.clip_score_labels_ensemble(
        pipeline,
        Image.new("RGB", (8, 8), color="green"),
        ["tomato", "potato"],
        label_type="crop",
    )

    assert best_label == "tomato"
    assert best_score > scores["potato"]
    assert any("Solanum lycopersicum" in prompt for prompt in captured_prompts)
    assert any("tomato leaf" in prompt for prompt in captured_prompts)


def test_encode_and_score_raises_on_scoring_error_when_strict(monkeypatch):
    pipeline = _build_pipeline()
    pipeline.bioclip_backend = "transformers"
    pipeline.bioclip = object()
    pipeline.bioclip_processor = object()
    pipeline.strict_scoring_errors = True

    monkeypatch.setattr(
        clip_runtime,
        "_score_processor_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="Prompt encoding failed"):
        clip_runtime.encode_and_score(
            pipeline,
            clip_runtime.ClipScoreRequest(image=Image.new("RGB", (8, 8), color="green")),
            ["tomato"],
        )


def test_encode_and_score_batch_returns_zeros_when_non_strict(monkeypatch):
    pipeline = _build_pipeline()
    pipeline.bioclip_backend = "transformers"
    pipeline.bioclip = object()
    pipeline.bioclip_processor = object()
    pipeline.strict_scoring_errors = False

    monkeypatch.setattr(
        clip_runtime,
        "_score_processor_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    output = clip_runtime.encode_and_score_batch(
        pipeline,
        [Image.new("RGB", (8, 8), color="green") for _ in range(2)],
        ["tomato", "potato"],
    )

    assert tuple(output.shape) == (2, 2)
    assert float(output.sum().item()) == 0.0


def test_clip_score_labels_ensemble_batch_reuses_open_clip_image_embeddings():
    pipeline = _build_pipeline()
    model = _attach_open_clip_runtime(pipeline)

    results = clip_runtime.clip_score_labels_ensemble_batch(
        pipeline,
        [Image.new("RGB", (8, 8), color="green") for _ in range(3)],
        ["tomato", "potato"],
        label_type="crop",
        num_prompts=2,
        image_batch_size=8,
    )

    assert len(results) == 3
    assert all(result[0] in {"tomato", "potato"} for result in results)
    assert model.encode_image_calls == 1


def test_open_set_debug_payload_is_lazy_when_debug_disabled(monkeypatch):
    pipeline = _build_pipeline()
    _attach_open_clip_runtime(pipeline)

    monkeypatch.setattr(clip_runtime.logger, "isEnabledFor", lambda _level: False)
    monkeypatch.setattr(
        clip_runtime.json,
        "dumps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("debug payload should stay lazy")),
    )

    label, confidence = clip_runtime.clip_score_labels(
        pipeline,
        Image.new("RGB", (8, 8), color="green"),
        ["tomato"],
        label_type="crop",
    )

    assert label in {"tomato", "unknown"}
    assert confidence >= 0.0


def test_analyze_image_sam3_delegates_to_runtime_helpers(monkeypatch):
    pipeline = _build_pipeline()
    expected = {"detections": [{"crop": "tomato"}], "image_size": (3, 8, 8)}
    sentinel_context = SimpleNamespace()

    def fake_build_request_context(runtime, **kwargs):
        assert runtime is pipeline
        assert kwargs["image_size"] == (3, 8, 8)
        return sentinel_context

    monkeypatch.setattr(sam3_runtime, "build_request_context", fake_build_request_context)
    monkeypatch.setattr(sam3_runtime, "analyze_sam3_image", lambda runtime, context: expected)

    result = pipeline._analyze_image_sam3(Image.new("RGB", (8, 8), color="green"), (3, 8, 8))

    assert result == expected


def test_build_request_context_parses_string_runtime_booleans():
    pipeline = _build_pipeline()
    pipeline.vlm_config["timing_logs_enabled"] = "false"
    pipeline.vlm_config["sam3_expand_with_supported_parts"] = "false"
    context = sam3_runtime.build_request_context(
        pipeline,
        pil_image=Image.new("RGB", (8, 8), color="green"),
        image_size=(3, 8, 8),
        confidence_threshold=0.25,
        max_detections=7,
    )

    assert context.timing_logs_enabled is False
    assert context.sam3_prompts == ("plant",)


def test_run_sam3_merges_results_across_prompt_sequence():
    pipeline = _build_pipeline()
    calls = []

    def fake_single(_image, prompt, threshold=0.7):
        calls.append((prompt, threshold))
        index = len(calls)
        return {
            "masks": torch.ones(1, 2, 2, dtype=torch.float32) * float(index),
            "boxes": torch.tensor([[index, index, index + 1, index + 1]], dtype=torch.float32),
            "scores": torch.tensor([0.9 - (0.1 * index)], dtype=torch.float32),
        }

    pipeline._run_sam3_single_prompt = fake_single

    result = pipeline._run_sam3(Image.new("RGB", (8, 8), color="green"), prompt=["plant", "leaf"], threshold=0.55)

    assert calls == [("plant", 0.55), ("leaf", 0.55)]
    assert tuple(result["boxes"].shape) == (2, 4)
    assert tuple(result["scores"].shape) == (2,)
    assert tuple(result["masks"].shape) == (2, 2, 2)


def test_analyze_image_reports_empty_sam3_diagnostics_with_prompt_sequence():
    pipeline = RouterPipeline(
        config={
            "router": {
                "crop_mapping": {"tomato": {"parts": ["leaf", "fruit", "whole"]}},
                "vlm": {
                    "enabled": True,
                    "confidence_threshold": 0.25,
                    "crop_labels": ["tomato"],
                    "part_labels": ["leaf", "fruit", "whole plant"],
                },
            },
        },
        device="cpu",
    )
    pipeline.models_loaded = True
    pipeline.actual_pipeline = "sam3"

    captured = {}

    def fake_run_sam3(_image, prompt, threshold=0.7):
        captured["prompt"] = prompt
        captured["threshold"] = threshold
        return {
            "masks": torch.tensor([]),
            "boxes": torch.tensor([]),
            "scores": torch.tensor([]),
        }

    pipeline._run_sam3 = fake_run_sam3

    result = pipeline.analyze_image_result(Image.new("RGB", (8, 8), color="green"))

    assert captured == {
        "prompt": ("plant", "leaf", "fruit", "whole plant"),
        "threshold": 0.6,
    }
    assert result.detections_count == 0
    assert result.message == "No SAM3 instances for prompts=plant,leaf,fruit,whole plant threshold=0.60."


def test_route_batch_preserves_order_and_uses_batched_runtime(monkeypatch):
    pipeline = _build_pipeline()
    pipeline.models_loaded = True
    pipeline.actual_pipeline = "sam3"

    analyses = [
        {"detections": [{"crop": "tomato", "part": "leaf", "bbox": [0, 0, 1, 1], "crop_confidence": 0.9}]},
        {"detections": [{"crop": "potato", "part": "leaf", "bbox": [1, 1, 2, 2], "crop_confidence": 0.8}]},
        {"detections": [{"crop": "pepper", "part": "leaf", "bbox": [2, 2, 3, 3], "crop_confidence": 0.7}]},
    ]

    monkeypatch.setattr(
        sam3_runtime,
        "analyze_sam3_batch",
        lambda runtime, batch, confidence_threshold=0.8, max_detections=None: analyses,
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("per-image path should not run")),
    )

    crops_out, confs = pipeline.route_batch(torch.zeros(3, 3, 8, 8))

    assert [item["crop"] for item in crops_out] == ["tomato", "potato", "pepper"]
    assert confs == [0.9, 0.8, 0.7]


def test_analyze_image_uses_config_default_request_options_when_omitted():
    pipeline = _build_pipeline()
    pipeline.models_loaded = True
    pipeline.actual_pipeline = "sam3"

    captured = {}

    def fake_analyze(pil_image, image_size, confidence_threshold=0.8, max_detections=None):
        del pil_image, image_size
        captured["confidence_threshold"] = confidence_threshold
        captured["max_detections"] = max_detections
        return {"detections": [{"crop": "tomato", "part": "leaf", "crop_confidence": 0.9}]}

    pipeline._analyze_image_sam3 = fake_analyze

    payload = pipeline.analyze_image(torch.zeros(3, 8, 8))

    assert captured == {"confidence_threshold": 0.25, "max_detections": 7}
    assert payload["primary_detection"]["crop"] == "tomato"
    assert payload["detections_count"] == 1


def test_process_image_uses_config_default_request_options_when_omitted():
    pipeline = _build_pipeline()
    pipeline.models_loaded = True
    pipeline.actual_pipeline = "sam3"

    captured = {}

    def fake_analyze(pil_image, image_size, confidence_threshold=0.8, max_detections=None):
        del pil_image, image_size
        captured["confidence_threshold"] = confidence_threshold
        captured["max_detections"] = max_detections
        return {"detections": [{"crop": "tomato", "part": "leaf", "crop_confidence": 0.88}]}

    pipeline._analyze_image_sam3 = fake_analyze

    payload = pipeline.process_image(torch.zeros(3, 8, 8))

    assert captured == {"confidence_threshold": 0.25, "max_detections": 7}
    assert payload["analysis"]["primary_detection"]["crop"] == "tomato"
    assert payload["scenario"] == "diagnostic_scouting"


def test_route_batch_uses_config_default_request_options_when_omitted(monkeypatch):
    pipeline = _build_pipeline()
    pipeline.models_loaded = True
    pipeline.actual_pipeline = "sam3"

    captured = {}

    def fake_batch(runtime, batch, confidence_threshold=0.8, max_detections=None):
        del runtime, batch
        captured["confidence_threshold"] = confidence_threshold
        captured["max_detections"] = max_detections
        return [
            {"detections": [{"crop": "tomato", "part": "leaf", "crop_confidence": 0.9}]},
            {"detections": [{"crop": "potato", "part": "leaf", "crop_confidence": 0.8}]},
        ]

    monkeypatch.setattr(sam3_runtime, "analyze_sam3_batch", fake_batch)

    crops_out, confs = pipeline.route_batch(torch.zeros(2, 3, 8, 8))

    assert captured == {"confidence_threshold": 0.25, "max_detections": 7}
    assert [item["crop"] for item in crops_out] == ["tomato", "potato"]
    assert confs == [0.9, 0.8]


def test_analyze_sam3_batch_reduces_sam_call_count_under_batch_support(monkeypatch):
    pipeline = _build_pipeline()
    pipeline.models_loaded = True
    pipeline.actual_pipeline = "sam3"
    pipeline.vlm_config["batch_chunk_size"] = 3

    batch_calls = {"count": 0}
    fallback_calls = {"count": 0}

    def fake_run_sam3_batch(images, prompt, threshold=0.7):
        batch_calls["count"] += 1
        return [{"masks": [], "boxes": [], "scores": []} for _ in images]

    monkeypatch.setattr(pipeline, "_run_sam3_batch", fake_run_sam3_batch)
    monkeypatch.setattr(
        sam3_runtime,
        "analyze_sam3_image",
        lambda *_args, **_kwargs: fallback_calls.__setitem__("count", fallback_calls["count"] + 1),
    )

    results = sam3_runtime.analyze_sam3_batch(pipeline, torch.zeros(5, 3, 8, 8))

    assert len(results) == 5
    assert batch_calls["count"] == 2
    assert fallback_calls["count"] == 0


def test_analyze_sam3_batch_falls_back_per_chunk_when_batched_sam3_is_unsupported(monkeypatch):
    pipeline = _build_pipeline()
    pipeline.models_loaded = True
    pipeline.actual_pipeline = "sam3"
    pipeline.vlm_config["batch_chunk_size"] = 2

    monkeypatch.setattr(
        pipeline,
        "_run_sam3_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("batch unsupported")),
    )

    calls = {"count": 0}

    def fake_analyze_sam3_image(_runtime, context):
        index = calls["count"]
        calls["count"] += 1
        return {
            "detections": [
                {
                    "crop": f"crop_{index}",
                    "part": "leaf",
                    "bbox": [0, 0, 1, 1],
                    "crop_confidence": 0.5 + index,
                }
            ],
            "image_size": context.image_size,
        }

    monkeypatch.setattr(sam3_runtime, "analyze_sam3_image", fake_analyze_sam3_image)

    results = sam3_runtime.analyze_sam3_batch(pipeline, torch.zeros(3, 3, 8, 8))

    assert len(results) == 3
    assert [result["detections"][0]["crop"] for result in results] == ["crop_0", "crop_1", "crop_2"]
    assert calls["count"] == 3


def test_set_runtime_profile_refreshes_profile_derived_controls():
    pipeline = RouterPipeline(
        config={
            "router": {
                "vlm": {
                    "enabled": True,
                    "open_set_enabled": False,
                    "profiles": {
                        "disabled": {"enabled": False},
                        "open_set": {"open_set_enabled": True},
                    },
                },
            },
        },
        device="cpu",
    )

    assert pipeline.enabled is True
    assert pipeline.open_set_enabled is False

    assert pipeline.set_runtime_profile("disabled") is True
    assert pipeline.enabled is False

    assert pipeline.set_runtime_profile("open_set") is True
    assert pipeline.open_set_enabled is True
