from scripts import calibrate_router_surface as calibrator
from pathlib import Path


def test_parse_sweep_spec_accepts_alias_and_coerces_int_values():
    parameter, values = calibrator.parse_sweep_spec("sam3_prompt_limit=4,6")

    assert parameter == "router.vlm.sam3_prompt_limit"
    assert values == [4, 6]


def test_parse_sweep_spec_accepts_boolean_input_guard_values():
    parameter, values = calibrator.parse_sweep_spec("input_guard_enabled=false,true")

    assert parameter == "inference.input_guard.enabled"
    assert values == [False, True]


def test_resolve_sweep_grid_includes_current_config_value():
    base_config = {
        "inference": {"router_min_confidence": 0.65},
        "router": {"vlm": {"confidence_threshold": 0.25}},
    }

    grid = calibrator.resolve_sweep_grid(
        base_config,
        preset="none",
        sweep_specs=["router_min_confidence=0.55,0.75"],
    )

    assert grid == {"inference.router_min_confidence": [0.65, 0.55, 0.75]}


def test_apply_overrides_mirrors_vlm_values_into_active_profile():
    base_config = {
        "router": {
            "vlm": {
                "profile": "balanced",
                "sam3_prompt_limit": 6,
                "profiles": {"balanced": {"sam3_prompt_limit": 6}},
            }
        }
    }

    config = calibrator.apply_overrides(base_config, {"router.vlm.sam3_prompt_limit": 4})

    assert config["router"]["vlm"]["sam3_prompt_limit"] == 4
    assert config["router"]["vlm"]["profiles"]["balanced"]["sam3_prompt_limit"] == 4
    assert base_config["router"]["vlm"]["profiles"]["balanced"]["sam3_prompt_limit"] == 6


def test_replay_variant_applies_confidence_and_margin_gates_without_model_rerun():
    samples = [
        {
            "group": "id",
            "expected_crop": "tomato",
            "expected_part": "leaf",
            "predicted_crop": "tomato",
            "predicted_part": "leaf",
            "router_handoff_crop": True,
            "handoff_crop": True,
            "crop_confidence": 0.70,
            "routing_margin": 0.04,
            "crop_correct": True,
            "part_correct": True,
            "part_abstained": False,
            "unsupported_part_emitted": False,
            "latency_ms": 10.0,
        },
        {
            "group": "off_crop",
            "expected_crop": "unknown",
            "expected_part": "unknown",
            "predicted_crop": "tomato",
            "predicted_part": "leaf",
            "router_handoff_crop": True,
            "handoff_crop": True,
            "crop_confidence": 0.66,
            "routing_margin": 0.20,
            "crop_correct": False,
            "part_correct": False,
            "part_abstained": False,
            "unsupported_part_emitted": False,
            "latency_ms": 10.0,
        },
    ]

    replayed = calibrator.replay_variant(
        samples,
        overrides={
            "inference.router_min_confidence": 0.65,
            "inference.router_min_margin": 0.10,
        },
    )

    rows = replayed["samples"]
    assert rows[0]["handoff_crop"] is False
    assert rows[0]["runtime_gate_reasons"] == ["router_min_margin"]
    assert rows[0]["predicted_part"] == "unknown"
    assert rows[1]["handoff_crop"] is True
    assert replayed["metrics"]["negative_false_accept_rate"] == 1.0


def test_replay_variant_can_apply_cached_input_guard_rejection():
    samples = [
        {
            "group": "non_plant",
            "expected_crop": "unknown",
            "expected_part": "unknown",
            "predicted_crop": "tomato",
            "predicted_part": "leaf",
            "router_handoff_crop": True,
            "handoff_crop": True,
            "crop_confidence": 0.90,
            "routing_margin": 0.40,
            "crop_correct": False,
            "part_correct": False,
            "part_abstained": False,
            "unsupported_part_emitted": False,
            "latency_ms": 10.0,
            "input_guard": {
                "plant_score": 0.30,
                "non_plant_score": 0.60,
            },
        }
    ]

    replayed = calibrator.replay_variant(
        samples,
        overrides={
            "inference.input_guard.enabled": True,
            "inference.input_guard.plant_min_score": 0.45,
            "inference.input_guard.negative_margin": 0.10,
        },
    )

    assert replayed["samples"][0]["handoff_crop"] is False
    assert replayed["samples"][0]["runtime_gate_reasons"] == ["input_guard_plant_min_score"]
    assert replayed["metrics"]["negative_false_accept_rate"] == 0.0


def test_rank_variants_prefers_eligible_low_false_accept_config():
    baseline = {
        "metrics": {
            "crop_accuracy": 1.0,
            "part_non_unknown_precision": 1.0,
            "negative_false_accept_rate": 0.5,
        }
    }
    variants = [
        {
            "variant_id": "bad",
            "metrics": {
                "negative_false_accept_rate": 0.2,
                "unsupported_part_emissions": 0,
                "wrong_part_rejection_rate": 1.0,
                "crop_accuracy": 1.0,
                "part_non_unknown_precision": 1.0,
                "part_recall": 1.0,
                "abstention_rate": 0.1,
                "mean_latency_ms": 10.0,
            },
        },
        {
            "variant_id": "good",
            "metrics": {
                "negative_false_accept_rate": 0.0,
                "unsupported_part_emissions": 0,
                "wrong_part_rejection_rate": 1.0,
                "crop_accuracy": 1.0,
                "part_non_unknown_precision": 1.0,
                "part_recall": 1.0,
                "abstention_rate": 0.2,
                "mean_latency_ms": 12.0,
            },
        },
    ]

    ranked = calibrator.annotate_and_rank_variants(
        variants,
        baseline=baseline,
        target_negative_false_accept_rate=0.05,
    )

    assert ranked[0]["variant_id"] == "good"
    assert ranked[0]["eligible"] is True
    assert ranked[1]["eligible"] is False


def test_rank_variants_rejects_safety_metric_regressions():
    baseline = {
        "metrics": {
            "crop_accuracy": 0.90,
            "part_non_unknown_precision": 0.90,
            "part_recall": 0.80,
            "wrong_part_rejection_rate": 1.0,
            "p95_latency_ms": 100.0,
        }
    }
    variants = [
        {
            "variant_id": "unsafe_accuracy_gain",
            "metrics": {
                "negative_false_accept_rate": 0.0,
                "unsupported_part_emissions": 0,
                "wrong_part_rejection_rate": 0.70,
                "crop_accuracy": 0.95,
                "part_non_unknown_precision": 0.90,
                "part_recall": 0.70,
                "abstention_rate": 0.1,
                "mean_latency_ms": 100.0,
                "p95_latency_ms": 140.0,
            },
        },
        {
            "variant_id": "safe",
            "metrics": {
                "negative_false_accept_rate": 0.0,
                "unsupported_part_emissions": 0,
                "wrong_part_rejection_rate": 0.99,
                "crop_accuracy": 0.90,
                "part_non_unknown_precision": 0.90,
                "part_recall": 0.79,
                "abstention_rate": 0.2,
                "mean_latency_ms": 90.0,
                "p95_latency_ms": 110.0,
            },
        },
    ]

    ranked = calibrator.annotate_and_rank_variants(variants, baseline=baseline)

    assert ranked[0]["variant_id"] == "safe"
    assert ranked[0]["eligible"] is True
    assert ranked[1]["variant_id"] == "unsafe_accuracy_gain"
    assert ranked[1]["eligible"] is False
    assert ranked[1]["eligibility_reasons"] == [
        "part_recall_drop",
        "wrong_part_rejection_drop",
        "p95_latency_regression",
    ]


def test_select_recommendation_returns_best_rejected_when_no_eligible_candidate():
    ranked = [
        {"variant_id": "baseline", "eligible": False},
        {
            "variant_id": "rejected_a",
            "eligible": False,
            "metrics": {"negative_false_accept_rate": 0.20},
        },
        {
            "variant_id": "rejected_b",
            "eligible": False,
            "metrics": {"negative_false_accept_rate": 0.10},
        },
    ]

    selection = calibrator.select_recommendation(ranked)

    assert selection["recommended"] == {}
    assert selection["best_rejected"]["variant_id"] == "rejected_a"
    assert selection["selection_summary"]["has_eligible_recommendation"] is False
    assert selection["selection_summary"]["best_rejected_variant_id"] == "rejected_a"


def test_build_failure_analysis_summarizes_false_accepts_and_false_rejects():
    analysis = calibrator.build_failure_analysis(
        [
            {
                "group": "non_plant",
                "expected_handoff": False,
                "handoff_crop": True,
                "predicted_crop": "tomato",
                "predicted_part": "leaf",
                "crop_confidence": 0.91,
                "routing_margin": 0.05,
                "crop_confidence_margin": 0.08,
                "runtime_gate_reasons": [],
                "input_guard": {"plant_score": 0.30, "non_plant_score": 0.70},
                "image_path": "d:/sample-1.png",
            },
            {
                "group": "id",
                "expected_handoff": True,
                "handoff_crop": False,
                "predicted_crop": "tomato",
                "predicted_part": "unknown",
                "crop_confidence": 0.60,
                "routing_margin": 0.12,
                "crop_confidence_margin": 0.03,
                "runtime_gate_reasons": ["input_guard_plant_min_score"],
                "image_path": "d:/sample-2.png",
            },
        ]
    )

    assert analysis["false_accept_count"] == 1
    assert analysis["false_accept_counts_by_group"]["non_plant"] == 1
    assert analysis["hardest_false_accept_examples"][0]["image_path"] == "d:/sample-1.png"
    assert any(
        item["cause"] == "input_guard_not_separating_negatives" for item in analysis["false_accept_failure_causes"]
    )
    assert analysis["false_reject_count"] == 1
    assert any(item["cause"] == "guard_too_aggressive" for item in analysis["false_reject_failure_causes"])


def test_adaptive_strategy_runs_on_synthetic_router_eval(tmp_path: Path, monkeypatch):
    # Create minimal router eval layout with one id and one negative
    id_dir = tmp_path / "id" / "tomato" / "leaf"
    id_dir.mkdir(parents=True)
    from PIL import Image as PILImage
    img1 = id_dir / "a.jpg"
    PILImage.new('RGB', (8, 8), color=(255, 0, 0)).save(img1)
    neg_dir = tmp_path / "negatives" / "off_crop" / "bg"
    neg_dir.mkdir(parents=True)
    img2 = neg_dir / "b.jpg"
    PILImage.new('RGB', (8, 8), color=(0, 255, 0)).save(img2)

    # Stub RouterPipeline to avoid heavy model deps
    class DummyDetection:
        def __init__(self):
            self.crop = 'tomato'
            self.part = 'leaf'
            self.crop_confidence = 0.9
            self.part_confidence = 0.9
            self.quality_score = None

        def to_dict(self):
            return {
                'crop': self.crop,
                'part': self.part,
                'crop_confidence': self.crop_confidence,
                'part_confidence': self.part_confidence,
            }

    class DummyAnalysis:
        def __init__(self):
            self.primary_detection = DummyDetection()
            self.detections = [DummyDetection()]
            self.detections_count = 1
            self.message = ''
            self.status = 'ok'
            self.processing_time_ms = 10.0

    class DummyRouter:
        def __init__(self, *args, **kwargs):
            self.config = {}
            self.vlm_config = {}
            self._base_vlm_config = {}

        def load_models(self):
            return None

        def is_ready(self):
            return True

        def analyze_image_result(self, image):
            return DummyAnalysis()
        def set_runtime_profile(self, *args, **kwargs):
            return None

    monkeypatch.setattr('scripts.calibrate_router_surface.RouterPipeline', DummyRouter)

    # Run adaptive calibration with a tiny preset (handoff)
    from scripts.calibrate_router_surface import calibrate_router_surface

    res = calibrate_router_surface(
        tmp_path,
        config_env='colab',
        device='cpu',
        preset='handoff',
        include_current=True,
        max_variants=128,
        strategy='adaptive',
        progress_every=1,
        collect_input_guard_scores=False,
    )

    assert isinstance(res, dict)
    assert 'selection_summary' in res
