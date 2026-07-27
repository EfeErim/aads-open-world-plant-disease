from PIL import Image

from src.router.runtime_surface import (
    coerce_non_negative_int,
    normalize_router_analysis_result,
    resolve_request_options,
    resolve_router_analyzer,
    resolve_runtime_controls,
)
from src.shared.contracts import RouterAnalysisResult, RouterRequestOptions


def test_resolve_runtime_controls_normalizes_router_surface():
    controls = resolve_runtime_controls(
        {"vlm_open_set_margin": "0.25"},
        {
            "enabled": True,
            "confidence_threshold": 0.3,
            "max_detections": "5",
            "open_set_enabled": True,
            "open_set_min_confidence": "0.4",
            "model_ids": {"sam": "custom/sam"},
        },
    )

    assert controls["enabled"] is True
    assert controls["confidence_threshold"] == 0.3
    assert controls["max_detections"] == 5
    assert controls["open_set_min_confidence"] == 0.4
    assert controls["open_set_margin"] == 0.25
    assert controls["model_ids"]["sam"] == "custom/sam"


def test_resolve_request_options_uses_defaults_when_not_overridden():
    options = resolve_request_options(
        default_confidence_threshold=0.35,
        default_max_detections=7,
    )

    assert options == RouterRequestOptions(confidence_threshold=0.35, max_detections=7)


def test_resolve_request_options_normalizes_non_positive_max_detections():
    options = resolve_request_options(
        default_confidence_threshold=0.35,
        default_max_detections=7,
        max_detections=0,
    )

    assert options == RouterRequestOptions(confidence_threshold=0.35, max_detections=None)


def test_resolve_runtime_controls_parses_string_booleans():
    controls = resolve_runtime_controls(
        {"vlm_enabled": "false", "vlm_open_set_enabled": "false"},
        {"enabled": True, "open_set_enabled": True},
    )

    assert controls["enabled"] is False
    assert controls["open_set_enabled"] is False


def test_coerce_non_negative_int_clamps_negative_default_on_parse_failure():
    assert coerce_non_negative_int("not-an-int", default=-4) == 0


def test_normalize_router_analysis_result_attaches_request_contract():
    request = RouterRequestOptions(confidence_threshold=0.2, max_detections=3)

    result = normalize_router_analysis_result(
        {"detections": [{"crop": "tomato", "crop_confidence": 0.9}]},
        request=request,
    )

    assert isinstance(result, RouterAnalysisResult)
    assert result.request == request
    assert result.primary_detection is not None
    assert result.primary_detection.crop == "tomato"


def test_resolve_router_analyzer_returns_active_callable():
    def _sam3(_image: Image.Image, _image_size, _threshold: float, _max_detections):
        return {"detections": []}

    def _other(_image: Image.Image, _image_size, _threshold: float, _max_detections):
        return {"detections": [{"crop": "potato"}]}

    selected = resolve_router_analyzer("other", {"sam3": _sam3, "other": _other})

    assert selected is _other
