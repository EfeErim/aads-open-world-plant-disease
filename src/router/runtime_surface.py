"""Runtime-surface helpers for request resolution and analyzer dispatch."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple

from PIL import Image

from src.router.pipeline_flow_utils import resolve_active_analyzer
from src.shared.contracts import RouterAnalysisResult, RouterRequestOptions

RouterAnalyzer = Callable[
    [Image.Image, Tuple[int, int, int], float, Optional[int]],
    Dict[str, Any] | RouterAnalysisResult,
]


def coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def coerce_float(value: Any, default: float) -> float:
    fallback = float(default)
    try:
        return fallback if value is None else float(value)
    except (TypeError, ValueError, OverflowError):
        return fallback


def coerce_non_negative_int(value: Any, default: int = 0) -> int:
    fallback = max(0, int(default))
    try:
        return max(0, int(fallback if value is None else value))
    except (TypeError, ValueError, OverflowError):
        return fallback


def resolve_runtime_controls(config: Dict[str, Any], vlm_config: Dict[str, Any]) -> Dict[str, Any]:
    configured_max = config.get("vlm_max_detections", vlm_config.get("max_detections", 0))
    configured_max_int = coerce_non_negative_int(configured_max, default=0)
    strict_from_env = (
        str(os.getenv("AADS_ULORA_STRICT_MODEL_LOADING", "0")).strip().lower() in {"1", "true", "yes", "on"}
    )
    raw_model_ids = vlm_config.get("model_ids", {})
    configured_ids = raw_model_ids if isinstance(raw_model_ids, dict) else {}
    return {
        "enabled": coerce_bool(config.get("vlm_enabled", vlm_config.get("enabled", False)), default=False),
        "confidence_threshold": config.get("vlm_confidence_threshold", vlm_config.get("confidence_threshold", 0.7)),
        "max_detections": None if configured_max_int <= 0 else configured_max_int,
        "open_set_enabled": coerce_bool(
            config.get("vlm_open_set_enabled", vlm_config.get("open_set_enabled", True)),
            default=True,
        ),
        "open_set_min_confidence": coerce_float(
            config.get("vlm_open_set_min_confidence", vlm_config.get("open_set_min_confidence", 0.55)),
            0.55,
        ),
        "open_set_margin": coerce_float(
            config.get("vlm_open_set_margin", vlm_config.get("open_set_margin", 0.10)),
            0.10,
        ),
        "strict_model_loading": coerce_bool(
            config.get(
                "vlm_strict_model_loading",
                vlm_config.get("strict_model_loading", strict_from_env),
            ),
            default=bool(strict_from_env),
        ),
        "strict_scoring_errors": coerce_bool(
            config.get(
                "vlm_strict_scoring_errors",
                vlm_config.get("strict_scoring_errors", True),
            ),
            default=True,
        ),
        "model_source": config.get("vlm_model_source", vlm_config.get("model_source", "huggingface")),
        "model_ids": {
            "sam": configured_ids.get("sam", "facebook/sam3"),
            "bioclip": configured_ids.get("bioclip", "imageomics/bioclip-2.5-vith14"),
        },
    }


def resolve_request_options(
    *,
    default_confidence_threshold: Any,
    default_max_detections: Optional[int],
    confidence_threshold: Optional[float] = None,
    max_detections: Optional[int] = None,
    options: Optional[RouterRequestOptions] = None,
) -> RouterRequestOptions:
    if options is not None:
        resolved_max = coerce_non_negative_int(options.max_detections, default=0)
        return RouterRequestOptions(
            confidence_threshold=coerce_float(options.confidence_threshold, 0.7),
            max_detections=(None if options.max_detections is None or resolved_max <= 0 else resolved_max),
        )
    resolved_default_threshold = coerce_float(default_confidence_threshold, 0.7)
    resolved_confidence_threshold = (
        resolved_default_threshold
        if confidence_threshold is None
        else coerce_float(confidence_threshold, resolved_default_threshold)
    )
    resolved_max_detections = default_max_detections if max_detections is None else max_detections
    resolved_max_detections_int = coerce_non_negative_int(resolved_max_detections, default=0)
    return RouterRequestOptions(
        confidence_threshold=float(resolved_confidence_threshold),
        max_detections=(
            None if resolved_max_detections is None or resolved_max_detections_int <= 0 else resolved_max_detections_int
        ),
    )


def normalize_router_analysis_result(
    analysis: RouterAnalysisResult | Dict[str, Any],
    *,
    request: RouterRequestOptions,
    status: str = "ok",
    message: str = "",
) -> RouterAnalysisResult:
    if isinstance(analysis, RouterAnalysisResult):
        result = analysis
    else:
        result = RouterAnalysisResult.from_dict(analysis)
    if not result.status:
        result.status = status
    if message and not result.message:
        result.message = message
    result.request = request
    return result


def resolve_router_analyzer(
    actual_pipeline: Optional[str],
    analyzers: Dict[str, RouterAnalyzer],
) -> Optional[RouterAnalyzer]:
    return resolve_active_analyzer(actual_pipeline, analyzers)
