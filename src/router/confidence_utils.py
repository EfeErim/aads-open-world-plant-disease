import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _clamp_unit_interval(value: Any, *, default: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        logger.debug(
            f"Failed to coerce value to float; using default {default}",
            exc_info=exc,
        )
        resolved = float(default)
    return max(0.0, min(1.0, resolved))


def _coerce_non_negative_float(value: Any, *, default: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        logger.debug(
            f"Failed to coerce value to float; using default {default}",
            exc_info=exc,
        )
        resolved = float(default)
    return max(0.0, resolved)


def resolve_effective_confidence_threshold(
    confidence_threshold: float,
    policy_value_fn: Callable[[str, str, Any], Any],
) -> float:
    """Resolve profile/policy-adjusted confidence threshold with optional clamps."""
    base_threshold = _clamp_unit_interval(confidence_threshold, default=0.0)

    multiplier_raw = policy_value_fn('execution', 'confidence_threshold_multiplier', 1.0)
    multiplier = _coerce_non_negative_float(multiplier_raw, default=1.0)

    adjusted = base_threshold * multiplier

    min_raw = policy_value_fn('execution', 'confidence_threshold_min', 0.0)
    max_raw = policy_value_fn('execution', 'confidence_threshold_max', 1.0)
    min_threshold = _clamp_unit_interval(min_raw, default=0.0)
    max_threshold = max(min_threshold, _clamp_unit_interval(max_raw, default=1.0))
    return max(min_threshold, min(max_threshold, adjusted))


def passes_open_set_gate(crop_label: str, crop_confidence: float, min_confidence: float) -> bool:
    """Evaluate open-set acceptance for a classified detection."""
    if str(crop_label).strip().lower() == 'unknown':
        return False
    return float(crop_confidence) >= float(min_confidence)


def build_open_set_rejection_reasons(
    *,
    label: str,
    confidence: float,
    second_confidence: float,
    unknown_confidence: float,
    min_confidence: float,
    margin_threshold: float,
    unknown_label: str = "unknown",
) -> list[str]:
    """Return ordered rejection reasons for a label-vs-unknown open-set decision."""
    resolved_label = str(label).strip().lower()
    resolved_unknown = str(unknown_label).strip().lower() or "unknown"
    reasons: list[str] = []
    if resolved_label == resolved_unknown:
        reasons.append(f"label resolved to {resolved_unknown}")

    best_confidence = float(confidence)
    unknown_score = float(unknown_confidence)
    if _unknown_dominates(unknown_score, best_confidence):
        reasons.append(f"unknown_confidence ({unknown_score:.4f}) >= confidence ({best_confidence:.4f})")

    threshold = _clamp_unit_interval(min_confidence, default=0.0)
    if best_confidence < threshold:
        reasons.append(f"confidence ({best_confidence:.4f}) < threshold ({threshold:.4f})")

    margin_floor = _coerce_non_negative_float(margin_threshold, default=0.0)
    margin = best_confidence - float(second_confidence)
    if _margin_too_small(margin, margin_floor):
        reasons.append(f"margin ({margin:.4f}) < threshold ({margin_floor:.4f})")
    return reasons


def _unknown_dominates(unknown_score: float, best_confidence: float) -> bool:
    return unknown_score >= best_confidence


def _margin_too_small(margin: float, floor: float) -> bool:
    return float(margin) < float(floor)
