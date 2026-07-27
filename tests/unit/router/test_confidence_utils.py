from src.router.confidence_utils import (
    build_open_set_rejection_reasons,
    passes_open_set_gate,
    resolve_effective_confidence_threshold,
)


def _policy_value_from(mapping):
    def _policy_value(stage, key, default):
        return mapping.get((stage, key), default)

    return _policy_value


def test_resolve_effective_confidence_threshold_applies_multiplier_and_clamps():
    policy_value = _policy_value_from(
        {
            ('execution', 'confidence_threshold_multiplier'): 1.5,
            ('execution', 'confidence_threshold_min'): 0.3,
            ('execution', 'confidence_threshold_max'): 0.8,
        }
    )

    value = resolve_effective_confidence_threshold(0.6, policy_value)
    assert value == 0.8


def test_resolve_effective_confidence_threshold_handles_invalid_policy_values():
    policy_value = _policy_value_from(
        {
            ('execution', 'confidence_threshold_multiplier'): 'bad',
            ('execution', 'confidence_threshold_min'): 'bad',
            ('execution', 'confidence_threshold_max'): 'bad',
        }
    )

    value = resolve_effective_confidence_threshold(0.55, policy_value)
    assert value == 0.55


def test_passes_open_set_gate_rejects_unknown_and_low_confidence():
    assert passes_open_set_gate('unknown', 0.99, 0.1) is False
    assert passes_open_set_gate('tomato', 0.49, 0.5) is False
    assert passes_open_set_gate('tomato', 0.5, 0.5) is True


def test_build_open_set_rejection_reasons_reports_unknown_confidence_and_margin():
    reasons = build_open_set_rejection_reasons(
        label="fruit",
        confidence=0.41,
        second_confidence=0.38,
        unknown_confidence=0.46,
        min_confidence=0.40,
        margin_threshold=0.05,
        unknown_label="unknown",
    )

    assert reasons == [
        "unknown_confidence (0.4600) >= confidence (0.4100)",
        "margin (0.0300) < threshold (0.0500)",
    ]
