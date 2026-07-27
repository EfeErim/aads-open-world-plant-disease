"""Report-only evidence-gate policy recommendations from calibration artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "v1_evidence_gate_policy_recommendations"
DEFAULT_CALIBRATION_INPUT = Path("docs/ablation_results/dual_view_inference/evidence_gate_calibration.json")
DEFAULT_FAILURE_ANALYSIS_INPUT = Path("docs/ablation_results/dual_view_inference/notebook16_failure_analysis.json")
DEFAULT_JSON_OUTPUT = Path("docs/ablation_results/dual_view_inference/evidence_gate_policy_recommendations.json")
DEFAULT_MARKDOWN_OUTPUT = Path("docs/ablation_results/dual_view_inference/evidence_gate_policy_recommendations.md")


def load_json_payload(path: str | Path) -> dict[str, Any]:
    """Load a JSON object payload."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_optional_json_payload(path: str | Path) -> dict[str, Any]:
    """Load an optional JSON object payload."""
    input_path = Path(path)
    if not input_path.is_file():
        return {}
    return load_json_payload(input_path)


def build_policy_recommendations(
    calibration_payload: dict[str, Any],
    *,
    failure_analysis_payload: dict[str, Any] | None = None,
    calibration_input: str | Path = DEFAULT_CALIBRATION_INPUT,
    failure_analysis_input: str | Path = DEFAULT_FAILURE_ANALYSIS_INPUT,
) -> dict[str, Any]:
    """Build report-only policy recommendations for every calibrated target."""
    target_policies = calibration_payload.get("target_policies")
    if not isinstance(target_policies, dict):
        raise ValueError("Calibration payload must include object-valued 'target_policies'.")
    failure_targets = {}
    if isinstance(failure_analysis_payload, dict):
        maybe_targets = failure_analysis_payload.get("targets")
        failure_targets = maybe_targets if isinstance(maybe_targets, dict) else {}
    target_recommendations = {
        target_id: _target_recommendation(target_id, policy, failure_targets.get(target_id, {}))
        for target_id, policy in sorted(target_policies.items())
        if isinstance(policy, dict)
    }
    categories = {
        "target_specific": _targets_with_status(target_recommendations, "target_specific"),
        "group_fallback": _targets_with_status(target_recommendations, "group_fallback"),
        "global_fallback": _targets_with_status(target_recommendations, "global_fallback"),
        "no_eligible_policy": _targets_with_status(target_recommendations, "no_eligible_policy"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "calibration_input": Path(calibration_input).as_posix(),
        "failure_analysis_input": Path(failure_analysis_input).as_posix(),
        "calibration_schema_version": calibration_payload.get("schema_version"),
        "constraints": calibration_payload.get("constraints", {}),
        "global_policy_status": _policy_status(calibration_payload.get("global_policy")),
        "runtime_recommendation": {
            "status": "advisory_only",
            "reason": "not_runtime_validated",
        },
        "summary": {
            "target_count": len(target_recommendations),
            "target_specific_count": len(categories["target_specific"]),
            "group_fallback_count": len(categories["group_fallback"]),
            "global_fallback_count": len(categories["global_fallback"]),
            "no_eligible_policy_count": len(categories["no_eligible_policy"]),
        },
        "categories": categories,
        "targets": target_recommendations,
        "audit_queue": _audit_queue_payload(calibration_payload),
    }


def render_policy_recommendations_markdown(payload: dict[str, Any]) -> str:
    """Render a compact policy recommendation handoff."""
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# Evidence Gate Policy Recommendations",
        "",
        f"Calibration artifact: `{payload.get('calibration_input')}`",
        f"Failure analysis artifact: `{payload.get('failure_analysis_input')}`",
        "",
        "## Summary",
        "",
        f"- target count: `{summary.get('target_count', 0)}`",
        f"- target-specific candidates: `{summary.get('target_specific_count', 0)}`",
        f"- group fallback candidates: `{summary.get('group_fallback_count', 0)}`",
        f"- global fallback candidates: `{summary.get('global_fallback_count', 0)}`",
        f"- no eligible policy: `{summary.get('no_eligible_policy_count', 0)}`",
        f"- global policy status: `{payload.get('global_policy_status')}`",
        "",
        "## Recommended Report-Only Candidates",
        "",
        "| Target | Recommendation | Source | Threshold | Calibration capture | Holdout capture | Holdout false-positive | Missed wrong | Note |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    targets = payload.get("targets") if isinstance(payload.get("targets"), dict) else {}
    for target_id in _ordered_targets(targets):
        target = targets[target_id]
        if target.get("recommendation") == "audit_required":
            continue
        lines.append(_target_table_row(target_id, target))
    pilot_lines = _tomato_leaf_pilot_lines(targets)
    if pilot_lines:
        lines.extend(["", "## `tomato__leaf` Pilot Decision", "", *pilot_lines])
    lines.extend(
        [
            "",
            "## Audit Required",
            "",
            "| Target | Reason | Wrong | Missed wrong | Calibration note |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for target_id in _ordered_targets(targets):
        target = targets[target_id]
        if target.get("recommendation") != "audit_required":
            continue
        lines.append(
            "| `{target}` | `{reason}` | `{wrong}` | `{missed}` | `{note}` |".format(
                target=target_id,
                reason=target.get("reason", ""),
                wrong=target.get("wrong_count", 0),
                missed=target.get("missed_wrong_count", 0),
                note=target.get("fallback_reason", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- `tomato__leaf` is the current report-only review-gate pilot candidate.",
            "- Treat this as report-only policy guidance.",
            "- Keep full-image adapter prediction as the final decision; use ROI/bbox and v2 calibration only as review/audit signals.",
            "- Do not hardcode per-adapter policy decisions manually.",
            "- Do not change runtime inference without a separate validation and promotion step.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_recommendation_outputs(payload: dict[str, Any], *, json_output: str | Path, markdown_output: str | Path) -> None:
    """Write policy recommendation JSON and Markdown outputs."""
    json_path = Path(json_output)
    markdown_path = Path(markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_policy_recommendations_markdown(payload), encoding="utf-8")


def _target_recommendation(target_id: str, policy_payload: dict[str, Any], failure_target: Any) -> dict[str, Any]:
    status = _policy_status(policy_payload)
    failure = failure_target if isinstance(failure_target, dict) else {}
    policy = policy_payload.get("policy") if isinstance(policy_payload.get("policy"), dict) else {}
    calibration_metrics = _dict_or_empty(policy_payload.get("calibration_metrics"))
    holdout_metrics = _dict_or_empty(policy_payload.get("holdout_metrics"))
    if status in {"target_specific", "group_fallback", "global_fallback"}:
        recommendation = "report_only_candidate"
        reason = "eligible_policy_available_but_not_runtime_validated"
    else:
        recommendation = "audit_required"
        reason = "no_safe_target_group_or_global_policy"
    return {
        "target_id": target_id,
        "recommendation": recommendation,
        "status": status,
        "source": _recommendation_source(policy_payload),
        "group_key": str(policy_payload.get("group_key") or ""),
        "reason": reason,
        "fallback_reason": str(policy_payload.get("fallback_reason") or ""),
        "policy": policy,
        "threshold": policy.get("full_confidence_threshold"),
        "calibration_wrong_capture_rate": calibration_metrics.get("wrong_capture_rate"),
        "calibration_false_positive_review_rate": calibration_metrics.get("false_positive_review_rate"),
        "holdout_wrong_capture_rate": holdout_metrics.get("wrong_capture_rate"),
        "holdout_false_positive_review_rate": holdout_metrics.get("false_positive_review_rate"),
        "holdout_review_rate": holdout_metrics.get("review_rate"),
        "wrong_count": failure.get("wrong_count", _nested_int(policy_payload, ("evidence_status", "total_wrong"))),
        "missed_wrong_count": failure.get("missed_wrong_count"),
        "review_capture_rate": failure.get("review_capture_rate"),
        "false_positive_review_rate": failure.get("false_positive_review_rate"),
    }


def _target_table_row(target_id: str, target: dict[str, Any]) -> str:
    return (
        "| `{target}` | `{recommendation}` | `{source}` | `{threshold}` | `{calibration_capture}` | "
        "`{holdout_capture}` | `{holdout_fp}` | `{missed}` | `{note}` |"
    ).format(
        target=target_id,
        recommendation=target.get("recommendation", ""),
        source=target.get("source", ""),
        threshold=_format_optional_float(target.get("threshold")),
        calibration_capture=_format_optional_float(target.get("calibration_wrong_capture_rate")),
        holdout_capture=_format_optional_float(target.get("holdout_wrong_capture_rate")),
        holdout_fp=_format_optional_float(target.get("holdout_false_positive_review_rate")),
        missed=_format_optional_int(target.get("missed_wrong_count")),
        note=target.get("fallback_reason", ""),
    )


def _tomato_leaf_pilot_lines(targets: dict[str, dict[str, Any]]) -> list[str]:
    target = targets.get("tomato__leaf")
    if not isinstance(target, dict) or target.get("recommendation") != "report_only_candidate":
        return []
    return [
        "- Keep `tomato__leaf` as a report-only candidate; do not promote it into runtime inference in this step.",
        "- Use the selected `0.95` full-confidence threshold only for review-gate analysis and audit prioritization.",
        "- Current evidence: missed wrong `{missed}`, calibration capture `{calibration}`, holdout capture `{holdout}`, "
        "holdout false-positive `{holdout_fp}`.".format(
            missed=_format_optional_int(target.get("missed_wrong_count")),
            calibration=_format_optional_float(target.get("calibration_wrong_capture_rate")),
            holdout=_format_optional_float(target.get("holdout_wrong_capture_rate")),
            holdout_fp=_format_optional_float(target.get("holdout_false_positive_review_rate")),
        ),
    ]


def _ordered_targets(targets: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        targets,
        key=lambda target_id: (
            targets[target_id].get("recommendation") == "audit_required",
            -_int_or_zero(targets[target_id].get("missed_wrong_count")),
            target_id,
        ),
    )


def _targets_with_status(targets: dict[str, dict[str, Any]], status: str) -> list[str]:
    return [target_id for target_id, target in sorted(targets.items()) if target.get("status") == status]


def _recommendation_source(policy_payload: dict[str, Any]) -> str:
    status = _policy_status(policy_payload)
    if status == "group_fallback":
        group_key = str(policy_payload.get("group_key") or "")
        return f"group:{group_key}" if group_key else "group"
    return status


def _audit_queue_payload(calibration_payload: dict[str, Any]) -> list[dict[str, Any]]:
    queue = calibration_payload.get("audit_queue")
    if not isinstance(queue, list):
        return []
    return [dict(item) for item in queue if isinstance(item, dict)]


def _policy_status(policy_payload: Any) -> str:
    if not isinstance(policy_payload, dict):
        return "not_available"
    return str(policy_payload.get("status") or "not_available")


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested_int(payload: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _int_or_none(value)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    parsed = _int_or_none(value)
    return 0 if parsed is None else parsed


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _format_optional_int(value: Any) -> str:
    parsed = _int_or_none(value)
    return "n/a" if parsed is None else str(parsed)
