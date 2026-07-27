"""Target-aware calibration helpers for Notebook 16 evidence-gate reports."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.pipeline.roi_evidence_analysis import target_key as resolve_target_key

SCHEMA_VERSION = "v1_evidence_gate_calibration"
V2_SCHEMA_VERSION = "v2_evidence_gate_calibration"
DEFAULT_SOURCE_REPORT = Path("docs/ablation_results/dual_view_inference/multi_target_report.json")
DEFAULT_OUTPUT = Path("docs/ablation_results/dual_view_inference/evidence_gate_calibration.json")
DEFAULT_SEED = 20260613
ROI_QUALITY_BAD = {"roi_too_large", "roi_too_small"}
V1_CONFIDENCE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]
V2_CONFIDENCE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]


def load_notebook16_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load rows from a Notebook 16 multi-target report."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Expected a report with a list-valued 'rows' field: {path}")
    return [dict(row) for row in rows]


def calibrate_evidence_gate_report(
    rows: Iterable[dict[str, Any]],
    *,
    source_report: str | Path = DEFAULT_SOURCE_REPORT,
    schema_version: str = SCHEMA_VERSION,
    min_capture: float = 0.70,
    max_false_positive_rate: float = 0.15,
    min_target_errors: int = 20,
    holdout_ratio: float = 0.30,
    seed: int = DEFAULT_SEED,
    include_samples: bool = False,
    max_review_rate: float = 0.25,
    min_calibration_errors: int = 10,
    min_holdout_errors: int = 5,
    max_holdout_capture_drop: float = 0.15,
    max_holdout_fp_increase: float = 0.10,
) -> dict[str, Any]:
    """Build a global plus per-target advisory evidence-gate calibration payload."""
    if schema_version not in {SCHEMA_VERSION, V2_SCHEMA_VERSION, "v1", "v2"}:
        raise ValueError("schema_version must be one of: v1, v2, v1_evidence_gate_calibration, v2_evidence_gate_calibration.")
    resolved_schema_version = V2_SCHEMA_VERSION if schema_version == "v2" else SCHEMA_VERSION if schema_version == "v1" else schema_version
    normalized_rows = [_normalize_row(row) for row in rows]
    calibration_rows, holdout_rows = split_rows(normalized_rows, holdout_ratio=holdout_ratio, seed=seed)
    constraints = {
        "min_capture": float(min_capture),
        "max_false_positive_rate": float(max_false_positive_rate),
        "min_target_errors": int(min_target_errors),
        "holdout_ratio": float(holdout_ratio),
        "seed": int(seed),
    }
    if resolved_schema_version == V2_SCHEMA_VERSION:
        constraints.update(
            {
                "max_review_rate": float(max_review_rate),
                "min_calibration_errors": int(min_calibration_errors),
                "min_holdout_errors": int(min_holdout_errors),
                "max_holdout_capture_drop": float(max_holdout_capture_drop),
                "max_holdout_fp_increase": float(max_holdout_fp_increase),
            }
        )

    if resolved_schema_version == V2_SCHEMA_VERSION:
        payload = _calibrate_v2_report(
            normalized_rows,
            calibration_rows,
            holdout_rows,
            source_report=source_report,
            constraints=constraints,
            include_samples=include_samples,
        )
        return payload

    global_selection = select_policy(
        calibration_rows,
        min_capture=min_capture,
        max_false_positive_rate=max_false_positive_rate,
    )
    global_policy = _policy_payload(global_selection, holdout_rows)

    rows_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    calibration_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    holdout_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        rows_by_target[row["target_key"]].append(row)
    for row in calibration_rows:
        calibration_by_target[row["target_key"]].append(row)
    for row in holdout_rows:
        holdout_by_target[row["target_key"]].append(row)

    target_policies = {}
    for target_key, target_rows in sorted(rows_by_target.items()):
        target_calibration_rows = calibration_by_target.get(target_key, [])
        target_holdout_rows = holdout_by_target.get(target_key, [])
        target_policies[target_key] = _target_policy_payload(
            target_rows,
            target_calibration_rows,
            target_holdout_rows,
            global_selection=global_selection,
            min_capture=min_capture,
            max_false_positive_rate=max_false_positive_rate,
            min_target_errors=min_target_errors,
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_report": _as_posix(source_report),
        "constraints": constraints,
        "global_policy": global_policy,
        "target_policies": target_policies,
    }
    if include_samples:
        split_rows_with_labels = [*calibration_rows, *holdout_rows]
        payload["samples"] = [
            {
                "image_path": row["image_path"],
                "target_key": row["target_key"],
                "split": row["split"],
                "expected_label": row["expected_label"],
                "diagnosis": row["diagnosis"],
                "is_correct": row["is_correct"],
            }
            for row in split_rows_with_labels
        ]
    return payload


def write_calibration_report(payload: dict[str, Any], path: str | Path) -> None:
    """Write a calibration report with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def split_rows(
    rows: Sequence[dict[str, Any]],
    *,
    holdout_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows deterministically with a stable image-path plus target-key hash."""
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be between 0 and 1.")
    calibration_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        hash_input = f"{seed}|{item.get('target_key', '')}|{item.get('image_path', '')}"
        value = int(hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
        item["split"] = "holdout" if value < holdout_ratio else "calibration"
        if item["split"] == "holdout":
            holdout_rows.append(item)
        else:
            calibration_rows.append(item)
    return calibration_rows, holdout_rows


def select_policy(
    rows: Sequence[dict[str, Any]],
    *,
    min_capture: float,
    max_false_positive_rate: float,
    max_review_rate: float | None = None,
    confidence_thresholds: Sequence[float] = V1_CONFIDENCE_THRESHOLDS,
) -> dict[str, Any]:
    """Select the highest-coverage eligible policy from the deterministic grid."""
    variants = [evaluate_policy(rows, policy) for policy in candidate_policies(confidence_thresholds=confidence_thresholds)]
    eligible = [
        variant
        for variant in variants
        if _rate_at_least(variant["metrics"]["wrong_capture_rate"], min_capture)
        and _rate_at_most(variant["metrics"]["false_positive_review_rate"], max_false_positive_rate)
        and (max_review_rate is None or _rate_at_most(variant["metrics"]["review_rate"], max_review_rate))
    ]
    if eligible:
        selected = sorted(eligible, key=_eligible_sort_key)[0]
        return {
            "status": "eligible",
            "selected": selected,
            "eligible_count": len(eligible),
            "evaluated_count": len(variants),
        }

    best_rejected = sorted(variants, key=_rejected_sort_key)[0] if variants else None
    return {
        "status": "no_eligible_policy",
        "best_rejected": best_rejected,
        "eligible_count": 0,
        "evaluated_count": len(variants),
    }


def evaluate_policy(rows: Sequence[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one candidate policy over normalized rows."""
    normalized_rows = [_ensure_normalized_row(row) for row in rows]
    reviewed = 0
    wrong_reviewed = 0
    wrong_count = 0
    correct_reviewed = 0
    correct_count = 0
    comparable_count = 0
    for row in normalized_rows:
        should_review = policy_requires_review(row, policy)
        if should_review:
            reviewed += 1
        if row["is_comparable"]:
            comparable_count += 1
            if row["is_correct"]:
                correct_count += 1
                if should_review:
                    correct_reviewed += 1
            else:
                wrong_count += 1
                if should_review:
                    wrong_reviewed += 1

    sample_count = len(normalized_rows)
    metrics = {
        "sample_count": sample_count,
        "comparable_count": comparable_count,
        "wrong_count": wrong_count,
        "correct_count": correct_count,
        "review_count": reviewed,
        "accepted_count": sample_count - reviewed,
        "review_rate": _safe_rate(reviewed, sample_count),
        "coverage": _safe_rate(sample_count - reviewed, sample_count),
        "wrong_capture_rate": _safe_rate(wrong_reviewed, wrong_count),
        "false_positive_review_rate": _safe_rate(correct_reviewed, correct_count),
        "wrong_missed_count": wrong_count - wrong_reviewed,
        "correct_reviewed_count": correct_reviewed,
    }
    return {"policy": dict(policy), "metrics": metrics}


def policy_requires_review(row: dict[str, Any], policy: dict[str, Any]) -> bool:
    """Return whether a normalized row should be reviewed under a candidate policy."""
    row = _ensure_normalized_row(row)
    if row["full_confidence"] < float(policy["full_confidence_threshold"]):
        return True
    if policy["review_on_roi_conflict"] and row["roi_evidence_status"] == "conflicts_with_full":
        return True
    if policy["review_on_roi_quality_bad"] and row["roi_quality_status"] in ROI_QUALITY_BAD:
        return True
    if policy["review_on_full_ood"] and row["full_ood_is_ood"]:
        return True
    return bool(policy["review_on_roi_ood"] and row["roi_ood_is_ood"])


def candidate_policies(
    *,
    confidence_thresholds: Sequence[float] = V1_CONFIDENCE_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Return the small auditable v1 candidate grid."""
    policies: list[dict[str, Any]] = []
    for threshold, roi_conflict, roi_quality_bad, full_ood, roi_ood in itertools.product(
        confidence_thresholds,
        [False, True],
        [False, True],
        [False, True],
        [False, True],
    ):
        policies.append(
            {
                "full_confidence_threshold": threshold,
                "review_on_roi_conflict": roi_conflict,
                "review_on_roi_quality_bad": roi_quality_bad,
                "review_on_full_ood": full_ood,
                "review_on_roi_ood": roi_ood,
            }
        )
    return policies


def _calibrate_v2_report(
    normalized_rows: Sequence[dict[str, Any]],
    calibration_rows: Sequence[dict[str, Any]],
    holdout_rows: Sequence[dict[str, Any]],
    *,
    source_report: str | Path,
    constraints: dict[str, Any],
    include_samples: bool,
) -> dict[str, Any]:
    global_selection = _select_policy_v2(calibration_rows, constraints)
    global_policy = _policy_payload_with_stability(global_selection, holdout_rows, constraints)
    group_members = _build_group_members(normalized_rows)
    group_policies = _build_group_policy_payloads(group_members, calibration_rows, holdout_rows, constraints)

    rows_by_target = _group_rows_by(normalized_rows, "target_key")
    calibration_by_target = _group_rows_by(calibration_rows, "target_key")
    holdout_by_target = _group_rows_by(holdout_rows, "target_key")
    target_policies: dict[str, dict[str, Any]] = {}
    audit_queue: list[dict[str, Any]] = []
    for target_key, target_rows in sorted(rows_by_target.items()):
        target_policy = _target_policy_payload_v2(
            target_key,
            target_rows,
            calibration_by_target.get(target_key, []),
            holdout_by_target.get(target_key, []),
            group_policies=group_policies,
            global_policy=global_policy,
            constraints=constraints,
        )
        target_policies[target_key] = target_policy
        audit_item = _audit_queue_item(target_key, target_rows, target_policy)
        if audit_item:
            audit_queue.append(audit_item)

    payload: dict[str, Any] = {
        "schema_version": V2_SCHEMA_VERSION,
        "source_report": _as_posix(source_report),
        "constraints": constraints,
        "policy_families": {
            "confidence_threshold_only": {
                "thresholds": V2_CONFIDENCE_THRESHOLDS,
                "description": "review when full_confidence is below threshold",
            },
            "confidence_roi_ood_grid": {
                "candidate_count": len(candidate_policies(confidence_thresholds=V2_CONFIDENCE_THRESHOLDS)),
                "description": "confidence thresholds combined with ROI conflict, ROI quality, full OOD, and ROI OOD gates",
            },
        },
        "global_policy": global_policy,
        "group_policies": group_policies,
        "target_policies": target_policies,
        "risk_coverage_curves": _risk_coverage_curves(normalized_rows, group_members),
        "calibration_diagnostics": _calibration_diagnostics(normalized_rows, group_members),
        "audit_queue": audit_queue,
        "runtime_recommendation": {
            "status": "advisory_only",
            "reason": "not_runtime_validated",
        },
    }
    if include_samples:
        payload["samples"] = _sample_split_payload([*calibration_rows, *holdout_rows])
    return payload


def _select_policy_v2(rows: Sequence[dict[str, Any]], constraints: dict[str, Any]) -> dict[str, Any]:
    return select_policy(
        rows,
        min_capture=float(constraints["min_capture"]),
        max_false_positive_rate=float(constraints["max_false_positive_rate"]),
        max_review_rate=float(constraints["max_review_rate"]),
        confidence_thresholds=V2_CONFIDENCE_THRESHOLDS,
    )


def _policy_payload_with_stability(
    selection: dict[str, Any],
    holdout_rows: Sequence[dict[str, Any]],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    payload = _policy_payload(selection, holdout_rows)
    if payload["status"] != "eligible":
        payload["holdout_stability"] = {"status": "not_evaluated", "reason": "no_eligible_calibration_policy"}
        return payload
    payload["holdout_stability"] = _holdout_stability_payload(
        payload["calibration_metrics"],
        payload["holdout_metrics"],
        constraints,
    )
    if payload["holdout_stability"]["status"] != "passed":
        payload["status"] = "no_eligible_policy"
        payload["rejection_reason"] = "holdout_stability_failed"
    return payload


def _holdout_stability_payload(
    calibration_metrics: dict[str, Any],
    holdout_metrics: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    holdout_wrong = int(holdout_metrics.get("wrong_count") or 0)
    if holdout_wrong < int(constraints["min_holdout_errors"]):
        return {
            "status": "failed",
            "reason": "holdout_wrong_count_below_min_holdout_errors",
            "holdout_wrong": holdout_wrong,
        }
    calibration_capture = _rate_or_zero(calibration_metrics.get("wrong_capture_rate"))
    holdout_capture = _rate_or_zero(holdout_metrics.get("wrong_capture_rate"))
    calibration_fp = _rate_or_zero(calibration_metrics.get("false_positive_review_rate"))
    holdout_fp = _rate_or_zero(holdout_metrics.get("false_positive_review_rate"))
    capture_drop = calibration_capture - holdout_capture
    fp_increase = holdout_fp - calibration_fp
    issues = []
    if capture_drop > float(constraints["max_holdout_capture_drop"]):
        issues.append("holdout_capture_drop_exceeds_limit")
    if fp_increase > float(constraints["max_holdout_fp_increase"]):
        issues.append("holdout_false_positive_increase_exceeds_limit")
    return {
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "capture_drop": capture_drop,
        "false_positive_increase": fp_increase,
    }


def _build_group_policy_payloads(
    group_members: dict[str, list[str]],
    calibration_rows: Sequence[dict[str, Any]],
    holdout_rows: Sequence[dict[str, Any]],
    constraints: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for group_key, members in sorted(group_members.items()):
        member_set = set(members)
        group_calibration_rows = [row for row in calibration_rows if row["target_key"] in member_set]
        group_holdout_rows = [row for row in holdout_rows if row["target_key"] in member_set]
        selection = _select_policy_v2(group_calibration_rows, constraints)
        payload = _policy_payload_with_stability(selection, group_holdout_rows, constraints)
        payload["members"] = sorted(member_set)
        payloads[group_key] = payload
    return payloads


def _target_policy_payload_v2(
    target_key: str,
    target_rows: Sequence[dict[str, Any]],
    calibration_rows: Sequence[dict[str, Any]],
    holdout_rows: Sequence[dict[str, Any]],
    *,
    group_policies: dict[str, dict[str, Any]],
    global_policy: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    total_wrong = _wrong_count(target_rows)
    calibration_wrong = _wrong_count(calibration_rows)
    holdout_wrong = _wrong_count(holdout_rows)
    evidence_status = {
        "total_wrong": total_wrong,
        "calibration_wrong": calibration_wrong,
        "holdout_wrong": holdout_wrong,
    }
    if (
        total_wrong >= int(constraints["min_target_errors"])
        and calibration_wrong >= int(constraints["min_calibration_errors"])
        and holdout_wrong >= int(constraints["min_holdout_errors"])
    ):
        target_selection = _select_policy_v2(calibration_rows, constraints)
        target_payload = _policy_payload_with_stability(target_selection, holdout_rows, constraints)
        if target_payload["status"] == "eligible":
            target_payload.update({"status": "target_specific", "fallback_reason": "", "evidence_status": evidence_status})
            return target_payload
        evidence_status["target_rejection_reason"] = target_payload.get("rejection_reason", target_payload["status"])

    for group_key in _fallback_group_keys(target_key):
        group_payload = group_policies.get(group_key, {})
        if group_payload.get("status") == "eligible":
            return {
                "status": "group_fallback",
                "group_key": group_key,
                "policy": group_payload["policy"],
                "calibration_metrics": group_payload["calibration_metrics"],
                "holdout_metrics": evaluate_policy(holdout_rows, group_payload["policy"])["metrics"],
                "holdout_stability": group_payload["holdout_stability"],
                "fallback_reason": _fallback_reason(evidence_status, constraints),
                "evidence_status": evidence_status,
            }

    if global_policy.get("status") == "eligible":
        return {
            "status": "global_fallback",
            "policy": global_policy["policy"],
            "calibration_metrics": global_policy["calibration_metrics"],
            "holdout_metrics": evaluate_policy(holdout_rows, global_policy["policy"])["metrics"],
            "holdout_stability": global_policy["holdout_stability"],
            "fallback_reason": _fallback_reason(evidence_status, constraints),
            "evidence_status": evidence_status,
        }

    return {
        "status": "no_eligible_policy",
        "policy": {},
        "calibration_metrics": {},
        "holdout_metrics": {},
        "holdout_stability": {"status": "not_evaluated", "reason": "no_safe_target_group_or_global_policy"},
        "fallback_reason": _fallback_reason(evidence_status, constraints),
        "best_rejected": global_policy.get("best_rejected"),
        "evidence_status": evidence_status,
    }


def _risk_coverage_curves(
    rows: Sequence[dict[str, Any]],
    group_members: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    curves = {"global": _risk_coverage_curve(rows)}
    for group_key, members in sorted(group_members.items()):
        member_set = set(members)
        curves[group_key] = _risk_coverage_curve([row for row in rows if row["target_key"] in member_set])
    return curves


def _risk_coverage_curve(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    curve = []
    for threshold in V2_CONFIDENCE_THRESHOLDS:
        policy = {
            "full_confidence_threshold": threshold,
            "review_on_roi_conflict": False,
            "review_on_roi_quality_bad": False,
            "review_on_full_ood": False,
            "review_on_roi_ood": False,
        }
        row = evaluate_policy(rows, policy)
        curve.append({"threshold": threshold, **row["metrics"]})
    return curve


def _calibration_diagnostics(
    rows: Sequence[dict[str, Any]],
    group_members: dict[str, list[str]],
) -> dict[str, Any]:
    diagnostics = {"global": _diagnostic_payload(rows)}
    for group_key, members in sorted(group_members.items()):
        member_set = set(members)
        diagnostics[group_key] = _diagnostic_payload([row for row in rows if row["target_key"] in member_set])
    return diagnostics


def _diagnostic_payload(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    comparable = [row for row in rows if row["is_comparable"]]
    wrong = [row for row in comparable if not row["is_correct"]]
    correct = [row for row in comparable if row["is_correct"]]
    high_conf_wrong = [row for row in wrong if row["full_confidence"] >= 0.90]
    return {
        "sample_count": len(rows),
        "comparable_count": len(comparable),
        "wrong_count": len(wrong),
        "correct_count": len(correct),
        "error_rate": _safe_rate(len(wrong), len(comparable)),
        "high_confidence_wrong_count": len(high_conf_wrong),
        "high_confidence_wrong_rate": _safe_rate(len(high_conf_wrong), len(wrong)),
        "mean_wrong_confidence": _mean(row["full_confidence"] for row in wrong),
        "mean_correct_confidence": _mean(row["full_confidence"] for row in correct),
    }


def _audit_queue_item(
    target_key: str,
    rows: Sequence[dict[str, Any]],
    target_policy: dict[str, Any],
) -> dict[str, Any] | None:
    reasons: list[str] = []
    if target_policy["status"] == "no_eligible_policy":
        reasons.append("no_calibration_policy_meets_constraints")
    evidence_status = target_policy.get("evidence_status", {})
    if int(evidence_status.get("total_wrong") or 0) < 20:
        reasons.append("target_error_count_too_small_for_reliable_policy_selection")
    diagnostics = _diagnostic_payload(rows)
    if _rate_or_zero(diagnostics["high_confidence_wrong_rate"]) >= 0.50:
        reasons.append("high_confidence_errors_dominate")
    if not reasons:
        return None
    return {
        "target_id": target_key,
        "reasons": reasons,
        "sample_count": diagnostics["sample_count"],
        "wrong_count": diagnostics["wrong_count"],
        "policy_status": target_policy["status"],
    }


def _build_group_members(rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    targets = sorted({row["target_key"] for row in rows})
    groups: dict[str, set[str]] = defaultdict(set)
    for target_key in targets:
        crop, part = _split_target_key(target_key)
        groups[f"{part}_targets"].add(target_key)
        groups[f"{crop}_targets"].add(target_key)
        groups[f"part:{part}"].add(target_key)
        groups[f"crop:{crop}"].add(target_key)
    return {key: sorted(value) for key, value in groups.items() if value}


def _fallback_group_keys(target_key: str) -> list[str]:
    crop, part = _split_target_key(target_key)
    return [f"{part}_targets", f"part:{part}", f"{crop}_targets", f"crop:{crop}"]


def _fallback_reason(evidence_status: dict[str, Any], constraints: dict[str, Any]) -> str:
    total_wrong = int(evidence_status.get("total_wrong") or 0)
    calibration_wrong = int(evidence_status.get("calibration_wrong") or 0)
    holdout_wrong = int(evidence_status.get("holdout_wrong") or 0)
    if total_wrong < int(constraints["min_target_errors"]):
        return f"target_wrong_count_below_min_target_errors:{total_wrong}<{constraints['min_target_errors']}"
    if calibration_wrong < int(constraints["min_calibration_errors"]):
        return f"target_calibration_wrong_count_below_min_calibration_errors:{calibration_wrong}<{constraints['min_calibration_errors']}"
    if holdout_wrong < int(constraints["min_holdout_errors"]):
        return f"target_holdout_wrong_count_below_min_holdout_errors:{holdout_wrong}<{constraints['min_holdout_errors']}"
    return "target_selection_no_eligible_or_unstable_policy"


def _group_rows_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


def _sample_split_payload(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "image_path": row["image_path"],
            "target_key": row["target_key"],
            "split": row["split"],
            "expected_label": row["expected_label"],
            "diagnosis": row["diagnosis"],
            "is_correct": row["is_correct"],
        }
        for row in rows
    ]


def _wrong_count(rows: Sequence[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row["is_comparable"] and not row["is_correct"])


def _target_policy_payload(
    target_rows: Sequence[dict[str, Any]],
    calibration_rows: Sequence[dict[str, Any]],
    holdout_rows: Sequence[dict[str, Any]],
    *,
    global_selection: dict[str, Any],
    min_capture: float,
    max_false_positive_rate: float,
    min_target_errors: int,
) -> dict[str, Any]:
    total_wrong = sum(1 for row in target_rows if row["is_comparable"] and not row["is_correct"])
    calibration_wrong = sum(1 for row in calibration_rows if row["is_comparable"] and not row["is_correct"])
    holdout_wrong = sum(1 for row in holdout_rows if row["is_comparable"] and not row["is_correct"])
    if total_wrong < min_target_errors:
        return _global_fallback_payload(
            global_selection,
            holdout_rows,
            fallback_reason=f"target_wrong_count_below_min_target_errors:{total_wrong}<{min_target_errors}",
        )
    if calibration_wrong < 1 or holdout_wrong < 1:
        return _global_fallback_payload(
            global_selection,
            holdout_rows,
            fallback_reason="target_split_missing_wrong_examples",
        )

    target_selection = select_policy(
        calibration_rows,
        min_capture=min_capture,
        max_false_positive_rate=max_false_positive_rate,
    )
    if target_selection["status"] == "eligible":
        selected = target_selection["selected"]
        return {
            "status": "target_specific",
            "policy": selected["policy"],
            "calibration_metrics": selected["metrics"],
            "holdout_metrics": evaluate_policy(holdout_rows, selected["policy"])["metrics"],
            "fallback_reason": "",
        }
    fallback = _global_fallback_payload(
        global_selection,
        holdout_rows,
        fallback_reason="target_selection_no_eligible_policy",
    )
    fallback["target_best_rejected"] = target_selection.get("best_rejected")
    return fallback


def _global_fallback_payload(
    global_selection: dict[str, Any],
    holdout_rows: Sequence[dict[str, Any]],
    *,
    fallback_reason: str,
) -> dict[str, Any]:
    if global_selection["status"] != "eligible":
        return {
            "status": "no_eligible_policy",
            "policy": {},
            "calibration_metrics": {},
            "holdout_metrics": {},
            "fallback_reason": fallback_reason,
            "best_rejected": global_selection.get("best_rejected"),
        }
    selected = global_selection["selected"]
    return {
        "status": "global_fallback",
        "policy": selected["policy"],
        "calibration_metrics": selected["metrics"],
        "holdout_metrics": evaluate_policy(holdout_rows, selected["policy"])["metrics"],
        "fallback_reason": fallback_reason,
    }


def _policy_payload(selection: dict[str, Any], holdout_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if selection["status"] != "eligible":
        return {
            "status": "no_eligible_policy",
            "policy": {},
            "calibration_metrics": {},
            "holdout_metrics": {},
            "best_rejected": selection.get("best_rejected"),
        }
    selected = selection["selected"]
    return {
        "status": "eligible",
        "policy": selected["policy"],
        "calibration_metrics": selected["metrics"],
        "holdout_metrics": evaluate_policy(holdout_rows, selected["policy"])["metrics"],
    }


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    expected = _normalize_label(row.get("expected_label"))
    diagnosis = _normalize_label(row.get("diagnosis") or row.get("full_diagnosis"))
    resolved_target_key = resolve_target_key(row)
    crop, part = _split_target_key(resolved_target_key)
    is_comparable = bool(expected and diagnosis)
    return {
        "image_path": str(row.get("image_path") or ""),
        "target_key": resolved_target_key,
        "crop": crop,
        "part": part,
        "expected_label": row.get("expected_label"),
        "diagnosis": row.get("diagnosis") or row.get("full_diagnosis"),
        "is_comparable": is_comparable,
        "is_correct": bool(is_comparable and expected == diagnosis),
        "full_confidence": _to_float(row.get("full_confidence", row.get("confidence")), default=0.0),
        "roi_evidence_status": _normalize_label(row.get("roi_evidence_status")),
        "roi_quality_status": _normalize_label(row.get("roi_quality_status")),
        "full_ood_is_ood": _is_true(row.get("full_ood_is_ood") if "full_ood_is_ood" in row else row.get("ood_is_ood")),
        "roi_ood_is_ood": _is_true(row.get("roi_ood_is_ood")),
    }


def _ensure_normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    if "is_comparable" in row and "target_key" in row:
        return dict(row)
    return _normalize_row(row)


def _split_target_key(target_key: str) -> tuple[str, str]:
    if "__" not in target_key:
        return target_key, "unknown"
    crop, part = target_key.split("__", 1)
    return crop or "unknown", part or "unknown"


def _eligible_sort_key(variant: dict[str, Any]) -> tuple[float, float, float, int, float]:
    metrics = variant["metrics"]
    policy = variant["policy"]
    return (
        -_rate_or_zero(metrics["coverage"]),
        -_rate_or_zero(metrics["wrong_capture_rate"]),
        _rate_or_one(metrics["false_positive_review_rate"]),
        _policy_complexity(policy),
        float(policy["full_confidence_threshold"]),
    )


def _rejected_sort_key(variant: dict[str, Any]) -> tuple[float, float, float, int, float]:
    metrics = variant["metrics"]
    policy = variant["policy"]
    return (
        -_rate_or_zero(metrics["wrong_capture_rate"]),
        _rate_or_one(metrics["false_positive_review_rate"]),
        -_rate_or_zero(metrics["coverage"]),
        _policy_complexity(policy),
        float(policy["full_confidence_threshold"]),
    )


def _policy_complexity(policy: dict[str, Any]) -> int:
    return sum(
        int(bool(policy[key]))
        for key in (
            "review_on_roi_conflict",
            "review_on_roi_quality_bad",
            "review_on_full_ood",
            "review_on_roi_ood",
        )
    )


def _rate_at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _rate_at_most(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def _rate_or_zero(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def _rate_or_one(value: float | None) -> float:
    return 1.0 if value is None else float(value)


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return (numerator / denominator) if denominator else None


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


def _to_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _normalize_label(value) in {"1", "true", "yes", "y"}


def _as_posix(path: str | Path) -> str:
    return Path(path).as_posix()
