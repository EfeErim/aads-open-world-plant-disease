"""Failure analysis helpers for ROI evidence-gate reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

FAILURE_BUCKETS = ("router", "bbox", "adapter", "confidence_ood", "review_gate")


def analyze_dual_view_evidence_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize router, bbox, adapter, confidence/OOD, and review-gate failure signals."""
    rows_list = [dict(row) for row in rows]
    row_analyses = [classify_dual_view_evidence_row(row) for row in rows_list]
    bucket_counts = Counter(bucket for item in row_analyses for bucket in item["failure_buckets"])
    reason_counts = Counter(reason for item in row_analyses for reason in item["reasons"])
    comparable = [item for item in row_analyses if item["is_comparable"]]
    incorrect = [item for item in comparable if not item["is_correct"]]
    correct = [item for item in comparable if item["is_correct"]]
    review_rows = [item for item in row_analyses if item["requires_review"]]

    per_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in row_analyses:
        per_target[item["target_key"]].append(item)

    return {
        "schema_version": "v1_dual_view_evidence_failure_analysis",
        "sample_count": len(row_analyses),
        "comparable_count": len(comparable),
        "incorrect_count": len(incorrect),
        "review_count": len(review_rows),
        "bucket_counts": {bucket: int(bucket_counts.get(bucket, 0)) for bucket in FAILURE_BUCKETS},
        "reason_counts": dict(sorted(reason_counts.items())),
        "review_capture_rate_on_errors": _safe_rate(
            sum(1 for item in incorrect if item["requires_review"]),
            len(incorrect),
        ),
        "review_false_positive_rate_on_correct": _safe_rate(
            sum(1 for item in correct if item["requires_review"]),
            len(correct),
        ),
        "per_target": {
            target: _summarize_target_items(items)
            for target, items in sorted(per_target.items())
        },
        "rows": row_analyses,
    }


def classify_dual_view_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    """Classify one Notebook 16 dual-view row into actionable failure buckets."""
    expected = _normalize_label(row.get("expected_label"))
    diagnosis = _normalize_label(row.get("diagnosis") or row.get("full_diagnosis"))
    is_comparable = bool(expected and diagnosis)
    is_correct = bool(is_comparable and expected == diagnosis)
    reasons: list[str] = []
    buckets: list[str] = []

    router_crop = _normalize_label(row.get("router_crop") or row.get("router_primary_crop"))
    router_part = _normalize_label(row.get("router_part") or row.get("router_primary_part"))
    adapter_crop = _normalize_label(row.get("crop"))
    adapter_part = _normalize_label(row.get("part"))
    router_status = _normalize_label(row.get("router_status"))
    if router_status and router_status not in {"ok", "trusted_hint_skipped", "skipped"}:
        _add_signal(buckets, reasons, "router", f"router_status:{router_status}")
    if adapter_crop and router_crop and adapter_crop != router_crop:
        _add_signal(buckets, reasons, "router", "router_crop_mismatch")
    if adapter_part and router_part and router_part != "unknown" and adapter_part != router_part:
        _add_signal(buckets, reasons, "router", "router_part_mismatch")
    if router_part == "unknown":
        _add_signal(buckets, reasons, "router", "router_part_unknown")

    roi_status = _normalize_label(row.get("roi_quality_status"))
    evidence_status = _normalize_label(row.get("roi_evidence_status"))
    detection_source = _normalize_label(row.get("selected_detection_source"))
    grounding_status = _normalize_label(row.get("grounding_dino_status"))
    if evidence_status == "target_detection_missing" or detection_source == "target_detection_missing":
        _add_signal(buckets, reasons, "bbox", "target_detection_missing")
    if grounding_status == "error":
        _add_signal(buckets, reasons, "bbox", "grounding_dino_error")
    if roi_status and roi_status not in {"roi_ok", "not_evaluated"}:
        _add_signal(buckets, reasons, "bbox", roi_status)
    if _is_false(row.get("semantic_roi_match")):
        _add_signal(buckets, reasons, "bbox", "semantic_mismatch")

    if is_comparable and not is_correct:
        _add_signal(buckets, reasons, "adapter", "full_image_prediction_error")
    if evidence_status == "conflicts_with_full":
        _add_signal(buckets, reasons, "adapter", "roi_conflicts_with_full")

    threshold = _to_float(row.get("full_confidence_review_threshold"), default=0.0)
    full_confidence = _to_float(row.get("full_confidence", row.get("confidence")), default=0.0)
    if threshold > 0.0 and full_confidence < threshold:
        _add_signal(buckets, reasons, "confidence_ood", "low_full_confidence")
    if _is_true(row.get("full_ood_is_ood") if "full_ood_is_ood" in row else row.get("ood_is_ood")):
        _add_signal(buckets, reasons, "confidence_ood", "full_image_marked_ood")
    if "roi_ood_is_ood" in row and _is_true(row.get("roi_ood_is_ood")):
        _add_signal(buckets, reasons, "confidence_ood", "roi_marked_ood")

    requires_review = _is_true(row.get("requires_review"))
    if is_comparable and not is_correct and not requires_review:
        _add_signal(buckets, reasons, "review_gate", "missed_prediction_error")
    if is_comparable and is_correct and requires_review:
        _add_signal(buckets, reasons, "review_gate", "review_false_positive")

    return {
        "image_path": str(row.get("image_path") or ""),
        "target_key": target_key(row),
        "expected_label": row.get("expected_label"),
        "diagnosis": row.get("diagnosis") or row.get("full_diagnosis"),
        "is_comparable": is_comparable,
        "is_correct": is_correct if is_comparable else None,
        "requires_review": requires_review,
        "failure_buckets": buckets,
        "reasons": reasons,
    }


def _summarize_target_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [item for item in items if item["is_comparable"]]
    incorrect = [item for item in comparable if not item["is_correct"]]
    bucket_counts = Counter(bucket for item in items for bucket in item["failure_buckets"])
    return {
        "sample_count": len(items),
        "comparable_count": len(comparable),
        "incorrect_count": len(incorrect),
        "review_count": sum(1 for item in items if item["requires_review"]),
        "bucket_counts": {bucket: int(bucket_counts.get(bucket, 0)) for bucket in FAILURE_BUCKETS},
        "review_capture_rate_on_errors": _safe_rate(
            sum(1 for item in incorrect if item["requires_review"]),
            len(incorrect),
        ),
    }


def _add_signal(buckets: list[str], reasons: list[str], bucket: str, reason: str) -> None:
    if bucket not in buckets:
        buckets.append(bucket)
    if reason not in reasons:
        reasons.append(reason)


def target_key(row: dict[str, Any]) -> str:
    if row.get("target_id"):
        return str(row["target_id"])
    if row.get("dataset_key"):
        return str(row["dataset_key"])
    crop = str(row.get("crop") or "unknown")
    part = str(row.get("part") or "unknown")
    return f"{crop}__{part}"


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


def _is_false(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return not bool(value)
    return _normalize_label(value) in {"0", "false", "no", "n"}


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return (numerator / denominator) if denominator else None
