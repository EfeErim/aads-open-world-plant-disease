"""Notebook 16 multi-target failure analysis helpers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.pipeline.roi_evidence_analysis import target_key

SCHEMA_VERSION = "v1_notebook16_failure_analysis"
DEFAULT_SOURCE_REPORT = Path("docs/ablation_results/dual_view_inference/multi_target_report.json")
DEFAULT_CALIBRATION_INPUT = Path("docs/ablation_results/dual_view_inference/evidence_gate_calibration.json")
DEFAULT_JSON_OUTPUT = Path("docs/ablation_results/dual_view_inference/notebook16_failure_analysis.json")
DEFAULT_MARKDOWN_OUTPUT = Path("docs/ablation_results/dual_view_inference/notebook16_failure_analysis.md")
DEFAULT_TARGET_AUDIT_JSON_OUTPUT = Path("docs/ablation_results/dual_view_inference/tomato_leaf_missed_wrong_audit.json")
DEFAULT_TARGET_AUDIT_CSV_OUTPUT = Path("docs/ablation_results/dual_view_inference/tomato_leaf_missed_wrong_audit.csv")
DEFAULT_TARGET_AUDIT_MARKDOWN_OUTPUT = Path("docs/ablation_results/dual_view_inference/tomato_leaf_missed_wrong_audit.md")
DEFAULT_FOCUS_TARGET = "tomato__leaf"
DEFAULT_DATA_AUDIT_TARGET = "strawberry__fruit"
DEFAULT_CONFIDENCE_THRESHOLD_SWEEP = (0.95, 0.98, 0.99)
TARGET_AUDIT_SCHEMA_VERSION = "v1_notebook16_target_missed_wrong_audit"
TARGET_AUDIT_FIELD_ORDER = (
    "rank",
    "target_id",
    "expected_label",
    "diagnosis",
    "confusion_pair",
    "full_confidence",
    "requires_review",
    "roi_evidence_status",
    "roi_quality_status",
    "router_confidence",
    "target_detection_source",
    "grounding_dino_status",
    "bbox_area_ratio",
    "local_exists",
    "local_path",
    "image_path",
    "suggested_action",
    "review_decision",
    "review_notes",
)


def load_notebook16_report(path: str | Path) -> list[dict[str, Any]]:
    """Load Notebook 16 rows from a multi-target report."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Expected a report with a list-valued 'rows' field: {path}")
    return [dict(row) for row in rows]


def load_calibration_statuses(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load target policy status details from an optional evidence-gate calibration artifact."""
    input_path = Path(path)
    if not input_path.is_file():
        return {}
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    target_policies = payload.get("target_policies")
    if not isinstance(target_policies, dict):
        return {}
    statuses = {}
    for target_id, policy in target_policies.items():
        if isinstance(policy, dict):
            statuses[str(target_id)] = {
                "status": str(policy.get("status") or ""),
                "group_key": str(policy.get("group_key") or ""),
                "fallback_reason": str(policy.get("fallback_reason") or ""),
            }
    return statuses


def build_notebook16_failure_analysis(
    rows: Iterable[dict[str, Any]],
    *,
    calibration_statuses: dict[str, dict[str, Any]] | None = None,
    focus_target: str = DEFAULT_FOCUS_TARGET,
    data_audit_target: str = DEFAULT_DATA_AUDIT_TARGET,
    top_examples: int = 25,
    source_report: str | Path = DEFAULT_SOURCE_REPORT,
    calibration_input: str | Path = DEFAULT_CALIBRATION_INPUT,
) -> dict[str, Any]:
    """Build a deterministic multi-target Notebook 16 failure-analysis payload."""
    normalized_rows = [_normalize_row(row) for row in rows]
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        grouped_rows[row["target_id"]].append(row)

    calibration_statuses = calibration_statuses or {}
    targets = {
        target_id: _target_summary(
            target_id,
            target_rows,
            calibration_statuses.get(target_id, {}),
            top_examples=top_examples,
        )
        for target_id, target_rows in sorted(grouped_rows.items())
    }
    ordered_target_ids = sorted(targets, key=lambda target_id: _target_sort_key(targets[target_id], target_id))
    return {
        "schema_version": SCHEMA_VERSION,
        "source_report": Path(source_report).as_posix(),
        "calibration_input": Path(calibration_input).as_posix(),
        "focus_target": focus_target,
        "data_audit_target": data_audit_target,
        "sample_count": len(normalized_rows),
        "target_count": len(targets),
        "ordered_targets": ordered_target_ids,
        "focus_target_summary": targets.get(focus_target, {}),
        "data_audit_target_summary": targets.get(data_audit_target, {}),
        "targets": targets,
    }


def render_notebook16_failure_markdown(payload: dict[str, Any]) -> str:
    """Render a compact handoff Markdown report from the analysis payload."""
    focus_target = str(payload.get("focus_target") or DEFAULT_FOCUS_TARGET)
    data_audit_target = str(payload.get("data_audit_target") or DEFAULT_DATA_AUDIT_TARGET)
    focus = payload.get("focus_target_summary") if isinstance(payload.get("focus_target_summary"), dict) else {}
    audit = payload.get("data_audit_target_summary") if isinstance(payload.get("data_audit_target_summary"), dict) else {}
    lines = [
        "# Notebook 16 Failure Analysis",
        "",
        f"Source report: `{payload.get('source_report')}`",
        f"Calibration artifact: `{payload.get('calibration_input')}`",
        "",
        f"## Focus Target: `{focus_target}`",
        "",
    ]
    if focus:
        lines.extend(_target_markdown_lines(focus))
        if focus_target == DEFAULT_FOCUS_TARGET:
            lines.extend(_focus_decision_lines(focus))
    else:
        lines.append("No rows found for this focus target.")
    lines.extend(
        [
            "",
            "## All Targets",
            "",
            "| Target | Samples | Wrong | Accuracy | Review capture | Missed wrong | False-positive review | Calibration |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    targets = payload.get("targets") if isinstance(payload.get("targets"), dict) else {}
    for target_id in payload.get("ordered_targets", []):
        target = targets.get(target_id, {})
        lines.append(
            "| `{target}` | `{samples}` | `{wrong}` | `{accuracy}` | `{capture}` | `{missed}` | `{fp}` | `{calibration}` |".format(
                target=target_id,
                samples=target.get("sample_count", 0),
                wrong=target.get("wrong_count", 0),
                accuracy=_format_rate(target.get("accuracy")),
                capture=_format_rate(target.get("review_capture_rate")),
                missed=target.get("missed_wrong_count", 0),
                fp=_format_rate(target.get("false_positive_review_rate")),
                calibration=target.get("calibration_status", ""),
            )
        )
    lines.extend(
        [
            "",
            f"## Data/Label Audit Target: `{data_audit_target}`",
            "",
        ]
    )
    if audit:
        lines.extend(_target_markdown_lines(audit))
    else:
        lines.append("No rows found for this data/label audit target.")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Keep this as analysis/reporting only.",
            f"- Treat `{focus_target}` as the review-gate focus target for this Notebook 16 pass.",
            "- Do not change Notebook 16 final-decision behavior from this artifact alone.",
            "- Do not promote v2 calibration policies into runtime without a separate validation decision.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_analysis_outputs(payload: dict[str, Any], *, json_output: str | Path, markdown_output: str | Path) -> None:
    """Write machine JSON and handoff Markdown outputs."""
    json_path = Path(json_output)
    markdown_path = Path(markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_notebook16_failure_markdown(payload), encoding="utf-8")


def build_target_missed_wrong_audit(
    rows: Iterable[dict[str, Any]],
    *,
    target_id: str = DEFAULT_FOCUS_TARGET,
    source_report: str | Path = DEFAULT_SOURCE_REPORT,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Build a full missed-wrong audit table for one Notebook 16 target."""
    repo_root_path = Path(repo_root)
    normalized_rows = [_normalize_row(row) for row in rows]
    target_wrong = [
        row
        for row in normalized_rows
        if row["target_id"] == target_id and row["is_comparable"] and not row["is_correct"]
    ]
    missed_wrong = [row for row in target_wrong if not row["requires_review"]]
    missed_wrong = sorted(
        missed_wrong,
        key=lambda row: (
            -float(row["full_confidence"]),
            str(row["expected_label"] or ""),
            str(row["diagnosis"] or ""),
            str(row["image_path"] or ""),
        ),
    )
    audit_rows = [
        _target_audit_row(index=index, row=row, repo_root=repo_root_path)
        for index, row in enumerate(missed_wrong, start=1)
    ]
    return {
        "schema_version": TARGET_AUDIT_SCHEMA_VERSION,
        "source_report": Path(source_report).as_posix(),
        "target_id": target_id,
        "wrong_count": len(target_wrong),
        "missed_wrong_count": len(missed_wrong),
        "local_available_count": sum(1 for row in audit_rows if row["local_exists"]),
        "confusion_pairs": _top_counter(row["confusion_pair"] for row in audit_rows),
        "roi_evidence_status_counts": _top_counter(row["roi_evidence_status"] for row in audit_rows),
        "roi_quality_status_counts": _top_counter(row["roi_quality_status"] for row in audit_rows),
        "rows": audit_rows,
    }


def render_target_missed_wrong_audit_markdown(payload: dict[str, Any]) -> str:
    """Render a compact handoff report for a target missed-wrong audit."""
    target_id = str(payload.get("target_id") or DEFAULT_FOCUS_TARGET)
    lines = [
        f"# `{target_id}` Missed-Wrong Audit",
        "",
        f"Source report: `{payload.get('source_report')}`",
        "",
        "## Summary",
        "",
        f"- wrong predictions: `{payload.get('wrong_count', 0)}`",
        f"- missed wrong predictions: `{payload.get('missed_wrong_count', 0)}`",
        f"- local files available: `{payload.get('local_available_count', 0)}`",
        "",
        "## Top Missed Confusions",
        "",
        "| Confusion pair | Count |",
        "| --- | ---: |",
    ]
    for item in payload.get("confusion_pairs", []):
        lines.append(f"| `{item.get('key')}` | `{item.get('count')}` |")
    lines.extend(
        [
            "",
            "## Review Guidance",
            "",
            "- Audit these rows as data/label quality first; do not promote a runtime policy from this table alone.",
            "- Prioritize high-count confusion pairs before one-off mistakes.",
            "- Keep full-image adapter prediction as final until a refreshed Notebook 16 artifact passes promotion gates.",
            "",
            "## First Rows",
            "",
            "| Rank | Expected | Predicted | Confidence | Local path |",
            "| ---: | --- | --- | ---: | --- |",
        ]
    )
    for row in payload.get("rows", [])[:20]:
        lines.append(
            "| `{rank}` | `{expected}` | `{diagnosis}` | `{confidence:.4f}` | `{path}` |".format(
                rank=row.get("rank"),
                expected=row.get("expected_label"),
                diagnosis=row.get("diagnosis"),
                confidence=float(row.get("full_confidence") or 0.0),
                path=row.get("local_path"),
            )
        )
    return "\n".join(lines) + "\n"


def write_target_missed_wrong_audit_outputs(
    payload: dict[str, Any],
    *,
    json_output: str | Path,
    csv_output: str | Path,
    markdown_output: str | Path,
) -> None:
    """Write JSON, CSV, and Markdown outputs for a target missed-wrong audit."""
    from src.shared.csv_utils import write_csv_rows_with_order

    json_path = Path(json_output)
    markdown_path = Path(markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv_rows_with_order(
        Path(csv_output),
        payload.get("rows", []),
        preferred_headers=TARGET_AUDIT_FIELD_ORDER,
        encoding="utf-8-sig",
    )
    markdown_path.write_text(render_target_missed_wrong_audit_markdown(payload), encoding="utf-8")


def _focus_decision_lines(target: dict[str, Any]) -> list[str]:
    threshold_095 = _threshold_sweep_entry(target, 0.95)
    if not threshold_095:
        return []
    return [
        "",
        "Review-gate focus decision:",
        "",
        f"- `{target.get('target_id')}` stays report-only; it is not a runtime promotion.",
        f"- missed wrong predictions: `{target.get('missed_wrong_count', 0)}`",
        "- `0.95` confidence-threshold simulation: review capture `{capture}`, "
        "false-positive review `{false_positive}`.".format(
            capture=_format_rate(threshold_095.get("wrong_capture_rate")),
            false_positive=_format_rate(threshold_095.get("false_positive_review_rate")),
        ),
    ]


def _threshold_sweep_entry(target: dict[str, Any], threshold: float) -> dict[str, Any]:
    sweep = target.get("confidence_threshold_sweep")
    if not isinstance(sweep, list):
        return {}
    for entry in sweep:
        if not isinstance(entry, dict):
            continue
        try:
            entry_threshold = float(entry.get("threshold"))
        except (TypeError, ValueError):
            continue
        if abs(entry_threshold - threshold) < 1e-9:
            return entry
    return {}


def _target_summary(
    target_id: str,
    rows: Sequence[dict[str, Any]],
    calibration_status: dict[str, Any],
    *,
    top_examples: int,
) -> dict[str, Any]:
    comparable = [row for row in rows if row["is_comparable"]]
    correct = [row for row in comparable if row["is_correct"]]
    wrong = [row for row in comparable if not row["is_correct"]]
    reviewed_correct = [row for row in correct if row["requires_review"]]
    reviewed_wrong = [row for row in wrong if row["requires_review"]]
    missed_wrong = [row for row in wrong if not row["requires_review"]]
    high_confidence_wrong = [row for row in wrong if row["full_confidence"] >= 0.90]
    return {
        "target_id": target_id,
        "sample_count": len(rows),
        "comparable_count": len(comparable),
        "correct_count": len(correct),
        "wrong_count": len(wrong),
        "accuracy": _safe_rate(len(correct), len(comparable)),
        "review_count": sum(1 for row in rows if row["requires_review"]),
        "review_capture_rate": _safe_rate(len(reviewed_wrong), len(wrong)),
        "missed_wrong_count": len(missed_wrong),
        "false_positive_review_rate": _safe_rate(len(reviewed_correct), len(correct)),
        "high_confidence_wrong_count": len(high_confidence_wrong),
        "confusion_pairs": _top_counter(_confusion_key(row) for row in wrong),
        "missed_wrong_confusion_pairs": _top_counter(_confusion_key(row) for row in missed_wrong),
        "roi_evidence_status_counts": _top_counter(row["roi_evidence_status"] for row in rows),
        "roi_quality_status_counts": _top_counter(row["roi_quality_status"] for row in rows),
        "missed_wrong_roi_evidence_status_counts": _top_counter(row["roi_evidence_status"] for row in missed_wrong),
        "missed_wrong_roi_quality_status_counts": _top_counter(row["roi_quality_status"] for row in missed_wrong),
        "missed_wrong_confidence_distribution": _confidence_distribution(missed_wrong),
        "review_reason_counts": _top_counter(reason for row in rows for reason in row["review_reasons"]),
        "confidence_threshold_sweep": _confidence_threshold_sweep(comparable, DEFAULT_CONFIDENCE_THRESHOLD_SWEEP),
        "top_missed_confusion_examples": _top_confusion_examples(missed_wrong, top_examples=5, examples_per_pair=5),
        "calibration_status": str(calibration_status.get("status") or "not_available"),
        "calibration_group_key": str(calibration_status.get("group_key") or ""),
        "calibration_fallback_reason": str(calibration_status.get("fallback_reason") or ""),
        "missed_wrong_examples": _example_rows(missed_wrong, top_examples),
        "high_confidence_wrong_examples": _example_rows(high_confidence_wrong, top_examples),
    }


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    expected = _normalize_label(row.get("expected_label"))
    diagnosis = _normalize_label(row.get("diagnosis") or row.get("full_diagnosis"))
    is_comparable = bool(expected and diagnosis)
    return {
        "image_path": str(row.get("image_path") or ""),
        "target_id": target_key(row),
        "expected_label": row.get("expected_label"),
        "diagnosis": row.get("diagnosis") or row.get("full_diagnosis"),
        "normalized_expected_label": expected,
        "normalized_diagnosis": diagnosis,
        "is_comparable": is_comparable,
        "is_correct": bool(is_comparable and expected == diagnosis),
        "requires_review": _is_true(row.get("requires_review")),
        "review_reasons": _normalize_reasons(row.get("review_reasons")),
        "full_confidence": _to_float(row.get("full_confidence", row.get("confidence")), default=0.0),
        "router_confidence": _to_float(row.get("router_confidence"), default=0.0),
        "bbox_area_ratio": _to_float(row.get("bbox_area_ratio"), default=0.0),
        "roi_evidence_status": _normalize_label(row.get("roi_evidence_status")) or "missing",
        "roi_quality_status": _normalize_label(row.get("roi_quality_status")) or "missing",
        "target_detection_source": str(row.get("target_detection_source") or ""),
        "grounding_dino_status": str(row.get("grounding_dino_status") or ""),
    }


def _target_audit_row(index: int, row: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    image_path = str(row.get("image_path") or "")
    local_path = image_path.replace("/content/bitirmeprojesi/", "")
    return {
        "rank": index,
        "target_id": row["target_id"],
        "expected_label": row["expected_label"],
        "diagnosis": row["diagnosis"],
        "confusion_pair": _confusion_key(row),
        "full_confidence": float(row["full_confidence"]),
        "requires_review": bool(row["requires_review"]),
        "roi_evidence_status": row["roi_evidence_status"],
        "roi_quality_status": row["roi_quality_status"],
        "router_confidence": float(row["router_confidence"]),
        "target_detection_source": row["target_detection_source"],
        "grounding_dino_status": row["grounding_dino_status"],
        "bbox_area_ratio": float(row["bbox_area_ratio"]),
        "local_exists": bool(local_path and (repo_root / local_path).is_file()),
        "local_path": local_path,
        "image_path": image_path,
        "suggested_action": "review_label_or_class_boundary",
        "review_decision": "",
        "review_notes": "",
    }


def _example_rows(rows: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["requires_review"],
            -float(row["full_confidence"]),
            row["image_path"],
        ),
    )
    return [
        {
            "image_path": row["image_path"],
            "target_id": row["target_id"],
            "expected_label": row["expected_label"],
            "diagnosis": row["diagnosis"],
            "full_confidence": row["full_confidence"],
            "roi_evidence_status": row["roi_evidence_status"],
            "roi_quality_status": row["roi_quality_status"],
            "requires_review": row["requires_review"],
            "review_reasons": list(row["review_reasons"]),
        }
        for row in ordered[: max(0, int(limit))]
    ]


def _target_markdown_lines(target: dict[str, Any]) -> list[str]:
    lines = [
        f"- samples: `{target.get('sample_count', 0)}`",
        f"- wrong predictions: `{target.get('wrong_count', 0)}`",
        f"- accuracy: `{_format_rate(target.get('accuracy'))}`",
        f"- review capture on wrong predictions: `{_format_rate(target.get('review_capture_rate'))}`",
        f"- missed wrong predictions: `{target.get('missed_wrong_count', 0)}`",
        f"- false-positive review rate: `{_format_rate(target.get('false_positive_review_rate'))}`",
        f"- calibration status: `{target.get('calibration_status', 'not_available')}`",
        "",
        "Top confusion pairs:",
        "",
    ]
    lines.extend(_counter_markdown_lines(target.get("confusion_pairs")))
    lines.extend(["", "ROI evidence status:", ""])
    lines.extend(_counter_markdown_lines(target.get("roi_evidence_status_counts")))
    lines.extend(["", "ROI quality status:", ""])
    lines.extend(_counter_markdown_lines(target.get("roi_quality_status_counts")))
    lines.extend(["", "### Missed-Wrong Drilldown", ""])
    lines.extend(["Missed wrong confidence distribution:", ""])
    lines.extend(_counter_markdown_lines(target.get("missed_wrong_confidence_distribution")))
    lines.extend(["", "Missed wrong ROI evidence status:", ""])
    lines.extend(_counter_markdown_lines(target.get("missed_wrong_roi_evidence_status_counts")))
    lines.extend(["", "Missed wrong ROI quality status:", ""])
    lines.extend(_counter_markdown_lines(target.get("missed_wrong_roi_quality_status_counts")))
    lines.extend(["", "Confidence threshold sweep over existing review decisions:", ""])
    lines.extend(_threshold_sweep_markdown_lines(target.get("confidence_threshold_sweep")))
    lines.extend(["", "Top missed confusion examples:", ""])
    lines.extend(_confusion_examples_markdown_lines(target.get("top_missed_confusion_examples")))
    return lines


def _counter_markdown_lines(items: Any) -> list[str]:
    if not isinstance(items, list) or not items:
        return ["- none"]
    return [f"- `{item.get('key')}`: `{item.get('count')}`" for item in items[:10]]


def _threshold_sweep_markdown_lines(items: Any) -> list[str]:
    if not isinstance(items, list) or not items:
        return ["- none"]
    lines = [
        "| Threshold | Review capture | Missed wrong | False-positive review | Review rate | Added reviews | Newly captured wrong |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in items:
        lines.append(
            "| `{threshold}` | `{capture}` | `{missed}` | `{fp}` | `{review}` | `{added}` | `{new_wrong}` |".format(
                threshold=_format_rate(item.get("threshold")),
                capture=_format_rate(item.get("wrong_capture_rate")),
                missed=item.get("missed_wrong_count"),
                fp=_format_rate(item.get("false_positive_review_rate")),
                review=_format_rate(item.get("review_rate")),
                added=item.get("added_review_count"),
                new_wrong=item.get("newly_captured_wrong_count"),
            )
        )
    return lines


def _confusion_examples_markdown_lines(items: Any) -> list[str]:
    if not isinstance(items, list) or not items:
        return ["- none"]
    lines = []
    for item in items[:5]:
        examples = item.get("examples") if isinstance(item.get("examples"), list) else []
        example_paths = ", ".join(f"`{example.get('image_path')}`" for example in examples[:3])
        lines.append(f"- `{item.get('confusion_pair')}`: `{item.get('count')}` examples; {example_paths}")
    return lines


def _target_sort_key(target: dict[str, Any], target_id: str) -> tuple[int, int, float, str]:
    return (
        -int(target.get("missed_wrong_count") or 0),
        -int(target.get("sample_count") or 0),
        _rate_or_one(target.get("false_positive_review_rate")),
        target_id,
    )


def _top_counter(values: Iterable[str], *, limit: int = 10) -> list[dict[str, Any]]:
    counts = Counter(value for value in values if value)
    return [
        {"key": key, "count": int(count)}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _confidence_distribution(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = [
        ("<0.50", lambda value: value < 0.50),
        ("0.50-0.70", lambda value: 0.50 <= value < 0.70),
        ("0.70-0.90", lambda value: 0.70 <= value < 0.90),
        ("0.90-0.95", lambda value: 0.90 <= value < 0.95),
        ("0.95-0.98", lambda value: 0.95 <= value < 0.98),
        ("0.98-0.99", lambda value: 0.98 <= value < 0.99),
        (">=0.99", lambda value: value >= 0.99),
    ]
    counts = []
    for label, predicate in bins:
        count = sum(1 for row in rows if predicate(float(row["full_confidence"])))
        if count:
            counts.append({"key": label, "count": count})
    return counts


def _confidence_threshold_sweep(
    rows: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    comparable = [row for row in rows if row["is_comparable"]]
    correct = [row for row in comparable if row["is_correct"]]
    wrong = [row for row in comparable if not row["is_correct"]]
    rows_list = list(rows)
    sweep = []
    for threshold in thresholds:
        simulated_reviewed = [
            row
            for row in rows_list
            if row["requires_review"] or float(row["full_confidence"]) < float(threshold)
        ]
        simulated_wrong_reviewed = [row for row in wrong if row in simulated_reviewed]
        simulated_correct_reviewed = [row for row in correct if row in simulated_reviewed]
        added_reviewed = [row for row in simulated_reviewed if not row["requires_review"]]
        newly_captured_wrong = [row for row in wrong if row in added_reviewed]
        added_false_positives = [row for row in correct if row in added_reviewed]
        sweep.append(
            {
                "threshold": float(threshold),
                "review_count": len(simulated_reviewed),
                "review_rate": _safe_rate(len(simulated_reviewed), len(rows_list)),
                "wrong_capture_rate": _safe_rate(len(simulated_wrong_reviewed), len(wrong)),
                "missed_wrong_count": len(wrong) - len(simulated_wrong_reviewed),
                "false_positive_review_rate": _safe_rate(len(simulated_correct_reviewed), len(correct)),
                "added_review_count": len(added_reviewed),
                "newly_captured_wrong_count": len(newly_captured_wrong),
                "added_false_positive_count": len(added_false_positives),
            }
        )
    return sweep


def _top_confusion_examples(
    rows: Sequence[dict[str, Any]],
    *,
    top_examples: int,
    examples_per_pair: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_confusion_key(row)].append(row)
    payload = []
    for confusion_pair, pair_rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))[:top_examples]:
        payload.append(
            {
                "confusion_pair": confusion_pair,
                "count": len(pair_rows),
                "examples": _example_rows(pair_rows, examples_per_pair),
            }
        )
    return payload


def _confusion_key(row: dict[str, Any]) -> str:
    return f"{row['expected_label']} -> {row['diagnosis']}"


def _normalize_reasons(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


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


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return (numerator / denominator) if denominator else None


def _rate_or_one(value: Any) -> float:
    return 1.0 if value is None else float(value)


def _format_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
