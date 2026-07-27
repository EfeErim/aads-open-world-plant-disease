"""Reporting helpers for the M2 demo checklist runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from scripts.demo_checklist_grading import (
    ABSTAIN_STATUSES,
    CLASSLESS_SUPPORTED_PROBE_MARKERS,
    opposite_part_label,
)


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def _class_matches(expected_class: str, diagnosis: Any) -> bool:
    expected_key = _norm(expected_class)
    diagnosis_key = _norm(diagnosis)
    if not expected_key or not diagnosis_key:
        return False
    return expected_key in diagnosis_key or diagnosis_key in expected_key


def summarize_results(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    answered = sum(1 for row in items if row.get("actual_status") == "success")
    passed = sum(1 for row in items if row.get("pass_fail") == "pass")
    failed = sum(1 for row in items if row.get("pass_fail") == "fail")
    abstained = sum(1 for row in items if str(row.get("actual_status") or "") in ABSTAIN_STATUSES)
    asset_ready = sum(1 for row in items if row.get("actual_status") == "asset_ready")
    buckets: dict[str, int] = {}
    targets: dict[str, dict[str, int]] = {}
    for row in items:
        bucket = str(row.get("failure_bucket") or "")
        if bucket:
            buckets[bucket] = buckets.get(bucket, 0) + 1
        target = str(row.get("expected_target") or "")
        target_summary = targets.setdefault(target, {"total": 0, "pass": 0, "fail": 0})
        target_summary["total"] += 1
        target_summary[str(row.get("pass_fail") or "fail")] += 1
    return {
        "total": len(items),
        "passed": passed,
        "failed": failed,
        "answered": answered,
        "abstained_or_reviewed": abstained,
        "asset_ready": asset_ready,
        "failure_buckets": dict(sorted(buckets.items())),
        "per_target": dict(sorted(targets.items())),
    }


def build_analysis_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    crop_counts = {"correct": 0, "incorrect": 0, "not_applicable": 0}
    part_counts = {"correct": 0, "incorrect": 0, "not_applicable": 0}
    class_counts = {"correct": 0, "incorrect": 0, "not_applicable": 0}
    adapter_unavailable = {"wrong_router": 0, "missing_adapter": 0, "unknown": 0}
    reconciliation_counts: dict[str, int] = {}
    per_target: dict[str, dict[str, int]] = {}
    opposite_part_rows: list[str] = []
    answered_wrong_by_target: dict[str, int] = {}
    answered_wrong_by_expected_class: dict[str, int] = {}
    prototype_correct_but_abstained: list[str] = []
    prototype_correct_but_abstained_by_target: dict[str, int] = {}
    negative_false_accepts: list[str] = []
    negative_false_accepts_by_target: dict[str, int] = {}
    policy_thresholds_by_target: dict[str, dict[str, Any]] = {}
    classless_supported_probes = {
        "total": 0,
        "answered": 0,
        "answered_target_correct": 0,
        "answered_target_incorrect": 0,
        "reviewed_or_abstained": 0,
        "failed": 0,
    }

    for row in items:
        target = str(row.get("expected_target") or "")
        target_summary = per_target.setdefault(
            target,
            {
                "total": 0,
                "answered": 0,
                "abstained_or_reviewed": 0,
                "pass": 0,
                "fail": 0,
                "exact_class_correct": 0,
                "opposite_part_disease_labels": 0,
            },
        )
        target_summary["total"] += 1
        pass_fail = str(row.get("pass_fail") or "fail")
        if pass_fail in {"pass", "fail"}:
            target_summary[pass_fail] += 1
        status = str(row.get("actual_status") or "")
        reconcile_decision = str(row.get("reconcile_decision") or "")
        if reconcile_decision:
            reconciliation_counts[reconcile_decision] = reconciliation_counts.get(reconcile_decision, 0) + 1
        if status == "success":
            target_summary["answered"] += 1
        if status in ABSTAIN_STATUSES:
            target_summary["abstained_or_reviewed"] += 1

        expected_crop = str(row.get("expected_crop") or "").strip().lower()
        expected_part = str(row.get("expected_part") or "").strip().lower()
        predicted_crop = str(row.get("predicted_crop") or "").strip().lower()
        predicted_part = str(row.get("predicted_part") or "").strip().lower()
        expected_class = str(row.get("expected_class") or "").strip()
        predicted_disease = row.get("predicted_disease")
        is_classless_supported_probe = (
            not expected_class
            and bool(expected_crop)
            and bool(expected_part)
            and any(
                marker in str(row.get("expected_behavior") or "").strip().lower()
                for marker in CLASSLESS_SUPPORTED_PROBE_MARKERS
            )
        )

        if expected_crop:
            crop_counts["correct" if predicted_crop == expected_crop else "incorrect"] += 1
        else:
            crop_counts["not_applicable"] += 1
        if expected_part:
            part_counts["correct" if predicted_part == expected_part else "incorrect"] += 1
        else:
            part_counts["not_applicable"] += 1
        if expected_class and status == "success":
            class_correct = _class_matches(expected_class, predicted_disease)
            class_counts["correct" if class_correct else "incorrect"] += 1
            if class_correct:
                target_summary["exact_class_correct"] += 1
            else:
                answered_wrong_by_target[target] = answered_wrong_by_target.get(target, 0) + 1
                answered_wrong_by_expected_class[expected_class] = answered_wrong_by_expected_class.get(
                    expected_class,
                    0,
                ) + 1
        else:
            class_counts["not_applicable"] += 1

        if is_classless_supported_probe:
            classless_supported_probes["total"] += 1
            if status == "success":
                classless_supported_probes["answered"] += 1
                if predicted_crop == expected_crop and predicted_part == expected_part:
                    classless_supported_probes["answered_target_correct"] += 1
                else:
                    classless_supported_probes["answered_target_incorrect"] += 1
            if status in ABSTAIN_STATUSES:
                classless_supported_probes["reviewed_or_abstained"] += 1
            if pass_fail == "fail":
                classless_supported_probes["failed"] += 1

        if opposite_part_label(expected_part, predicted_disease):
            opposite_part_rows.append(str(row.get("image_id") or ""))
            target_summary["opposite_part_disease_labels"] += 1

        if status == "adapter_unavailable":
            if expected_crop and expected_part:
                if predicted_crop != expected_crop or predicted_part != expected_part:
                    adapter_unavailable["wrong_router"] += 1
                else:
                    adapter_unavailable["missing_adapter"] += 1
            else:
                adapter_unavailable["unknown"] += 1

        prototype_target = str(row.get("prototype_crop") or "")
        prototype_part = str(row.get("prototype_part") or "")
        if prototype_target and prototype_part:
            prototype_target = f"{prototype_target}__{prototype_part}"
        else:
            prototype_target = str(row.get("prototype_target") or "")
        if target and prototype_target == target and reconcile_decision == "abstain":
            prototype_correct_but_abstained.append(str(row.get("image_id") or ""))
            prototype_correct_but_abstained_by_target[target] = (
                prototype_correct_but_abstained_by_target.get(target, 0) + 1
            )
        if target in {"unknown_crop", "non_plant"} or target.endswith("__unknown_part"):
            has_disease_answer = bool(str(predicted_disease or "").strip())
            if status == "success" or has_disease_answer:
                negative_false_accepts.append(str(row.get("image_id") or ""))
                negative_false_accepts_by_target[target] = negative_false_accepts_by_target.get(target, 0) + 1
        if target and row.get("prototype_similarity") is not None:
            policy_thresholds_by_target.setdefault(
                target,
                {
                    "min_similarity": row.get("prototype_min_similarity"),
                    "min_margin": row.get("prototype_min_margin"),
                    "min_negative_gap": row.get("prototype_min_negative_gap"),
                },
            )

    return {
        "schema_version": "v1_m2_demo_analysis_summary",
        "total": len(items),
        "router_crop_correctness": crop_counts,
        "router_part_correctness": part_counts,
        "normalized_disease_class_correctness": class_counts,
        "classless_supported_probes": classless_supported_probes,
        "adapter_unavailable": adapter_unavailable,
        "prototype_reconciliation": dict(sorted(reconciliation_counts.items())),
        "answered_wrong_by_target": dict(sorted(answered_wrong_by_target.items())),
        "answered_wrong_by_expected_class": dict(sorted(answered_wrong_by_expected_class.items())),
        "prototype_correct_but_abstained": {
            "count": len(prototype_correct_but_abstained),
            "image_ids": prototype_correct_but_abstained[:100],
            "truncated": len(prototype_correct_but_abstained) > 100,
            "by_target": dict(sorted(prototype_correct_but_abstained_by_target.items())),
        },
        "negative_false_accepts": {
            "count": len(negative_false_accepts),
            "image_ids": negative_false_accepts[:100],
            "truncated": len(negative_false_accepts) > 100,
            "by_target": dict(sorted(negative_false_accepts_by_target.items())),
        },
        "policy_thresholds_by_target": dict(sorted(policy_thresholds_by_target.items())),
        "opposite_part_disease_labels": {
            "count": len(opposite_part_rows),
            "image_ids": opposite_part_rows[:100],
            "truncated": len(opposite_part_rows) > 100,
        },
        "per_target": dict(sorted(per_target.items())),
    }


def write_analysis_markdown(analysis: dict[str, Any], output_path: Path) -> None:
    json_fields = (
        "router_crop_correctness",
        "router_part_correctness",
        "normalized_disease_class_correctness",
        "classless_supported_probes",
        "adapter_unavailable",
        "prototype_reconciliation",
        "answered_wrong_by_target",
    )
    lines = [
        "# M2 Demo Analysis Summary",
        "",
        f"- total: {analysis['total']}",
        *[f"- {key}: `{json.dumps(analysis.get(key, {}), sort_keys=True)}`" for key in json_fields],
        (
            "- prototype_correct_but_abstained: "
            f"{analysis.get('prototype_correct_but_abstained', {}).get('count', 0)} "
            f"`{json.dumps(analysis.get('prototype_correct_but_abstained', {}).get('by_target', {}), sort_keys=True)}`"
        ),
        (
            "- negative_false_accepts: "
            f"{analysis.get('negative_false_accepts', {}).get('count', 0)} "
            f"`{json.dumps(analysis.get('negative_false_accepts', {}).get('by_target', {}), sort_keys=True)}`"
        ),
        f"- opposite_part_disease_labels: {analysis['opposite_part_disease_labels']['count']}",
        "",
        "## Per Target",
        "",
        "| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, values in analysis["per_target"].items():
        lines.append(
            f"| {target} | {values['total']} | {values['answered']} | {values['abstained_or_reviewed']} | "
            f"{values['pass']} | {values['fail']} | {values['exact_class_correct']} | "
            f"{values['opposite_part_disease_labels']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# M2 Demo Checklist Run",
        "",
        f"- started_at: `{report.get('started_at', '')}`",
        f"- finished_at: `{report.get('finished_at', '')}`",
        f"- elapsed: `{report.get('elapsed_human', '')}` ({report.get('elapsed_seconds', 0):.3f}s)",
        f"- generated_at: `{report['generated_at']}`",
        f"- checklist: `{report['checklist']}`",
        f"- device: `{report['device']}`",
        f"- adapter_root: `{report['adapter_root']}`",
        f"- mode: `{report['mode']}`",
        f"- batch_size: `{report.get('batch_size', '')}`",
        f"- adapter_batch_size: `{report.get('adapter_batch_size', '')}`",
        f"- handoff_cache: `{json.dumps(report.get('handoff_cache', {}), sort_keys=True)}`",
        "",
        "## Summary",
        "",
        f"- total: {summary['total']}",
        f"- passed: {summary['passed']}",
        f"- failed: {summary['failed']}",
        f"- answered: {summary['answered']}",
        f"- abstained_or_reviewed: {summary['abstained_or_reviewed']}",
        f"- asset_ready: {summary['asset_ready']}",
        f"- failure_buckets: `{json.dumps(summary['failure_buckets'], sort_keys=True)}`",
        "",
        "## Rows",
        "",
        "| image_id | status | pass_fail | failure_bucket | predicted | message |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        predicted = " / ".join(
            str(row.get(key) or "") for key in ("predicted_crop", "predicted_part", "predicted_disease")
        )
        message = str(row.get("message") or "").replace("\n", " ")[:220]
        lines.append(
            "| {image_id} | {status} | {pass_fail} | {bucket} | {predicted} | {message} |".format(
                image_id=row["image_id"],
                status=row.get("actual_status") or "",
                pass_fail=row.get("pass_fail") or "",
                bucket=row.get("failure_bucket") or "",
                predicted=predicted,
                message=message,
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_artifacts(
    report: dict[str, Any],
    *,
    output_path: Path,
    markdown_output_path: Path,
    analysis_output_path: Path | None = None,
    analysis_markdown_output_path: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Persist the JSON and Markdown artifacts for one demo-checklist run."""
    analysis = report["analysis_summary"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(report, markdown_output_path)

    resolved_analysis_output = analysis_output_path or (output_path.parent / "analysis_summary.json")
    resolved_analysis_markdown_output = analysis_markdown_output_path or (output_path.parent / "analysis_summary.md")
    resolved_analysis_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_analysis_output.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    resolved_analysis_markdown_output.parent.mkdir(parents=True, exist_ok=True)
    write_analysis_markdown(analysis, resolved_analysis_markdown_output)
    return output_path, markdown_output_path, resolved_analysis_output, resolved_analysis_markdown_output
