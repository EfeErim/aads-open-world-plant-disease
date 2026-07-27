#!/usr/bin/env python3
"""Compare two M2 Notebook 8 demo result summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_TARGETS = ("tomato__leaf", "tomato__fruit", "apricot__fruit", "strawberry__fruit")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enrich_summary_manifest_sha256(payload: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Fill manifest_sha256 when the summary manifest is locally available."""
    if _manifest_sha256(payload):
        return payload
    manifest = _normalized_manifest(payload)
    if not manifest:
        return payload
    manifest_path = (repo_root / manifest).resolve()
    try:
        manifest_path.relative_to(repo_root.resolve())
    except ValueError:
        return payload
    if manifest_path.is_file():
        payload["manifest_sha256"] = _sha256_file(manifest_path)
        payload["manifest_sha256_source"] = "local_manifest_enriched"
    return payload


def _read_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    summary = payload.get("summary")
    analysis = payload.get("analysis_summary")
    if not isinstance(summary, dict) or not isinstance(analysis, dict):
        raise ValueError(f"{path} must include object-valued 'summary' and 'analysis_summary'")
    return enrich_summary_manifest_sha256(payload, repo_root=Path.cwd())


def _int_at(payload: dict[str, Any], *keys: str, default: int = 0) -> int:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    try:
        return int(current)
    except (TypeError, ValueError):
        return default


def _target_metric(payload: dict[str, Any], target: str, metric: str) -> int:
    return _int_at(payload, "summary", "per_target", target, metric)


def _target_delta(baseline: dict[str, Any], candidate: dict[str, Any], target: str) -> dict[str, int]:
    return {
        "total_delta": _target_metric(candidate, target, "total") - _target_metric(baseline, target, "total"),
        "pass_delta": _target_metric(candidate, target, "pass") - _target_metric(baseline, target, "pass"),
        "fail_delta": _target_metric(candidate, target, "fail") - _target_metric(baseline, target, "fail"),
    }


def _normalized_manifest(payload: dict[str, Any]) -> str:
    value = payload.get("manifest")
    if value is None:
        return ""
    return str(value).replace("\\", "/")


def _manifest_sha256(payload: dict[str, Any]) -> str:
    value = payload.get("manifest_sha256")
    return "" if value is None else str(value).strip().lower()


def compare_results(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    targets: tuple[str, ...] = DEFAULT_TARGETS,
) -> dict[str, Any]:
    baseline_manifest = _normalized_manifest(baseline)
    candidate_manifest = _normalized_manifest(candidate)
    baseline_manifest_sha256 = _manifest_sha256(baseline)
    candidate_manifest_sha256 = _manifest_sha256(candidate)
    hashes_comparable = bool(baseline_manifest_sha256 and candidate_manifest_sha256)
    metrics = {
        "total_delta": _int_at(candidate, "summary", "total") - _int_at(baseline, "summary", "total"),
        "passed_delta": _int_at(candidate, "summary", "passed") - _int_at(baseline, "summary", "passed"),
        "failed_delta": _int_at(candidate, "summary", "failed") - _int_at(baseline, "summary", "failed"),
        "router_failure_delta": _int_at(candidate, "summary", "failure_buckets", "router")
        - _int_at(baseline, "summary", "failure_buckets", "router"),
        "negative_false_accept_delta": _int_at(candidate, "analysis_summary", "negative_false_accepts", "count")
        - _int_at(baseline, "analysis_summary", "negative_false_accepts", "count"),
        "opposite_part_delta": _int_at(candidate, "analysis_summary", "opposite_part_disease_labels", "count")
        - _int_at(baseline, "analysis_summary", "opposite_part_disease_labels", "count"),
        "prototype_correct_but_abstained_delta": _int_at(
            candidate, "analysis_summary", "prototype_correct_but_abstained", "count"
        )
        - _int_at(baseline, "analysis_summary", "prototype_correct_but_abstained", "count"),
    }
    target_deltas = {target: _target_delta(baseline, candidate, target) for target in targets}
    totals_match = metrics["total_delta"] == 0
    manifest_sha256_match = not hashes_comparable or baseline_manifest_sha256 == candidate_manifest_sha256
    warnings: list[str] = []
    if hashes_comparable and manifest_sha256_match and not totals_match:
        warnings.append("manifest_hash_matches_but_total_rows_differ")
    for side, payload in (("baseline", baseline), ("candidate", candidate)):
        if str(payload.get("manifest_sha256_source") or "") == "local_manifest_enriched":
            warnings.append(f"{side}_manifest_sha256_enriched_from_local_manifest")
    checks = {
        "manifests_match": bool(baseline_manifest) and baseline_manifest == candidate_manifest,
        "manifest_sha256_match": manifest_sha256_match,
        "totals_match": totals_match,
        "focus_target_totals_match": all(delta["total_delta"] == 0 for delta in target_deltas.values()),
        "failed_not_increased": metrics["failed_delta"] <= 0,
        "router_failures_not_increased": metrics["router_failure_delta"] <= 0,
        "negative_false_accepts_not_increased": metrics["negative_false_accept_delta"] <= 0,
        "opposite_part_not_increased": metrics["opposite_part_delta"] <= 0,
        "at_least_one_focus_target_improved": any(delta["pass_delta"] > 0 for delta in target_deltas.values()),
    }
    return {
        "schema_version": "m2_demo_result_comparison.v1",
        "baseline_created_at": baseline.get("created_at"),
        "candidate_created_at": candidate.get("created_at"),
        "baseline_manifest": baseline_manifest,
        "candidate_manifest": candidate_manifest,
        "baseline_manifest_sha256": baseline_manifest_sha256,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "baseline_manifest_sha256_source": str(baseline.get("manifest_sha256_source") or "summary"),
        "candidate_manifest_sha256_source": str(candidate.get("manifest_sha256_source") or "summary"),
        "metrics": metrics,
        "target_deltas": target_deltas,
        "checks": checks,
        "warnings": warnings,
        "status": "pass" if all(checks.values()) else "fail",
    }


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "pass" if value else "fail"
    return str(value)


def comparison_markdown(comparison: dict[str, Any]) -> str:
    """Render a compact human-readable comparison report."""
    metrics = comparison.get("metrics", {})
    target_deltas = comparison.get("target_deltas", {})
    checks = comparison.get("checks", {})
    lines = [
        "# M2 Demo Result Comparison",
        "",
        f"- Status: `{comparison.get('status', 'unknown')}`",
        f"- Baseline: `{comparison.get('baseline_created_at') or 'unknown'}`",
        f"- Candidate: `{comparison.get('candidate_created_at') or 'unknown'}`",
        f"- Baseline manifest: `{comparison.get('baseline_manifest') or 'unknown'}`",
        f"- Candidate manifest: `{comparison.get('candidate_manifest') or 'unknown'}`",
        f"- Baseline manifest SHA-256: `{comparison.get('baseline_manifest_sha256') or 'unknown'}`",
        f"- Candidate manifest SHA-256: `{comparison.get('candidate_manifest_sha256') or 'unknown'}`",
        f"- Baseline manifest SHA-256 source: `{comparison.get('baseline_manifest_sha256_source') or 'unknown'}`",
        f"- Candidate manifest SHA-256 source: `{comparison.get('candidate_manifest_sha256_source') or 'unknown'}`",
        "",
        "## Metric Deltas",
        "",
        "| Metric | Delta |",
        "| --- | ---: |",
    ]
    for name in (
        "total_delta",
        "passed_delta",
        "failed_delta",
        "router_failure_delta",
        "negative_false_accept_delta",
        "opposite_part_delta",
        "prototype_correct_but_abstained_delta",
    ):
        lines.append(f"| `{name}` | {_format_value(metrics.get(name, 0))} |")
    lines.extend(
        [
            "",
            "## Focus Target Deltas",
            "",
            "| Target | Total Delta | Pass Delta | Fail Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for target, deltas in target_deltas.items():
        if not isinstance(deltas, dict):
            continue
        lines.append(
            f"| `{target}` | {_format_value(deltas.get('total_delta', 0))} | "
            f"{_format_value(deltas.get('pass_delta', 0))} | "
            f"{_format_value(deltas.get('fail_delta', 0))} |"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Check | Result |",
            "| --- | --- |",
        ]
    )
    for name, passed in checks.items():
        lines.append(f"| `{name}` | {_format_value(passed)} |")
    warnings = comparison.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- `{warning}`")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--focus-target", action="append", default=[])
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    targets = tuple(args.focus_target) if args.focus_target else DEFAULT_TARGETS
    comparison = compare_results(
        baseline=_read_summary(args.baseline),
        candidate=_read_summary(args.candidate),
        targets=targets,
    )
    text = json.dumps(comparison, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(comparison_markdown(comparison), encoding="utf-8")
    print(text)
    return 1 if args.fail_on_regression and comparison["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
