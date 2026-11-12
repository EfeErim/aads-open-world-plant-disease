#!/usr/bin/env python3
"""Evaluate Notebook 8 router readiness on supported and open-world manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SUPPORTED_TARGETS = {
    "apricot__fruit",
    "apricot__leaf",
    "grape__fruit",
    "grape__leaf",
    "strawberry__fruit",
    "strawberry__leaf",
    "tomato__fruit",
    "tomato__leaf",
}
OPEN_WORLD_REQUIRED_COLUMNS = {
    "image_id",
    "source",
    "expected_target",
    "expected_crop",
    "expected_part",
    "expected_behavior",
    "ood_slice",
    "origin_url",
}
NEGATIVE_TARGET_MARKERS = ("unknown", "unsupported", "non_plant", "off_crop", "wrong_part")
NEGATIVE_BEHAVIOR_MARKERS = ("abstain", "review", "unknown", "unsafe", "negative", "unsupported", "ood")
REVIEW_STATUSES = {
    "asset_missing",
    "router_uncertain",
    "router_unavailable",
    "input_rejected",
    "dependency_access",
    "adapter_unavailable",
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _target_parts(target: str) -> tuple[str, str]:
    if "__" not in target:
        return "", ""
    crop, part = target.split("__", 1)
    return crop, part


def _expected_crop_part(row: dict[str, Any]) -> tuple[str, str]:
    expected_target = str(row.get("expected_target") or "").strip()
    crop, part = _target_parts(expected_target)
    return str(row.get("expected_crop") or crop).strip(), str(row.get("expected_part") or part).strip()


def _predicted_crop_part(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("predicted_crop") or "").strip(), str(row.get("predicted_part") or "").strip()


def _is_success(row: dict[str, Any]) -> bool:
    return str(row.get("actual_status") or "").strip() == "success"


def _is_review_or_abstain(row: dict[str, Any]) -> bool:
    status = str(row.get("actual_status") or "").strip()
    return status in REVIEW_STATUSES or not _is_success(row)


def is_negative_expected(row: dict[str, Any]) -> bool:
    target = str(row.get("expected_target") or "").strip().lower()
    behavior = str(row.get("expected_behavior") or "").strip().lower()
    if target and target not in SUPPORTED_TARGETS:
        return True
    return any(marker in target for marker in NEGATIVE_TARGET_MARKERS) or any(
        marker in behavior for marker in NEGATIVE_BEHAVIOR_MARKERS
    )


def _zero_failure_upper_bound(sample_count: int, *, confidence: float = 0.95) -> float | None:
    if sample_count <= 0:
        return None
    alpha = 1.0 - confidence
    return 1.0 - math.pow(alpha, 1.0 / sample_count)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_source(source: str, *, repo_root: Path) -> Path | None:
    value = source.strip()
    for prefix in ("staged_external:", "local_test_pool:"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    if not value or value.startswith(("http://", "https://")):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def validate_open_world_manifest(
    manifest: Path,
    *,
    repo_root: Path,
    min_rows: int = 300,
    disjoint_roots: list[Path] | None = None,
) -> dict[str, Any]:
    rows = read_manifest_rows(manifest)
    fieldnames = set(rows[0].keys()) if rows else set()
    missing_columns = sorted(OPEN_WORLD_REQUIRED_COLUMNS - fieldnames)
    issues: list[dict[str, Any]] = []
    if len(rows) < min_rows:
        issues.append({"code": "too_few_rows", "row_count": len(rows), "min_rows": min_rows})
    if missing_columns:
        issues.append({"code": "missing_columns", "columns": missing_columns})

    seen_ids: set[str] = set()
    local_hashes: dict[str, str] = {}
    duplicate_hashes: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=2):
        image_id = str(row.get("image_id") or "").strip()
        if not image_id:
            issues.append({"code": "missing_image_id", "line": index})
        elif image_id in seen_ids:
            issues.append({"code": "duplicate_image_id", "line": index, "image_id": image_id})
        seen_ids.add(image_id)

        for column in OPEN_WORLD_REQUIRED_COLUMNS:
            if not str(row.get(column) or "").strip():
                issues.append({"code": "missing_required_value", "line": index, "image_id": image_id, "column": column})
        if not str(row.get("notes") or row.get("provenance_notes") or "").strip():
            issues.append({"code": "missing_provenance_notes", "line": index, "image_id": image_id})

        source_path = _resolve_manifest_source(str(row.get("source") or ""), repo_root=repo_root)
        if source_path and source_path.is_file():
            digest = _sha256_file(source_path)
            prior = local_hashes.get(digest)
            if prior:
                duplicate_hashes.append({"sha256": digest, "first_image_id": prior, "image_id": image_id})
            else:
                local_hashes[digest] = image_id
    if duplicate_hashes:
        issues.append({"code": "duplicate_local_sha256", "duplicates": duplicate_hashes})

    disjoint_overlap: list[dict[str, str]] = []
    if disjoint_roots:
        manifest_hashes = set(local_hashes)
        for root in disjoint_roots:
            root_path = root if root.is_absolute() else repo_root / root
            for path in _iter_image_files(root_path):
                try:
                    digest = _sha256_file(path)
                except OSError:
                    continue
                if digest in manifest_hashes:
                    try:
                        source = path.relative_to(repo_root).as_posix()
                    except ValueError:
                        source = path.as_posix()
                    disjoint_overlap.append(
                        {
                            "sha256": digest,
                            "image_id": local_hashes[digest],
                            "overlap_path": source,
                        }
                    )
        if disjoint_overlap:
            issues.append({"code": "disjoint_sha256_overlap", "overlaps": disjoint_overlap[:100]})

    return {
        "schema_version": "router_open_world_manifest_audit.v1",
        "manifest": str(manifest),
        "row_count": len(rows),
        "min_rows": min_rows,
        "local_hashed_image_count": len(local_hashes),
        "disjoint_root_count": len(disjoint_roots or []),
        "disjoint_overlap_count": len(disjoint_overlap),
        "issue_count": len(issues),
        "issues": issues,
        "status": "pass" if not issues else "fail",
    }


def _iter_image_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
    ]


def _manifest_by_image_id(manifest: Path | None) -> dict[str, dict[str, str]]:
    if not manifest:
        return {}
    return {str(row.get("image_id") or ""): row for row in read_manifest_rows(manifest)}


def _merge_manifest_fields(
    rows: list[dict[str, Any]],
    manifest_rows: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    merged = []
    for row in rows:
        image_id = str(row.get("image_id") or "")
        enriched = dict(manifest_rows.get(image_id, {}))
        enriched.update(row)
        merged.append(enriched)
    return merged


def _failure_row(row: dict[str, Any], *, reason: str) -> dict[str, Any]:
    expected_crop, expected_part = _expected_crop_part(row)
    predicted_crop, predicted_part = _predicted_crop_part(row)
    return {
        "image_id": row.get("image_id"),
        "reason": reason,
        "ood_slice": row.get("ood_slice") or "",
        "actual_status": row.get("actual_status") or "",
        "expected_target": row.get("expected_target") or "",
        "expected_crop": expected_crop,
        "expected_part": expected_part,
        "predicted_crop": predicted_crop,
        "predicted_part": predicted_part,
        "predicted_disease": row.get("predicted_disease") or "",
        "reconcile_decision": row.get("reconcile_decision") or "",
        "reconcile_reason": row.get("reconcile_reason") or "",
    }


def _find_latency_ms(payload: dict[str, Any]) -> float | None:
    candidates: list[Any] = [
        payload.get("p95_latency_ms"),
        payload.get("router_p95_latency_ms"),
        payload.get("elapsed_p95_latency_ms"),
    ]
    for key in ("summary", "metrics", "timing", "latency"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.extend(
                [
                    value.get("p95_latency_ms"),
                    value.get("router_p95_latency_ms"),
                    value.get("p95_ms"),
                ]
            )
    for candidate in candidates:
        try:
            if candidate is not None:
                return float(candidate)
        except (TypeError, ValueError):
            continue
    total = None
    summary = payload.get("summary")
    if isinstance(summary, dict):
        total = summary.get("total")
    if total is None:
        rows = payload.get("rows")
        if isinstance(rows, list):
            total = len(rows)
    elapsed = payload.get("runner_elapsed_seconds") or payload.get("elapsed_seconds")
    if total and elapsed:
        return float(elapsed) * 1000.0 / float(total)
    return None


def evaluate_readiness(
    *,
    supported_payload: dict[str, Any],
    open_world_payload: dict[str, Any],
    supported_manifest: Path | None = None,
    open_world_manifest: Path | None = None,
    repo_root: Path | None = None,
    open_world_disjoint_roots: list[Path] | None = None,
    min_open_world_rows: int = 300,
    min_supported_route_coverage: float = 0.80,
    baseline_p95_latency_ms: float | None = None,
    candidate_p95_latency_ms: float | None = None,
    max_latency_regression: float = 0.25,
    require_latency_baseline: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root or Path.cwd()
    supported_rows = _merge_manifest_fields(_rows(supported_payload), _manifest_by_image_id(supported_manifest))
    open_world_rows = _merge_manifest_fields(_rows(open_world_payload), _manifest_by_image_id(open_world_manifest))

    supported_eval_rows = [
        row for row in supported_rows if str(row.get("expected_target") or "").strip() in SUPPORTED_TARGETS
    ]
    supported_answered = [row for row in supported_eval_rows if _is_success(row)]
    wrong_handoffs = []
    for row in supported_eval_rows:
        predicted_crop, predicted_part = _predicted_crop_part(row)
        if not predicted_crop and not predicted_part:
            continue
        expected_crop, expected_part = _expected_crop_part(row)
        if (predicted_crop, predicted_part) != (expected_crop, expected_part):
            wrong_handoffs.append(_failure_row(row, reason="wrong_supported_target_handoff"))

    negative_rows = [row for row in open_world_rows if is_negative_expected(row)]
    negative_false_accepts = [
        _failure_row(row, reason="negative_false_accept") for row in negative_rows if _is_success(row)
    ]
    wrong_part_false_accepts = []
    for row in open_world_rows:
        if not _is_success(row):
            continue
        expected_crop, expected_part = _expected_crop_part(row)
        _predicted_crop, predicted_part = _predicted_crop_part(row)
        if expected_part and predicted_part and expected_part != predicted_part:
            wrong_part_false_accepts.append(_failure_row(row, reason="wrong_part_false_accept"))

    per_slice: dict[str, dict[str, int]] = {}
    for row in open_world_rows:
        ood_slice = str(row.get("ood_slice") or "unspecified")
        item = per_slice.setdefault(
            ood_slice,
            {
                "total": 0,
                "accepted": 0,
                "negative_false_accepts": 0,
                "wrong_part_false_accepts": 0,
                "reviewed": 0,
            },
        )
        item["total"] += 1
        if _is_success(row):
            item["accepted"] += 1
        if _is_review_or_abstain(row):
            item["reviewed"] += 1
    for row in negative_false_accepts:
        per_slice.setdefault(str(row.get("ood_slice") or "unspecified"), {}).setdefault("negative_false_accepts", 0)
        per_slice[str(row.get("ood_slice") or "unspecified")]["negative_false_accepts"] += 1
    for row in wrong_part_false_accepts:
        per_slice.setdefault(str(row.get("ood_slice") or "unspecified"), {}).setdefault("wrong_part_false_accepts", 0)
        per_slice[str(row.get("ood_slice") or "unspecified")]["wrong_part_false_accepts"] += 1

    manifest_audit = None
    if open_world_manifest:
        manifest_audit = validate_open_world_manifest(
            open_world_manifest,
            repo_root=repo_root,
            min_rows=min_open_world_rows,
            disjoint_roots=open_world_disjoint_roots,
        )

    if candidate_p95_latency_ms is None:
        candidate_p95_latency_ms = _find_latency_ms(open_world_payload)
    latency_limit = None
    latency_not_regressed = True
    latency_warning = ""
    if baseline_p95_latency_ms is not None:
        latency_limit = baseline_p95_latency_ms * (1.0 + max_latency_regression)
        latency_not_regressed = candidate_p95_latency_ms is not None and candidate_p95_latency_ms <= latency_limit
    elif require_latency_baseline:
        latency_not_regressed = False
        latency_warning = "latency_baseline_missing"
    else:
        latency_warning = "latency_baseline_not_provided"

    supported_total = len(supported_eval_rows)
    supported_route_coverage = len(supported_answered) / supported_total if supported_total else 0.0
    checks = {
        "open_world_min_rows": len(negative_rows) >= min_open_world_rows,
        "manifest_valid": manifest_audit is None or manifest_audit["status"] == "pass",
        "wrong_supported_target_handoffs_zero": len(wrong_handoffs) == 0,
        "negative_false_accepts_zero": len(negative_false_accepts) == 0,
        "wrong_part_false_accepts_zero": len(wrong_part_false_accepts) == 0,
        "supported_route_coverage_min": supported_route_coverage >= min_supported_route_coverage,
        "latency_not_regressed": latency_not_regressed,
    }
    warnings = [latency_warning] if latency_warning else []
    return {
        "schema_version": "router_open_world_readiness.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "supported": {
            "row_count": supported_total,
            "answered_count": len(supported_answered),
            "route_coverage": supported_route_coverage,
            "min_route_coverage": min_supported_route_coverage,
            "wrong_supported_target_handoff_count": len(wrong_handoffs),
        },
        "open_world": {
            "row_count": len(open_world_rows),
            "negative_row_count": len(negative_rows),
            "min_negative_rows": min_open_world_rows,
            "negative_false_accept_count": len(negative_false_accepts),
            "wrong_part_false_accept_count": len(wrong_part_false_accepts),
            "zero_failure_95_upper_bound": _zero_failure_upper_bound(len(negative_rows))
            if not negative_false_accepts
            else None,
            "per_slice": dict(sorted(per_slice.items())),
        },
        "latency": {
            "candidate_p95_latency_ms": candidate_p95_latency_ms,
            "baseline_p95_latency_ms": baseline_p95_latency_ms,
            "max_regression": max_latency_regression,
            "limit_p95_latency_ms": latency_limit,
        },
        "manifest_audit": manifest_audit,
        "failures": {
            "wrong_supported_target_handoffs": wrong_handoffs,
            "negative_false_accepts": negative_false_accepts,
            "wrong_part_false_accepts": wrong_part_false_accepts,
        },
        "warnings": warnings,
    }


def readiness_markdown(report: dict[str, Any]) -> str:
    supported = report["supported"]
    open_world = report["open_world"]
    latency = report["latency"]
    lines = [
        "# Router Open-World Readiness",
        "",
        f"- Status: `{report['status']}`",
        f"- Supported route coverage: `{supported['route_coverage']:.4f}` "
        f"({supported['answered_count']}/{supported['row_count']})",
        f"- Wrong supported handoffs: `{supported['wrong_supported_target_handoff_count']}`",
        f"- Open-world negatives: `{open_world['negative_row_count']}`",
        f"- Negative false accepts: `{open_world['negative_false_accept_count']}`",
        f"- Wrong-part false accepts: `{open_world['wrong_part_false_accept_count']}`",
        f"- Zero-failure 95% upper bound: `{open_world['zero_failure_95_upper_bound']}`",
        f"- Candidate p95 latency ms: `{latency['candidate_p95_latency_ms']}`",
        f"- Baseline p95 latency ms: `{latency['baseline_p95_latency_ms']}`",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"| `{name}` | {'pass' if passed else 'fail'} |")
    lines.extend(
        [
            "",
            "## Per-Slice Open-World Results",
            "",
            "| Slice | Total | Accepted | False Accepts | Wrong-Part | Reviewed |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, values in open_world["per_slice"].items():
        lines.append(
            f"| `{name}` | {values.get('total', 0)} | {values.get('accepted', 0)} | "
            f"{values.get('negative_false_accepts', 0)} | {values.get('wrong_part_false_accepts', 0)} | "
            f"{values.get('reviewed', 0)} |"
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- `{warning}`")
    return "\n".join(lines) + "\n"


def write_failure_csvs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "reason",
        "ood_slice",
        "actual_status",
        "expected_target",
        "expected_crop",
        "expected_part",
        "predicted_crop",
        "predicted_part",
        "predicted_disease",
        "reconcile_decision",
        "reconcile_reason",
    ]
    for key, rows in report["failures"].items():
        path = output_dir / f"{key}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supported-report", type=Path, required=True)
    parser.add_argument("--open-world-report", type=Path, required=True)
    parser.add_argument("--supported-manifest", type=Path)
    parser.add_argument("--open-world-manifest", type=Path)
    parser.add_argument("--open-world-disjoint-root", action="append", type=Path, default=[])
    parser.add_argument("--min-open-world-rows", type=int, default=300)
    parser.add_argument("--min-supported-route-coverage", type=float, default=0.80)
    parser.add_argument("--baseline-p95-latency-ms", type=float)
    parser.add_argument("--candidate-p95-latency-ms", type=float)
    parser.add_argument("--max-latency-regression", type=float, default=0.25)
    parser.add_argument("--require-latency-baseline", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--failure-dir", type=Path)
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_readiness(
        supported_payload=read_json(args.supported_report),
        open_world_payload=read_json(args.open_world_report),
        supported_manifest=args.supported_manifest,
        open_world_manifest=args.open_world_manifest,
        open_world_disjoint_roots=list(args.open_world_disjoint_root),
        min_open_world_rows=args.min_open_world_rows,
        min_supported_route_coverage=args.min_supported_route_coverage,
        baseline_p95_latency_ms=args.baseline_p95_latency_ms,
        candidate_p95_latency_ms=args.candidate_p95_latency_ms,
        max_latency_regression=args.max_latency_regression,
        require_latency_baseline=args.require_latency_baseline,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(readiness_markdown(report), encoding="utf-8")
    if args.failure_dir:
        write_failure_csvs(report, args.failure_dir)
    print(text)
    return 1 if args.fail_on_not_ready and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
