#!/usr/bin/env python3
"""Validate a completed router open-world readiness result folder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REQUIRED_FILES = (
    "supported_balanced_run.json",
    "supported_balanced_run.md",
    "supported_balanced_analysis.json",
    "supported_balanced_analysis.md",
    "open_world_run.json",
    "open_world_run.md",
    "open_world_analysis.json",
    "open_world_analysis.md",
    "router_open_world_readiness.json",
    "router_open_world_readiness.md",
    "failures/wrong_supported_target_handoffs.csv",
    "failures/negative_false_accepts.csv",
    "failures/wrong_part_false_accepts.csv",
    "provenance/supported_manifest.csv",
    "provenance/open_world_manifest.csv",
)
OPTIONAL_PROVENANCE_FILES = (
    "provenance/baseline_summary.json",
    "provenance/prototype_bank.json",
    "provenance/taxonomy_registry.json",
    "provenance/router_prototype_calibration.json",
)
REQUIRED_PRODUCTION_PROVENANCE_FILES = (
    "provenance/baseline_summary.json",
    "provenance/prototype_bank.json",
    "provenance/taxonomy_registry.json",
    "provenance/router_prototype_calibration.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _csv_data_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def validate_result_dir(
    result_dir: Path,
    *,
    min_open_world_rows: int = 300,
    min_supported_route_coverage: float = 0.80,
    require_pass: bool = True,
    require_production_provenance: bool = True,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for relative in REQUIRED_FILES:
        if not (result_dir / relative).is_file():
            issues.append({"code": "missing_required_file", "path": relative})

    readiness_path = result_dir / "router_open_world_readiness.json"
    readiness: dict[str, Any] = {}
    if readiness_path.is_file():
        readiness = _read_json(readiness_path)
    else:
        readiness = {}

    if require_pass and readiness.get("status") != "pass":
        issues.append({"code": "readiness_not_pass", "status": readiness.get("status")})

    checks = readiness.get("checks")
    if isinstance(checks, dict):
        failed_checks = sorted(key for key, value in checks.items() if value is not True)
        if failed_checks:
            issues.append({"code": "failed_readiness_checks", "checks": failed_checks})
    elif readiness_path.is_file():
        issues.append({"code": "readiness_checks_missing"})

    open_world = readiness.get("open_world") if isinstance(readiness.get("open_world"), dict) else {}
    supported = readiness.get("supported") if isinstance(readiness.get("supported"), dict) else {}
    latency = readiness.get("latency") if isinstance(readiness.get("latency"), dict) else {}
    runner_exit_codes = readiness.get("runner_exit_codes") if isinstance(readiness.get("runner_exit_codes"), dict) else {}
    if int(open_world.get("negative_row_count") or 0) < min_open_world_rows:
        issues.append(
            {
                "code": "open_world_negative_rows_below_min",
                "actual": int(open_world.get("negative_row_count") or 0),
                "minimum": min_open_world_rows,
            }
        )
    if float(supported.get("route_coverage") or 0.0) < min_supported_route_coverage:
        issues.append(
            {
                "code": "supported_route_coverage_below_min",
                "actual": float(supported.get("route_coverage") or 0.0),
                "minimum": min_supported_route_coverage,
            }
        )
    if int(supported.get("wrong_supported_target_handoff_count") or 0) != 0:
        issues.append(
            {
                "code": "wrong_supported_target_handoffs_nonzero",
                "actual": int(supported.get("wrong_supported_target_handoff_count") or 0),
            }
        )
    if int(open_world.get("negative_false_accept_count") or 0) != 0:
        issues.append(
            {
                "code": "negative_false_accepts_nonzero",
                "actual": int(open_world.get("negative_false_accept_count") or 0),
            }
        )
    if int(open_world.get("wrong_part_false_accept_count") or 0) != 0:
        issues.append(
            {
                "code": "wrong_part_false_accepts_nonzero",
                "actual": int(open_world.get("wrong_part_false_accept_count") or 0),
            }
        )
    if latency.get("candidate_p95_latency_ms") is None:
        issues.append({"code": "candidate_latency_missing"})
    if latency.get("baseline_p95_latency_ms") is None:
        issues.append({"code": "baseline_latency_missing"})
    for name in ("supported", "open_world"):
        if int(runner_exit_codes.get(name) or 0) != 0:
            issues.append(
                {
                    "code": "runner_exit_code_nonzero",
                    "name": name,
                    "actual": runner_exit_codes.get(name),
                }
            )

    for relative in (
        "failures/wrong_supported_target_handoffs.csv",
        "failures/negative_false_accepts.csv",
        "failures/wrong_part_false_accepts.csv",
    ):
        path = result_dir / relative
        if path.is_file() and _csv_data_row_count(path) != 0:
            issues.append({"code": "failure_csv_not_empty", "path": relative})

    copied_provenance = readiness.get("copied_provenance")
    if isinstance(copied_provenance, dict):
        for key, copied_path in copied_provenance.items():
            if copied_path and not Path(copied_path).is_file():
                issues.append({"code": "copied_provenance_missing", "key": key, "path": copied_path})
    if require_production_provenance:
        for relative in REQUIRED_PRODUCTION_PROVENANCE_FILES:
            if not (result_dir / relative).is_file():
                issues.append({"code": "missing_production_provenance", "path": relative})

    return {
        "schema_version": "router_open_world_result_validation.v1",
        "result_dir": result_dir.as_posix(),
        "status": "pass" if not issues else "fail",
        "issue_count": len(issues),
        "issues": issues,
        "required_file_count": len(REQUIRED_FILES),
        "optional_provenance_present": [
            relative for relative in OPTIONAL_PROVENANCE_FILES if (result_dir / relative).is_file()
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--min-open-world-rows", type=int, default=300)
    parser.add_argument("--min-supported-route-coverage", type=float, default=0.80)
    parser.add_argument("--allow-fail-status", action="store_true")
    parser.add_argument("--allow-missing-production-provenance", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-invalid", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_result_dir(
        args.result_dir,
        min_open_world_rows=args.min_open_world_rows,
        min_supported_route_coverage=args.min_supported_route_coverage,
        require_pass=not args.allow_fail_status,
        require_production_provenance=not args.allow_missing_production_provenance,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if args.fail_on_invalid and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
