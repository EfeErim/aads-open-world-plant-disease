#!/usr/bin/env python3
"""Validate that Notebook 8 is prepared to launch the open-world router gate."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_router_open_world_readiness import validate_open_world_manifest  # noqa: E402
from src.data.dataset_release_runtime import DEFAULT_MATERIALIZED_DATASET_ROOT  # noqa: E402

DEFAULT_RUN_STATE = Path("docs/notebook8_m2_run_state.json")
DEFAULT_NOTEBOOK = Path("colab_notebooks/8_auto_router_adapter_prediction.ipynb")
DEFAULT_SUPPORTED_MANIFEST = Path("docs/demo_assets/m2_full_image_set/manifests/m2_balanced_80_run_manifest.csv")
DEFAULT_OPEN_WORLD_MANIFEST = Path("docs/demo_assets/open_world_router/manifests/m2_open_world_router_manifest.csv")
DEFAULT_OPEN_WORLD_SUMMARY = Path("docs/demo_assets/open_world_router/manifests/m2_open_world_router_summary.json")
DEFAULT_OPEN_WORLD_DISJOINT_ROOTS = [
    Path("docs/demo_assets/m2_full_image_set/images"),
    DEFAULT_MATERIALIZED_DATASET_ROOT,
    Path("docs/demo_assets/prototype_curation"),
]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _repo_relative(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def _load_json_from_worktree_or_head(path: Path) -> tuple[dict[str, Any] | None, str]:
    if path.is_file():
        return _load_json(path), "worktree"
    repo_relative = _repo_relative(path)
    if not repo_relative:
        return None, "missing"
    completed = subprocess.run(
        ["git", "show", f"HEAD:{repo_relative}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return None, "missing"
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"HEAD:{repo_relative} must contain a JSON object")
    return payload, "git_head"


def _path_exists_in_worktree_or_head(path: Path) -> tuple[bool, str]:
    if path.is_file():
        return True, "worktree"
    repo_relative = _repo_relative(path)
    if not repo_relative:
        return False, "missing"
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{repo_relative}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.returncode == 0, "git_head" if completed.returncode == 0 else "missing"


def _read_csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def _add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _notebook_source_contains(path: Path, snippet: str) -> bool:
    try:
        payload = _load_json(path)
    except Exception:
        return False
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if snippet in "".join(cell.get("source", [])):
            return True
    return False


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    run_state_path = args.run_state
    notebook_path = args.notebook
    supported_manifest = args.supported_manifest
    open_world_manifest = args.open_world_manifest
    open_world_summary_path = args.open_world_summary

    run_state: dict[str, Any] = {}
    try:
        run_state = _load_json(run_state_path)
        _add_check(checks, "run_state_json_valid", True, str(run_state_path))
    except Exception as exc:
        _add_check(checks, "run_state_json_valid", False, f"{exc.__class__.__name__}: {exc}")

    mode = str(run_state.get("mode") or "")
    _add_check(checks, "run_state_mode_open_world_only", mode == "open_world_only", mode)
    _add_check(
        checks,
        "run_state_full_demo_disabled",
        run_state.get("m2_run_full_demo") is False,
        str(run_state.get("m2_run_full_demo")),
    )
    _add_check(
        checks,
        "run_state_open_world_only_enabled",
        run_state.get("m2_open_world_only") is True,
        str(run_state.get("m2_open_world_only")),
    )
    _add_check(
        checks,
        "run_state_problem_only_disabled",
        run_state.get("m2_run_problem_only_demo") is False,
        str(run_state.get("m2_run_problem_only_demo")),
    )
    _add_check(
        checks,
        "run_state_open_world_enabled",
        run_state.get("m2_run_open_world_router_validation") is True,
        str(run_state.get("m2_run_open_world_router_validation")),
    )
    _add_check(
        checks,
        "visible_notebook_open_world_enabled",
        _notebook_source_contains(notebook_path, "M2_RUN_OPEN_WORLD_ROUTER_VALIDATION = True"),
        str(notebook_path),
    )
    _add_check(
        checks,
        "visible_notebook_open_world_only_enabled",
        _notebook_source_contains(notebook_path, "M2_OPEN_WORLD_ONLY = True"),
        str(notebook_path),
    )

    artifact_dir_value = str(run_state.get("m2_open_world_prototype_artifact_dir") or "")
    artifact_dir = (REPO_ROOT / artifact_dir_value).resolve() if artifact_dir_value else None
    for filename in ("prototype_bank.json", "taxonomy_registry.json", "router_prototype_calibration.json"):
        artifact_path = artifact_dir / filename if artifact_dir else Path(filename)
        exists, source = _path_exists_in_worktree_or_head(artifact_path) if artifact_dir else (False, "missing")
        _add_check(
            checks,
            f"open_world_prototype_artifact_{filename.removesuffix('.json')}_available",
            exists,
            f"{artifact_dir_value}/{filename}; source={source}" if artifact_dir_value else "missing",
        )

    baseline_value = str(run_state.get("m2_open_world_baseline_summary") or run_state.get("m2_comparison_baseline") or "")
    baseline_path = (REPO_ROOT / baseline_value).resolve() if baseline_value else None
    baseline_payload = None
    baseline_source = "missing"
    if baseline_path:
        try:
            baseline_payload, baseline_source = _load_json_from_worktree_or_head(baseline_path)
        except Exception as exc:
            _add_check(
                checks,
                "open_world_baseline_summary_available",
                False,
                f"{baseline_value}; {exc.__class__.__name__}: {exc}",
            )
    _add_check(
        checks,
        "open_world_baseline_summary_available",
        baseline_payload is not None,
        f"{baseline_value}; source={baseline_source}",
    )
    if baseline_payload is not None:
        try:
            has_timing = any(key in baseline_payload for key in ("runner_elapsed_seconds", "elapsed_seconds"))
            has_summary_total = isinstance(baseline_payload.get("summary"), dict) and bool(
                baseline_payload["summary"].get("total")
            )
            has_p95 = any(key in baseline_payload for key in ("p95_latency_ms", "router_p95_latency_ms"))
            _add_check(
                checks,
                "open_world_baseline_latency_derivable",
                has_p95 or (has_timing and has_summary_total),
                baseline_value,
            )
        except Exception as exc:
            _add_check(
                checks,
                "open_world_baseline_latency_derivable",
                False,
                f"{exc.__class__.__name__}: {exc}",
            )

    _add_check(checks, "supported_manifest_exists", supported_manifest.is_file(), str(supported_manifest))
    if supported_manifest.is_file():
        supported_count = _read_csv_count(supported_manifest)
        _add_check(
            checks,
            "supported_manifest_has_rows",
            supported_count > 0,
            f"row_count={supported_count}",
        )

    _add_check(checks, "open_world_manifest_exists", open_world_manifest.is_file(), str(open_world_manifest))
    open_world_audit = None
    if open_world_manifest.is_file():
        open_world_audit = validate_open_world_manifest(
            open_world_manifest,
            repo_root=REPO_ROOT,
            min_rows=args.min_open_world_rows,
            disjoint_roots=list(args.open_world_disjoint_root),
        )
        _add_check(
            checks,
            "open_world_manifest_audit_pass",
            open_world_audit["status"] == "pass",
            f"row_count={open_world_audit['row_count']}; issues={open_world_audit['issue_count']}",
        )

    _add_check(checks, "open_world_summary_exists", open_world_summary_path.is_file(), str(open_world_summary_path))
    if open_world_summary_path.is_file():
        try:
            summary = _load_json(open_world_summary_path)
            _add_check(
                checks,
                "open_world_summary_min_rows",
                int(summary.get("row_count") or 0) >= args.min_open_world_rows,
                f"row_count={summary.get('row_count')}",
            )
            _add_check(
                checks,
                "open_world_summary_no_duplicate_hashes",
                int(summary.get("duplicate_hash_count") or 0) == 0,
                f"duplicate_hash_count={summary.get('duplicate_hash_count')}",
            )
        except Exception as exc:
            _add_check(checks, "open_world_summary_readable", False, f"{exc.__class__.__name__}: {exc}")

    failed = [check for check in checks if not check["passed"]]
    return {
        "schema_version": "router_open_world_preflight.v1",
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "inputs": {
            "run_state": str(run_state_path),
            "notebook": str(notebook_path),
            "supported_manifest": str(supported_manifest),
            "open_world_manifest": str(open_world_manifest),
            "open_world_summary": str(open_world_summary_path),
            "baseline_summary": baseline_value,
        },
        "open_world_manifest_audit": open_world_audit,
    }


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Router Open-World Preflight",
        "",
        f"- Status: `{report['status']}`",
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "| --- | --- | --- |",
    ]
    for check in report["checks"]:
        result = "pass" if check["passed"] else "fail"
        detail = str(check.get("detail") or "").replace("|", "\\|")
        lines.append(f"| `{check['name']}` | {result} | {detail} |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-state", type=Path, default=DEFAULT_RUN_STATE)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--supported-manifest", type=Path, default=DEFAULT_SUPPORTED_MANIFEST)
    parser.add_argument("--open-world-manifest", type=Path, default=DEFAULT_OPEN_WORLD_MANIFEST)
    parser.add_argument("--open-world-summary", type=Path, default=DEFAULT_OPEN_WORLD_SUMMARY)
    parser.add_argument("--open-world-disjoint-root", action="append", type=Path, default=list(DEFAULT_OPEN_WORLD_DISJOINT_ROOTS))
    parser.add_argument("--min-open-world-rows", type=int, default=300)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--fail-on-invalid", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(report_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if args.fail_on_invalid and report["status"] != "pass" else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
