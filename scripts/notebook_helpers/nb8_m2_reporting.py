"""Notebook 8 M2 reporting helpers."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.compare_m2_demo_results import compare_results, comparison_markdown, enrich_summary_manifest_sha256


def _relative_repo_path(path: str | Path, repo_root: Path) -> str:
    return Path(path).relative_to(repo_root).as_posix()


def format_elapsed_seconds(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def copy_existing_artifacts(
    *,
    source_paths: Iterable[str | Path | None],
    repo_results_dir: Path,
    repo_root: Path,
) -> list[str]:
    copied_paths: list[str] = []
    for source_path in source_paths:
        if source_path is None:
            continue
        path = Path(source_path)
        if not path.is_file():
            continue
        destination = repo_results_dir / path.name
        shutil.copy2(path, destination)
        copied_paths.append(destination.relative_to(repo_root).as_posix())
    return copied_paths


def write_m2_failure_summary(
    *,
    repo_root: Path,
    results_root: str | Path,
    phase: str,
    error: Any,
    started_at: datetime,
    manifest_path: str | Path | None = None,
    calibration_manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
    markdown_output_path: str | Path | None = None,
    analysis_output_path: str | Path | None = None,
    analysis_markdown_output_path: str | Path | None = None,
    prototype_calibration_output_path: str | Path | None = None,
    runner_exit_code: int | None = None,
    problem_only: bool = False,
    batch_size: int = 1,
    adapter_batch_size: int = 1,
    pytorch_cuda_alloc_conf: str = "",
    prototype_reconciler: Mapping[str, Any] | None = None,
    stamp: str | None = None,
    finished_at: datetime | None = None,
) -> dict[str, Any]:
    created_at = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    repo_results_rel = Path(results_root) / created_at
    repo_results_dir = repo_root / repo_results_rel
    repo_results_dir.mkdir(parents=True, exist_ok=True)
    finished = finished_at or datetime.now(timezone.utc)
    elapsed_seconds = (finished - started_at).total_seconds()
    copied_paths = copy_existing_artifacts(
        source_paths=(
            output_path,
            markdown_output_path,
            analysis_output_path,
            analysis_markdown_output_path,
            prototype_calibration_output_path,
        ),
        repo_results_dir=repo_results_dir,
        repo_root=repo_root,
    )
    summary_path = repo_results_dir / "summary.json"
    summary_payload: dict[str, Any] = {
        "created_at": created_at,
        "started_at": started_at.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_human": format_elapsed_seconds(elapsed_seconds),
        "status": "failed_before_complete_report",
        "failure_phase": str(phase),
        "error": str(error),
        "runner_exit_code": runner_exit_code,
        "manifest": _relative_repo_path(manifest_path, repo_root) if manifest_path else "",
        "problem_only": bool(problem_only),
        "prototype_calibration_manifest": _relative_repo_path(calibration_manifest_path, repo_root)
        if calibration_manifest_path
        else "",
        "batch_size": int(max(1, batch_size)),
        "adapter_batch_size": int(max(1, adapter_batch_size)),
        "pytorch_cuda_alloc_conf": str(pytorch_cuda_alloc_conf),
        "prototype_reconciler": dict(prototype_reconciler or {}),
        "copied_artifacts": copied_paths,
    }
    copied_paths.append(summary_path.relative_to(repo_root).as_posix())
    summary_payload["copied_artifacts"] = copied_paths
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "run_dir": repo_results_rel.as_posix(),
        "summary": summary_payload,
        "summary_path": summary_path,
        "copied_artifacts": copied_paths,
    }


def write_m2_result_comparison(
    *,
    repo_root: Path,
    repo_results_dir: Path,
    candidate_summary_payload: dict[str, Any],
    comparison_baseline: str,
    copied_paths: list[str],
) -> dict[str, Any]:
    comparison_baseline_path = (repo_root / comparison_baseline).resolve() if comparison_baseline else None
    if not comparison_baseline_path:
        return {
            "baseline": comparison_baseline,
            "enabled": False,
            "written": False,
            "status": "not_run",
            "checks": {},
        }
    if not comparison_baseline_path.is_file():
        return {
            "baseline": comparison_baseline,
            "enabled": True,
            "written": False,
            "status": "baseline_missing",
            "checks": {},
        }

    comparison_path = repo_results_dir / "m2_result_comparison.json"
    comparison_markdown_path = repo_results_dir / "m2_result_comparison.md"
    baseline_payload = enrich_summary_manifest_sha256(
        json.loads(comparison_baseline_path.read_text(encoding="utf-8")),
        repo_root=repo_root,
    )
    comparison_payload = compare_results(
        baseline=baseline_payload,
        candidate=candidate_summary_payload,
    )
    comparison_path.write_text(json.dumps(comparison_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    comparison_markdown_path.write_text(comparison_markdown(comparison_payload), encoding="utf-8")
    comparison_output = _relative_repo_path(comparison_path, repo_root)
    comparison_markdown_output = _relative_repo_path(comparison_markdown_path, repo_root)
    copied_paths.append(comparison_output)
    copied_paths.append(comparison_markdown_output)
    comparison_summary = {
        "baseline": _relative_repo_path(comparison_baseline_path, repo_root),
        "output": comparison_output,
        "markdown_output": comparison_markdown_output,
        "status": comparison_payload.get("status"),
        "checks": comparison_payload.get("checks", {}),
    }
    candidate_summary_payload["comparison"] = comparison_summary
    candidate_summary_payload["copied_artifacts"] = copied_paths
    return {
        **comparison_summary,
        "enabled": True,
        "written": True,
    }
