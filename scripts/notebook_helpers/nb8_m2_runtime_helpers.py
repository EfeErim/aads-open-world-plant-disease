from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.compare_m2_demo_results import compare_results, comparison_markdown, enrich_summary_manifest_sha256
from src.colab.repo_bootstrap import push_repo_paths_to_github
from src.shared.hash_utils import sha256_file


def build_m2_demo_checklist_command(
    *,
    python_executable: str,
    repo_root: Path,
    manifest_path: Path,
    device: str,
    config_env: str,
    adapter_root_path: Path,
    output_path: Path,
    markdown_output_path: Path,
    analysis_output_path: Path,
    analysis_markdown_output_path: Path,
    batch_size: int,
    adapter_batch_size: int,
    handoff_cache_path: Path,
    demo_limit: Any = None,
    stop_on_dependency_blocker: bool = True,
    refresh_handoff_cache: bool = True,
    enable_prototype_reconciler: bool = False,
    prototype_bank: str = "",
    taxonomy_registry: str = "",
    prototype_min_similarity: Any = None,
    prototype_min_margin: Any = None,
    prototype_min_negative_gap: Any = None,
    prototype_calibration_output_path: Path | None = None,
) -> list[str]:
    command = [
        python_executable,
        str(repo_root / "scripts" / "run_demo_checklist.py"),
        "--no-checklist",
        "--extra-manifest",
        str(manifest_path),
        "--device",
        str(device),
        "--config-env",
        str(config_env),
        "--adapter-root",
        str(adapter_root_path),
        "--output",
        str(output_path),
        "--markdown-output",
        str(markdown_output_path),
        "--analysis-output",
        str(analysis_output_path),
        "--analysis-markdown-output",
        str(analysis_markdown_output_path),
        "--batch-size",
        str(max(1, int(batch_size))),
        "--adapter-batch-size",
        str(max(1, int(adapter_batch_size))),
        "--handoff-cache",
        str(handoff_cache_path),
    ]
    if demo_limit is not None:
        command.extend(["--limit", str(int(demo_limit))])
    if stop_on_dependency_blocker:
        command.append("--stop-on-dependency-blocker")
    if refresh_handoff_cache:
        command.append("--refresh-handoff-cache")
    if enable_prototype_reconciler:
        command.append("--enable-prototype-reconciler")
        if prototype_bank:
            command.extend(["--prototype-bank", str((repo_root / prototype_bank).resolve())])
        if taxonomy_registry:
            command.extend(["--taxonomy-registry", str((repo_root / taxonomy_registry).resolve())])
        if prototype_min_similarity is not None:
            command.extend(["--prototype-min-similarity", str(float(prototype_min_similarity))])
        if prototype_min_margin is not None:
            command.extend(["--prototype-min-margin", str(float(prototype_min_margin))])
        if prototype_min_negative_gap is not None:
            command.extend(["--prototype-min-negative-gap", str(float(prototype_min_negative_gap))])
        if prototype_calibration_output_path and prototype_calibration_output_path.is_file():
            command.extend(["--prototype-calibration-report", str(prototype_calibration_output_path)])
    return command

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

def m2_subprocess_env(settings: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    pytorch_cuda_alloc_conf = str(settings.get("M2_PYTORCH_CUDA_ALLOC_CONF") or "").strip()
    if pytorch_cuda_alloc_conf:
        existing = str(env.get("PYTORCH_CUDA_ALLOC_CONF") or "").strip()
        if not existing:
            env["PYTORCH_CUDA_ALLOC_CONF"] = pytorch_cuda_alloc_conf
    return env


def expected_calibration_constraints(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "min_precision": settings["M2_PROTOTYPE_CALIBRATION_MIN_PRECISION"],
        "min_coverage": settings["M2_PROTOTYPE_CALIBRATION_MIN_COVERAGE"],
        "require_zero_non_plant_false_accepts": not settings["M2_ALLOW_NON_PLANT_FALSE_ACCEPTS"],
        "max_negative_false_accepts": settings["M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPTS"],
        "max_negative_false_accept_rate": settings["M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPT_RATE"],
        "target_min_precision": settings["M2_PROTOTYPE_TARGET_MIN_PRECISION"],
        "target_max_supported_wrong": int(settings["M2_PROTOTYPE_TARGET_MAX_SUPPORTED_WRONG"]),
        "target_max_cross_part_supported_wrong": settings["M2_PROTOTYPE_TARGET_MAX_CROSS_PART_SUPPORTED_WRONG"],
        "target_policy_negative_mode": settings["M2_PROTOTYPE_TARGET_POLICY_NEGATIVE_MODE"],
        "target_class_min_accepted": settings["M2_PROTOTYPE_TARGET_CLASS_MIN_ACCEPTED"],
        "class_part_conflict_override": "clean_fruit_class",
        "expected_class_rescue": "clean_exact_class_v2_ignore_hard_negative",
        "promotion_mode": "prototype_override",
    }


def same_scalar(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) is bool(right)
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        try:
            return abs(float(left) - float(right)) < 1e-9
        except (TypeError, ValueError):
            return False
    return str(left) == str(right)


def constraints_match(payload: object, settings: dict[str, Any]) -> bool:
    constraints = payload.get("constraints") if isinstance(payload, dict) else {}
    if not isinstance(constraints, dict):
        return False
    expected = expected_calibration_constraints(settings)
    return all(same_scalar(constraints.get(key), value) for key, value in expected.items())


def restore_latest_handoff_cache(repo_root: Path, cache_path: Path, settings: dict[str, Any]) -> Path | None:
    if settings["M2_REFRESH_HANDOFF_CACHE"] or cache_path.is_file():
        return None
    results_root = repo_root / str(settings["M2_REPO_RESULTS_ROOT"])
    if not results_root.is_dir():
        return None
    for candidate_dir in sorted((path for path in results_root.iterdir() if path.is_dir()), reverse=True):
        candidate_cache = candidate_dir / cache_path.name
        if not candidate_cache.is_file():
            continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_cache, cache_path)
        return candidate_cache
    return None


def summary_manifest_sha(candidate_dir: Path) -> str:
    summary_path = candidate_dir / "summary.json"
    if not summary_path.is_file():
        return ""
    try:
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(summary_payload.get("manifest_sha256") or "")


def path_matches_repo_suffix(saved_path: object, current_path: Path, repo_root: Path) -> bool:
    try:
        current_rel = current_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return False
    normalized_saved = str(saved_path or "").replace("\\", "/")
    return normalized_saved == current_rel or normalized_saved.endswith(f"/{current_rel}")


def copy_reusable_calibration_if_available(
    repo_root: Path,
    manifest_path: Path,
    prototype_bank_path: Path,
    output_path: Path,
    settings: dict[str, Any],
) -> Path | None:
    if not settings["M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION"]:
        return None
    if not manifest_path.is_file() or not prototype_bank_path.is_file():
        return None
    manifest_sha256 = sha256_file(manifest_path)
    prototype_sha256 = sha256_file(prototype_bank_path)
    for candidate_dir in sorted((repo_root / str(settings["M2_REPO_RESULTS_ROOT"])).glob("*"), reverse=True):
        calibration_path = candidate_dir / "router_prototype_calibration.json"
        if not calibration_path.is_file():
            continue
        try:
            payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[M2] Skipping unreadable calibration file: {calibration_path}")
            continue
        if not constraints_match(payload, settings):
            continue
        candidate_manifest_sha = str(payload.get("manifest_sha256") or summary_manifest_sha(candidate_dir))
        if candidate_manifest_sha and candidate_manifest_sha != manifest_sha256:
            continue
        if not candidate_manifest_sha and not path_matches_repo_suffix(payload.get("manifest"), manifest_path, repo_root):
            continue
        candidate_prototype_sha = str(payload.get("prototype_bank_sha256") or "")
        co_located_prototype = candidate_dir / "prototype_bank.json"
        if not candidate_prototype_sha and co_located_prototype.is_file():
            candidate_prototype_sha = sha256_file(co_located_prototype)
        if candidate_prototype_sha and candidate_prototype_sha != prototype_sha256:
            continue
        if not candidate_prototype_sha and not path_matches_repo_suffix(
            payload.get("prototype_bank"),
            prototype_bank_path,
            repo_root,
        ):
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(calibration_path, output_path)
        return calibration_path
    return None


def discard_stale_calibration(output_path: Path) -> None:
    try:
        output_path.unlink()
    except FileNotFoundError:
        pass


def normalize_repo_path(value: object, repo_root: Path) -> str:
    normalized = str(value or "").replace("\\", "/").strip()
    if not normalized:
        return ""
    path = Path(normalized)
    try:
        if path.is_absolute():
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
    return normalized


def validate_curated_prototype_bank(repo_root: Path, prototype_bank_ref: str, curation_root: str) -> None:
    if not curation_root:
        return
    if not prototype_bank_ref:
        raise RuntimeError("M2_PROTOTYPE_CURATION_ROOT is set but no prototype bank was selected.")
    prototype_bank_path = (repo_root / prototype_bank_ref).resolve()
    if not prototype_bank_path.is_file():
        raise FileNotFoundError(
            "M2_PROTOTYPE_CURATION_ROOT is set but the selected prototype bank does not exist: "
            f"{prototype_bank_path}"
        )
    payload = json.loads(prototype_bank_path.read_text(encoding="utf-8"))
    source_roots = payload.get("source_roots") if isinstance(payload, dict) else {}
    summary = payload.get("summary") if isinstance(payload, dict) else {}
    saved_curation_root = normalize_repo_path(source_roots.get("curation_root"), repo_root)
    expected_curation_root = normalize_repo_path(curation_root, repo_root)
    if saved_curation_root != expected_curation_root:
        raise RuntimeError(
            "Selected prototype bank was not built from the requested M2 curation root: "
            f"bank={prototype_bank_path}, bank_curation_root={saved_curation_root or '<missing>'}, "
            f"requested_curation_root={expected_curation_root}"
        )
    curation_positive_count = int(summary.get("curation_positive_count") or 0)
    hard_negative_count = int(summary.get("hard_negative_count") or 0)
    if curation_positive_count + hard_negative_count <= 0:
        raise RuntimeError(
            "Selected prototype bank contains zero usable curated rows despite M2_PROTOTYPE_CURATION_ROOT being set: "
            f"bank={prototype_bank_path}, curation_positive_count={curation_positive_count}, "
            f"hard_negative_count={hard_negative_count}"
        )
    print(
        "[M2] Curated prototype bank validated: "
        f"curation_positive_count={curation_positive_count}, hard_negative_count={hard_negative_count}."
    )


def prototype_bank_matches_curation_root(repo_root: Path, prototype_bank_path: Path, curation_root: str) -> bool:
    if not curation_root:
        return True
    try:
        payload = json.loads(prototype_bank_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    source_roots = payload.get("source_roots") if isinstance(payload, dict) else {}
    if not isinstance(source_roots, dict):
        return False
    saved_curation_root = normalize_repo_path(source_roots.get("curation_root"), repo_root)
    expected_curation_root = normalize_repo_path(curation_root, repo_root)
    return bool(saved_curation_root and saved_curation_root == expected_curation_root)


def run_open_world_router_validation(
    repo_root: Path,
    stamp: str,
    adapter_root_path: Path,
    settings: dict[str, Any],
    prototype_calibration_output_path: Path | None = None,
) -> dict[str, Any]:
    open_world_output_root = (repo_root / str(settings["M2_OPEN_WORLD_OUTPUT_ROOT"])).resolve()
    open_world_run_dir = open_world_output_root / stamp
    open_world_command = [
        sys.executable,
        str(repo_root / "scripts" / "run_router_open_world_validation.py"),
        "--run-id",
        stamp,
        "--output-root",
        str(open_world_output_root),
        "--supported-manifest",
        str((repo_root / str(settings["M2_OPEN_WORLD_SUPPORTED_MANIFEST"])).resolve()),
        "--open-world-manifest",
        str((repo_root / str(settings["M2_OPEN_WORLD_MANIFEST"])).resolve()),
        "--device",
        str(settings["DEVICE"]),
        "--config-env",
        str(settings["CONFIG_ENV"]),
        "--adapter-root",
        str(adapter_root_path),
        "--batch-size",
        str(max(1, int(settings["M2_BATCH_SIZE"]))),
        "--adapter-batch-size",
        str(max(1, int(settings["M2_ADAPTER_BATCH_SIZE"]))),
        "--handoff-cache",
        str((repo_root / str(settings["M2_OPEN_WORLD_HANDOFF_CACHE"])).resolve()),
        "--min-open-world-rows",
        str(max(1, int(settings["M2_OPEN_WORLD_MIN_ROWS"]))),
        "--min-supported-route-coverage",
        str(float(settings["M2_OPEN_WORLD_MIN_SUPPORTED_ROUTE_COVERAGE"])),
    ]
    if settings["M2_OPEN_WORLD_PROTOTYPE_ARTIFACT_DIR"]:
        open_world_command.extend(
            [
                "--prototype-artifact-dir",
                str((repo_root / str(settings["M2_OPEN_WORLD_PROTOTYPE_ARTIFACT_DIR"])).resolve()),
            ]
        )
    if settings["M2_ENABLE_PROTOTYPE_RECONCILER"]:
        open_world_command.append("--enable-prototype-reconciler")
    if settings["M2_PROTOTYPE_BANK"]:
        open_world_command.extend(["--prototype-bank", str((repo_root / str(settings["M2_PROTOTYPE_BANK"])).resolve())])
    if settings["M2_TAXONOMY_REGISTRY"]:
        open_world_command.extend(
            ["--taxonomy-registry", str((repo_root / str(settings["M2_TAXONOMY_REGISTRY"])).resolve())]
        )
    if (
        not settings["M2_OPEN_WORLD_PROTOTYPE_ARTIFACT_DIR"]
        and prototype_calibration_output_path
        and prototype_calibration_output_path.is_file()
    ):
        open_world_command.extend(["--prototype-calibration-report", str(prototype_calibration_output_path)])
    if settings["M2_PROTOTYPE_MIN_SIMILARITY"] is not None:
        open_world_command.extend(["--prototype-min-similarity", str(float(settings["M2_PROTOTYPE_MIN_SIMILARITY"]))])
    if settings["M2_PROTOTYPE_MIN_MARGIN"] is not None:
        open_world_command.extend(["--prototype-min-margin", str(float(settings["M2_PROTOTYPE_MIN_MARGIN"]))])
    if settings["M2_PROTOTYPE_MIN_NEGATIVE_GAP"] is not None:
        open_world_command.extend(
            ["--prototype-min-negative-gap", str(float(settings["M2_PROTOTYPE_MIN_NEGATIVE_GAP"]))]
        )
    if settings["M2_OPEN_WORLD_BASELINE_SUMMARY"]:
        open_world_command.extend(
            ["--baseline-summary", str((repo_root / str(settings["M2_OPEN_WORLD_BASELINE_SUMMARY"])).resolve())]
        )
    if settings["M2_REFRESH_HANDOFF_CACHE"]:
        open_world_command.append("--refresh-handoff-cache")
    if settings["M2_STOP_ON_DEPENDENCY_BLOCKER"]:
        open_world_command.append("--stop-on-dependency-blocker")
    if settings["M2_OPEN_WORLD_REQUIRE_LATENCY_BASELINE"]:
        open_world_command.append("--require-latency-baseline")
    if settings["M2_OPEN_WORLD_FAIL_ON_NOT_READY"]:
        open_world_command.append("--fail-on-not-ready")
    print(f"[M2] Starting open-world router validation gate: {open_world_run_dir.relative_to(repo_root)}")
    open_world_completed = subprocess.run(
        open_world_command,
        cwd=repo_root,
        check=False,
        env=m2_subprocess_env(settings),
    )
    open_world_readiness_path = open_world_run_dir / "router_open_world_readiness.json"
    report = {
        "enabled": True,
        "run": True,
        "exit_code": int(open_world_completed.returncode),
        "run_dir": open_world_run_dir.relative_to(repo_root).as_posix(),
        "readiness": open_world_readiness_path.relative_to(repo_root).as_posix(),
        "status": "missing_readiness",
        "ready": False,
        "checks": {},
    }
    if open_world_readiness_path.is_file():
        open_world_readiness = json.loads(open_world_readiness_path.read_text(encoding="utf-8"))
        report["status"] = str(open_world_readiness.get("status") or "unknown")
        report["ready"] = open_world_completed.returncode == 0 and open_world_readiness.get("status") == "pass"
        report["checks"] = open_world_readiness.get("checks", {})
    return report


def write_and_push_m2_failure_summary(
    *,
    repo_root: Path,
    settings: dict[str, Any],
    stamp: str,
    prototype_calibration_output_path: Path,
    phase: str,
    error: str,
    started_at: object,
    manifest_path: Path | None = None,
    calibration_manifest_path: Path | None = None,
    output_path: Path | None = None,
    markdown_output_path: Path | None = None,
    analysis_output_path: Path | None = None,
    analysis_markdown_output_path: Path | None = None,
    runner_exit_code: int | None = None,
) -> dict[str, Any]:
    failure_report = write_m2_failure_summary(
        repo_root=repo_root,
        results_root=str(settings["M2_REPO_RESULTS_ROOT"]),
        phase=phase,
        error=error,
        started_at=started_at,
        manifest_path=manifest_path,
        calibration_manifest_path=calibration_manifest_path,
        output_path=output_path,
        markdown_output_path=markdown_output_path,
        analysis_output_path=analysis_output_path,
        analysis_markdown_output_path=analysis_markdown_output_path,
        prototype_calibration_output_path=prototype_calibration_output_path,
        runner_exit_code=runner_exit_code,
        problem_only=bool(settings["M2_RUN_PROBLEM_ONLY_DEMO"]),
        batch_size=int(settings["M2_BATCH_SIZE"]),
        adapter_batch_size=int(settings["M2_ADAPTER_BATCH_SIZE"]),
        pytorch_cuda_alloc_conf=str(settings["M2_PYTORCH_CUDA_ALLOC_CONF"]),
        prototype_reconciler={
            "enabled": bool(settings["M2_ENABLE_PROTOTYPE_RECONCILER"]),
            "prototype_curation_root": settings["M2_PROTOTYPE_CURATION_ROOT"],
            "prototype_bank": settings["M2_PROTOTYPE_BANK"],
            "taxonomy_registry": settings["M2_TAXONOMY_REGISTRY"],
        },
        stamp=stamp,
    )
    repo_results_rel = Path(str(failure_report["run_dir"]))
    publish_report = {
        "enabled": bool(settings["M2_AUTO_PUSH_RESULTS"]),
        "pushed": False,
        "path": repo_results_rel.as_posix(),
    }
    if settings["M2_AUTO_PUSH_RESULTS"]:
        try:
            publish_report = push_repo_paths_to_github(
                repo_root=repo_root,
                relative_paths=[repo_results_rel.as_posix()],
                remote_name=str(settings["M2_AUTO_PUSH_REMOTE_NAME"]),
                branch=settings["M2_AUTO_PUSH_BRANCH"],
                commit_message=f"Add failed M2 demo report {stamp}",
                print_fn=print,
            )
        except Exception as exc:
            publish_report = {
                "enabled": True,
                "pushed": False,
                "path": repo_results_rel.as_posix(),
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            print(f"[GIT] Failed M2 report auto-push failed: {publish_report['error']}")
    print(f"[M2] Failure report: {repo_results_rel.as_posix()}")
    return {"run_dir": repo_results_rel.as_posix(), "summary": failure_report["summary"], "publish": publish_report}
