#!/usr/bin/env python3
"""Run the supported and open-world router validation manifests, then gate readiness."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_router_open_world_readiness import (  # noqa: E402
    evaluate_readiness,
    read_json,
    readiness_markdown,
    write_failure_csvs,
)
from src.data.dataset_release_runtime import DEFAULT_MATERIALIZED_DATASET_ROOT  # noqa: E402

DEFAULT_SUPPORTED_MANIFEST = Path("docs/demo_assets/m2_full_image_set/manifests/m2_balanced_80_run_manifest.csv")
DEFAULT_OPEN_WORLD_MANIFEST = Path("docs/demo_assets/open_world_router/manifests/m2_open_world_router_manifest.csv")
DEFAULT_OUTPUT_ROOT = Path("docs/demo_results/router_open_world")
DEFAULT_OPEN_WORLD_DISJOINT_ROOTS = [
    Path("docs/demo_assets/m2_full_image_set/images"),
    DEFAULT_MATERIALIZED_DATASET_ROOT,
    Path("docs/demo_assets/prototype_curation"),
]
PROTOTYPE_ARTIFACT_FILES = {
    "prototype_bank": "prototype_bank.json",
    "taxonomy_registry": "taxonomy_registry.json",
    "prototype_calibration_report": "router_prototype_calibration.json",
}
PROVENANCE_FILENAMES = {
    "supported_manifest": "supported_manifest.csv",
    "open_world_manifest": "open_world_manifest.csv",
    "baseline_summary": "baseline_summary.json",
    "prototype_bank": "prototype_bank.json",
    "taxonomy_registry": "taxonomy_registry.json",
    "prototype_calibration_report": "router_prototype_calibration.json",
}


def _coalesce_artifact_path(explicit_path: Path | None, artifact_dir: Path | None, filename: str) -> Path | None:
    if explicit_path:
        return explicit_path
    if not artifact_dir:
        return None
    candidate = artifact_dir / filename
    return candidate if candidate.is_file() else None


def _derive_p95_latency_from_summary(path: Path | None) -> float | None:
    if not path:
        return None
    payload = read_json(path)
    for key in ("p95_latency_ms", "router_p95_latency_ms"):
        value = payload.get(key)
        if value is not None:
            return float(value)
    for section_key in ("summary", "metrics", "timing", "latency"):
        section = payload.get(section_key)
        if not isinstance(section, dict):
            continue
        for key in ("p95_latency_ms", "router_p95_latency_ms", "p95_ms"):
            value = section.get(key)
            if value is not None:
                return float(value)
    total = None
    summary = payload.get("summary")
    if isinstance(summary, dict):
        total = summary.get("total")
    elapsed = payload.get("runner_elapsed_seconds") or payload.get("elapsed_seconds")
    if total and elapsed:
        return float(elapsed) * 1000.0 / float(total)
    return None


def _read_report_or_empty(path: Path, *, label: str) -> tuple[dict, str]:
    if not path.is_file():
        return {"rows": [], "summary": {"total": 0}}, f"{label}_report_missing"
    try:
        return read_json(path), ""
    except Exception as exc:
        return {
            "rows": [],
            "summary": {"total": 0},
            "report_read_error": f"{exc.__class__.__name__}: {exc}",
        }, f"{label}_report_invalid"


def _copy_if_present(source: Path | None, destination: Path) -> str:
    if not source or not source.is_file():
        return ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination.as_posix()


def _write_run_provenance(args: argparse.Namespace, run_dir: Path) -> dict[str, str]:
    provenance_dir = run_dir / "provenance"
    return {
        "supported_manifest": _copy_if_present(
            args.supported_manifest,
            provenance_dir / PROVENANCE_FILENAMES["supported_manifest"],
        ),
        "open_world_manifest": _copy_if_present(
            args.open_world_manifest,
            provenance_dir / PROVENANCE_FILENAMES["open_world_manifest"],
        ),
        "baseline_summary": _copy_if_present(
            args.baseline_summary,
            provenance_dir / PROVENANCE_FILENAMES["baseline_summary"],
        ),
        "prototype_bank": _copy_if_present(
            args.resolved_prototype_bank,
            provenance_dir / PROVENANCE_FILENAMES["prototype_bank"],
        ),
        "taxonomy_registry": _copy_if_present(
            args.resolved_taxonomy_registry,
            provenance_dir / PROVENANCE_FILENAMES["taxonomy_registry"],
        ),
        "prototype_calibration_report": _copy_if_present(
            args.resolved_prototype_calibration_report,
            provenance_dir / PROVENANCE_FILENAMES["prototype_calibration_report"],
        ),
    }


def _run_demo_command(
    *,
    manifest: Path,
    output_json: Path,
    output_markdown: Path,
    analysis_json: Path,
    analysis_markdown: Path,
    args: argparse.Namespace,
) -> int:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_demo_checklist.py"),
        "--no-checklist",
        "--extra-manifest",
        str(manifest),
        "--output",
        str(output_json),
        "--markdown-output",
        str(output_markdown),
        "--analysis-output",
        str(analysis_json),
        "--analysis-markdown-output",
        str(analysis_markdown),
        "--mode",
        str(args.mode),
        "--config-env",
        str(args.config_env),
        "--device",
        str(args.device),
        "--adapter-root",
        str(args.adapter_root),
        "--batch-size",
        str(max(1, int(args.batch_size))),
        "--adapter-batch-size",
        str(max(1, int(args.adapter_batch_size))),
        "--handoff-cache",
        str(args.handoff_cache),
    ]
    if args.enable_prototype_reconciler:
        command.append("--enable-prototype-reconciler")
    if args.resolved_prototype_bank:
        command.extend(["--prototype-bank", str(args.resolved_prototype_bank)])
    if args.resolved_taxonomy_registry:
        command.extend(["--taxonomy-registry", str(args.resolved_taxonomy_registry)])
    if args.resolved_prototype_calibration_report:
        command.extend(["--prototype-calibration-report", str(args.resolved_prototype_calibration_report)])
    if args.prototype_min_similarity is not None:
        command.extend(["--prototype-min-similarity", str(args.prototype_min_similarity)])
    if args.prototype_min_margin is not None:
        command.extend(["--prototype-min-margin", str(args.prototype_min_margin)])
    if args.prototype_min_negative_gap is not None:
        command.extend(["--prototype-min-negative-gap", str(args.prototype_min_negative_gap)])
    if args.refresh_handoff_cache:
        command.append("--refresh-handoff-cache")
    if args.limit is not None:
        command.extend(["--limit", str(max(0, int(args.limit)))])
    if args.stop_on_dependency_blocker:
        command.append("--stop-on-dependency-blocker")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--supported-manifest", type=Path, default=DEFAULT_SUPPORTED_MANIFEST)
    parser.add_argument("--open-world-manifest", type=Path, default=DEFAULT_OPEN_WORLD_MANIFEST)
    parser.add_argument(
        "--open-world-disjoint-root",
        action="append",
        type=Path,
        default=list(DEFAULT_OPEN_WORLD_DISJOINT_ROOTS),
    )
    parser.add_argument("--mode", choices=("official", "asset-audit"), default="official")
    parser.add_argument("--adapter-root", type=Path, default=Path("runs"))
    parser.add_argument("--config-env", default="colab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--enable-prototype-reconciler", action="store_true")
    parser.add_argument("--prototype-artifact-dir", type=Path)
    parser.add_argument("--prototype-bank", type=Path)
    parser.add_argument("--taxonomy-registry", type=Path)
    parser.add_argument("--prototype-calibration-report", type=Path)
    parser.add_argument("--prototype-min-similarity", type=float)
    parser.add_argument("--prototype-min-margin", type=float)
    parser.add_argument("--prototype-min-negative-gap", type=float)
    parser.add_argument("--handoff-cache", type=Path, default=Path(".runtime_tmp/router_open_world_handoff_cache.json"))
    parser.add_argument("--refresh-handoff-cache", action="store_true")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--adapter-batch-size", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-open-world-rows", type=int, default=300)
    parser.add_argument("--min-supported-route-coverage", type=float, default=0.80)
    parser.add_argument("--baseline-p95-latency-ms", type=float)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--candidate-p95-latency-ms", type=float)
    parser.add_argument("--max-latency-regression", type=float, default=0.25)
    parser.add_argument("--require-latency-baseline", action="store_true")
    parser.add_argument("--stop-on-dependency-blocker", action="store_true")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.resolved_prototype_bank = _coalesce_artifact_path(
        args.prototype_bank,
        args.prototype_artifact_dir,
        PROTOTYPE_ARTIFACT_FILES["prototype_bank"],
    )
    args.resolved_taxonomy_registry = _coalesce_artifact_path(
        args.taxonomy_registry,
        args.prototype_artifact_dir,
        PROTOTYPE_ARTIFACT_FILES["taxonomy_registry"],
    )
    args.resolved_prototype_calibration_report = _coalesce_artifact_path(
        args.prototype_calibration_report,
        args.prototype_artifact_dir,
        PROTOTYPE_ARTIFACT_FILES["prototype_calibration_report"],
    )
    baseline_p95_latency_ms = args.baseline_p95_latency_ms
    if baseline_p95_latency_ms is None:
        baseline_p95_latency_ms = _derive_p95_latency_from_summary(args.baseline_summary)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / run_id
    copied_provenance = _write_run_provenance(args, run_dir)
    supported_report = run_dir / "supported_balanced_run.json"
    open_world_report = run_dir / "open_world_run.json"
    supported_exit = _run_demo_command(
        manifest=args.supported_manifest,
        output_json=supported_report,
        output_markdown=run_dir / "supported_balanced_run.md",
        analysis_json=run_dir / "supported_balanced_analysis.json",
        analysis_markdown=run_dir / "supported_balanced_analysis.md",
        args=args,
    )
    open_world_exit = _run_demo_command(
        manifest=args.open_world_manifest,
        output_json=open_world_report,
        output_markdown=run_dir / "open_world_run.md",
        analysis_json=run_dir / "open_world_analysis.json",
        analysis_markdown=run_dir / "open_world_analysis.md",
        args=args,
    )

    supported_payload, supported_report_warning = _read_report_or_empty(supported_report, label="supported")
    open_world_payload, open_world_report_warning = _read_report_or_empty(open_world_report, label="open_world")
    readiness = evaluate_readiness(
        supported_payload=supported_payload,
        open_world_payload=open_world_payload,
        supported_manifest=args.supported_manifest,
        open_world_manifest=args.open_world_manifest,
        open_world_disjoint_roots=list(args.open_world_disjoint_root),
        min_open_world_rows=args.min_open_world_rows,
        min_supported_route_coverage=args.min_supported_route_coverage,
        baseline_p95_latency_ms=baseline_p95_latency_ms,
        candidate_p95_latency_ms=args.candidate_p95_latency_ms,
        max_latency_regression=args.max_latency_regression,
        require_latency_baseline=args.require_latency_baseline,
    )
    readiness["run_id"] = run_id
    readiness["run_dir"] = run_dir.as_posix()
    readiness["runner_exit_codes"] = {
        "supported": supported_exit,
        "open_world": open_world_exit,
    }
    readiness["checks"]["supported_report_written"] = not supported_report_warning
    readiness["checks"]["open_world_report_written"] = not open_world_report_warning
    for warning in (supported_report_warning, open_world_report_warning):
        if warning:
            readiness.setdefault("warnings", []).append(warning)
            readiness["status"] = "fail"
    readiness["resolved_inputs"] = {
        "prototype_bank": str(args.resolved_prototype_bank or ""),
        "taxonomy_registry": str(args.resolved_taxonomy_registry or ""),
        "prototype_calibration_report": str(args.resolved_prototype_calibration_report or ""),
        "baseline_summary": str(args.baseline_summary or ""),
        "baseline_p95_latency_ms_source": "explicit" if args.baseline_p95_latency_ms is not None else "summary",
    }
    readiness["copied_provenance"] = copied_provenance
    if supported_exit != 0 or open_world_exit != 0:
        readiness["status"] = "fail"
        readiness["checks"]["runner_exit_codes_zero"] = False
    else:
        readiness["checks"]["runner_exit_codes_zero"] = True

    readiness_json = run_dir / "router_open_world_readiness.json"
    readiness_md = run_dir / "router_open_world_readiness.md"
    readiness_json.write_text(json.dumps(readiness, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    readiness_md.write_text(readiness_markdown(readiness), encoding="utf-8")
    write_failure_csvs(readiness, run_dir / "failures")
    print(json.dumps(readiness, indent=2, ensure_ascii=False))
    return 1 if args.fail_on_not_ready and readiness["status"] != "pass" else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
