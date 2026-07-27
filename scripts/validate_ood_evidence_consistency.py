"""Validate that production readiness verdicts are backed by OOD evidence.

This is a lightweight automation guard for existing run artifacts. It does not
recompute OOD metrics; it checks that each `production_readiness.json` verdict
is internally consistent with the repo's evidence contract:

- `ready` requires real held-out OOD evidence.
- `provisional` may use the held-out benchmark fallback, but must include the
  benchmark summary artifact.
- deployable verdicts must not report missing real-OOD deployment evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEPLOYMENT_STATUSES = {"ready", "provisional"}
REAL_OOD_SOURCE = "real_ood_split"
FALLBACK_SOURCE = "held_out_benchmark"


@dataclass(frozen=True)
class EvidenceIssue:
    severity: str
    path: str
    code: str
    message: str


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _normalise_status(payload: dict[str, Any]) -> str:
    return str(payload.get("status") or payload.get("readiness_status") or "").strip().lower()


def _normalise_source(payload: dict[str, Any]) -> str:
    source = payload.get("ood_evidence_source")
    if not source and isinstance(payload.get("ood_evidence"), dict):
        source = payload["ood_evidence"].get("source")
    return str(source or "").strip()


def _metric_number(payload: dict[str, Any], key: str) -> float | None:
    candidates: list[Any] = []
    ood_evidence = payload.get("ood_evidence")
    if isinstance(ood_evidence, dict):
        metrics = ood_evidence.get("metrics")
        if isinstance(metrics, dict):
            candidates.append(metrics.get(key))
        candidates.append(ood_evidence.get(key))
    metrics = payload.get("ood_metrics")
    if isinstance(metrics, dict):
        candidates.append(metrics.get(key))
    candidates.append(payload.get(key))

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return float(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _evidence_sample_count(payload: dict[str, Any]) -> int | None:
    for key in ("ood_samples", "sample_count", "image_count", "n_ood"):
        value = _metric_number(payload, key)
        if value is not None:
            return int(value)
    return None


def _benchmark_summary_path(readiness_path: Path) -> Path:
    return readiness_path.parent / "ood_benchmark" / "summary.json"


def validate_readiness_file(
    readiness_path: Path,
    *,
    min_real_ood_images: int,
) -> list[EvidenceIssue]:
    payload = _read_json(readiness_path)
    status = _normalise_status(payload)
    source = _normalise_source(payload)
    issues: list[EvidenceIssue] = []
    path_text = str(readiness_path)

    if status not in DEPLOYMENT_STATUSES:
        return issues

    missing_deployment = payload.get("missing_deployment_requirements")
    if isinstance(missing_deployment, list) and "real_ood_evidence" in missing_deployment and status == "ready":
        issues.append(
            EvidenceIssue(
                severity="error",
                path=path_text,
                code="ready_missing_real_ood_requirement",
                message="ready verdict still lists real_ood_evidence as a missing deployment requirement",
            )
        )

    if status == "ready" and source != REAL_OOD_SOURCE:
        issues.append(
            EvidenceIssue(
                severity="error",
                path=path_text,
                code="ready_without_real_ood",
                message=f"ready verdict must use {REAL_OOD_SOURCE}; found {source or '<empty>'}",
            )
        )

    if source == REAL_OOD_SOURCE:
        sample_count = _evidence_sample_count(payload)
        if sample_count is None:
            issues.append(
                EvidenceIssue(
                    severity="error",
                    path=path_text,
                    code="real_ood_sample_count_missing",
                    message="real OOD evidence is reported but no OOD sample count was found",
                )
            )
        elif sample_count < min_real_ood_images:
            issues.append(
                EvidenceIssue(
                    severity="error",
                    path=path_text,
                    code="real_ood_sample_count_too_low",
                    message=f"real OOD evidence has {sample_count} samples; expected at least {min_real_ood_images}",
                )
            )
    elif source == FALLBACK_SOURCE:
        benchmark_summary = _benchmark_summary_path(readiness_path)
        if not benchmark_summary.exists():
            issues.append(
                EvidenceIssue(
                    severity="error",
                    path=path_text,
                    code="fallback_summary_missing",
                    message=f"fallback OOD evidence requires {benchmark_summary}",
                )
            )
        if status == "ready":
            issues.append(
                EvidenceIssue(
                    severity="error",
                    path=path_text,
                    code="fallback_marked_ready",
                    message="held-out benchmark fallback may be provisional, not ready",
                )
            )
    else:
        issues.append(
            EvidenceIssue(
                severity="error",
                path=path_text,
                code="ood_evidence_source_missing",
                message=f"{status} verdict must declare real or fallback OOD evidence; found {source or '<empty>'}",
            )
        )

    return issues


def collect_readiness_files(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return sorted(
        path
        for path in runs_root.rglob("production_readiness.json")
        if "_index" not in path.relative_to(runs_root).parts
    )


def build_report(
    *,
    runs_root: Path,
    readiness_files: list[Path],
    issues: list[EvidenceIssue],
    min_real_ood_images: int,
) -> dict[str, Any]:
    error_count = sum(1 for issue in issues if issue.severity == "error")
    return {
        "schema_version": "v1_ood_evidence_consistency_report",
        "ok": error_count == 0,
        "runs_root": str(runs_root),
        "readiness_file_count": len(readiness_files),
        "min_real_ood_images": int(min_real_ood_images),
        "error_count": error_count,
        "issues": [asdict(issue) for issue in issues],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"), help="Root containing run artifact folders.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runtime_tmp") / "ood_consistency_report.json",
        help="Path for the JSON report.",
    )
    parser.add_argument(
        "--min-real-ood-images",
        type=int,
        default=10,
        help="Minimum real held-out OOD samples required for deployable evidence.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    readiness_files = collect_readiness_files(args.runs_root)
    issues: list[EvidenceIssue] = []
    for readiness_file in readiness_files:
        try:
            issues.extend(validate_readiness_file(readiness_file, min_real_ood_images=args.min_real_ood_images))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                EvidenceIssue(
                    severity="error",
                    path=str(readiness_file),
                    code="readiness_file_unreadable",
                    message=str(exc),
                )
            )

    report = build_report(
        runs_root=args.runs_root,
        readiness_files=readiness_files,
        issues=issues,
        min_real_ood_images=args.min_real_ood_images,
    )
    _write_json(args.output, report)
    if report["ok"]:
        print(f"PASS: {len(readiness_files)} production_readiness.json files checked")
        return 0
    print(f"FAIL: {report['error_count']} OOD evidence consistency issue(s); report: {args.output}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
