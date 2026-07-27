#!/usr/bin/env python3
"""Validate that router calibration results stay within deployment guardrails.

This is the lightweight Tier 1B automation guard from the SOTA guide. By
default it validates an existing `scripts/calibrate_router_surface.py` JSON
report. If `data/router_eval/` is not present, the guard emits a skipped report
and passes so CI can run in source-only checkouts. If router eval images are
present, a missing or unstable calibration report is a failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class CalibrationIssue:
    severity: str
    code: str
    message: str


from scripts.utils.reporting import read_json, write_json  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _metric(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return default
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def count_router_eval_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def validate_calibration_payload(
    payload: dict[str, Any],
    *,
    target_negative_far: float,
    max_crop_accuracy_drop: float,
    max_part_precision_drop: float,
    max_abstention_rate: float,
    min_samples: int,
) -> list[CalibrationIssue]:
    issues: list[CalibrationIssue] = []
    sample_count = int(payload.get("sample_count") or 0)
    if sample_count < min_samples:
        issues.append(
            CalibrationIssue(
                severity="error",
                code="sample_count_too_low",
                message=f"router calibration used {sample_count} samples; expected at least {min_samples}",
            )
        )

    baseline = payload.get("baseline")
    recommended = payload.get("recommended")
    best_rejected = payload.get("best_rejected")
    if not isinstance(baseline, dict):
        issues.append(
            CalibrationIssue(
                severity="error",
                code="baseline_missing",
                message="calibration report must include a baseline variant",
            )
        )
        baseline = {}
    if not isinstance(recommended, dict):
        if isinstance(best_rejected, dict) and best_rejected:
            issues.append(
                CalibrationIssue(
                    severity="error",
                    code="no_eligible_recommendation",
                    message="calibration report did not produce an eligible recommendation",
                )
            )
        else:
            issues.append(
                CalibrationIssue(
                    severity="error",
                    code="recommended_missing",
                    message="calibration report must include a recommended variant",
                )
            )
        recommended = {}

    if not recommended:
        return issues

    negative_far = _metric(recommended, "negative_false_accept_rate")
    if negative_far > target_negative_far:
        issues.append(
            CalibrationIssue(
                severity="error",
                code="negative_false_accept_rate_above_target",
                message=(
                    f"recommended negative_false_accept_rate={negative_far:.4f}; "
                    f"target <= {target_negative_far:.4f}"
                ),
            )
        )

    abstention_rate = _metric(recommended, "abstention_rate")
    if abstention_rate > max_abstention_rate:
        issues.append(
            CalibrationIssue(
                severity="error",
                code="abstention_rate_above_target",
                message=f"recommended abstention_rate={abstention_rate:.4f}; target <= {max_abstention_rate:.4f}",
            )
        )

    crop_drop = _metric(baseline, "crop_accuracy") - _metric(recommended, "crop_accuracy")
    if crop_drop > max_crop_accuracy_drop:
        issues.append(
            CalibrationIssue(
                severity="error",
                code="crop_accuracy_drop",
                message=f"recommended crop accuracy dropped by {crop_drop:.4f}; limit {max_crop_accuracy_drop:.4f}",
            )
        )

    part_precision_drop = _metric(baseline, "part_non_unknown_precision") - _metric(
        recommended, "part_non_unknown_precision"
    )
    if part_precision_drop > max_part_precision_drop:
        issues.append(
            CalibrationIssue(
                severity="error",
                code="part_precision_drop",
                message=(
                    f"recommended part precision dropped by {part_precision_drop:.4f}; "
                    f"limit {max_part_precision_drop:.4f}"
                ),
            )
        )

    if recommended.get("eligible") is False:
        reasons = recommended.get("eligibility_reasons") or []
        issues.append(
            CalibrationIssue(
                severity="error",
                code="recommended_variant_ineligible",
                message=f"recommended variant is marked ineligible: {reasons}",
            )
        )

    return issues


def build_report(
    *,
    router_eval_root: Path,
    calibration_results: Path | None,
    eval_image_count: int,
    issues: list[CalibrationIssue],
    skipped: bool,
) -> dict[str, Any]:
    error_count = sum(1 for issue in issues if issue.severity == "error")
    return {
        "schema_version": "v1_router_calibration_stability_report",
        "ok": error_count == 0,
        "skipped": skipped,
        "router_eval_root": str(router_eval_root),
        "calibration_results": None if calibration_results is None else str(calibration_results),
        "eval_image_count": eval_image_count,
        "error_count": error_count,
        "issues": [asdict(issue) for issue in issues],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-eval-root", type=Path, default=Path("data/router_eval"))
    parser.add_argument(
        "--calibration-results",
        type=Path,
        default=Path(".runtime_tmp/router_calibration.json"),
        help="JSON report produced by scripts/calibrate_router_surface.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runtime_tmp/router_calibration_stability_report.json"),
    )
    parser.add_argument("--target-negative-far", type=float, default=0.05)
    parser.add_argument("--max-crop-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--max-part-precision-drop", type=float, default=0.02)
    parser.add_argument("--max-abstention-rate", type=float, default=0.30)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument(
        "--fail-if-missing-eval",
        action="store_true",
        help="Fail instead of skipping when router eval images are absent.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_image_count = count_router_eval_images(args.router_eval_root)
    issues: list[CalibrationIssue] = []
    skipped = False

    if eval_image_count == 0:
        skipped = not args.fail_if_missing_eval
        if args.fail_if_missing_eval:
            issues.append(
                CalibrationIssue(
                    severity="error",
                    code="router_eval_images_missing",
                    message=f"no router eval images found under {args.router_eval_root}",
                )
            )
    elif not args.calibration_results.exists():
        issues.append(
            CalibrationIssue(
                severity="error",
                code="calibration_results_missing",
                message=(
                    f"router eval images exist but {args.calibration_results} is missing; "
                    "run scripts/calibrate_router_surface.py first"
                ),
            )
        )
    else:
        try:
            payload = _read_json(args.calibration_results)
            issues.extend(
                validate_calibration_payload(
                    payload,
                    target_negative_far=args.target_negative_far,
                    max_crop_accuracy_drop=args.max_crop_accuracy_drop,
                    max_part_precision_drop=args.max_part_precision_drop,
                    max_abstention_rate=args.max_abstention_rate,
                    min_samples=args.min_samples,
                )
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                CalibrationIssue(
                    severity="error",
                    code="calibration_results_unreadable",
                    message=str(exc),
                )
            )

    report = build_report(
        router_eval_root=args.router_eval_root,
        calibration_results=args.calibration_results,
        eval_image_count=eval_image_count,
        issues=issues,
        skipped=skipped,
    )
    _write_json(args.output, report)
    if report["ok"]:
        if skipped:
            print(f"SKIP: no router eval images found; report: {args.output}")
        else:
            print(f"PASS: router calibration stability checked; report: {args.output}")
        return 0
    print(f"FAIL: {report['error_count']} router calibration stability issue(s); report: {args.output}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
