#!/usr/bin/env python3
"""Calibrate prototype-router reconciliation thresholds on a held-out manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_demo_checklist import parse_manifest_rows, resolve_image_path  # noqa: E402
from src.router.prototype_calibration import ScoredRow, calibrate, has_runtime_policy  # noqa: E402
from src.router.prototype_reconciler import nearest_target  # noqa: E402
from src.shared.json_utils import read_json, write_json  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_float_grid(value: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not value:
        return default
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def score_manifest(
    *,
    manifest_path: Path,
    prototype_bank_path: Path,
    repo_root: Path,
    limit: int | None = None,
) -> list[ScoredRow]:
    prototype_payload = read_json(prototype_bank_path, default={}, expect_type=dict)
    rows = parse_manifest_rows(manifest_path)
    if limit is not None:
        rows = rows[: max(0, int(limit))]

    scored: list[ScoredRow] = []
    for row in rows:
        image_path, asset_status = resolve_image_path(row.source, repo_root)
        if asset_status != "ok" or image_path is None:
            scored.append(
                ScoredRow(
                    image_id=row.image_id,
                    expected_target=row.expected_target,
                    expected_behavior=row.expected_behavior,
                    predicted_target=None,
                    similarity=0.0,
                    margin=0.0,
                    resolved_image="" if image_path is None else str(image_path),
                    status=asset_status,
                    prototype_class_label=None,
                    prototype_level="unavailable",
                    expected_class=row.expected_class,
                )
            )
            continue
        try:
            match = nearest_target(image_path, prototype_payload)
            scored.append(
                ScoredRow(
                    image_id=row.image_id,
                    expected_target=row.expected_target,
                    expected_behavior=row.expected_behavior,
                    predicted_target=match.target_id,
                    similarity=match.similarity,
                    margin=match.margin,
                    resolved_image=str(image_path),
                    status="ok",
                    prototype_class_label=match.class_label,
                    prototype_level=match.prototype_level,
                    expected_class=row.expected_class,
                )
            )
        except Exception as exc:
            scored.append(
                ScoredRow(
                    image_id=row.image_id,
                    expected_target=row.expected_target,
                    expected_behavior=row.expected_behavior,
                    predicted_target=None,
                    similarity=0.0,
                    margin=0.0,
                    resolved_image=str(image_path),
                    status=f"error:{exc}",
                    prototype_class_label=None,
                    prototype_level="error",
                    expected_class=row.expected_class,
                )
            )
    return scored


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prototype-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(".runtime_tmp/router_prototype_calibration.json"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--limit", type=int)
    parser.add_argument("--similarity-grid", default="0.20,0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--margin-grid", default="0.00,0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--negative-gap-grid", default="0.00,0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--min-precision", type=float, default=0.985)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--allow-non-plant-false-accepts", action="store_true")
    parser.add_argument("--max-negative-false-accepts", type=int, default=0)
    parser.add_argument("--max-negative-false-accept-rate", type=float, default=0.05)
    parser.add_argument(
        "--target-min-precision",
        type=float,
        default=0.98,
        help=(
            "Precision floor for target-specific policies. The global policy still uses --min-precision; "
            "the default recovers near-miss targets without promoting noisy targets."
        ),
    )
    parser.add_argument(
        "--target-max-supported-wrong",
        type=int,
        default=1,
        help="Maximum wrong supported rows allowed for a target-specific policy.",
    )
    parser.add_argument(
        "--target-max-cross-part-supported-wrong",
        type=int,
        default=0,
        help=(
            "Maximum supported rows from the opposite fruit/leaf part that a target or class policy may accept "
            "when checked against the full calibration set."
        ),
    )
    parser.add_argument(
        "--target-policy-negative-mode",
        choices=("all", "none"),
        default="all",
        help=(
            "Use all negative rows when selecting each target policy, or select target policies from target rows only. "
            "The global selected policy always keeps the full negative guard."
        ),
    )
    parser.add_argument(
        "--target-class-min-accepted",
        type=int,
        default=5,
        help=(
            "Minimum accepted supported rows before a class-specific target policy can be used. "
            "Class policies are evaluated against the full calibration set to preserve false-accept guards."
        ),
    )
    parser.add_argument("--fail-on-no-policy", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = score_manifest(
        manifest_path=args.manifest,
        prototype_bank_path=args.prototype_bank,
        repo_root=args.repo_root,
        limit=args.limit,
    )
    calibration = calibrate(
        rows,
        similarity_grid=_parse_float_grid(args.similarity_grid, (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)),
        margin_grid=_parse_float_grid(args.margin_grid, (0.00, 0.02, 0.04, 0.06, 0.08, 0.10)),
        negative_gap_grid=_parse_float_grid(args.negative_gap_grid, (0.00, 0.02, 0.04, 0.06, 0.08, 0.10)),
        min_precision=args.min_precision,
        min_coverage=args.min_coverage,
        require_zero_non_plant_false_accepts=not args.allow_non_plant_false_accepts,
        max_negative_false_accepts=args.max_negative_false_accepts,
        max_negative_false_accept_rate=args.max_negative_false_accept_rate,
        target_min_precision=args.target_min_precision,
        target_max_supported_wrong=args.target_max_supported_wrong,
        target_max_cross_part_supported_wrong=args.target_max_cross_part_supported_wrong,
        target_policy_negative_mode=args.target_policy_negative_mode,
        target_class_min_accepted=args.target_class_min_accepted,
    )
    payload = {
        "schema_version": "router_prototype_calibration.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest) if args.manifest.is_file() else "",
        "prototype_bank": str(args.prototype_bank),
        "prototype_bank_sha256": _sha256_file(args.prototype_bank) if args.prototype_bank.is_file() else "",
        "constraints": {
            "min_precision": args.min_precision,
            "min_coverage": args.min_coverage,
            "require_zero_non_plant_false_accepts": not args.allow_non_plant_false_accepts,
            "max_negative_false_accepts": args.max_negative_false_accepts,
            "max_negative_false_accept_rate": args.max_negative_false_accept_rate,
            "target_min_precision": args.target_min_precision,
            "target_max_supported_wrong": args.target_max_supported_wrong,
            "target_max_cross_part_supported_wrong": args.target_max_cross_part_supported_wrong,
            "target_policy_negative_mode": args.target_policy_negative_mode,
            "target_class_min_accepted": args.target_class_min_accepted,
            "class_part_conflict_override": "clean_fruit_class",
            "expected_class_rescue": "clean_exact_class_v2_ignore_hard_negative",
            "promotion_mode": "prototype_override",
        },
        "summary": {
            "rows": len(rows),
            "ok_rows": sum(1 for row in rows if row.status == "ok"),
            "selected_for_runtime": has_runtime_policy(calibration),
        },
        **calibration,
        "rows": [row.__dict__ for row in rows],
    }
    output = write_json(args.output, payload, ensure_ascii=False, sort_keys=False)
    print(json.dumps({"output": str(output), **payload["summary"]}, indent=2, ensure_ascii=False))
    if args.fail_on_no_policy and not payload["summary"]["selected_for_runtime"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
