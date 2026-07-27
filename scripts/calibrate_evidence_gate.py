"""Calibrate an advisory evidence-gate policy from Notebook 16 reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.evidence_gate_calibration import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_SEED,
    DEFAULT_SOURCE_REPORT,
    calibrate_evidence_gate_report,
    load_notebook16_rows,
    write_calibration_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_SOURCE_REPORT), help="Notebook 16 multi-target report JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Calibration artifact output JSON.")
    parser.add_argument("--schema-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--min-capture", type=float, default=0.70)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.15)
    parser.add_argument("--min-target-errors", type=int, default=20)
    parser.add_argument("--holdout-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-review-rate", type=float, default=0.25)
    parser.add_argument("--min-calibration-errors", type=int, default=10)
    parser.add_argument("--min-holdout-errors", type=int, default=5)
    parser.add_argument("--max-holdout-capture-drop", type=float, default=0.15)
    parser.add_argument("--max-holdout-fp-increase", type=float, default=0.10)
    parser.add_argument("--include-samples", action="store_true", help="Include sample-level split rows for debugging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = load_notebook16_rows(input_path)
    payload = calibrate_evidence_gate_report(
        rows,
        source_report=input_path,
        schema_version=args.schema_version,
        min_capture=args.min_capture,
        max_false_positive_rate=args.max_false_positive_rate,
        min_target_errors=args.min_target_errors,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
        include_samples=args.include_samples,
        max_review_rate=args.max_review_rate,
        min_calibration_errors=args.min_calibration_errors,
        min_holdout_errors=args.min_holdout_errors,
        max_holdout_capture_drop=args.max_holdout_capture_drop,
        max_holdout_fp_increase=args.max_holdout_fp_increase,
    )
    write_calibration_report(payload, output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "schema_version": payload["schema_version"],
                "global_status": payload["global_policy"]["status"],
                "target_count": len(payload["target_policies"]),
                "audit_queue_count": len(payload.get("audit_queue", [])),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
