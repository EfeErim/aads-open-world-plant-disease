#!/usr/bin/env python3
"""Evaluate whether a staged adapter OOD/OE candidate may replace the baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ood.recovery_promotion import evaluate_candidate_promotion  # noqa: E402


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--integrity-report", type=Path, required=True)
    parser.add_argument("--reload-parity", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs") / "ablation_results" / "adapter_ood_oe_promotion.json",
    )
    parser.add_argument("--max-worst-slice-fpr-regression", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = evaluate_candidate_promotion(
            _read(args.baseline),
            _read(args.candidate),
            integrity_report=_read(args.integrity_report),
            reload_parity={key: bool(value) for key, value in _read(args.reload_parity).items()},
            max_worst_slice_fpr_regression=args.max_worst_slice_fpr_regression,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": "v1_adapter_ood_oe_promotion_report",
            "overall_promote": False,
            "error": str(exc),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{'PROMOTE' if report['overall_promote'] else 'REJECT'}: "
        f"{report.get('promoted_target_count', 0)}/8 targets; report={args.output}"
    )
    return 0 if report["overall_promote"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
