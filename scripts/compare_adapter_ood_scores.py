#!/usr/bin/env python3
"""Compare ensemble, energy, and kNN scores from adapter OOD-dev JSONL evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ood.score_comparison import build_score_comparison  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="JSONL rows with target, split_role, ood_label, ood_type, scores.")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs") / "ablation_results" / "adapter_ood_score_comparison.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = [
            json.loads(line)
            for line in args.evidence.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report = build_score_comparison(rows, target_fpr=args.target_fpr)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": "v1_adapter_ood_score_comparison",
            "ok": False,
            "issues": [{"code": "evidence_unreadable", "message": str(exc)}],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{'PASS' if report['ok'] else 'FAIL'}: selected={report.get('selected_target_count', 0)}/8 "
        f"report={args.output}"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
