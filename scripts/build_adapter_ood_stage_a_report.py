#!/usr/bin/env python3
"""Build validation-selected Stage-A OOD score recommendations from run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ood.stage_a import build_stage_a_report, render_stage_a_markdown  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs") / "ablation_results" / "adapter_ood_stage_a.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs") / "ablation_results" / "adapter_ood_stage_a.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_stage_a_report(args.runs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_stage_a_markdown(report), encoding="utf-8")
    print(
        f"{'PASS' if report['ok'] else 'FAIL'}: selected={report['selected_target_count']}/8 "
        f"pooled_test_gate_pass={report['pooled_test_gate_pass_target_count']}/8 "
        f"promotable={report['stage_a_promotable_target_count']}/8"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
