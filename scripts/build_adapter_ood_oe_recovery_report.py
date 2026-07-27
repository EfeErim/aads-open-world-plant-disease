#!/usr/bin/env python3
"""Build the strict eight-target adapter behavioral recovery report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ood.recovery import (  # noqa: E402
    build_recovery_report,
    paths_from_arguments,
    render_recovery_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--acceptance-file", type=Path, action="append", default=[])
    parser.add_argument("--readiness-file", type=Path, action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs") / "ablation_results" / "adapter_ood_oe_recovery_baseline.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs") / "ablation_results" / "adapter_ood_oe_recovery_baseline.md",
    )
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--require-ready", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    explicit_files = [*args.acceptance_file, *args.readiness_file]
    selected = paths_from_arguments(explicit_files) if explicit_files else None
    report = build_recovery_report(args.runs_root, selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_recovery_markdown(report), encoding="utf-8")
    print(
        f"{report['overall_status']}: {report['passed_target_count']}/{report['required_target_count']} targets passed; "
        f"JSON={args.output}; Markdown={args.markdown_output}"
    )
    require_pass = bool(args.require_pass or args.require_ready)
    return 1 if require_pass and report["overall_status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
