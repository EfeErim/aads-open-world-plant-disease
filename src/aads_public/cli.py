"""Public command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .evidence import EXPECTED_ANSWERED, EXPECTED_REVIEWED, load_acceptance_report


def _default_evidence() -> Path:
    return Path(__file__).resolve().parent / "data" / "controlled_demo_summary.json"


def replay(path: Path) -> int:
    report = load_acceptance_report(path)
    status = "PASS" if report.controlled_demo_passed else "FAIL"
    print("AADS | GPU-free row snapshot validation")
    print("-----------------------------------------")
    print(f"Run                     {report.run_id}")
    print(f"Manifest identity       {'PASS' if report.identity_verified else 'FAIL'}")
    print(f"Sanitized rows          {report.total}/{report.total}  {'PASS' if report.rows_verified else 'FAIL'}")
    print(f"Controlled acceptance   {report.passed}/{report.total}  {status}")
    print(f"Disease answers         {report.answered}/{EXPECTED_ANSWERED}")
    print(f"Safe review/abstain     {report.reviewed}/{EXPECTED_REVIEWED}")
    print(f"Negative false accepts  {report.negative_false_accepts}")
    print(f"Wrong-part labels       {report.wrong_part_disease_labels}")
    print("Scope                    recorded decisions, not fresh inference | NOT production-ready")
    return 0 if report.controlled_demo_passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aads", description="AADS public engineering demo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay_parser = subparsers.add_parser(
        "replay",
        help="validate the frozen row-level decision snapshot (does not rerun inference)",
    )
    replay_parser.add_argument("--evidence", type=Path, default=_default_evidence())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "replay":
        return replay(args.evidence)
    raise AssertionError("unreachable")
