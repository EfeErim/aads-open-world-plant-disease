#!/usr/bin/env python3
"""Replay a few checked-in controlled-demo decisions without model downloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / "evidence" / "controlled_demo_summary.json"
DEFAULT_ROWS = REPO_ROOT / "evidence" / "controlled_demo_rows.json"


def load_demo_evidence(
    summary_path: Path = DEFAULT_SUMMARY,
    rows_path: Path = DEFAULT_ROWS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    rows_payload = json.loads(Path(rows_path).read_text(encoding="utf-8"))
    rows = [dict(row) for row in rows_payload.get("rows", [])]

    if str(summary.get("run_id", "")) != str(rows_payload.get("run_id", "")):
        raise ValueError("Controlled-demo summary and rows use different run IDs.")
    if int(summary.get("total", -1)) != len(rows):
        raise ValueError("Controlled-demo row count does not match the summary.")
    if int(summary.get("passed", -1)) != sum(bool(row.get("passed")) for row in rows):
        raise ValueError("Controlled-demo pass count does not match the rows.")
    return dict(summary), rows


def select_replay_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selectors = (
        lambda row: row.get("actual_outcome") == "answer",
        lambda row: row.get("expected_target") == "unknown_crop" and row.get("actual_outcome") == "review",
        lambda row: row.get("expected_target") == "non_plant" and row.get("actual_outcome") == "review",
    )
    selected: list[dict[str, Any]] = []
    for selector in selectors:
        match = next((row for row in rows if selector(row)), None)
        if match is not None:
            selected.append(match)
    return selected


def build_replay_payload(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mode": "recorded_controlled_demo_replay",
        "fresh_inference": False,
        "run_id": summary["run_id"],
        "examples": select_replay_rows(rows),
        "summary": {
            "passed": summary["passed"],
            "total": summary["total"],
            "answered": summary["answered"],
            "reviewed_or_abstained": summary["reviewed_or_abstained"],
            "negative_false_accepts": summary["negative_false_accepts"],
            "wrong_part_disease_labels": summary["wrong_part_disease_labels"],
            "production_ready": summary["production_ready"],
        },
    }


def _display_target(row: dict[str, Any]) -> str:
    return str(row.get("actual_target") or row.get("expected_target") or "unknown").replace("__", " / ")


def format_replay(payload: dict[str, Any]) -> str:
    lines = [
        "AADS controlled-demo replay",
        f"run: {payload['run_id']}",
        "mode: checked-in decisions (not fresh model inference)",
        "",
    ]
    for row in payload["examples"]:
        row_id = str(row.get("row_id", "unknown"))
        outcome = str(row.get("actual_outcome", "unknown")).upper()
        if outcome == "ANSWER":
            detail = f"{_display_target(row)} -> {row.get('predicted_class')}"
        else:
            detail = f"{_display_target(row)} -> {row.get('actual_status')}"
        lines.append(f"[{outcome:6}] {row_id}: {detail}")

    summary = payload["summary"]
    lines.extend(
        [
            "",
            (
                f"summary: {summary['passed']}/{summary['total']} passed | "
                f"{summary['answered']} answered | {summary['reviewed_or_abstained']} review/abstain"
            ),
            (
                f"safety: {summary['negative_false_accepts']} false accepts | "
                f"{summary['wrong_part_disease_labels']} wrong-part labels"
            ),
            "production safety gate: not passed",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay representative rows from the checked-in 48-row controlled demo.",
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--json", action="store_true", help="Print the replay payload as JSON.")
    args = parser.parse_args(argv)

    summary, rows = load_demo_evidence(args.summary, args.rows)
    payload = build_replay_payload(summary, rows)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(format_replay(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
