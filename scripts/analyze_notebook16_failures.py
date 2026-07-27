"""Analyze Notebook 16 multi-target review-gate failures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.notebook16_failure_analysis import (  # noqa: E402
    DEFAULT_CALIBRATION_INPUT,
    DEFAULT_FOCUS_TARGET,
    DEFAULT_JSON_OUTPUT,
    DEFAULT_MARKDOWN_OUTPUT,
    DEFAULT_SOURCE_REPORT,
    build_notebook16_failure_analysis,
    load_calibration_statuses,
    load_notebook16_report,
    write_analysis_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_SOURCE_REPORT), help="Notebook 16 multi-target report JSON.")
    parser.add_argument(
        "--calibration-input",
        default=str(DEFAULT_CALIBRATION_INPUT),
        help="Optional evidence-gate calibration artifact JSON.",
    )
    parser.add_argument("--json-output", default=str(DEFAULT_JSON_OUTPUT), help="Machine-readable analysis JSON.")
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT), help="Handoff Markdown report.")
    parser.add_argument("--focus-target", default=DEFAULT_FOCUS_TARGET)
    parser.add_argument("--top-examples", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    calibration_input = Path(args.calibration_input)
    rows = load_notebook16_report(input_path)
    calibration_statuses = load_calibration_statuses(calibration_input)
    payload = build_notebook16_failure_analysis(
        rows,
        calibration_statuses=calibration_statuses,
        focus_target=args.focus_target,
        top_examples=args.top_examples,
        source_report=input_path,
        calibration_input=calibration_input,
    )
    write_analysis_outputs(payload, json_output=args.json_output, markdown_output=args.markdown_output)
    focus = payload.get("focus_target_summary") if isinstance(payload.get("focus_target_summary"), dict) else {}
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "markdown_output": str(args.markdown_output),
                "schema_version": payload["schema_version"],
                "target_count": payload["target_count"],
                "focus_target": payload["focus_target"],
                "focus_missed_wrong_count": focus.get("missed_wrong_count"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
