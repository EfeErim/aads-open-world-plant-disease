"""Generate report-only evidence-gate policy recommendations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.evidence_gate_policy_recommendations import (  # noqa: E402
    DEFAULT_CALIBRATION_INPUT,
    DEFAULT_FAILURE_ANALYSIS_INPUT,
    DEFAULT_JSON_OUTPUT,
    DEFAULT_MARKDOWN_OUTPUT,
    build_policy_recommendations,
    load_json_payload,
    load_optional_json_payload,
    write_recommendation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-input", default=str(DEFAULT_CALIBRATION_INPUT))
    parser.add_argument("--failure-analysis-input", default=str(DEFAULT_FAILURE_ANALYSIS_INPUT))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    calibration_input = Path(args.calibration_input)
    failure_analysis_input = Path(args.failure_analysis_input)
    calibration_payload = load_json_payload(calibration_input)
    failure_payload = load_optional_json_payload(failure_analysis_input)
    payload = build_policy_recommendations(
        calibration_payload,
        failure_analysis_payload=failure_payload,
        calibration_input=calibration_input,
        failure_analysis_input=failure_analysis_input,
    )
    write_recommendation_outputs(payload, json_output=args.json_output, markdown_output=args.markdown_output)
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "markdown_output": str(args.markdown_output),
                "schema_version": payload["schema_version"],
                **payload["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
