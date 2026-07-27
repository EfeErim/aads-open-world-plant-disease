"""Download and verify the public controlled-demo adapter release."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.pipeline.adapter_release import fetch_adapter_release

REPOSITORY = "EfeErim/bitirmeprojesi"
DEFAULT_MANIFEST = Path("docs/evidence/current/demo_release/release_manifest.json")
DEFAULT_DESTINATION = Path("models/adapters")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the public AADS adapters and verify their release manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    return parser


def main() -> int:
    args = _parser().parse_args()
    os.environ.setdefault("GITHUB_RELEASE_REPOSITORY", REPOSITORY)
    receipt = fetch_adapter_release(
        args.manifest.resolve(),
        args.destination.resolve(),
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
