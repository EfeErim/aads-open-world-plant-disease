#!/usr/bin/env python3
"""Restore the latest published Notebook 5 router calibration artifact locally."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_PUBLISHED_ROOT = Path("runs/_index/router_calibration")
DEFAULT_OUTPUT = Path(".runtime_tmp/router_calibration.json")


def _published_calibration_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("*/router_calibration.json"), key=lambda path: path.parent.name, reverse=True)


def restore_latest_router_calibration(published_root: Path, output: Path) -> dict[str, str]:
    candidates = list(_published_calibration_files(published_root))
    if not candidates:
        raise FileNotFoundError(
            f"No published router_calibration.json files found under {published_root}. "
            "Run Notebook 5 in Colab with PUBLISH_RESULTS_TO_GIT=True first."
        )

    source = candidates[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    return {
        "source": str(source),
        "output": str(output),
        "published_run": source.parent.name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-root", type=Path, default=DEFAULT_PUBLISHED_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", help="Print a machine-readable restore summary.")
    args = parser.parse_args()

    try:
        summary = restore_latest_router_calibration(args.published_root, args.output)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            "Restored router calibration artifact: "
            f"{summary['source']} -> {summary['output']} (published_run={summary['published_run']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
