#!/usr/bin/env python3
"""Create the deterministic public smoke-training dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.public_sample_dataset import materialize_public_sample_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/public_sample_runtime_datasets")
    parser.add_argument("--target", default="tomato__leaf")
    args = parser.parse_args()
    result = materialize_public_sample_dataset(Path(args.root), target=args.target)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
