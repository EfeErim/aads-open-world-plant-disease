"""Build the public, checksum-pinned asset manifest from archived release metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aads_public.manifest import build_public_asset_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repository", default="EfeErim/bitirmeprojesi")
    parser.add_argument("--release-tag", default="aads-public-demo-v1.0.0")
    args = parser.parse_args()
    payload = build_public_asset_manifest(args.source, repository=args.repository, release_tag=args.release_tag)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(payload['assets'])} assets to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
