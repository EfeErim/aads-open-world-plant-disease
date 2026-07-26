"""Replace placeholder adapter cards in a release asset directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from aads_public.model_card import build_target_model_card


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset_directory", type=Path)
    args = parser.parse_args()
    root = args.asset_directory.resolve()
    targets = sorted(
        path.name.removesuffix("--production_readiness.json")
        for path in root.glob("*--production_readiness.json")
    )
    if len(targets) != 8:
        raise ValueError(f"expected 8 targets, found {len(targets)}")
    for target in targets:
        card = build_target_model_card(
            root / f"{target}--production_readiness.json",
            root / f"{target}--adapter_config.json",
            root / f"{target}--adapter_meta.json",
            target=target,
        )
        (root / f"{target}--README.md").write_text(card, encoding="utf-8", newline="\n")
    print(f"Generated {len(targets)} evidence-bound model cards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
