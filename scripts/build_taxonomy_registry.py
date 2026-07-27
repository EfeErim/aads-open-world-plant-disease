"""Build the supported-target taxonomy registry artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_release_runtime import DEFAULT_MATERIALIZED_DATASET_ROOT  # noqa: E402
from src.router.taxonomy_registry import build_taxonomy_registry, write_taxonomy_registry  # noqa: E402

DEFAULT_OUTPUT = Path("runs/_index/router_prototypes/latest_taxonomy_registry.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_MATERIALIZED_DATASET_ROOT)
    parser.add_argument("--adapter-root", type=Path, default=Path("runs"))
    parser.add_argument("--taxonomy-path", type=Path, default=Path("config/plant_taxonomy.json"))
    parser.add_argument("--overrides-path", type=Path, default=None)
    parser.add_argument("--external-cache-path", type=Path, default=Path("runs/_index/router_prototypes/external_taxonomy_cache.json"))
    parser.add_argument("--refresh-gbif", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-adapter-discovery", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_taxonomy_registry(
        dataset_root=args.dataset_root,
        adapter_root=None if args.no_adapter_discovery else args.adapter_root,
        taxonomy_path=args.taxonomy_path,
        overrides_path=args.overrides_path,
        external_cache_path=args.external_cache_path,
        refresh_gbif=bool(args.refresh_gbif),
    )
    output = write_taxonomy_registry(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(output),
                **payload["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
