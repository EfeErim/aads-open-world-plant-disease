"""Build taxonomy registry and router prototype-bank artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_release_runtime import DEFAULT_MATERIALIZED_DATASET_ROOT  # noqa: E402
from src.router.prototype_bank import (  # noqa: E402
    DEFAULT_BACKEND,
    build_prototype_bank,
    write_prototype_bank,
    write_router_prototype_summary,
)
from src.router.taxonomy_registry import (  # noqa: E402
    build_taxonomy_registry,
    now_utc_timestamp,
    write_taxonomy_registry,
)

DEFAULT_OUTPUT_ROOT = Path("runs/_index/router_prototypes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_MATERIALIZED_DATASET_ROOT)
    parser.add_argument("--adapter-root", type=Path, default=Path("runs"))
    parser.add_argument("--taxonomy-path", type=Path, default=Path("config/plant_taxonomy.json"))
    parser.add_argument("--overrides-path", type=Path, default=None)
    parser.add_argument("--external-cache-path", type=Path, default=Path("runs/_index/router_prototypes/external_taxonomy_cache.json"))
    parser.add_argument("--refresh-gbif", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--curation-root",
        type=Path,
        default=None,
        help="Optional reviewed prototype curation directory with prototype_positive_manifest.csv and prototype_hard_negative_manifest.csv.",
    )
    parser.add_argument("--embedding-backend", default=DEFAULT_BACKEND)
    parser.add_argument("--embedding-model-id", default="imageomics/bioclip-2.5-vith14")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--split", action="append", dest="splits", default=None)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--include-ood", action="store_true")
    parser.add_argument("--no-adapter-discovery", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or now_utc_timestamp()
    output_dir = args.output_root / run_id
    created_at = now_utc_timestamp()
    registry_payload = build_taxonomy_registry(
        dataset_root=args.dataset_root,
        adapter_root=None if args.no_adapter_discovery else args.adapter_root,
        taxonomy_path=args.taxonomy_path,
        overrides_path=args.overrides_path,
        external_cache_path=args.external_cache_path,
        refresh_gbif=bool(args.refresh_gbif),
        created_at=created_at,
    )
    prototype_payload = build_prototype_bank(
        dataset_root=args.dataset_root,
        curation_root=args.curation_root,
        repo_root=ROOT,
        embedding_backend=args.embedding_backend,
        embedding_model_id=args.embedding_model_id,
        device=args.embedding_device,
        splits=tuple(args.splits) if args.splits else ("train", "val", "test", "continual"),
        include_ood=args.include_ood,
        max_images_per_class=args.max_images_per_class,
        created_at=created_at,
    )
    registry_output = write_taxonomy_registry(registry_payload, output_dir / "taxonomy_registry.json")
    prototype_output = write_prototype_bank(prototype_payload, output_dir / "prototype_bank.json")
    summary_output = write_router_prototype_summary(
        output_path=output_dir / "summary.md",
        registry_payload=registry_payload,
        prototype_payload=prototype_payload,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "taxonomy_registry": str(registry_output),
                "prototype_bank": str(prototype_output),
                "summary": str(summary_output),
                "registry": registry_payload["summary"],
                "prototype_bank_summary": prototype_payload["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
