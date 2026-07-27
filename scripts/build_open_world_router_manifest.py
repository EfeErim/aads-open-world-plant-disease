#!/usr/bin/env python3
"""Build a fresh open-world negative manifest for Notebook 8 router validation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.enrich_m2_demo_image_set import (  # noqa: E402
    IMAGE_EXT,
    AcceptedImage,
    ImagePlan,
    _collect_external_images,
    _load_reject_ids,
    _sha256,
)
from src.data.dataset_release_runtime import DEFAULT_MATERIALIZED_DATASET_ROOT  # noqa: E402

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_OUTPUT_ROOT = Path("docs/demo_assets/open_world_router")
DEFAULT_MANIFEST = DEFAULT_OUTPUT_ROOT / "manifests" / "m2_open_world_router_manifest.csv"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_ROOT / "manifests" / "m2_open_world_router_summary.json"
DEFAULT_STAGING_DIR = Path(".runtime_tmp/open_world_router_downloads")
DEFAULT_DISJOINT_ROOTS = (
    Path("docs/demo_assets/m2_full_image_set/images"),
    DEFAULT_MATERIALIZED_DATASET_ROOT,
    Path("docs/demo_assets/prototype_curation"),
)
OPEN_WORLD_BEHAVIOR = "open-world negative; abstain or review expected, no supported disease label"


@dataclass(frozen=True)
class OpenWorldPlan:
    ood_slice: str
    expected_target: str
    expected_crop: str
    expected_part: str
    taxon_name: str
    queries: tuple[str, ...]
    count: int
    notes: str


OPEN_WORLD_PLAN: tuple[OpenWorldPlan, ...] = (
    OpenWorldPlan(
        "unsupported_crop",
        "unknown_crop",
        "unknown",
        "unknown",
        "Malus domestica",
        ("apple leaf disease", "apple scab leaf", "apple fruit disease", "Malus domestica leaf"),
        35,
        "unsupported apple crop",
    ),
    OpenWorldPlan(
        "unsupported_crop",
        "unknown_crop",
        "unknown",
        "unknown",
        "Capsicum annuum",
        ("pepper fruit disease", "pepper bacterial spot", "Capsicum annuum leaf", "pepper plant disease"),
        35,
        "unsupported pepper crop",
    ),
    OpenWorldPlan(
        "unsupported_crop",
        "unknown_crop",
        "unknown",
        "unknown",
        "Solanum tuberosum",
        ("potato late blight leaf", "potato plant disease", "Solanum tuberosum leaf"),
        35,
        "unsupported potato crop",
    ),
    OpenWorldPlan(
        "unsupported_crop",
        "unknown_crop",
        "unknown",
        "unknown",
        "Citrus",
        ("citrus leaf disease", "orange tree leaf", "citrus fruit disease", "Citrus leaf"),
        35,
        "unsupported citrus crop",
    ),
    OpenWorldPlan(
        "unsupported_crop",
        "unknown_crop",
        "unknown",
        "unknown",
        "Cucumis sativus",
        ("cucumber leaf disease", "cucumber fruit", "cucumber plant", "Cucumis sativus"),
        30,
        "unsupported cucumber crop",
    ),
    OpenWorldPlan(
        "unsupported_crop",
        "unknown_crop",
        "unknown",
        "unknown",
        "Pyrus",
        ("pear leaf disease", "pear fruit", "pear tree leaf", "Pyrus leaf"),
        30,
        "unsupported pear crop",
    ),
    OpenWorldPlan(
        "unsupported_crop",
        "unknown_crop",
        "unknown",
        "unknown",
        "Prunus persica",
        ("peach leaf disease", "peach fruit", "peach tree leaf", "Prunus persica"),
        30,
        "unsupported peach crop",
    ),
    OpenWorldPlan(
        "same_crop_wrong_part",
        "tomato__unknown_part",
        "tomato",
        "unknown",
        "Solanum lycopersicum",
        ("tomato flower", "tomato stem", "tomato seedling", "tomato whole plant"),
        20,
        "supported tomato crop but unsupported or ambiguous plant part",
    ),
    OpenWorldPlan(
        "same_crop_wrong_part",
        "grape__unknown_part",
        "grape",
        "unknown",
        "Vitis vinifera",
        ("grapevine tendril", "grape flower", "grape trunk", "Vitis vinifera bark"),
        20,
        "supported grape crop but unsupported or ambiguous plant part",
    ),
    OpenWorldPlan(
        "same_crop_wrong_part",
        "strawberry__unknown_part",
        "strawberry",
        "unknown",
        "Fragaria",
        ("strawberry flower", "strawberry runner", "Fragaria flower", "strawberry seedling"),
        20,
        "supported strawberry crop but unsupported or ambiguous plant part",
    ),
    OpenWorldPlan(
        "same_crop_wrong_part",
        "apricot__unknown_part",
        "apricot",
        "unknown",
        "Prunus armeniaca",
        ("apricot flower", "apricot bark", "apricot tree trunk", "Prunus armeniaca flower"),
        20,
        "supported apricot crop but unsupported or ambiguous plant part",
    ),
    OpenWorldPlan(
        "plant_like_non_target",
        "unknown_crop",
        "unknown",
        "unknown",
        "Rosa",
        ("rose leaf spot", "rose leaf", "rose plant disease", "Rosa leaf"),
        30,
        "plant-like non-target ornamental leaves",
    ),
    OpenWorldPlan(
        "plant_like_non_target",
        "unknown_crop",
        "unknown",
        "unknown",
        "Quercus",
        ("oak leaf disease", "oak leaves", "Quercus leaf", "oak gall leaf"),
        30,
        "plant-like non-target tree leaves",
    ),
    OpenWorldPlan(
        "plant_like_non_target",
        "unknown_crop",
        "unknown",
        "unknown",
        "Acer",
        ("maple leaf", "maple leaves", "Acer leaf", "maple leaf disease"),
        30,
        "plant-like non-target maple leaves",
    ),
    OpenWorldPlan(
        "plant_like_non_target",
        "unknown_crop",
        "unknown",
        "unknown",
        "Taraxacum",
        ("dandelion leaf", "dandelion flower", "Taraxacum", "dandelion plant"),
        30,
        "plant-like non-target weed or flower",
    ),
    OpenWorldPlan(
        "non_plant_distractor",
        "non_plant",
        "non_plant",
        "unknown",
        "Fungi",
        ("mold fungus", "mushroom close up", "fungal growth", "Fungi"),
        25,
        "non-plant biological distractor",
    ),
    OpenWorldPlan(
        "non_plant_distractor",
        "non_plant",
        "non_plant",
        "unknown",
        "Insecta",
        ("insect close up", "aphid close up", "beetle close up", "Insecta"),
        25,
        "non-plant insect distractor",
    ),
    OpenWorldPlan(
        "low_quality_ambiguous",
        "unknown_crop",
        "unknown",
        "unknown",
        "Plantae",
        ("blurry plant", "damaged leaf close up", "plant close up", "ambiguous plant"),
        30,
        "ambiguous user-photo-style plant image",
    ),
)


def _iter_image_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
        return [root]
    if not root.is_dir():
        return []
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def collect_disjoint_hashes(roots: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for root in roots:
        for path in _iter_image_files(root):
            try:
                hashes.add(_sha256(path))
            except OSError:
                continue
    return hashes


def _load_manifest_photo_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            str(row.get("photo_id") or "").strip()
            for row in rows
            if str(row.get("photo_id") or "").strip()
        }


def _read_existing_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _next_open_world_id(rows: list[dict[str, str]], start_id: int) -> int:
    numbers = []
    for row in rows:
        image_id = str(row.get("image_id") or "")
        if not image_id.startswith("ow_"):
            continue
        try:
            numbers.append(int(image_id.split("_", 1)[1]))
        except ValueError:
            continue
    return max(numbers, default=max(0, start_id - 1)) + 1


def _compatible_plan(plan: OpenWorldPlan) -> ImagePlan:
    return ImagePlan(
        expected_target=plan.expected_target,
        expected_crop=plan.expected_crop,
        expected_part=plan.expected_part,
        taxon_name=plan.taxon_name,
        queries=plan.queries,
        count=plan.count,
    )


def _row_for_image(
    *,
    image_id: str,
    final_path: Path,
    plan: OpenWorldPlan,
    accepted: AcceptedImage,
) -> dict[str, str]:
    candidate = accepted.candidate
    provenance = (
        f"source=iNaturalist; photo_id={candidate.photo_id}; "
        f"observation={candidate.observation_url}; license={candidate.license_code}; "
        f"attribution={candidate.attribution}"
    )
    return {
        "image_id": image_id,
        "source": f"staged_external:{final_path.as_posix()}",
        "expected_target": plan.expected_target,
        "expected_crop": plan.expected_crop,
        "expected_part": plan.expected_part,
        "expected_class": "",
        "expected_behavior": OPEN_WORLD_BEHAVIOR,
        "ood_slice": plan.ood_slice,
        "origin_url": candidate.observation_url,
        "notes": f"{plan.notes}; query={candidate.query}; taxon={candidate.taxon_name}",
        "provenance_notes": provenance,
        "resolved_source_path": final_path.as_posix(),
        "photo_url": candidate.source_url,
        "taxon_name": candidate.taxon_name,
        "query": candidate.query,
        "photo_id": candidate.photo_id,
        "license_code": candidate.license_code,
        "attribution": candidate.attribution,
        "sha256": accepted.sha256,
        "width": str(accepted.width),
        "height": str(accepted.height),
        "contrast": f"{accepted.contrast:.2f}",
        "source_backend": "inaturalist",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "image_id",
        "source",
        "expected_target",
        "expected_crop",
        "expected_part",
        "expected_class",
        "expected_behavior",
        "ood_slice",
        "origin_url",
        "notes",
        "provenance_notes",
        "resolved_source_path",
        "photo_url",
        "taxon_name",
        "query",
        "photo_id",
        "license_code",
        "attribution",
        "sha256",
        "width",
        "height",
        "contrast",
        "source_backend",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_open_world_manifest(
    *,
    output_manifest: Path,
    output_image_dir: Path,
    staging_dir: Path,
    disjoint_roots: list[Path],
    reject_ids_path: Path | None,
    start_id: int,
    per_page: int,
    max_pages: int,
    timeout: int,
    sleep_seconds: float,
    min_side: int,
    min_bytes: int,
    min_contrast: float,
    max_aspect_ratio: float,
    min_rows: int,
) -> dict[str, object]:
    output_image_dir.mkdir(parents=True, exist_ok=True)
    existing_rows = _read_existing_manifest(output_manifest)
    existing_hashes = {str(row.get("sha256") or "").strip() for row in existing_rows if row.get("sha256")}
    used_hashes = collect_disjoint_hashes(disjoint_roots + [output_image_dir]) | existing_hashes
    reject_ids = _load_reject_ids(reject_ids_path) if reject_ids_path else set()
    reject_ids |= _load_manifest_photo_ids(output_manifest)
    rows: list[dict[str, str]] = list(existing_rows)
    next_id = _next_open_world_id(existing_rows, start_id)
    slice_counts: Counter[str] = Counter(str(row.get("ood_slice") or "") for row in rows)
    target_counts: Counter[str] = Counter(str(row.get("expected_target") or "") for row in rows)

    for plan in OPEN_WORLD_PLAN:
        if len(rows) >= min_rows:
            break
        accepted_images = _collect_external_images(
            _compatible_plan(plan),
            staging_dir=staging_dir / plan.ood_slice / plan.expected_target,
            used_hashes=used_hashes,
            reject_ids=reject_ids,
            per_page=per_page,
            max_pages=max_pages,
            timeout=timeout,
            sleep_seconds=sleep_seconds,
            min_side=min_side,
            min_bytes=min_bytes,
            min_contrast=min_contrast,
            max_aspect_ratio=max_aspect_ratio,
        )
        if len(accepted_images) != plan.count:
            print(
                "[WARN] "
                f"{plan.ood_slice}/{plan.expected_target} selected "
                f"{len(accepted_images)}/{plan.count} external images"
            )
        for accepted in accepted_images:
            image_id = f"ow_{next_id:04d}"
            filename = f"{image_id}_{plan.ood_slice}_{plan.expected_target.replace('__', '_')}{IMAGE_EXT}"
            final_path = output_image_dir / filename
            shutil.copy2(accepted.path, final_path)
            row = _row_for_image(image_id=image_id, final_path=final_path, plan=plan, accepted=accepted)
            rows.append(row)
            slice_counts[plan.ood_slice] += 1
            target_counts[plan.expected_target] += 1
            reject_ids.add(accepted.candidate.photo_id)
            next_id += 1

    _write_csv(output_manifest, rows)
    if len(rows) < min_rows:
        raise RuntimeError(f"open-world manifest selected {len(rows)}/{min_rows} required rows")
    summary = {
        "schema_version": "open_world_router_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": output_manifest.as_posix(),
        "image_dir": output_image_dir.as_posix(),
        "row_count": len(rows),
        "min_rows": int(min_rows),
        "target_row_count": sum(plan.count for plan in OPEN_WORLD_PLAN),
        "slice_counts": dict(sorted(slice_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "duplicate_hash_count": len(rows) - len({row["sha256"] for row in rows}),
        "disjoint_root_count": len(disjoint_roots),
        "disjoint_hash_count": len(used_hashes),
        "quality_filters": {
            "min_side": int(min_side),
            "min_bytes": int(min_bytes),
            "min_contrast": float(min_contrast),
            "max_aspect_ratio": float(max_aspect_ratio),
        },
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-image-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "images")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--disjoint-root", action="append", type=Path, default=list(DEFAULT_DISJOINT_ROOTS))
    parser.add_argument(
        "--reject-ids",
        type=Path,
        default=Path("docs/demo_assets/m2_full_image_set/manifests/m2_external_inaturalist_reject_photo_ids.txt"),
    )
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--per-page", type=int, default=80)
    parser.add_argument("--max-pages", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--min-side", type=int, default=320)
    parser.add_argument("--min-bytes", type=int, default=20_000)
    parser.add_argument("--min-contrast", type=float, default=14.0)
    parser.add_argument("--max-aspect-ratio", type=float, default=3.0)
    parser.add_argument("--min-rows", type=int, default=300)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_open_world_manifest(
        output_manifest=args.output_manifest,
        output_image_dir=args.output_image_dir,
        staging_dir=args.staging_dir,
        disjoint_roots=list(args.disjoint_root),
        reject_ids_path=args.reject_ids,
        start_id=max(1, int(args.start_id)),
        per_page=max(1, int(args.per_page)),
        max_pages=max(1, int(args.max_pages)),
        timeout=max(1, int(args.timeout)),
        sleep_seconds=max(0.0, float(args.sleep_seconds)),
        min_side=max(1, int(args.min_side)),
        min_bytes=max(1, int(args.min_bytes)),
        min_contrast=float(args.min_contrast),
        max_aspect_ratio=float(args.max_aspect_ratio),
        min_rows=max(1, int(args.min_rows)),
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
