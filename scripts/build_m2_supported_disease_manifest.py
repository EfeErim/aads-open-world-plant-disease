#!/usr/bin/env python3
"""Build an M2 manifest that covers every supported disease class per adapter."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_release_runtime import DEFAULT_MATERIALIZED_DATASET_ROOT  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_PRIORITY = ("test", "val", "continual", "train")


def _label_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower().replace("ı", "i"))
    return "".join(ch for ch in normalized if ch.isalnum() and not unicodedata.combining(ch))


def is_healthy_class(class_name: str) -> bool:
    key = _label_key(class_name)
    return "healthy" in key or "saglikli" in key


def class_images(class_dir: Path) -> list[Path]:
    return sorted(
        (path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )


def build_rows(dataset_root: Path, *, start_id: int, images_per_class: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    next_id = int(start_id)
    for target_root in sorted(dataset_root.iterdir(), key=lambda path: path.name):
        test_root = target_root / "test"
        if not target_root.is_dir() or not test_root.is_dir():
            continue
        for class_dir in sorted(test_root.iterdir(), key=lambda path: _label_key(path.name)):
            if not class_dir.is_dir() or is_healthy_class(class_dir.name):
                continue
            candidate_images: list[Path] = []
            seen_paths: set[Path] = set()
            for split in SPLIT_PRIORITY:
                split_dir = target_root / split
                split_class_dir = split_dir / class_dir.name
                if not split_class_dir.is_dir():
                    matching = [
                        child for child in split_dir.iterdir()
                        if child.is_dir() and _label_key(child.name) == _label_key(class_dir.name)
                    ] if split_dir.is_dir() else []
                    split_class_dir = matching[0] if matching else split_class_dir
                for image_path in class_images(split_class_dir) if split_class_dir.is_dir() else []:
                    resolved = image_path.resolve()
                    if resolved not in seen_paths:
                        candidate_images.append(image_path)
                        seen_paths.add(resolved)
                    if len(candidate_images) >= images_per_class:
                        break
                if len(candidate_images) >= images_per_class:
                    break
            for image_path in candidate_images[:images_per_class]:
                image_id = f"demo_{next_id:03d}"
                try:
                    split_name = image_path.relative_to(target_root).parts[0]
                except ValueError:
                    split_name = ""
                rows.append(
                    {
                        "image_id": image_id,
                        "source": f"local_test_pool:{image_path.as_posix()}",
                        "expected_target": target_root.name,
                        "expected_crop": target_root.name.split("__", 1)[0],
                        "expected_part": target_root.name.split("__", 1)[1] if "__" in target_root.name else "",
                        "expected_class": class_dir.name,
                        "expected_behavior": (
                            f"known supported disease class: {class_dir.name}; disease answer or review expected"
                        ),
                        "notes": f"supported disease coverage for {target_root.name}; split={split_name}",
                        "disease_class": class_dir.name,
                    }
                )
                next_id += 1
    return rows


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    fieldnames = [
        "image_id",
        "source",
        "expected_target",
        "expected_crop",
        "expected_part",
        "expected_class",
        "expected_behavior",
        "notes",
        "disease_class",
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_MATERIALIZED_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=Path(".runtime_tmp/m2_supported_disease_coverage_manifest.csv"))
    parser.add_argument("--summary", type=Path, default=Path(".runtime_tmp/m2_supported_disease_coverage_summary.json"))
    parser.add_argument("--start-id", type=int, default=145)
    parser.add_argument("--images-per-class", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = build_rows(args.dataset_root, start_id=args.start_id, images_per_class=max(1, int(args.images_per_class)))
    write_manifest(rows, args.manifest)
    per_target: dict[str, int] = {}
    per_class: dict[str, dict[str, int]] = {}
    for row in rows:
        target = row["expected_target"]
        per_target[target] = per_target.get(target, 0) + 1
        per_class.setdefault(target, {})
        disease_class = row["disease_class"]
        per_class[target][disease_class] = per_class[target].get(disease_class, 0) + 1
    summary = {
        "schema_version": "v1_m2_supported_disease_coverage_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root),
        "manifest": str(args.manifest),
        "images_per_class": max(1, int(args.images_per_class)),
        "row_count": len(rows),
        "per_target": dict(sorted(per_target.items())),
        "per_class": {target: dict(sorted(counts.items())) for target, counts in sorted(per_class.items())},
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
