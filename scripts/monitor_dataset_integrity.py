#!/usr/bin/env python3
"""Monitor prepared runtime datasets for split leakage and weak OOD/OE pools.

This is a fast integrity guard for generated Notebook 0 runtime datasets. It
checks exact file-hash overlap across `continual`, `val`, `test`, `ood`, and
`oe`, reports class-balance counts, and keeps `ood` and `oe` disjoint. Missing
dataset roots are reported as `skipped` so source-only CI can run without local
generated data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_release_runtime import DEFAULT_MATERIALIZED_DATASET_ROOT  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("continual", "val", "test", "ood", "oe")


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_dataset_roots(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if any((root / split).is_dir() for split in SPLITS):
        return [root]
    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and any((child / split).is_dir() for split in SPLITS)
    )


def _class_counts(dataset_root: Path, split: str) -> dict[str, int]:
    split_root = dataset_root / split
    if not split_root.is_dir():
        return {}
    counts: dict[str, int] = {}
    for class_dir in sorted(child for child in split_root.iterdir() if child.is_dir()):
        counts[class_dir.name] = sum(1 for path in class_dir.rglob("*") if _is_image(path))
    root_images = [path for path in split_root.iterdir() if _is_image(path)]
    if root_images:
        counts["__root__"] = len(root_images)
    return counts


def _hash_rows(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_root = dataset_root / split
        if not split_root.is_dir():
            continue
        for image_path in sorted(path for path in split_root.rglob("*") if _is_image(path)):
            class_name = image_path.parent.name if image_path.parent != split_root else "__root__"
            rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "path": image_path.relative_to(dataset_root).as_posix(),
                    "sha256": _sha256(image_path),
                }
            )
    return rows


def _find_cross_split_duplicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_hash.setdefault(str(row["sha256"]), []).append(row)

    duplicates: list[dict[str, Any]] = []
    for digest, digest_rows in sorted(by_hash.items()):
        splits = sorted({str(row["split"]) for row in digest_rows})
        if len(splits) <= 1:
            continue
        duplicates.append(
            {
                "sha256": digest,
                "splits": splits,
                "paths": [str(row["path"]) for row in digest_rows],
            }
        )
    return duplicates


def _balance_warnings(counts: dict[str, dict[str, int]], min_class_samples: int) -> list[str]:
    warnings: list[str] = []
    train_counts = counts.get("continual", {})
    for class_name, count in sorted(train_counts.items()):
        if count < min_class_samples:
            warnings.append(
                f"continual/{class_name} has {count} images; below min_class_samples={min_class_samples}"
            )
    positive_counts = [count for count in train_counts.values() if count > 0]
    if len(positive_counts) >= 2 and min(positive_counts) > 0:
        ratio = max(positive_counts) / min(positive_counts)
        if ratio > 1.5:
            warnings.append(f"continual class balance ratio is {ratio:.2f}; above 1.50")
    return warnings


def inspect_dataset_root(
    dataset_root: Path,
    *,
    min_class_samples: int = 100,
    min_ood_images: int = 20,
    min_oe_images: int = 20,
) -> dict[str, Any]:
    counts = {split: _class_counts(dataset_root, split) for split in SPLITS}
    rows = _hash_rows(dataset_root)
    duplicates = _find_cross_split_duplicates(rows)

    errors: list[str] = []
    warnings = _balance_warnings(counts, min_class_samples)
    if duplicates:
        examples = "; ".join(
            f"{item['sha256'][:12]} in {','.join(item['splits'])}" for item in duplicates[:5]
        )
        errors.append(f"Exact image hash overlap across splits: {examples}")

    ood_count = sum(counts.get("ood", {}).values())
    oe_count = sum(counts.get("oe", {}).values())
    if ood_count and ood_count < min_ood_images:
        warnings.append(f"ood pool has {ood_count} images; below min_ood_images={min_ood_images}")
    if oe_count and oe_count < min_oe_images:
        warnings.append(f"oe pool has {oe_count} images; below min_oe_images={min_oe_images}")

    return {
        "dataset_root": str(dataset_root),
        "status": "fail" if errors else "warn" if warnings else "pass",
        "counts": counts,
        "image_count": len(rows),
        "duplicate_count": len(duplicates),
        "duplicates": duplicates[:50],
        "errors": errors,
        "warnings": warnings,
    }


def build_report(
    root: Path,
    *,
    min_class_samples: int,
    min_ood_images: int,
    min_oe_images: int,
) -> dict[str, Any]:
    dataset_roots = _iter_dataset_roots(root)
    datasets = [
        inspect_dataset_root(
            dataset_root,
            min_class_samples=min_class_samples,
            min_ood_images=min_ood_images,
            min_oe_images=min_oe_images,
        )
        for dataset_root in dataset_roots
    ]
    fail_count = sum(1 for item in datasets if item["status"] == "fail")
    warn_count = sum(1 for item in datasets if item["status"] == "warn")
    return {
        "status": "fail" if fail_count else "warn" if warn_count else "skipped" if not datasets else "pass",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "dataset_count": len(datasets),
        "fail_count": fail_count,
        "warn_count": warn_count,
        "datasets": datasets,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_MATERIALIZED_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=Path(".runtime_tmp/dataset_integrity.json"))
    parser.add_argument("--min-class-samples", type=int, default=100)
    parser.add_argument("--min-ood-images", type=int, default=20)
    parser.add_argument("--min-oe-images", type=int, default=20)
    parser.add_argument("--strict", action="store_true", help="Return non-zero on warnings as well as errors.")
    args = parser.parse_args(argv)

    report = build_report(
        args.root,
        min_class_samples=args.min_class_samples,
        min_ood_images=args.min_ood_images,
        min_oe_images=args.min_oe_images,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"dataset_integrity status={report['status']} datasets={report['dataset_count']} "
        f"failures={report['fail_count']} warnings={report['warn_count']} output={args.output}"
    )
    if report["fail_count"]:
        return 1
    if args.strict and report["warn_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
