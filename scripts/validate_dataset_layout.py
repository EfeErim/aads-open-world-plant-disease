#!/usr/bin/env python3
"""Validate prepared runtime dataset layout and split-leakage invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_dataset_layout import evaluate_layout
from src.shared.json_utils import read_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RUNTIME_SPLITS = ("continual", "val", "test")
GENERATED_DIR_NAME = "_offline_aug"


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _row_split(row: Dict[str, Any]) -> str:
    split = str(row.get("split", "") or "").strip().lower()
    if split == "train":
        return "continual"
    return split


def _row_family_key(row: Dict[str, Any]) -> str:
    for field_name in (
        "grouped_family_id",
        "family_bundle_key",
        "family_canonical_relative_path",
        "family_id",
    ):
        value = str(row.get(field_name, "") or "").strip()
        if value:
            return f"{field_name}:{value}"

    variant_parent = str(row.get("variant_of", "") or "").strip()
    if variant_parent:
        return f"variant_of:{variant_parent}"

    if bool(row.get("generated_offline_augmentation")):
        source_path = str(row.get("source_runtime_relative_path", "") or "").strip()
        if source_path:
            return f"source_runtime_relative_path:{source_path}"

    return ""


def validate_runtime_dataset_layout(root: Path, *, check_leakage: bool = True) -> Dict[str, Any]:
    """Validate a Notebook 0/Notebook 2 prepared runtime dataset root."""

    root = Path(root)
    errors: List[str] = []
    warnings: List[str] = []
    split_counts: Dict[str, Dict[str, int]] = {
        split_name: {"real": 0, "generated": 0, "disk_images": 0}
        for split_name in RUNTIME_SPLITS
    }

    if not root.exists():
        return {
            "ok": False,
            "root": str(root),
            "expected_layout": "<root>/{continual,val,test}/<class>/<images>",
            "errors": [f"Runtime dataset root does not exist: {root}"],
            "warnings": [],
            "summary": {},
        }
    if not root.is_dir():
        return {
            "ok": False,
            "root": str(root),
            "expected_layout": "<root>/{continual,val,test}/<class>/<images>",
            "errors": [f"Runtime dataset root is not a directory: {root}"],
            "warnings": [],
            "summary": {},
        }

    for split_name in RUNTIME_SPLITS:
        split_root = root / split_name
        if not split_root.is_dir():
            errors.append(f"Missing required runtime split directory: {split_name}")
            continue
        disk_images = [path for path in split_root.rglob("*") if _is_image(path)]
        split_counts[split_name]["disk_images"] = len(disk_images)
        if check_leakage and split_name in {"val", "test"}:
            leaked_generated = [
                path.relative_to(root).as_posix()
                for path in disk_images
                if GENERATED_DIR_NAME in set(path.relative_to(root).parts)
            ]
            if leaked_generated:
                errors.append(
                    f"Found generated augmentation files under {split_name}: "
                    + ", ".join(leaked_generated[:5])
                )

    manifest_path = root / "split_manifest.json"
    manifest: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    if not manifest_path.exists():
        errors.append(f"Missing split_manifest.json at {manifest_path}")
    else:
        try:
            manifest = read_json(manifest_path, default={}, expect_type=dict)
        except Exception as exc:
            errors.append(f"Could not read split_manifest.json: {exc}")
        raw_rows = manifest.get("rows", []) if isinstance(manifest, dict) else []
        if isinstance(raw_rows, list):
            rows = [row for row in raw_rows if isinstance(row, dict)]
        else:
            errors.append("split_manifest.json field 'rows' must be a list when present.")

    family_splits: Dict[str, set[str]] = {}
    generated_rows_outside_train: List[str] = []
    for row_index, row in enumerate(rows):
        if bool(row.get("runtime_skipped")):
            continue
        split_name = _row_split(row)
        if split_name in split_counts:
            key = "generated" if bool(row.get("generated_offline_augmentation")) else "real"
            split_counts[split_name][key] += 1

        if bool(row.get("generated_offline_augmentation")) and split_name != "continual":
            row_path = str(
                row.get("runtime_relative_path")
                or row.get("relative_path")
                or row.get("original_image_name")
                or f"manifest row {row_index}"
            )
            generated_rows_outside_train.append(f"{row_path} ({split_name or 'missing split'})")

        family_key = _row_family_key(row)
        if family_key and split_name in RUNTIME_SPLITS:
            family_splits.setdefault(family_key, set()).add(split_name)

    leakage_families = {
        family_key: sorted(splits)
        for family_key, splits in sorted(family_splits.items())
        if len(splits) > 1
    }
    if check_leakage and generated_rows_outside_train:
        errors.append(
            "Generated offline augmentation rows must stay in continual split; found: "
            + ", ".join(generated_rows_outside_train[:5])
        )
    if check_leakage and leakage_families:
        examples = [
            f"{family_key} -> {','.join(splits)}"
            for family_key, splits in list(leakage_families.items())[:5]
        ]
        errors.append("Family leakage detected across runtime splits: " + "; ".join(examples))

    if rows:
        manifest_disk_total = sum(item["real"] + item["generated"] for item in split_counts.values())
        disk_total = sum(item["disk_images"] for item in split_counts.values())
        if disk_total and manifest_disk_total and disk_total != manifest_disk_total:
            warnings.append(
                "Manifest row count for continual/val/test does not match image files on disk "
                f"({manifest_disk_total} manifest rows vs {disk_total} disk images)."
            )

    return {
        "ok": len(errors) == 0,
        "root": str(root),
        "expected_layout": "<root>/{continual,val,test}/<class>/<images>",
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "manifest_rows": len(rows),
            "families_checked": len(family_splits),
            "leakage_family_count": len(leakage_families),
            "generated_rows_outside_train": len(generated_rows_outside_train),
            "splits": split_counts,
        },
    }


def _print_runtime_result(result: Dict[str, Any]) -> None:
    print("=" * 60)
    print("RUNTIME DATASET VALIDATION")
    print("=" * 60)
    print(f"Root: {result['root']}")
    print(f"Expected Layout: {result['expected_layout']}")
    print("")

    summary = result.get("summary", {})
    splits = summary.get("splits", {}) if isinstance(summary, dict) else {}
    if splits:
        print("Split composition:")
        for split_name in RUNTIME_SPLITS:
            stats = splits.get(split_name, {})
            print(
                f"  {split_name:10} real={stats.get('real', 0):4} "
                f"generated={stats.get('generated', 0):4} "
                f"disk_images={stats.get('disk_images', 0):4}"
            )
        print("")
        print(f"Families checked: {summary.get('families_checked', 0)}")
        print(f"Leakage families: {summary.get('leakage_family_count', 0)}")
        print("")

    if result["warnings"]:
        print("Warnings:")
        for item in result["warnings"]:
            print(f"  - {item}")
        print("")

    if result["errors"]:
        print("Errors:")
        for item in result["errors"]:
            print(f"  - {item}")
        print("")
        print("Result: FAIL")
    else:
        print("Result: PASS")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dataset root to validate.")
    parser.add_argument(
        "--check-leakage",
        action="store_true",
        help="Validate prepared runtime split leakage invariants instead of flat class-root layout.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON result payload.")
    parser.add_argument("--min-classes", type=int, default=1, help="Flat-layout minimum class count.")
    parser.add_argument(
        "--min-images-per-class",
        type=int,
        default=1,
        help="Flat-layout warning threshold per class.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.check_leakage:
        result = validate_runtime_dataset_layout(args.root, check_leakage=True)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_runtime_result(result)
    else:
        result = evaluate_layout(
            root=args.root,
            min_classes=int(args.min_classes),
            min_images_per_class=int(args.min_images_per_class),
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result, indent=2))

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
