#!/usr/bin/env python3
"""Evaluate flat class-root dataset layout for grouped-prep notebook flows."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class ClassSummary:
    class_name: str
    image_files: int
    est_train: int
    est_val: int
    est_test: int


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _estimate_split_counts(total: int) -> tuple[int, int, int]:
    """Estimate an upper bound assuming all images survive grouped prep as independent families."""
    if total <= 0:
        return 0, 0, 0
    if total < 3:
        return total, 0, 0

    train_count = max(1, int(total * 0.8))
    val_count = max(1, int(total * 0.1))

    if train_count + val_count >= total:
        val_count = 1
        train_count = max(1, total - 2)

    test_count = total - train_count - val_count
    if test_count < 1:
        test_count = 1
        if train_count > 1:
            train_count -= 1
        elif val_count > 0:
            val_count -= 1

    return train_count, val_count, test_count


def evaluate_layout(
    root: Path,
    min_classes: int = 1,
    min_images_per_class: int = 1,
) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []
    class_summaries: List[ClassSummary] = []

    if not root.exists():
        errors.append(f"Dataset root does not exist: {root}")
        return {
            "ok": False,
            "root": str(root),
            "expected_layout": "<root>/<class_name>/<images>",
            "errors": errors,
            "warnings": warnings,
            "classes": [],
        }

    if not root.is_dir():
        errors.append(f"Dataset root is not a directory: {root}")
        return {
            "ok": False,
            "root": str(root),
            "expected_layout": "<root>/<class_name>/<images>",
            "errors": errors,
            "warnings": warnings,
            "classes": [],
        }

    split_dirs_present = [name for name in SPLIT_NAMES if (root / name).is_dir()]
    if split_dirs_present:
        errors.append(
            "This pipeline now requires flat class-root layout, not pre-split layout. "
            f"Found split folder(s): {split_dirs_present}"
        )

    root_level_files = [p.name for p in root.iterdir() if p.is_file()]
    if root_level_files:
        warnings.append(
            f"Root has {len(root_level_files)} non-directory file(s): {root_level_files[:5]}"
        )

    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    class_dirs = [p for p in class_dirs if p.name.lower() not in set(SPLIT_NAMES)]

    if len(class_dirs) < min_classes:
        errors.append(
            f"Found {len(class_dirs)} class directory(ies), but minimum required is {min_classes}."
        )

    for class_dir in class_dirs:
        image_count = sum(1 for p in class_dir.rglob("*") if _is_image(p))
        if image_count == 0:
            errors.append(f"Class '{class_dir.name}' has zero image files.")
            train_n = val_n = test_n = 0
        else:
            if image_count < min_images_per_class:
                warnings.append(
                    f"Class '{class_dir.name}' has only {image_count} image(s); "
                    f"minimum requested is {min_images_per_class}."
                )
            train_n, val_n, test_n = _estimate_split_counts(image_count)

        class_summaries.append(
            ClassSummary(
                class_name=class_dir.name,
                image_files=image_count,
                est_train=train_n,
                est_val=val_n,
                est_test=test_n,
            )
        )

    total_images = sum(item.image_files for item in class_summaries)
    total_train = sum(item.est_train for item in class_summaries)
    total_val = sum(item.est_val for item in class_summaries)
    total_test = sum(item.est_test for item in class_summaries)

    return {
        "ok": len(errors) == 0,
        "root": str(root),
        "expected_layout": "<root>/<class_name>/<images>",
        "split_policy": "Grouped family-aware split after audit (raw image counts are an upper bound only)",
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "num_classes": len(class_summaries),
            "total_images": total_images,
            "estimated_train_images": total_train,
            "estimated_val_images": total_val,
            "estimated_test_images": total_test,
        },
        "classes": [asdict(item) for item in class_summaries],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate flat class-root dataset layout. "
            "Required format: <root>/<class_name>/<images>. "
            "Notebook grouped prep will derive continual/val/test after duplicate-aware audit."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/class_root_dataset"),
        help="Dataset root to evaluate (default: data/class_root_dataset)",
    )
    parser.add_argument(
        "--min-classes",
        type=int,
        default=1,
        help="Minimum required number of class folders (default: 1)",
    )
    parser.add_argument(
        "--min-images-per-class",
        type=int,
        default=1,
        help="Warn when class image count is below this value (default: 1)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON result payload.",
    )
    args = parser.parse_args()

    result = evaluate_layout(
        root=args.root,
        min_classes=args.min_classes,
        min_images_per_class=args.min_images_per_class,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("DATASET LAYOUT EVALUATION")
        print("=" * 60)
        print(f"Root: {result['root']}")
        print(f"Expected Layout: {result['expected_layout']}")
        print(f"Split Policy: {result['split_policy']}")
        print("")

        summary = result.get("summary", {})
        if summary:
            print(f"Classes: {summary.get('num_classes', 0)}")
            print(f"Images:  {summary.get('total_images', 0)}")
            print(
                "Estimated split counts: "
                f"train={summary.get('estimated_train_images', 0)}, "
                f"val={summary.get('estimated_val_images', 0)}, "
                f"test={summary.get('estimated_test_images', 0)}"
            )
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

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
