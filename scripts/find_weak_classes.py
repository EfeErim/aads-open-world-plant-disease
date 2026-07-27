#!/usr/bin/env python3
"""Detect class folders with few images in a class-root dataset."""
from pathlib import Path
import json
import sys

ROOT = Path("data/class_root_dataset")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def count_images(path: Path) -> int:
    return sum(1 for f in path.rglob("*") if f.is_file() and f.suffix.lower() in EXTS)


def main():
    if not ROOT.exists():
        print("[]")
        return 0
    results = []
    for crop_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()], key=lambda p: p.name):
        for class_dir in sorted([p for p in crop_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            cnt = count_images(class_dir)
            results.append({"crop": crop_dir.name, "class": class_dir.name, "count": cnt, "path": str(class_dir)})
    # print classes with count < 50
    weak = [r for r in results if r["count"] < 50]
    print(json.dumps({"weak_classes": weak, "all": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
