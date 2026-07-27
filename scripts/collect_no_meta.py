#!/usr/bin/env python3
"""Collect images into a target folder using SHA256 dedupe.
Optional: use sidecar metadata JSON files for filtering when available.

Usage examples:
  .\scripts\python.cmd scripts\collect_no_meta.py \
    --sources data/internet_image_candidates data/internet_image_candidates/extra \
    --target data/ood_dataset/final/tomato__leaf_ood_final \
    --max 500 --verbose

The script writes `manifest_collected.json` in the target folder and
updates `outputs/collector_hash_index.json` to avoid cross-run duplicates.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional


def sha256_file(p: Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(p: Path):
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def find_images(sources):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    for s in sources:
        p = Path(s)
        if p.is_dir():
            for f in p.rglob("*"):
                if f.suffix.lower() in exts and f.is_file():
                    yield f
        elif p.is_file() and p.suffix.lower() in exts:
            yield p


def read_sidecar_json(img_path: Path) -> Optional[dict]:
    j = img_path.with_suffix(img_path.suffix + ".json")
    if j.exists():
        try:
            return load_json(j)
        except Exception:
            return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--max", type=int, default=500)
    parser.add_argument("--use-metadata", action="store_true", help="Prefer images matching sidecar metadata filters if provided")
    parser.add_argument("--filter-key", type=str, help="Metadata key to filter on (e.g. crop)")
    parser.add_argument("--filter-value", type=str, help="Metadata value to match (e.g. tomato)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    global_index = ROOT / "outputs" / "collector_hash_index.json"
    index = load_json(global_index)
    if not isinstance(index, dict):
        index = {}

    # preload existing hashes in target
    existing_hashes = set()
    for f in target.rglob("*.*"):
        if f.is_file():
            try:
                h = sha256_file(f)
                existing_hashes.add(h)
            except Exception:
                continue

    collected = []
    count = 0
    for img in find_images(args.sources):
        if args.verbose:
            print("Checking:", img)
        try:
            h = sha256_file(img)
        except Exception:
            if args.verbose:
                print("  skip (read error):", img)
            continue
        if h in existing_hashes or h in index:
            if args.verbose:
                print("  skip (duplicate hash)")
            continue

        # optional metadata filtering
        if args.use_metadata and (args.filter_key and args.filter_value):
            meta = read_sidecar_json(img)
            if meta is None or str(meta.get(args.filter_key, "")).lower() != args.filter_value.lower():
                if args.verbose:
                    print("  skip (metadata mismatch)")
                continue

        # copy file
        dest_name = f"{img.stem}_{h[:8]}{img.suffix.lower()}"
        dest = target / dest_name
        try:
            shutil.copy2(img, dest)
        except Exception as e:
            if args.verbose:
                print("  copy failed:", e)
            continue

        existing_hashes.add(h)
        index[h] = str(dest.relative_to(ROOT))
        collected.append({"filename": str(dest.relative_to(ROOT)), "sha256": h})
        count += 1
        if args.verbose:
            print("  copied ->", dest)
        if count >= args.max:
            break

    # write manifest in target
    save_json(target / "manifest_collected.json", collected)
    save_json(global_index, index)
    print(f"Collected {len(collected)} files into: {target}")


if __name__ == "__main__":
    main()
