#!/usr/bin/env python3
"""Create provenance-preserving crops from existing staged OOD/OE evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
OFFSETS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_urls(manifest_root: Path) -> dict[str, str]:
    urls: dict[str, str] = {}
    for path in sorted(manifest_root.glob("*_manifest.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                relative_path = str(row.get("relative_path") or "").strip().replace("\\", "/")
                source_url = str(row.get("source_url") or "").strip()
                if relative_path and source_url:
                    urls[relative_path] = source_url
    return urls


def crop_box(width: int, height: int, index: int, reuse_round: int) -> tuple[int, int, int, int]:
    fraction = 0.86 - 0.04 * (reuse_round % 3)
    crop_width = max(8, min(width, round(width * fraction)))
    crop_height = max(8, min(height, round(height * fraction)))
    x_ratio, y_ratio = OFFSETS[index % len(OFFSETS)]
    left = round((width - crop_width) * x_ratio)
    top = round((height - crop_height) * y_ratio)
    return left, top, left + crop_width, top + crop_height


def derive_crops(
    *,
    evidence_root: Path,
    manifest_root: Path,
    output_dir: Path,
    candidate_manifest: Path,
    target: str,
    disease_id: str,
    role: str,
    count: int,
) -> list[dict[str, str | int]]:
    source_dir = evidence_root / target / role / disease_id
    sources = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    )
    if not sources:
        raise ValueError(f"no staged source images: {source_dir}")
    source_urls = load_source_urls(manifest_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    for index in range(count):
        source = sources[index % len(sources)]
        relative_path = source.relative_to(evidence_root).as_posix()
        source_url = source_urls.get(relative_path)
        if not source_url:
            raise ValueError(f"missing source URL for staged evidence: {relative_path}")
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            box = crop_box(rgb.width, rgb.height, index, index // len(sources))
            crop = rgb.crop(box)
            output_path = output_dir / f"{index + 1:02d}_{source.stem[:12]}_crop.jpg"
            crop.save(output_path, quality=90 + index % 3, optimize=True)
        rows.append(
            {
                "filename": output_path.name,
                "image_url": relative_path,
                "source_page_url": source_url,
                "title": f"Derived crop from {relative_path}",
                "sha256": sha256_file(output_path),
                "width": crop.width,
                "height": crop.height,
                "parent_relative_path": relative_path,
            }
        )
    candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
    with candidate_manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--disease-id", required=True)
    parser.add_argument("--role", choices=("oe_train", "ood_dev", "ood_test"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, default=Path("data/adapter_ood_oe_evidence"))
    parser.add_argument(
        "--manifest-root", type=Path, default=Path("data/internet_image_candidates/turkey_adapter_ood")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = derive_crops(
        evidence_root=args.evidence_root,
        manifest_root=args.manifest_root,
        output_dir=args.output_dir,
        candidate_manifest=args.candidate_manifest,
        target=args.target,
        disease_id=args.disease_id,
        role=args.role,
        count=args.count,
    )
    print(f"PASS: derived={len(rows)} role={args.role} target={args.target} disease={args.disease_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
