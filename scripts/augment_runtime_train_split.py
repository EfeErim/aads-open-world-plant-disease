#!/usr/bin/env python3
"""Create a runtime dataset copy with offline augmentations only in `continual/`."""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_layout import normalize_class_name
from src.shared.json_utils import read_json, write_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
GENERATED_DIR_NAME = "_offline_aug"
POLICY_NAME = "train_split_only_pil_v1"





def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _is_generated_path(path: Path) -> bool:
    return GENERATED_DIR_NAME in set(path.parts)


def _image_paths(root: Path, *, include_generated: bool = True) -> List[Path]:
    paths = [path for path in root.rglob("*") if _is_image(path)]
    if not include_generated:
        paths = [path for path in paths if not _is_generated_path(path)]
    return sorted(paths, key=lambda item: item.as_posix().lower())


def _split_class_counts(dataset_root: Path, *, include_generated: bool) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for split_name in ("continual", "val", "test"):
        split_root = dataset_root / split_name
        split_counts: Dict[str, int] = {}
        if split_root.is_dir():
            for class_dir in sorted((item for item in split_root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
                class_name = normalize_class_name(class_dir.name)
                split_counts[class_name] = len(_image_paths(class_dir, include_generated=include_generated))
        counts[split_name] = split_counts
    return counts


def _class_names_from_counts(*count_maps: Dict[str, Dict[str, int]]) -> List[str]:
    names: set[str] = set()
    for split_counts in count_maps:
        for class_counts in split_counts.values():
            names.update(str(name) for name in class_counts)
    return sorted(names)


def _manifest_reference_counts(source_root: Path, fallback_counts: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    manifest_path = source_root / "split_manifest.json"
    payload = read_json(manifest_path, default={}, expect_type=dict) if manifest_path.exists() else {}
    classes = payload.get("classes", []) if isinstance(payload, dict) else []
    counts: Dict[str, int] = {}
    if isinstance(classes, list):
        for entry in classes:
            if not isinstance(entry, dict):
                continue
            class_name = normalize_class_name(entry.get("class_name", ""))
            if not class_name:
                continue
            raw_count = entry.get("reference_image_count", entry.get("image_count"))
            try:
                counts[class_name] = int(raw_count)
            except (TypeError, ValueError):
                continue
    if counts:
        return counts
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            split_name = str(row.get("split", "")).strip().lower()
            if split_name not in {"continual", "val", "test"}:
                continue
            if bool(row.get("runtime_skipped")) or bool(row.get("generated_offline_augmentation")):
                continue
            if bool(row.get("synthetic_hint")):
                continue
            class_name = normalize_class_name(row.get("normalized_class_name", row.get("class_name", "")))
            if not class_name:
                continue
            counts[class_name] = int(counts.get(class_name, 0)) + 1
    if counts:
        return counts
    return {
        class_name: int(sum(split_counts.get(class_name, 0) for split_counts in fallback_counts.values()))
        for class_name in _class_names_from_counts(fallback_counts)
    }


def _variant_seed(path: Path, variant_index: int, seed: int) -> int:
    payload = f"{path.as_posix()}::{variant_index}::{seed}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _apply_variant(image: Image.Image, *, source_path: Path, variant_index: int, seed: int) -> Image.Image:
    rng = random.Random(_variant_seed(source_path, variant_index, seed))
    output = ImageOps.exif_transpose(image).convert("RGB")

    if rng.random() < 0.5:
        output = ImageOps.mirror(output)

    width, height = output.size
    scale = rng.uniform(0.86, 1.0)
    crop_width = max(1, int(width * scale))
    crop_height = max(1, int(height * scale))
    left = 0 if crop_width >= width else rng.randint(0, width - crop_width)
    top = 0 if crop_height >= height else rng.randint(0, height - crop_height)
    output = output.crop((left, top, left + crop_width, top + crop_height)).resize((width, height), Image.Resampling.BICUBIC)

    angle = rng.uniform(-18.0, 18.0)
    output = output.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    output = ImageEnhance.Brightness(output).enhance(rng.uniform(0.86, 1.14))
    output = ImageEnhance.Contrast(output).enhance(rng.uniform(0.88, 1.18))
    output = ImageEnhance.Color(output).enhance(rng.uniform(0.90, 1.16))
    output = ImageEnhance.Sharpness(output).enhance(rng.uniform(0.92, 1.20))
    return output


def _save_variant(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(destination, quality=92, optimize=True)
    else:
        image.save(destination)


def _copy_runtime_dataset(source_root: Path, output_root: Path, *, overwrite: bool) -> None:
    if source_root.resolve() == output_root.resolve():
        raise ValueError("source_root and output_root must be different.")
    try:
        output_root.resolve().relative_to(source_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output_root must not be inside source_root.")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output runtime dataset already exists: {output_root}")
        shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root)


def _generated_filename(source_path: Path, class_root: Path, variant_index: int) -> Path:
    relative_source = source_path.relative_to(class_root)
    digest = hashlib.sha1(relative_source.as_posix().encode("utf-8")).hexdigest()[:10]
    parent = relative_source.parent
    suffix = source_path.suffix.lower() if source_path.suffix.lower() in IMAGE_EXTENSIONS else ".jpg"
    filename = f"{source_path.stem}__aug_{digest}_{variant_index:02d}{suffix}"
    return Path(GENERATED_DIR_NAME) / parent / filename


def augment_runtime_train_split(
    *,
    source_root: Path,
    output_root: Path,
    variants_per_image: int = 2,
    seed: int = 42,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Copy a runtime dataset and add deterministic offline variants to `continual/` only."""

    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source runtime dataset not found: {source_root}")
    for split_name in ("continual", "val", "test"):
        if not (source_root / split_name).is_dir():
            raise RuntimeError(f"Source runtime dataset is missing required split: {split_name}")

    variants = int(variants_per_image)
    if variants < 1:
        raise ValueError("variants_per_image must be at least 1.")

    source_non_aug_counts = _split_class_counts(source_root, include_generated=False)
    reference_counts = _manifest_reference_counts(source_root, source_non_aug_counts)
    _copy_runtime_dataset(source_root, output_root, overwrite=bool(overwrite))

    generated_rows: List[Dict[str, Any]] = []
    for class_dir in sorted((item for item in (output_root / "continual").iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        class_name = normalize_class_name(class_dir.name)
        for source_path in _image_paths(class_dir, include_generated=False):
            if _is_generated_path(source_path):
                continue
            with Image.open(source_path) as image:
                for variant_index in range(1, variants + 1):
                    destination_relative = _generated_filename(source_path, class_dir, variant_index)
                    destination_path = class_dir / destination_relative
                    variant = _apply_variant(image, source_path=source_path, variant_index=variant_index, seed=int(seed))
                    _save_variant(variant, destination_path)
                    generated_rows.append(
                        {
                            "relative_path": destination_path.relative_to(output_root).as_posix(),
                            "runtime_relative_path": destination_path.relative_to(output_root).as_posix(),
                            "source_runtime_relative_path": source_path.relative_to(output_root).as_posix(),
                            "raw_class_name": class_dir.name,
                            "normalized_class_name": class_name,
                            "split": "continual",
                            "family_assignment": "continual",
                            "synthetic_hint": True,
                            "generated_offline_augmentation": True,
                            "offline_augmentation_policy": POLICY_NAME,
                            "train_only_routed": True,
                            "train_only_route_reason": "offline_train_augmentation",
                            "canonical_eval_safe": False,
                            "augmentation_parent_split": "continual",
                            "augmentation_variant_index": int(variant_index),
                        }
                    )

    actual_counts = _split_class_counts(output_root, include_generated=True)
    non_aug_counts = _split_class_counts(output_root, include_generated=False)
    classes = []
    for class_name in _class_names_from_counts(actual_counts, non_aug_counts):
        actual_split_counts = {
            split_name: int(actual_counts.get(split_name, {}).get(class_name, 0))
            for split_name in ("continual", "val", "test")
        }
        non_aug_split_counts = {
            split_name: int(non_aug_counts.get(split_name, {}).get(class_name, 0))
            for split_name in ("continual", "val", "test")
        }
        classes.append(
            {
                "class_name": class_name,
                "image_count": int(sum(actual_split_counts.values())),
                "reference_image_count": int(reference_counts.get(class_name, sum(non_aug_split_counts.values()))),
                "split_counts": actual_split_counts,
                "non_augmented_split_counts": non_aug_split_counts,
                "offline_augmented_count": int(actual_split_counts["continual"] - non_aug_split_counts["continual"]),
            }
        )

    source_manifest_path = source_root / "split_manifest.json"
    manifest = read_json(source_manifest_path, default={}, expect_type=dict) if source_manifest_path.exists() else {}
    manifest = dict(manifest) if isinstance(manifest, dict) else {}
    base_rows = list(manifest.get("rows", [])) if isinstance(manifest.get("rows", []), list) else []
    manifest["classes"] = classes
    manifest["rows"] = base_rows + generated_rows
    manifest["offline_train_augmentation"] = {
        "schema_version": "v1_train_split_only_offline_augmentation",
        "policy": POLICY_NAME,
        "source_runtime_dataset_root": str(source_root),
        "output_runtime_dataset_root": str(output_root),
        "variants_per_image": int(variants),
        "seed": int(seed),
        "generated_image_count": int(len(generated_rows)),
        "leakage_policy": "val/test/ood are copied unchanged; generated variants are derived only from continual split images.",
        "reference_count_policy": "reference_image_count excludes generated offline augmentations.",
    }
    write_json(output_root / "split_manifest.json", manifest, ensure_ascii=False)

    report = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "variants_per_image": int(variants),
        "seed": int(seed),
        "generated_image_count": int(len(generated_rows)),
        "classes": classes,
    }
    write_json(output_root / "offline_train_augmentation_report.json", report, ensure_ascii=False)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Prepared runtime dataset root to copy.")
    parser.add_argument("--output-root", type=Path, default=None, help="Output runtime dataset root.")
    parser.add_argument("--suffix", type=str, default="_train_aug", help="Suffix used when --output-root is omitted.")
    parser.add_argument("--variants-per-image", type=int, default=2, help="Number of generated train variants per continual image.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic augmentation seed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output runtime dataset if it already exists.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    source_root = Path(args.source_root)
    output_root = Path(args.output_root) if args.output_root is not None else source_root.with_name(source_root.name + str(args.suffix))
    report = augment_runtime_train_split(
        source_root=source_root,
        output_root=output_root,
        variants_per_image=int(args.variants_per_image),
        seed=int(args.seed),
        overwrite=bool(args.overwrite),
    )
    print(
        "[AUG] wrote "
        f"{report['output_root']} generated={report['generated_image_count']} "
        f"variants_per_image={report['variants_per_image']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
