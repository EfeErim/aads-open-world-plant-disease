"""Helpers for resolving adapter bundle paths."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

ADAPTER_BUNDLE_DIR_NAME = "continual_sd_lora_adapter"
DEFAULT_PART_NAME = "unspecified"


def normalize_adapter_name(value: object, *, default: str = DEFAULT_PART_NAME) -> str:
    normalized = str(value or "").strip().lower()
    return normalized or str(default)


def build_adapter_bundle_root(
    base_dir: str | Path,
    crop_name: str,
    part_name: Optional[str] = None,
) -> Path:
    root = Path(base_dir)
    crop_key = normalize_adapter_name(crop_name)
    part_key = normalize_adapter_name(part_name)
    return root / crop_key / part_key


def build_adapter_bundle_dir(
    base_dir: str | Path,
    crop_name: str,
    part_name: Optional[str] = None,
) -> Path:
    return build_adapter_bundle_root(base_dir, crop_name, part_name) / ADAPTER_BUNDLE_DIR_NAME


def is_adapter_bundle_dir(path: str | Path) -> bool:
    root = Path(path)
    return root.is_dir() and (root / "adapter_meta.json").exists()


def iter_adapter_bundle_candidates(
    base_dir: str | Path,
    crop_name: Optional[str] = None,
    part_name: Optional[str] = None,
) -> Iterable[Path]:
    root = Path(base_dir)
    crop_key = normalize_adapter_name(crop_name) if crop_name is not None else ""
    part_key = normalize_adapter_name(part_name) if part_name is not None else ""

    if crop_key:
        if part_key:
            yield build_adapter_bundle_dir(root, crop_key, part_key)
        yield build_adapter_bundle_dir(root, crop_key, None)
        if part_key and part_key != DEFAULT_PART_NAME:
            yield build_adapter_bundle_root(root, crop_key, part_key)
        yield build_adapter_bundle_root(root, crop_key, None)

    yield root / ADAPTER_BUNDLE_DIR_NAME
    yield root


def resolve_adapter_bundle_dir(
    base_dir: str | Path,
    crop_name: Optional[str] = None,
    part_name: Optional[str] = None,
) -> Path:
    base_path = Path(base_dir)
    # If base_dir itself is an adapter bundle, return it directly
    if is_adapter_bundle_dir(base_path):
        return base_path
    for candidate in iter_adapter_bundle_candidates(base_dir, crop_name=crop_name, part_name=part_name):
        if is_adapter_bundle_dir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not resolve an adapter bundle from {Path(base_dir)}"
    )