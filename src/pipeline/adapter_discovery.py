"""Adapter discovery and metadata helpers extracted from RouterAdapterRuntime.

Provides resolver helpers for locating adapter bundle directories and reading
adapter metadata state. This isolates filesystem discovery logic for easier
testing and reuse.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.shared.adapter_paths import ADAPTER_BUNDLE_DIR_NAME, normalize_adapter_name, resolve_adapter_bundle_dir
from src.shared.json_utils import read_json_dict

_logger = logging.getLogger(__name__)


def discover_fallback_adapter_dir(search_root: Path, crop_name: str, part_name: Optional[str], allow_cross_part: bool) -> Optional[Path]:
    if not search_root.exists() or not search_root.is_dir():
        return None

    crop_key = normalize_adapter_name(crop_name)
    requested_part = normalize_adapter_name(part_name) if part_name else ""
    ranked_candidates: list[tuple[int, Path]] = []

    def _infer_path_metadata(adapter_bundle_dir: Path) -> tuple[str, str]:
        try:
            relative_parts = adapter_bundle_dir.relative_to(search_root).parts
        except ValueError:
            return "", ""
        if len(relative_parts) >= 3 and relative_parts[-1] == ADAPTER_BUNDLE_DIR_NAME:
            return (
                normalize_adapter_name(relative_parts[0], default=""),
                normalize_adapter_name(relative_parts[1], default=""),
            )
        return "", ""

    for meta_path in sorted(search_root.rglob("adapter_meta.json")):
        adapter_bundle_dir = meta_path.parent
        if adapter_bundle_dir.name != ADAPTER_BUNDLE_DIR_NAME:
            continue
        try:
            meta = read_json_dict(meta_path)
        except Exception as exc:
            _logger.debug(f"Failed to read adapter metadata from {meta_path}; skipping", exc_info=exc)
            continue
        path_crop, path_part = _infer_path_metadata(adapter_bundle_dir)
        meta_crop = normalize_adapter_name(meta.get("crop_name"), default="") if meta.get("crop_name") else ""
        resolved_crop = meta_crop or path_crop
        if not resolved_crop:
            continue
        if resolved_crop != crop_key:
            continue
        meta_part = normalize_adapter_name(meta.get("part_name"), default="") if meta.get("part_name") else ""
        resolved_part = meta_part or path_part
        if requested_part and resolved_part == requested_part:
            ranked_candidates.append((0, adapter_bundle_dir))
        elif requested_part and not allow_cross_part:
            continue
        elif not requested_part and (not resolved_part or resolved_part == "unspecified"):
            ranked_candidates.append((1, adapter_bundle_dir))
        elif not requested_part:
            ranked_candidates.append((2, adapter_bundle_dir))
        elif resolved_part in {"", "unspecified"}:
            ranked_candidates.append((1, adapter_bundle_dir))
        else:
            ranked_candidates.append((2, adapter_bundle_dir))

    if not ranked_candidates:
        return None
    ranked_candidates.sort(key=lambda item: (item[0], str(item[1])))
    return ranked_candidates[0][1]


def resolve_adapter_dir(adapter_root: Path, crop_name: str, part_name: Optional[str], allow_cross_part: bool, adapter_dir_override: Optional[Path] = None) -> Path:
    if adapter_dir_override is not None:
        root = Path(adapter_dir_override)
        try:
            return resolve_adapter_bundle_dir(root, crop_name=crop_name, part_name=part_name)
        except FileNotFoundError:
            fallback = discover_fallback_adapter_dir(root, crop_name=crop_name, part_name=part_name, allow_cross_part=allow_cross_part)
            if fallback is not None:
                return fallback
            raise FileNotFoundError(f"Could not resolve adapter bundle from {root}")
    try:
        return resolve_adapter_bundle_dir(adapter_root, crop_name=crop_name, part_name=part_name)
    except FileNotFoundError:
        fallback = discover_fallback_adapter_dir(adapter_root, crop_name=crop_name, part_name=part_name, allow_cross_part=allow_cross_part)
        if fallback is not None:
            return fallback
        expected_part = normalize_adapter_name(part_name) if part_name else "unspecified"
        raise FileNotFoundError(f"Adapter not found for crop '{crop_name}' part '{expected_part}' under {adapter_root}")


def adapter_meta_state(adapter_dir: Path) -> tuple[Path, str, int, int]:
    resolved_dir = adapter_dir.resolve()
    meta_path = resolved_dir / "adapter_meta.json"
    try:
        meta_stat = meta_path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"No adapter assets found under {resolved_dir}") from exc
    meta = read_json_dict(meta_path)
    part_name = normalize_adapter_name(meta.get("part_name")) if meta.get("part_name") else "unspecified"
    return (resolved_dir, part_name, int(meta_stat.st_mtime_ns), int(meta_stat.st_size))
