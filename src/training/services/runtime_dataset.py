"""Helpers for resolving runtime dataset roots for training workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.shared.json_utils import read_json


def normalize_runtime_crop_name(name: str) -> str:
    normalized = str(name or "").strip().lower()
    for token in (" ", "-", "/", "\\"):
        normalized = normalized.replace(token, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _read_runtime_manifest(crop_root: Path) -> Dict[str, Any]:
    manifest_path = crop_root / "split_manifest.json"
    if not manifest_path.exists():
        return {}
    payload = read_json(manifest_path, default={}, expect_type=dict)
    return dict(payload) if isinstance(payload, dict) else {}


@dataclass(frozen=True)
class ResolvedRuntimeDataset:
    dataset_key: str
    crop_root: Path
    resolution_source: str


def _looks_like_runtime_dataset_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / split_name).is_dir() for split_name in ("continual", "val", "test")) or (
        path / "split_manifest.json"
    ).exists()


def resolve_runtime_dataset(*, data_dir: str | Path, crop_name: str) -> ResolvedRuntimeDataset:
    resolved_data_dir = Path(data_dir)
    crop_key = normalize_runtime_crop_name(crop_name)

    if _looks_like_runtime_dataset_root(resolved_data_dir):
        manifest_payload = _read_runtime_manifest(resolved_data_dir)
        manifest_crop_key = normalize_runtime_crop_name(manifest_payload.get("crop_name", ""))
        dataset_key = str(resolved_data_dir.name).strip().lower()
        dataset_matches_crop = dataset_key == crop_key or dataset_key.startswith(f"{crop_key}__")
        manifest_matches_crop = bool(manifest_crop_key and manifest_crop_key == crop_key)
        if dataset_matches_crop or manifest_matches_crop:
            return ResolvedRuntimeDataset(
                dataset_key=str(resolved_data_dir.name),
                crop_root=resolved_data_dir,
                resolution_source="exact_root",
            )
        raise ValueError(
            f"Runtime dataset root {resolved_data_dir} does not match crop '{crop_key}'. "
            "Point data_dir at the matching dataset root or adjust crop_name."
        )

    direct_root = resolved_data_dir / crop_key

    manifest_matches: list[Path] = []
    prefix_matches: list[Path] = []
    if resolved_data_dir.is_dir():
        for candidate in sorted(resolved_data_dir.iterdir(), key=lambda item: item.name.lower()):
            if not candidate.is_dir():
                continue
            candidate_key = str(candidate.name).strip().lower()
            if candidate_key.startswith(f"{crop_key}__"):
                prefix_matches.append(candidate)
            manifest_payload = _read_runtime_manifest(candidate)
            manifest_crop_key = normalize_runtime_crop_name(manifest_payload.get("crop_name", ""))
            if manifest_payload and manifest_crop_key == crop_key:
                manifest_matches.append(candidate)

    candidate_sources: dict[Path, set[str]] = {}

    def _record_candidate(path: Path, source: str) -> None:
        candidate_sources.setdefault(path, set()).add(source)

    if direct_root.is_dir():
        _record_candidate(direct_root, "direct")
    for crop_root in manifest_matches:
        _record_candidate(crop_root, "manifest")
    for crop_root in prefix_matches:
        _record_candidate(crop_root, "prefix")

    if len(candidate_sources) == 1:
        crop_root, sources = next(iter(candidate_sources.items()))
        resolution_source = (
            "direct"
            if "direct" in sources
            else "manifest:split_manifest.json"
            if "manifest" in sources
            else "directory_prefix"
        )
        return ResolvedRuntimeDataset(
            dataset_key=str(crop_root.name),
            crop_root=crop_root,
            resolution_source=resolution_source,
        )

    if len(candidate_sources) > 1:
        matches = ", ".join(str(path.name) for path in sorted(candidate_sources.keys(), key=lambda item: item.name.lower()))
        raise ValueError(
            f"Multiple runtime datasets matched crop '{crop_key}' under {resolved_data_dir}: {matches}. "
            "Point data_dir at the specific runtime root or remove the ambiguity."
        )

    return ResolvedRuntimeDataset(
        dataset_key=crop_key,
        crop_root=direct_root,
        resolution_source="default",
    )
