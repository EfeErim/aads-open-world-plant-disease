"""Persistent, slice-aware real-OOD split assignment helpers."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from src.shared.hash_utils import sha256_file

OOD_SPLIT_MANIFEST_SCHEMA = "v1_real_ood_split_manifest"
FROZEN_OOD_SPLIT_MANIFEST_SCHEMA = "v2_family_frozen_ood_split_manifest"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()





def _image_paths(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def infer_ood_slice(path: Path, root: Path) -> str:
    """Return the top-level OOD slice for a file under ``root``."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "unlabeled"
    parts = list(relative.parts)
    if len(parts) <= 1:
        return "unlabeled"
    return str(parts[0] or "unlabeled")


def _collect_entries(root: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in _image_paths(root):
        relative_path = path.relative_to(root).as_posix()
        entries.append(
            {
                "relative_path": relative_path,
                "slice": infer_ood_slice(path, root),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return entries


def _manifest_matches(
    manifest: Mapping[str, Any],
    entries: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    dev_fraction: float,
    min_per_slice: int,
    min_total_for_dev_test: int,
) -> bool:
    if int(manifest.get("seed", -1)) != int(seed):
        return False
    if abs(float(manifest.get("dev_fraction", -1.0)) - float(dev_fraction)) > 1e-9:
        return False
    if int(manifest.get("min_per_slice", -1)) != int(min_per_slice):
        return False
    if int(manifest.get("min_total_for_dev_test", -1)) != int(min_total_for_dev_test):
        return False
    manifest_entries = dict(manifest.get("entries", {})) if isinstance(manifest.get("entries"), Mapping) else {}
    current = {
        str(entry.get("relative_path")): str(entry.get("sha256"))
        for entry in entries
    }
    recorded = {
        str(path): str(payload.get("sha256"))
        for path, payload in manifest_entries.items()
        if isinstance(payload, Mapping)
    }
    return bool(current) and recorded == current


def _manifest_entries_match(manifest: Mapping[str, Any], entries: Iterable[Mapping[str, Any]]) -> bool:
    manifest_entries = dict(manifest.get("entries", {})) if isinstance(manifest.get("entries"), Mapping) else {}
    current = {str(entry.get("relative_path")): str(entry.get("sha256")) for entry in entries}
    recorded = {
        str(path): str(payload.get("sha256"))
        for path, payload in manifest_entries.items()
        if isinstance(payload, Mapping)
    }
    return bool(current) and recorded == current


def _reconcile_frozen_manifest_entries(
    manifest: Mapping[str, Any], entries: Iterable[Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Bind frozen split assignments to relocated immutable members by hash."""
    current_entries = [dict(entry) for entry in entries]
    recorded_entries = dict(manifest.get("entries", {})) if isinstance(manifest.get("entries"), Mapping) else {}
    current_by_hash: Dict[str, Dict[str, Any]] = {}
    recorded_by_hash: Dict[str, Dict[str, Any]] = {}
    for entry in current_entries:
        digest = str(entry.get("sha256") or "")
        if not digest or digest in current_by_hash:
            return None
        current_by_hash[digest] = entry
    for relative_path, payload in recorded_entries.items():
        if not isinstance(payload, Mapping):
            return None
        digest = str(payload.get("sha256") or "")
        if not digest or digest in recorded_by_hash:
            return None
        recorded_by_hash[digest] = {"relative_path": str(relative_path), **dict(payload)}
    if not current_by_hash or set(current_by_hash) != set(recorded_by_hash):
        return None
    reconciled_entries: Dict[str, Dict[str, Any]] = {}
    for digest, current in sorted(current_by_hash.items(), key=lambda item: str(item[1].get("relative_path"))):
        current_path = str(current["relative_path"])
        current_slice = str(current.get("slice") or "unlabeled")
        payload = dict(recorded_by_hash[digest])
        payload.pop("relative_path", None)
        payload["slice"] = current_slice
        if "ood_type" in payload:
            payload["ood_type"] = current_slice
        reconciled_entries[current_path] = payload
    reconciled = dict(manifest)
    reconciled["entries"] = reconciled_entries
    reconciled["runtime_path_reconciled_by_sha256"] = True
    return reconciled


def _load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") not in {OOD_SPLIT_MANIFEST_SCHEMA, FROZEN_OOD_SPLIT_MANIFEST_SCHEMA}:
        return None
    return payload


def _assign_splits(
    entries: List[Dict[str, Any]],
    *,
    seed: int,
    dev_fraction: float,
    min_per_slice: int,
    min_total_for_dev_test: int,
) -> Tuple[Dict[str, str], Dict[str, Any], str]:
    rng = random.Random(int(seed))
    by_slice: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_slice[str(entry["slice"])].append(entry)

    assignments: Dict[str, str] = {}
    slice_summaries: Dict[str, Any] = {}
    total_count = len(entries)
    if total_count < int(min_total_for_dev_test):
        for slice_name, slice_entries in sorted(by_slice.items(), key=lambda item: item[0]):
            for entry in slice_entries:
                assignments[str(entry["relative_path"])] = "test"
            slice_summaries[slice_name] = {
                "total": len(slice_entries),
                "dev": 0,
                "test": len(slice_entries),
                "split_policy": "test_only_below_min_total_for_dev_test",
            }
        return assignments, slice_summaries, "test_only_below_min_total_for_dev_test"

    for slice_name, slice_entries in sorted(by_slice.items(), key=lambda item: item[0]):
        shuffled = list(slice_entries)
        rng.shuffle(shuffled)
        count = len(shuffled)
        if count < int(min_per_slice):
            for entry in shuffled:
                assignments[str(entry["relative_path"])] = "test"
            slice_summaries[slice_name] = {
                "total": count,
                "dev": 0,
                "test": count,
                "split_policy": "test_only_below_min_per_slice",
            }
            continue

        dev_count = int(round(float(count) * float(dev_fraction)))
        dev_count = max(1, min(count - 1, dev_count))
        dev_paths = {str(entry["relative_path"]) for entry in shuffled[:dev_count]}
        for entry in shuffled:
            relative_path = str(entry["relative_path"])
            assignments[relative_path] = "dev" if relative_path in dev_paths else "test"
        slice_summaries[slice_name] = {
            "total": count,
            "dev": dev_count,
            "test": count - dev_count,
            "split_policy": "slice_stratified",
        }
    return assignments, slice_summaries, "slice_stratified_dev_test"


def ensure_ood_split_manifest(
    ood_root: str | Path,
    *,
    manifest_name: str = "ood_split_manifest.json",
    seed: int = 42,
    dev_fraction: float = 0.4,
    min_per_slice: int = 2,
    min_total_for_dev_test: int = 30,
) -> Dict[str, Any]:
    """Create or reuse a persistent manifest for real-OOD dev/test splits."""
    root = Path(ood_root).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"OOD root is not a directory: {root}")

    entries = _collect_entries(root)
    manifest_path = root / str(manifest_name or "ood_split_manifest.json")
    existing = _load_manifest(manifest_path)
    resolved_dev_fraction = max(0.05, min(0.95, float(dev_fraction)))
    resolved_min_per_slice = max(2, int(min_per_slice))
    resolved_min_total_for_dev_test = max(0, int(min_total_for_dev_test))
    if existing is not None and existing.get("schema_version") == FROZEN_OOD_SPLIT_MANIFEST_SCHEMA:
        reconciled = _reconcile_frozen_manifest_entries(existing, entries)
        if reconciled is None:
            raise ValueError("Frozen OOD split manifest does not match the current OOD image inventory.")
        return reconciled
    if existing is not None and _manifest_matches(
        existing,
        entries,
        seed=int(seed),
        dev_fraction=resolved_dev_fraction,
        min_per_slice=resolved_min_per_slice,
        min_total_for_dev_test=resolved_min_total_for_dev_test,
    ):
        return existing

    assignments, slice_summaries, assignment_policy = _assign_splits(
        entries,
        seed=int(seed),
        dev_fraction=resolved_dev_fraction,
        min_per_slice=resolved_min_per_slice,
        min_total_for_dev_test=resolved_min_total_for_dev_test,
    )
    split_counts = {"dev": 0, "test": 0}
    entry_payload: Dict[str, Any] = {}
    for entry in entries:
        relative_path = str(entry["relative_path"])
        split = str(assignments.get(relative_path, "test"))
        split_counts[split] = int(split_counts.get(split, 0)) + 1
        entry_payload[relative_path] = {
            "split": split,
            "slice": str(entry["slice"]),
            "sha256": str(entry["sha256"]),
            "size_bytes": int(entry["size_bytes"]),
        }

    manifest = {
        "schema_version": OOD_SPLIT_MANIFEST_SCHEMA,
        "created_at": _utc_now_iso(),
        "source_root": str(root.resolve()),
        "split_policy": "phase1_real_ood_dev_test_if_large_enough",
        "assignment_policy": assignment_policy,
        "seed": int(seed),
        "dev_fraction": resolved_dev_fraction,
        "min_per_slice": resolved_min_per_slice,
        "min_total_for_dev_test": resolved_min_total_for_dev_test,
        "split_counts": split_counts,
        "slice_summaries": slice_summaries,
        "entries": entry_payload,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def select_manifest_paths(
    ood_root: str | Path,
    manifest: Mapping[str, Any],
    split: str,
) -> List[Path]:
    root = Path(ood_root).expanduser()
    target_split = str(split or "").strip().lower()
    entries = dict(manifest.get("entries", {})) if isinstance(manifest.get("entries"), Mapping) else {}
    selected: List[Path] = []
    for relative_path, payload in sorted(entries.items(), key=lambda item: str(item[0])):
        if not isinstance(payload, Mapping):
            continue
        if str(payload.get("split", "")).strip().lower() != target_split:
            continue
        selected.append(root / str(relative_path))
    return selected


def manifest_slice_map(
    ood_root: str | Path,
    manifest: Mapping[str, Any],
) -> Dict[str, str]:
    root = Path(ood_root).expanduser()
    entries = dict(manifest.get("entries", {})) if isinstance(manifest.get("entries"), Mapping) else {}
    result: Dict[str, str] = {}
    for relative_path, payload in entries.items():
        if not isinstance(payload, Mapping):
            continue
        result[str((root / str(relative_path)).resolve(strict=False))] = str(payload.get("slice", "unlabeled") or "unlabeled")
    return result


def find_ood_oe_hash_overlaps(
    ood_root: str | Path,
    oe_root: str | Path,
) -> List[Dict[str, str]]:
    """Return exact image-hash overlaps between real-OOD and OE pools."""
    resolved_ood_root = Path(ood_root).expanduser()
    resolved_oe_root = Path(oe_root).expanduser()
    ood_by_hash: Dict[str, List[str]] = defaultdict(list)
    for entry in _collect_entries(resolved_ood_root):
        ood_by_hash[str(entry["sha256"])].append(str(entry["relative_path"]))

    overlaps: List[Dict[str, str]] = []
    for entry in _collect_entries(resolved_oe_root):
        digest = str(entry["sha256"])
        for ood_relative_path in ood_by_hash.get(digest, []):
            overlaps.append(
                {
                    "sha256": digest,
                    "ood_relative_path": ood_relative_path,
                    "oe_relative_path": str(entry["relative_path"]),
                }
            )
    return overlaps


def validate_ood_oe_disjoint(
    ood_root: str | Path,
    oe_root: str | Path,
) -> None:
    """Raise if exact images are reused between final OOD evidence and OE."""
    overlaps = find_ood_oe_hash_overlaps(ood_root, oe_root)
    if not overlaps:
        return
    preview = "; ".join(
        f"{item['ood_relative_path']} == {item['oe_relative_path']}"
        for item in overlaps[:5]
    )
    suffix = "" if len(overlaps) <= 5 else f"; +{len(overlaps) - 5} more"
    raise ValueError(
        "Exact image overlap between real OOD evidence and OE training pool is not allowed: "
        + preview
        + suffix
    )
