#!/usr/bin/env python3
"""Helpers for converting flat class-root datasets into runtime split layouts."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.shared.json_utils import read_json
from src.shared.string_utils import normalize_class_name

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MATERIALIZATION_STRATEGIES = {"auto", "copy", "symlink", "hardlink"}
REPO_LOCAL_DATASET_PARENT_NAMES = {
    "class_root_dataset",
    "ood_dataset",
    "prepared_class_root_datasets",
    "prepared_runtime_datasets",
    "runtime_notebook_datasets",
}
REPO_DATASET_ARCHIVE_SUFFIXES = {".zip"}
_CLASS_ALIAS_GROUPS = (
    {"healthy", "healthy_leaf"},
    {"gray_mold", "botrytis_gray_mold"},
    {"yellow_leaf_curl", "yellow_leaf_curl_virus"},
    {"spotted_wilt_virus", "tomato_spotted_wilt_virus"},
)





def _dataset_display_name(path: Path) -> str:
    if path.suffix.lower() in REPO_DATASET_ARCHIVE_SUFFIXES:
        return path.stem
    return path.name


def _repo_dataset_cache_root(*, repo_root: str | Path, repo_relative_root: str | Path) -> Path:
    resolved_repo_root = Path(repo_root).expanduser().resolve()
    raw_relative = str(repo_relative_root or "").strip()
    if not raw_relative:
        raise RuntimeError("Repo dataset root cannot be empty.")
    relative_path = Path(raw_relative).expanduser()
    if relative_path.is_absolute():
        raise RuntimeError("Repo dataset root must stay repo-relative.")
    cache_root = (resolved_repo_root / ".runtime_tmp" / "dataset_cache" / relative_path).resolve()
    try:
        cache_root.relative_to(resolved_repo_root)
    except ValueError as exc:
        raise RuntimeError(f"Repo dataset cache escapes the repo: {raw_relative}") from exc
    return cache_root


def _archive_signature(archive_path: Path) -> Dict[str, Any]:
    stat_result = archive_path.stat()
    return {
        "source_path": str(archive_path.resolve()),
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
    }


def _archive_metadata_path(target_root: Path) -> Path:
    return target_root / ".archive_source.json"


def _archive_cache_is_current(target_root: Path, archive_path: Path) -> bool:
    metadata_path = _archive_metadata_path(target_root)
    if not target_root.is_dir() or not metadata_path.is_file():
        return False
    try:
        payload = read_json(metadata_path, default={}, expect_type=dict)
    except Exception:
        return False
    expected = _archive_signature(archive_path)
    return (
        isinstance(payload, dict)
        and payload.get("source_path") == expected["source_path"]
        and int(payload.get("size", -1)) == int(expected["size"])
        and int(payload.get("mtime_ns", -1)) == int(expected["mtime_ns"])
    )


def _safe_extract_zip_archive(*, archive_path: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    target_root_resolved = target_root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_name = str(member.filename or "").strip()
            if not member_name:
                continue
            destination = (target_root / member_name).resolve()
            try:
                destination.relative_to(target_root_resolved)
            except ValueError as exc:
                raise RuntimeError(f"Zip archive contains an unsafe path: {member_name}") from exc
        archive.extractall(target_root)


def _materialize_repo_dataset_archive(*, archive_path: Path, cache_root: Path) -> Path:
    target_root = cache_root / archive_path.stem
    if _archive_cache_is_current(target_root, archive_path):
        return target_root

    if target_root.exists():
        shutil.rmtree(target_root)

    _safe_extract_zip_archive(archive_path=archive_path, target_root=target_root)
    _archive_metadata_path(target_root).write_text(
        json.dumps(_archive_signature(archive_path), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target_root


def _repo_dataset_root_candidates(*, repo_root: str | Path, repo_relative_root: str | Path) -> list[Path]:
    base_root = resolve_repo_relative_root(repo_root=repo_root, repo_relative_root=repo_relative_root)
    if not base_root.is_dir():
        raise RuntimeError(f"Repo dataset parent not found: {base_root}")

    candidates = [path for path in base_root.iterdir() if path.is_dir()]
    cache_root = _repo_dataset_cache_root(repo_root=repo_root, repo_relative_root=repo_relative_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    for archive_path in sorted(base_root.iterdir(), key=lambda path: path.name.lower()):
        if not archive_path.is_file() or archive_path.suffix.lower() not in REPO_DATASET_ARCHIVE_SUFFIXES:
            continue

        extracted_root = _materialize_repo_dataset_archive(archive_path=archive_path, cache_root=cache_root)
        try:
            child_dirs = [path for path in extracted_root.iterdir() if path.is_dir() and not path.name.startswith(".")]
        except OSError:
            child_dirs = []

        child_candidates = [child for child in child_dirs if looks_like_class_root_dataset(child)]
        if child_candidates:
            candidates.extend(child_candidates)
            continue

        if looks_like_class_root_dataset(extracted_root):
            candidates.append(extracted_root)

    unique_candidates: dict[str, Path] = {}
    for path in candidates:
        unique_candidates.setdefault(str(path.resolve()), path)
    return sorted(unique_candidates.values(), key=lambda path: _dataset_display_name(path).lower())


def resolve_repo_relative_root(*, repo_root: str | Path, repo_relative_root: str | Path) -> Path:
    raw_repo_root = Path(repo_root).expanduser().resolve()
    raw_relative = str(repo_relative_root or "").strip()
    if not raw_relative:
        raise RuntimeError("Repo dataset root cannot be empty.")
    candidate = Path(raw_relative).expanduser()
    if candidate.is_absolute():
        raise RuntimeError("Repo dataset root must stay repo-relative.")
    resolved = (raw_repo_root / candidate).resolve()
    try:
        resolved.relative_to(raw_repo_root)
    except ValueError as exc:
        raise RuntimeError(f"Repo dataset root escapes the repo: {raw_relative}") from exc
    return resolved


def list_repo_dataset_directories(
    *,
    repo_root: str | Path,
    repo_relative_root: str | Path,
) -> list[Path]:
    return _repo_dataset_root_candidates(repo_root=repo_root, repo_relative_root=repo_relative_root)


def _has_image_file(directory: Path) -> bool:
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return True
    return False


def looks_like_class_root_dataset(dataset_root: str | Path) -> bool:
    """Return True when a directory looks like a flat class-root dataset."""
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        return False
    try:
        class_dirs = [path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")]
    except OSError:
        return False
    return bool(class_dirs) and any(_has_image_file(class_dir) for class_dir in class_dirs)


def list_dataset_directories_from_parent(
    *,
    dataset_parent: str | Path,
    allow_direct_dataset_root: bool = True,
) -> list[Path]:
    """List selectable dataset directories under a local or Drive parent path."""
    parent = Path(dataset_parent).expanduser()
    if not parent.is_dir():
        raise RuntimeError(f"Dataset parent not found: {parent}")
    if allow_direct_dataset_root and looks_like_class_root_dataset(parent):
        return [parent]
    return sorted(
        [path for path in parent.iterdir() if path.is_dir() and not path.name.startswith(".")],
        key=lambda path: path.name.lower(),
    )


def resolve_dataset_directory_from_parent(
    *,
    dataset_parent: str | Path,
    requested_name: str = "",
    prompt_label: str = "dataset",
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    allow_direct_dataset_root: bool = True,
) -> tuple[str, Path, list[str]]:
    """Resolve a dataset directory by explicit name, numeric choice, or prompt."""
    parent = Path(dataset_parent).expanduser()
    dataset_dirs = list_dataset_directories_from_parent(
        dataset_parent=parent,
        allow_direct_dataset_root=allow_direct_dataset_root,
    )
    dataset_names = [path.name for path in dataset_dirs]
    if not dataset_names:
        raise RuntimeError(f"No dataset directories were found under {parent}")

    requested = str(requested_name or "").strip()
    if requested:
        if requested.isdigit():
            selected_index = int(requested) - 1
            if selected_index < 0 or selected_index >= len(dataset_names):
                raise RuntimeError(
                    f"Requested {prompt_label} index is out of range: {requested}. "
                    f"Available options: {dataset_names}"
                )
            selected_name = dataset_names[selected_index]
        elif requested in dataset_names:
            selected_name = requested
        else:
            raise RuntimeError(
                f"Requested {prompt_label} '{requested}' was not found under {parent}. "
                f"Available options: {dataset_names}"
            )
    elif len(dataset_names) == 1:
        selected_name = dataset_names[0]
        print_fn(f"[DATASET] Only one {prompt_label} bulundu, otomatik secildi: {selected_name}")
    else:
        print_fn(f"[DATASET] Kullanilabilir {prompt_label} secenekleri ({parent}):")
        for index, dataset_name in enumerate(dataset_names, start=1):
            print_fn(f"  [{index}] {dataset_name}")
        raw_choice = str(
            input_fn(
                f"Kullanilacak {prompt_label} icin isim ya da numara girin "
                f"(1-{len(dataset_names)}): "
            )
        ).strip()
        if not raw_choice:
            raise RuntimeError(f"{prompt_label.capitalize()} secimi bos birakilamaz.")
        return resolve_dataset_directory_from_parent(
            dataset_parent=parent,
            requested_name=raw_choice,
            prompt_label=prompt_label,
            input_fn=input_fn,
            print_fn=print_fn,
            allow_direct_dataset_root=allow_direct_dataset_root,
        )

    selected_path = next(path for path in dataset_dirs if path.name == selected_name)
    return selected_name, selected_path, dataset_names


def resolve_direct_repo_dataset_root(
    *,
    repo_root: str | Path,
    repo_relative_root: str | Path,
) -> tuple[str, Path] | None:
    base_root = resolve_repo_relative_root(repo_root=repo_root, repo_relative_root=repo_relative_root)
    data_root = Path(repo_root).expanduser().resolve() / "data"
    parent = base_root.parent
    if parent.parent == data_root and parent.name in REPO_LOCAL_DATASET_PARENT_NAMES:
        return base_root.name, base_root
    return None


def resolve_repo_dataset_directory(
    *,
    repo_root: str | Path,
    repo_relative_root: str | Path,
    requested_name: str = "",
    prompt_label: str = "dataset",
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> tuple[str, Path, list[str]]:
    requested = str(requested_name or "").strip()
    direct_dataset = resolve_direct_repo_dataset_root(
        repo_root=repo_root,
        repo_relative_root=repo_relative_root,
    )
    if direct_dataset is not None:
        selected_name, selected_path = direct_dataset
        if requested and requested != selected_name:
            raise RuntimeError(
                f"Requested {prompt_label} '{requested}' does not match the explicit dataset root {selected_path}."
            )
        print_fn(f"[DATASET] Repo {prompt_label} dogrudan kok olarak kullaniliyor: {selected_path}")
        return selected_name, selected_path, [selected_name]

    dataset_dirs = list_repo_dataset_directories(
        repo_root=repo_root,
        repo_relative_root=repo_relative_root,
    )
    dataset_names = [_dataset_display_name(path) for path in dataset_dirs]
    if not dataset_names:
        raise RuntimeError(
            "No dataset directories were found under "
            f"{resolve_repo_relative_root(repo_root=repo_root, repo_relative_root=repo_relative_root)}"
        )

    if requested:
        if requested.isdigit():
            selected_index = int(requested) - 1
            if selected_index < 0 or selected_index >= len(dataset_names):
                raise RuntimeError(
                    f"Requested {prompt_label} index is out of range: {requested}. "
                    f"Available options: {dataset_names}"
                )
            selected_name = dataset_names[selected_index]
        elif requested in dataset_names:
            selected_name = requested
        else:
            raise RuntimeError(
                f"Requested {prompt_label} '{requested}' was not found under "
                f"{resolve_repo_relative_root(repo_root=repo_root, repo_relative_root=repo_relative_root)}. "
                f"Available options: {dataset_names}"
            )
    elif len(dataset_names) == 1:
        selected_name = dataset_names[0]
        print_fn(f"[DATASET] Only one repo {prompt_label} bulundu, otomatik secildi: {selected_name}")
    else:
        print_fn(
            f"[DATASET] Repo icinde kullanilabilir {prompt_label} secenekleri "
            f"({resolve_repo_relative_root(repo_root=repo_root, repo_relative_root=repo_relative_root)}):"
        )
        for index, dataset_name in enumerate(dataset_names, start=1):
            print_fn(f"  [{index}] {dataset_name}")
        raw_choice = str(
            input_fn(
                f"Kullanilacak {prompt_label} icin isim ya da numara girin "
                f"(1-{len(dataset_names)}): "
            )
        ).strip()
        if not raw_choice:
            raise RuntimeError(f"{prompt_label.capitalize()} secimi bos birakilamaz.")
        return resolve_repo_dataset_directory(
            repo_root=repo_root,
            repo_relative_root=repo_relative_root,
            requested_name=raw_choice,
            prompt_label=prompt_label,
            input_fn=input_fn,
            print_fn=print_fn,
        )

    selected_path = next(path for path in dataset_dirs if _dataset_display_name(path) == selected_name)
    return selected_name, selected_path, dataset_names


def _build_alias_lookup() -> Dict[str, set[str]]:
    lookup: Dict[str, set[str]] = {}
    for group in _CLASS_ALIAS_GROUPS:
        resolved_group = {normalize_class_name(item) for item in group if normalize_class_name(item)}
        for alias in resolved_group:
            lookup[alias] = set(resolved_group)
    return lookup


_CLASS_ALIAS_LOOKUP = _build_alias_lookup()


def class_name_aliases(name: str, *, crop_name: str) -> set[str]:
    normalized = normalize_class_name(name)
    crop_key = normalize_class_name(crop_name)
    aliases = {normalized}

    if crop_key:
        crop_prefix = f"{crop_key}_"
        if normalized.startswith(crop_prefix):
            stripped = normalized[len(crop_prefix):]
            if stripped:
                aliases.add(stripped)
        aliases.add(crop_prefix + normalized if not normalized.startswith(crop_prefix) else normalized)

    if normalized.endswith("_leaf"):
        aliases.add(normalized[: -len("_leaf")])
    if normalized in {"healthy_leaf", f"{crop_key}_healthy_leaf", f"{crop_key}_healthy"}:
        aliases.add("healthy")

    expanded = set(aliases)
    for alias in list(aliases):
        expanded.update(_CLASS_ALIAS_LOOKUP.get(alias, set()))
    return {item for item in expanded if item}


def resolve_notebook_training_classes(
    *,
    available_classes: Iterable[str],
    crop_name: str,
    taxonomy: Optional[Dict[str, Any]] = None,
    taxonomy_path: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved_available = sorted(
        {
            normalize_class_name(name)
            for name in list(available_classes)
            if normalize_class_name(name)
        }
    )
    crop_key = normalize_class_name(crop_name)
    resolution: Dict[str, Any] = {
        "selected_classes": list(resolved_available),
        "used_taxonomy_filter": False,
        "reason": "no_available_classes" if not resolved_available else "taxonomy_unavailable",
        "expected_classes": [],
        "matched_classes": [],
        "unmatched_classes": [],
    }
    if not resolved_available:
        return resolution

    taxonomy_payload: Dict[str, Any] = {}
    if isinstance(taxonomy, dict):
        taxonomy_payload = dict(taxonomy)
    elif taxonomy_path is not None and Path(taxonomy_path).exists():
        loaded = read_json(Path(taxonomy_path), default={}, expect_type=dict)
        if isinstance(loaded, dict):
            taxonomy_payload = dict(loaded)

    expected = sorted(
        {
            normalize_class_name(item)
            for item in (
                taxonomy_payload.get("crop_specific_diseases", {}).get(crop_key, [])
                if isinstance(taxonomy_payload.get("crop_specific_diseases", {}), dict)
                else []
            )
            if normalize_class_name(item)
        }
        | {"healthy"}
    )
    resolution["expected_classes"] = list(expected)
    if not expected:
        resolution["reason"] = "crop_not_in_taxonomy"
        return resolution

    expected_set = set(expected)
    matched: List[str] = []
    unmatched: List[str] = []
    for class_name in resolved_available:
        aliases = class_name_aliases(class_name, crop_name=crop_key)
        if aliases & expected_set:
            matched.append(class_name)
        else:
            unmatched.append(class_name)

    resolution["matched_classes"] = list(matched)
    resolution["unmatched_classes"] = list(unmatched)
    if matched and not unmatched:
        resolution["selected_classes"] = list(matched)
        resolution["used_taxonomy_filter"] = True
        resolution["reason"] = "full_taxonomy_alignment"
        return resolution

    resolution["reason"] = "partial_taxonomy_alignment_fallback"
    return resolution


def estimate_split_counts(total: int) -> tuple[int, int, int]:
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


def _fingerprint_paths(paths: Iterable[Path], *, root: Path) -> str:
    digest = hashlib.sha1()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
