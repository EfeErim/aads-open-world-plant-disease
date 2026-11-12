"""Router/prototype handoff cache helpers for the M2 demo checklist."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HANDOFF_CACHE_SCHEMA_VERSION = "m2_router_prototype_handoff_cache.v1"


def _stable_json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _fingerprint_path_value(path: Path, *, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _path_fingerprint(path: Path | None, *, repo_root: Path) -> dict[str, Any]:
    if path is None:
        return {}
    resolved_path = path.resolve()
    try:
        stat = resolved_path.stat()
    except OSError:
        return {"path": _fingerprint_path_value(resolved_path, repo_root=repo_root), "exists": False}
    return {
        "path": _fingerprint_path_value(resolved_path, repo_root=repo_root),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _handoff_cache_key(
    *,
    row: Any,
    image_path: Path,
    config_env: str,
    device: str,
    enable_prototype_reconciler: bool,
    prototype_bank_path: Path | None,
    taxonomy_registry_path: Path | None,
    prototype_min_similarity: float | None,
    prototype_min_margin: float | None,
    prototype_min_negative_gap: float | None,
    prototype_target_policies: dict[str, Any] | None,
    expected_target_id: str | None,
    expected_class_label: str | None,
    repo_root: Path,
    runner_path: Path,
    auto_handoff_path: Path,
    prototype_reconciler_path: Path,
) -> str:
    material = {
        "schema": HANDOFF_CACHE_SCHEMA_VERSION,
        "image": _path_fingerprint(image_path, repo_root=repo_root),
        "image_id": row.image_id,
        "source": row.source,
        "config_env": config_env,
        "device": device,
        "enable_prototype_reconciler": enable_prototype_reconciler,
        "prototype_bank": _path_fingerprint(prototype_bank_path, repo_root=repo_root),
        "taxonomy_registry": _path_fingerprint(taxonomy_registry_path, repo_root=repo_root),
        "prototype_min_similarity": prototype_min_similarity,
        "prototype_min_margin": prototype_min_margin,
        "prototype_min_negative_gap": prototype_min_negative_gap,
        "prototype_target_policy_hash": _stable_json_hash(prototype_target_policies or {}),
        "expected_target_id": expected_target_id,
        "expected_class_label": expected_class_label,
        "code": {
            "runner": _path_fingerprint(runner_path, repo_root=repo_root),
            "auto_handoff": _path_fingerprint(auto_handoff_path, repo_root=repo_root),
            "prototype_reconciler": _path_fingerprint(prototype_reconciler_path, repo_root=repo_root),
        },
    }
    return _stable_json_hash(material)


def _new_handoff_cache() -> dict[str, Any]:
    return {
        "schema_version": HANDOFF_CACHE_SCHEMA_VERSION,
        "entries": {},
        "stats": {"hits": 0, "misses": 0, "writes": 0},
    }


def _load_handoff_cache(cache_path: Path | None, *, refresh: bool = False) -> dict[str, Any]:
    if cache_path is None or refresh or not cache_path.exists():
        return _new_handoff_cache()
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, dict):
        entries = {}
    cache = _new_handoff_cache()
    cache["entries"] = entries
    return cache


def _write_handoff_cache(cache_path: Path | None, cache: dict[str, Any]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _lookup_cached_handoff_by_key(*, cache: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    if cache is None:
        return None
    entry = cache.setdefault("entries", {}).get(key)
    if not isinstance(entry, dict) or not isinstance(entry.get("handoff"), dict):
        return None
    stats = cache.setdefault("stats", {"hits": 0, "misses": 0, "writes": 0})
    stats["hits"] = int(stats.get("hits", 0)) + 1
    return dict(entry["handoff"])


def _cached_handoff_by_key(
    *,
    cache: dict[str, Any] | None,
    key: str,
    row: Any,
    image_path: Path,
    resolver: Callable[..., dict[str, Any]],
    resolver_kwargs: dict[str, Any],
) -> dict[str, Any]:
    if cache is None:
        return resolver(image_path, **resolver_kwargs)
    entries = cache.setdefault("entries", {})
    stats = cache.setdefault("stats", {"hits": 0, "misses": 0, "writes": 0})
    entry = entries.get(key)
    if isinstance(entry, dict) and isinstance(entry.get("handoff"), dict):
        stats["hits"] = int(stats.get("hits", 0)) + 1
        return dict(entry["handoff"])
    stats["misses"] = int(stats.get("misses", 0)) + 1
    handoff = resolver(image_path, **resolver_kwargs)
    entries[key] = {
        "image_id": row.image_id,
        "image": str(image_path),
        "handoff": handoff,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    stats["writes"] = int(stats.get("writes", 0)) + 1
    return handoff
