"""Shared JSON, path, and merge helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

_MISSING = object()


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def read_json(
    path: str | Path,
    *,
    default: Any = _MISSING,
    expect_type: type[Any] | tuple[type[Any], ...] | None = None,
) -> Any:
    resolved = Path(path)
    if not resolved.exists():
        if default is _MISSING:
            raise FileNotFoundError(resolved)
        return copy.deepcopy(default)

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if expect_type is not None and not isinstance(payload, expect_type):
        raise ValueError(f"Expected {expect_type} JSON in {resolved}, got {type(payload)!r}")
    return payload


def read_json_dict(path: str | Path, *, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = read_json(path, default=(default or {}), expect_type=dict)
    return dict(payload)


def write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> Path:
    resolved = ensure_parent(path)
    resolved.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys),
        encoding="utf-8",
    )
    return resolved


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base) if isinstance(base, dict) else {}
    if not isinstance(override, dict):
        return merged
    return _deep_merge_dicts(merged, override)


def _deep_merge_dicts(merged: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `merged` and return the result."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def clone_jsonable(payload: Any) -> Any:
    """Create a deep clone of a JSON-serializable object.

    Uses round-trip serialization to ensure compatibility.

    Args:
        payload: Any JSON-serializable object.

    Returns:
        A cloned copy of the object.
    """
    return json.loads(json.dumps(payload))


def coerce_like_value(reference: Any, value: Any) -> Any:
    """Coerce a value to match the type of a reference value.

    Attempts type-aware conversion for bool, int, float, and other types.
    Handles string representations like "true", "1", "false", etc.

    Args:
        reference: The reference value whose type to match.
        value: The value to coerce.

    Returns:
        The coerced value, or the original reference if coercion fails.
    """
    if isinstance(reference, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(value)
    if isinstance(reference, int) and not isinstance(reference, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return reference
    if isinstance(reference, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return reference
    return value
