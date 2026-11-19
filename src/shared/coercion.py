"""Lenient conversions for values read from configs, JSON payloads and model outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def coerce_float(value: Any, default: float = 0.0) -> float:
    """Return ``value`` as a float, or ``default`` when it is missing or not numeric."""
    fallback = float(default)
    try:
        return fallback if value is None else float(value)
    except (TypeError, ValueError, OverflowError):
        return fallback


def optional_float(value: Any) -> Optional[float]:
    """Return ``value`` as a float, or None when it is missing, empty or not numeric."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with offset."""
    return datetime.now(timezone.utc).isoformat()
