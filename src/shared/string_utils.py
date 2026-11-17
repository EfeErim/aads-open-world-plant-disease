"""Shared string normalization utilities."""

from __future__ import annotations

import re


def normalize_class_name(name: str) -> str:
    """Normalize a class name for dataset layout consistency."""
    normalized = str(name or "").strip().lower()
    for token in (" ", "-", "/", "\\"):
        normalized = normalized.replace(token, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def slug_label_component(value: str, *, default: str = "unspecified") -> str:
    """Convert a label component to a URL-safe slug."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or default
