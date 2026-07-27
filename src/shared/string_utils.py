"""Shared string normalization utilities."""

from __future__ import annotations

import re


def normalize_class_name(name: str) -> str:
    """Normalize a class name for dataset layout consistency.

    Converts to lowercase, replaces separators with underscores,
    collapses consecutive underscores, and strips leading/trailing underscores.

    Args:
        name: Raw class name (e.g., 'Grape Leaf', 'berry-spot', 'fruit/type').

    Returns:
        Normalized class name (e.g., 'grape_leaf', 'berry_spot', 'fruit_type').
    """
    normalized = str(name or "").strip().lower()
    for token in (" ", "-", "/", "\\"):
        normalized = normalized.replace(token, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def slug_label_component(value: str, *, default: str = "unspecified") -> str:
    """Convert a label component to a URL-safe slug.

    Lowercases, keeps only alphanumeric + underscores, collapses runs,
    strips boundaries.

    Args:
        value: Label text (e.g., 'Leaf Spot', 'Disease-Type-1').
        default: Fallback if result is empty.

    Returns:
        Slugified label (e.g., 'leaf_spot', 'disease_type_1').
    """
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or default


def normalize_notebook_identifier(name: str) -> str:
    """Normalize a notebook or experiment identifier.

    Converts to lowercase, replaces non-alphanumeric with underscores,
    collapses runs, strips boundaries.

    Args:
        name: Raw identifier (e.g., 'Notebook 2 - Training').

    Returns:
        Normalized identifier (e.g., 'notebook_2_training').
    """
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name or "").strip())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")
