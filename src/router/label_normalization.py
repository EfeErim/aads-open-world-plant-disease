"""Shared normalization helpers for router-facing labels."""

from __future__ import annotations

from typing import Any


def normalize_part_label(label: Any) -> str:
    """Normalize semantically equivalent part aliases to one maintained surface."""
    normalized = str(label).strip().lower()
    if not normalized:
        return ""
    aliases = {
        "whole": "whole plant",
        "entire plant": "whole plant",
    }
    return aliases.get(normalized, normalized)
