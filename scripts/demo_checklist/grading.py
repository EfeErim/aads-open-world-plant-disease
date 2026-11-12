"""Shared grading constants and helpers for M2 demo checklist surfaces."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

ABSTAIN_STATUSES = {
    "unknown_crop",
    "router_uncertain",
    "adapter_unavailable",
    "non_plant_rejected",
    "router_unavailable",
}

CLASSLESS_SUPPORTED_PROBE_MARKERS = (
    "disease answer or review expected",
    "supported crop/part image",
)


def opposite_part_label(expected_part: str, diagnosis: Any) -> bool:
    expected = str(expected_part or "").strip().lower()
    diagnosis_key = unicodedata.normalize("NFKD", str(diagnosis or "").lower()).encode("ascii", "ignore").decode()
    diagnosis_tokens = [token for token in re.split(r"[^a-z0-9]+", diagnosis_key) if token]
    while diagnosis_tokens and diagnosis_tokens[-1].isdigit():
        diagnosis_tokens.pop()
    predicted_part = next(
        (token for token in reversed(diagnosis_tokens) if token in {"leaf", "yaprak", "fruit", "meyve"}),
        "",
    )
    if expected == "fruit":
        return predicted_part in {"leaf", "yaprak"}
    if expected == "leaf":
        return predicted_part in {"fruit", "meyve"}
    return False
