#!/usr/bin/env python3
"""Quantization policy guards for v6 continual SD-LoRA training."""

from __future__ import annotations

from typing import Any

PROHIBITED_4BIT_KEYS = {
    "load_in_" + "4bit",
    "bnb_" + "4bit_compute_dtype",
    "bnb_" + "4bit_quant_type",
    "bnb_" + "4bit_use_double_quant",
}

PROHIBITED_4BIT_TOKENS = (
    "4bit",
    "4-bit",
    "n" + "f4",
    "q" + "lora",
    "q-lora",
)


def find_prohibited_4bit_flags(config: Any, prefix: str = "") -> list[str]:
    """Return all paths that contain prohibited 4-bit quantization settings."""
    hits: list[str] = []
    if isinstance(config, dict):
        for key, value in config.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            key_lower = str(key).lower()
            if key_lower in PROHIBITED_4BIT_KEYS or any(token in key_lower for token in PROHIBITED_4BIT_TOKENS):
                hits.append(path)
            if isinstance(value, str):
                value_lower = value.lower()
                if any(token in value_lower for token in PROHIBITED_4BIT_TOKENS):
                    hits.append(path)
            elif isinstance(value, (dict, list)):
                hits.extend(find_prohibited_4bit_flags(value, path))
    elif isinstance(config, list):
        for idx, item in enumerate(config):
            path = f"{prefix}[{idx}]"
            hits.extend(find_prohibited_4bit_flags(item, path))
    elif isinstance(config, str):
        value_lower = config.lower()
        if any(token in value_lower for token in PROHIBITED_4BIT_TOKENS):
            hits.append(prefix or "<root>")
    return sorted(set(hits))


def assert_no_prohibited_4bit_flags(config: Any) -> None:
    hits = find_prohibited_4bit_flags(config)
    if hits:
        raise ValueError(
            "v6 forbids low-bit adapter settings. Remove keys/values at: "
            + ", ".join(hits)
        )
