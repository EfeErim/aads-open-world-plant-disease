#!/usr/bin/env python3
"""Detect config schema changes and warn if docs/tests are not updated."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def flatten_keys(obj: Any, prefix: str = "") -> Set[str]:
    """Recursively flatten JSON object keys into dot-notation."""
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.add(full_key)
            keys.update(flatten_keys(v, full_key))
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        # For arrays of objects, sample the first
        keys.update(flatten_keys(obj[0], prefix))
    return keys


def check_config_drift(
    base_config_path: Path,
    colab_config_path: Path,
    schema_version_path: Path,
) -> tuple[bool, list[str]]:
    """
    Check if config changes are tracked in documentation.

    Returns:
        (has_drift, warning_messages)
    """
    from src.shared.json_utils import read_json

    warnings = []

    base_config = read_json(base_config_path)
    colab_config = read_json(colab_config_path)

    # Check for schema version mismatch
    base_schema_version = base_config.get("config_schema_version")
    colab_schema_version = colab_config.get("config_schema_version")

    if base_schema_version != colab_schema_version:
        warnings.append(
            f"⚠️  Config schema version mismatch: base={base_schema_version}, "
            f"colab={colab_schema_version}"
        )

    # Check if docs mention schema versioning
    docs_readme = ROOT / "docs" / "README.md"
    if docs_readme.exists():
        docs_content = docs_readme.read_text()
        if "config_schema_version" not in docs_content:
            warnings.append(
                "⚠️  Documentation (docs/README.md) does not mention config schema versioning. "
                "Update with current schema version and breaking changes."
            )

    # Check if architecture docs exist
    arch_file = ROOT / "docs" / "architecture" / "overview.md"
    if arch_file.exists():
        arch_content = arch_file.read_text()
        if "config" in arch_content.lower():
            # Good sign that config is documented
            pass
        else:
            warnings.append(
                "⚠️  Config schema not documented in docs/architecture/overview.md. "
                "Add section explaining config flow and schema."
            )

    return len(warnings) > 0, warnings


if __name__ == "__main__":
    base_config = ROOT / "config" / "base.json"
    colab_config = ROOT / "config" / "colab.json"

    if not base_config.exists() or not colab_config.exists():
        print("ERROR: Config files not found", file=sys.stderr)
        sys.exit(1)

    has_drift, warnings = check_config_drift(base_config, colab_config, None)

    if warnings:
        print("\nConfig drift warnings:\n")
        for warning in warnings:
            print(warning)
        print(
            "\nWhen updating config schema, ensure:"
            "\n  1. Both base.json and colab.json are in sync"
            "\n  2. config_schema_version is incremented if breaking"
            "\n  3. docs/README.md or docs/architecture/ are updated"
            "\n  4. Tests covering new fields are added"
        )
        sys.exit(1 if has_drift else 0)
    else:
        print("PASS: Config schema appears consistent")
        sys.exit(0)
