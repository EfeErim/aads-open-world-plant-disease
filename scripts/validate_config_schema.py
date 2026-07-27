#!/usr/bin/env python3
"""Validate schema version declarations for tracked config surface files."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _safe_print(*args, **kwargs):
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        converted = [str(a).encode("ascii", errors="replace").decode("ascii") for a in args]
        builtins.print(*converted, **kwargs)


print = _safe_print


def _is_versioned_config_surface(payload: Dict[str, Any]) -> bool:
    from src.core.config_migrations import is_versioned_config_surface_payload

    return is_versioned_config_surface_payload(payload)


def _iter_config_payloads(config_dir: str | Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    from src.shared.json_utils import read_json_dict

    for path in sorted(Path(config_dir).glob("*.json")):
        yield path, read_json_dict(path)


def validate_config_schema_versions(config_dir: str | Path = ROOT / "config") -> Tuple[List[Path], List[str]]:
    from src.core.config_migrations import CONFIG_SCHEMA_VERSION_KEY, CURRENT_CONFIG_SCHEMA_VERSION

    checked_paths: List[Path] = []
    errors: List[str] = []

    for path, payload in _iter_config_payloads(config_dir):
        if not _is_versioned_config_surface(payload):
            continue
        checked_paths.append(path)
        raw_version = payload.get(CONFIG_SCHEMA_VERSION_KEY)
        if raw_version is None:
            errors.append(
                f"{path.name}: missing {CONFIG_SCHEMA_VERSION_KEY}; expected {CURRENT_CONFIG_SCHEMA_VERSION}."
            )
            continue
        try:
            version = int(raw_version)
        except (TypeError, ValueError):
            errors.append(
                f"{path.name}: {CONFIG_SCHEMA_VERSION_KEY} must be integer-compatible, got {raw_version!r}."
            )
            continue
        if version != CURRENT_CONFIG_SCHEMA_VERSION:
            errors.append(
                f"{path.name}: {CONFIG_SCHEMA_VERSION_KEY}={version}; expected {CURRENT_CONFIG_SCHEMA_VERSION}."
            )

    return checked_paths, errors


def main() -> int:
    from src.core.config_migrations import CONFIG_SCHEMA_VERSION_KEY, CURRENT_CONFIG_SCHEMA_VERSION

    checked_paths, errors = validate_config_schema_versions()

    print("=" * 60)
    print("AADS v6 Config Schema Validation")
    print("=" * 60)
    for path in checked_paths:
        print(f"CHECK {path.relative_to(ROOT)}")

    if errors:
        print("\nFAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        f"\nPASS: {len(checked_paths)} config surface file(s) declare "
        f"{CONFIG_SCHEMA_VERSION_KEY}={CURRENT_CONFIG_SCHEMA_VERSION}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
