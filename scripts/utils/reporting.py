"""Common reporting utilities for scripts.

Small helpers to write/read JSON reports in a consistent way so scripts
can reuse the same behavior (mkdir, encoding, pretty print).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    if str(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
