"""Filesystem helpers for deterministic artifact persistence."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .json_utils import ensure_parent, write_json


class ArtifactStore:
    """Persist JSON, text, and copied artifacts under a root directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str | Path) -> Path:
        return self.root / Path(relative_path)

    def write_json(self, relative_path: str | Path, payload: Any) -> Path:
        return write_json(self.resolve(relative_path), payload, ensure_ascii=False)

    def write_text(self, relative_path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
        target = ensure_parent(self.resolve(relative_path))
        target.write_text(text, encoding=encoding)
        return target

    def write_bytes(self, relative_path: str | Path, content: bytes) -> Path:
        target = ensure_parent(self.resolve(relative_path))
        target.write_bytes(content)
        return target

    def copy_file(self, source_path: str | Path, relative_path: str | Path) -> Path:
        source = Path(source_path)
        target = ensure_parent(self.resolve(relative_path))
        shutil.copy2(source, target)
        return target
