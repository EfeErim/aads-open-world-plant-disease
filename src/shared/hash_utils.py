"""Shared file hashing utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, *, chunk_size: int = 65536) -> str:
    """Compute SHA256 hash of a file.

    Args:
        path: Path to the file to hash.
        chunk_size: Bytes to read per iteration (default 65536 ≈ 64KB).

    Returns:
        Hex digest of SHA256 hash.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
