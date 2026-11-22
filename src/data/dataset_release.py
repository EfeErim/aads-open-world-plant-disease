"""Dataset archive safety checks and hashing helpers shared by the dataset release code."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from src.shared.hash_utils import sha256_file as _sha256_file

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_DATASET_MEMBER_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | {".csv", ".json"}


@dataclass(frozen=True)
class DatasetResourceLimits:
    max_file_bytes: int = 100 * 1024 * 1024
    max_total_bytes: int = 20 * 1024 * 1024 * 1024
    max_file_count: int = 100_000
    max_archive_expansion_ratio: float = 100.0
    max_shard_bytes: int = 1_900_000_000


class DatasetContractError(ValueError):
    """Raised when a dataset manifest or input violates a fail-closed contract."""


def _io_path(path: Path) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, reachable through Windows long (>260 char) paths."""
    return _sha256_file(_io_path(path))


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DatasetContractError(f"Expected a JSON object: {path}")
    return payload


def _normalized_relative_path(value: str) -> str:
    raw = str(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DatasetContractError(f"Unsafe relative path: {value!r}")
    if ":" in path.parts[0]:
        raise DatasetContractError(f"Drive-qualified path is not allowed: {value!r}")
    return path.as_posix()


def validate_archive(path: Path, limits: DatasetResourceLimits | None = None) -> dict[str, Any]:
    """Validate ZIP members without extracting attacker-controlled content."""
    limits = limits or DatasetResourceLimits()
    compressed_total = 0
    expanded_total = 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > limits.max_file_count:
            raise DatasetContractError("Archive file count exceeds the configured limit")
        seen: set[str] = set()
        for member in members:
            member_path = _normalized_relative_path(member.filename.rstrip("/"))
            if member_path in seen:
                raise DatasetContractError(f"Duplicate archive member: {member_path}")
            seen.add(member_path)
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise DatasetContractError(f"Archive symlink member is not allowed: {member_path}")
            if member.is_dir():
                continue
            if Path(member_path).suffix.lower() not in ALLOWED_DATASET_MEMBER_EXTENSIONS:
                raise DatasetContractError(f"Archive member extension is not allowed: {member_path}")
            if member.file_size > limits.max_file_bytes:
                raise DatasetContractError(f"Archive member exceeds the file-size limit: {member_path}")
            compressed_total += member.compress_size
            expanded_total += member.file_size
            if expanded_total > limits.max_total_bytes:
                raise DatasetContractError("Archive expanded size exceeds the configured limit")
        ratio = expanded_total / max(compressed_total, 1)
        if ratio > limits.max_archive_expansion_ratio:
            raise DatasetContractError("Archive expansion ratio exceeds the configured limit")
    return {"member_count": len(members), "expanded_bytes": expanded_total, "expansion_ratio": ratio}


