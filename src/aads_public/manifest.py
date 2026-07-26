"""Build checksum-pinned manifests from immutable GitHub Release metadata."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_TAG_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_coordinates(repository: str, release_tag: str) -> None:
    if not _REPOSITORY_PATTERN.fullmatch(repository):
        raise ValueError("repository must be a canonical owner/name identifier")
    if not _TAG_PATTERN.fullmatch(release_tag):
        raise ValueError("release_tag contains unsupported characters")


def _manifest(repository: str, release_tag: str, rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("source contains no assets")
    return {
        "schema_version": "aads.public_assets.v1",
        "access_mode": "public",
        "release_tag": release_tag,
        "production_ready": False,
        "scope": "controlled_demo",
        "assets": rows,
    }


def _asset_row(repository: str, release_tag: str, name: str, size: int, sha256: str) -> dict:
    if not name or Path(name).name != name:
        raise ValueError("source asset name must be a single safe filename")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"source asset has invalid size: {name}")
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError(f"source asset has invalid SHA-256: {name}")
    return {
        "name": name,
        "url": (
            f"https://github.com/{repository}/releases/download/"
            f"{quote(release_tag, safe='')}/{quote(name, safe='')}"
        ),
        "size_bytes": size,
        "sha256": sha256,
    }


def build_public_asset_manifest(source: Path, *, repository: str, release_tag: str) -> dict:
    _validate_coordinates(repository, release_tag)
    release = json.loads(source.read_text(encoding="utf-8"))
    if release.get("draft") is not False or release.get("immutable") is not True:
        raise ValueError("source release must be published and immutable")
    if release.get("tag_name") != release_tag:
        raise ValueError("source release tag does not match release_tag")
    rows = []
    for asset in sorted(release.get("assets") or [], key=lambda row: str(row.get("name") or "")):
        digest = str(asset.get("digest") or "")
        if not digest.startswith("sha256:"):
            raise ValueError(f"source asset has no GitHub SHA-256 digest: {asset.get('name')}")
        name = str(asset["name"])
        size = asset.get("size")
        rows.append(_asset_row(repository, release_tag, name, size, digest.removeprefix("sha256:")))
    return _manifest(repository, release_tag, rows)


def build_public_asset_manifest_from_directory(source: Path, *, repository: str, release_tag: str) -> dict:
    """Build a candidate manifest before uploading an immutable release."""

    _validate_coordinates(repository, release_tag)
    if not source.is_dir():
        raise ValueError("source must be an existing directory")
    rows = []
    for asset in sorted(source.iterdir(), key=lambda path: path.name):
        if not asset.is_file():
            raise ValueError(f"source must contain files only: {asset.name}")
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        rows.append(_asset_row(repository, release_tag, asset.name, asset.stat().st_size, digest))
    return _manifest(repository, release_tag, rows)
