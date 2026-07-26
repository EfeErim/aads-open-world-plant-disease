"""Checksum-pinned download helpers for public GitHub Release assets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

ALLOWED_ASSET_HOSTS = frozenset(
    {"github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"}
)
MAX_ASSET_COUNT = 128
MAX_ASSET_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class PublicAsset:
    name: str
    url: str
    size_bytes: int
    sha256: str


def _validate_asset(row: dict, *, release_tag: str) -> PublicAsset:
    name = str(row.get("name") or "")
    url = str(row.get("url") or "")
    digest = str(row.get("sha256") or "")
    size = row.get("size_bytes")
    parsed = urlparse(url)
    if not name or Path(name).name != name:
        raise ValueError("asset name must be a single safe filename")
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or f"/releases/download/{quote(release_tag, safe='')}/" not in parsed.path
        or not parsed.path.endswith(f"/{quote(name, safe='')}")
    ):
        raise ValueError("asset URL must be a canonical GitHub Release download URL")
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_ASSET_BYTES:
        raise ValueError("asset size_bytes must be positive and within the per-asset limit")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("asset sha256 must be a lowercase digest")
    return PublicAsset(name, url, size, digest)


def load_public_manifest(path: str | Path) -> tuple[PublicAsset, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "aads.public_assets.v1":
        raise ValueError("unsupported public asset manifest")
    if (
        payload.get("access_mode") != "public"
        or payload.get("production_ready") is not False
        or payload.get("scope") != "controlled_demo"
    ):
        raise ValueError("public demo assets must be public and explicitly not production-ready")
    release_tag = payload.get("release_tag")
    if not isinstance(release_tag, str) or not release_tag.strip():
        raise ValueError("public demo assets must declare a release_tag")
    assets = tuple(_validate_asset(dict(row), release_tag=release_tag) for row in payload.get("assets") or [])
    if not assets or len(assets) > MAX_ASSET_COUNT or len({asset.name for asset in assets}) != len(assets):
        raise ValueError("manifest must contain unique assets")
    if sum(asset.size_bytes for asset in assets) > MAX_TOTAL_BYTES:
        raise ValueError("manifest assets exceed the total download-size limit")
    return assets


def download_asset(asset: PublicAsset, destination: str | Path) -> Path:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    request = Request(asset.url, headers={"User-Agent": "aads-public/1.1"})
    digest = hashlib.sha256()
    size = 0
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            final_url = urlparse(response.geturl())
            if final_url.scheme != "https" or final_url.hostname not in ALLOWED_ASSET_HOSTS:
                raise ValueError(f"download redirected to an unapproved host for {asset.name}")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != asset.size_bytes:
                raise ValueError(f"download size header does not match the manifest for {asset.name}")
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if size > asset.size_bytes:
                    raise ValueError(f"download exceeded the manifest size for {asset.name}")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if size != asset.size_bytes or digest.hexdigest() != asset.sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"download verification failed for {asset.name}")
    temporary.replace(target)
    return target
