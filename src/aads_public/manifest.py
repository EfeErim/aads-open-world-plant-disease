"""Build checksum-pinned manifests from immutable GitHub Release metadata."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote


def build_public_asset_manifest(source: Path, *, repository: str, release_tag: str) -> dict:
    release = json.loads(source.read_text(encoding="utf-8"))
    if release.get("draft") is not False or release.get("immutable") is not True:
        raise ValueError("source release must be published and immutable")
    rows = []
    for asset in release.get("assets") or []:
        digest = str(asset.get("digest") or "")
        if not digest.startswith("sha256:"):
            raise ValueError(f"source asset has no GitHub SHA-256 digest: {asset.get('name')}")
        name = str(asset["name"])
        rows.append(
            {
                "name": name,
                "url": (
                    f"https://github.com/{repository}/releases/download/"
                    f"{quote(release_tag, safe='')}/{quote(name, safe='')}"
                ),
                "size_bytes": int(asset["size"]),
                "sha256": digest.removeprefix("sha256:"),
            }
        )
    if not rows:
        raise ValueError("source release contains no assets")
    return {
        "schema_version": "aads.public_assets.v1",
        "access_mode": "public",
        "release_tag": release_tag,
        "production_ready": False,
        "scope": "controlled_demo",
        "assets": rows,
    }
