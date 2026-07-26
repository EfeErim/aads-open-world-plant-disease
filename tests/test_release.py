import json
from pathlib import Path

import pytest

from aads_public.release import load_public_manifest


def _manifest() -> dict:
    return {
        "schema_version": "aads.public_assets.v1",
        "access_mode": "public",
        "production_ready": False,
        "assets": [
            {
                "name": "adapter.safetensors",
                "url": "https://github.com/owner/repo/releases/download/v1/adapter.safetensors",
                "size_bytes": 12,
                "sha256": "a" * 64,
            }
        ],
    }


def test_public_asset_manifest_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    (asset,) = load_public_manifest(path)
    assert asset.name == "adapter.safetensors"


def test_non_github_asset_host_is_rejected(tmp_path: Path) -> None:
    payload = _manifest()
    payload["assets"][0]["url"] = "https://example.com/adapter.safetensors"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="GitHub HTTPS host"):
        load_public_manifest(path)


def test_asset_name_cannot_escape_destination(tmp_path: Path) -> None:
    payload = _manifest()
    payload["assets"][0]["name"] = "../adapter.safetensors"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="safe filename"):
        load_public_manifest(path)
