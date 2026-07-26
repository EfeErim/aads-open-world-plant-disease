import json
from pathlib import Path

import pytest

from aads_public.release import MAX_ASSET_BYTES, download_asset, load_public_manifest


def _manifest() -> dict:
    return {
        "schema_version": "aads.public_assets.v1",
        "access_mode": "public",
        "production_ready": False,
        "scope": "controlled_demo",
        "release_tag": "v1",
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
    with pytest.raises(ValueError, match="canonical GitHub Release"):
        load_public_manifest(path)


def test_asset_url_must_match_declared_release_tag(tmp_path: Path) -> None:
    payload = _manifest()
    payload["release_tag"] = "v2"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical GitHub Release"):
        load_public_manifest(path)


def test_asset_name_cannot_escape_destination(tmp_path: Path) -> None:
    payload = _manifest()
    payload["assets"][0]["name"] = "../adapter.safetensors"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="safe filename"):
        load_public_manifest(path)


def test_manifest_requires_controlled_demo_scope_and_release_tag(tmp_path: Path) -> None:
    payload = _manifest()
    payload["scope"] = "production"
    payload.pop("release_tag")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="public and explicitly not production-ready"):
        load_public_manifest(path)


def test_manifest_rejects_oversized_asset(tmp_path: Path) -> None:
    payload = _manifest()
    payload["assets"][0]["size_bytes"] = MAX_ASSET_BYTES + 1
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="per-asset limit"):
        load_public_manifest(path)


def test_download_rejects_redirect_to_unapproved_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        headers = {"Content-Length": "12"}

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://example.com/adapter.safetensors"

        def read(self, _size: int) -> bytes:
            return b""

    monkeypatch.setattr("aads_public.release.urlopen", lambda *_args, **_kwargs: Response())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    (asset,) = load_public_manifest(manifest_path)
    destination = tmp_path / asset.name
    with pytest.raises(ValueError, match="unapproved host"):
        download_asset(asset, destination)
    assert not destination.exists()
    assert not destination.with_suffix(destination.suffix + ".partial").exists()
