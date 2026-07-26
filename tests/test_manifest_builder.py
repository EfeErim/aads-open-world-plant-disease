import json
from pathlib import Path

import pytest

from aads_public.manifest import build_public_asset_manifest


def _release(path: Path, *, immutable: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "draft": not immutable,
                "immutable": immutable,
                "assets": [
                    {
                        "name": "target--adapter.safetensors",
                        "size": 42,
                        "digest": f"sha256:{'a' * 64}",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_builder_preserves_github_digest_and_public_scope(tmp_path: Path) -> None:
    payload = build_public_asset_manifest(
        _release(tmp_path / "release.json"),
        repository="owner/repo",
        release_tag="public-v1",
    )
    assert payload["production_ready"] is False
    assert payload["assets"][0]["sha256"] == "a" * 64
    assert payload["assets"][0]["url"].endswith("/public-v1/target--adapter.safetensors")


def test_builder_rejects_mutable_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="published and immutable"):
        build_public_asset_manifest(
            _release(tmp_path / "release.json", immutable=False),
            repository="owner/repo",
            release_tag="public-v1",
        )
