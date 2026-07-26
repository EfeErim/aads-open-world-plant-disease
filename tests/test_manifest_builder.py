import json
from pathlib import Path

import pytest

from aads_public.manifest import build_public_asset_manifest, build_public_asset_manifest_from_directory


def _release(path: Path, *, immutable: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "draft": not immutable,
                "immutable": immutable,
                "tag_name": "public-v1",
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


def test_builder_rejects_repository_path_injection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="owner/name"):
        build_public_asset_manifest(
            _release(tmp_path / "release.json"),
            repository="owner/repo/extra",
            release_tag="public-v1",
        )


def test_builder_rejects_mismatched_release_tag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        build_public_asset_manifest(
            _release(tmp_path / "release.json"),
            repository="owner/repo",
            release_tag="different-tag",
        )


def test_builder_can_hash_candidate_directory(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "adapter_model.safetensors").write_bytes(b"safe")

    payload = build_public_asset_manifest_from_directory(
        assets,
        repository="EfeErim/bitirmeprojesi",
        release_tag="public-v2",
    )

    assert payload["release_tag"] == "public-v2"
    assert payload["production_ready"] is False
    assert payload["assets"] == [
        {
            "name": "adapter_model.safetensors",
            "url": (
                "https://github.com/EfeErim/bitirmeprojesi/releases/download/"
                "public-v2/adapter_model.safetensors"
            ),
            "size_bytes": 4,
            "sha256": "8b3369944dd2a3fab39e32d1aeb1f763946a458ae3e6368a46432adc8f3a0860",
        }
    ]


def test_directory_builder_rejects_nested_content(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "nested").mkdir()

    with pytest.raises(ValueError, match="files only"):
        build_public_asset_manifest_from_directory(
            assets,
            repository="EfeErim/bitirmeprojesi",
            release_tag="public-v2",
        )
