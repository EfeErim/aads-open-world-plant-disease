from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import src.pipeline.adapter_release as adapter_release
from src.pipeline.adapter_release import (
    APPROVAL_SCHEMA,
    RELEASE_SCHEMA,
    create_adapter_draft,
    fetch_adapter_release,
    prepare_notebook_adapter_root,
    promote_adapter_pointer,
    record_draft_release,
    validate_promotion_approval,
    validate_release_manifest,
    verify_adapter_draft,
    write_release_receipt,
)


def _manifest(file_bytes: bytes, *, state: str = "local", access_mode: str = "private") -> dict:
    release_id = 321 if state != "local" else None
    asset_id = 654 if state != "local" else None
    return {
        "schema_version": RELEASE_SCHEMA,
        "targets": ["tomato__leaf"],
        "github_release": {
            "repository": "example/private-repo",
            "access_mode": access_mode,
            "tag": "aads-adapters-v1.0.0",
            "tag_commit_sha": "a" * 40,
            "release_id": release_id,
            "draft": state != "immutable",
            "immutable": state == "immutable",
        },
        "files": [
            {
                "target_id": "tomato__leaf",
                "asset_name": "tomato__leaf--adapter_meta.json",
                "asset_id": asset_id,
                "local_path": "tomato/leaf/continual_sd_lora_adapter/adapter_meta.json",
                "source_path": "source/adapter_meta.json",
                "size_bytes": len(file_bytes),
                "sha256": hashlib.sha256(file_bytes).hexdigest(),
            }
        ],
    }


class FakeGitHub:
    def __init__(
        self,
        content: bytes,
        *,
        immutable: bool = False,
        release_exists: bool = False,
        private: bool = True,
    ) -> None:
        self.content = content
        self.immutable = immutable
        self.release_exists = release_exists or immutable
        self.private = private
        self.uploaded: list[str] = []
        self.immutable_setting_checks = 0

    def get_repository(self, repository: str) -> dict:
        return {
            "id": 99,
            "full_name": repository,
            "private": self.private,
            "permissions": {"pull": True, "push": True},
        }

    def immutable_releases_enabled(self, repository: str) -> bool:
        self.immutable_setting_checks += 1
        return True

    def create_draft(self, repository: str, tag: str, commit: str, *, name: str, body: str) -> dict:
        return {
            "id": 321,
            "tag_name": tag,
            "draft": True,
            "immutable": False,
            "upload_url": "https://uploads.example/assets{?name,label}",
            "html_url": "https://github.example/releases/321",
        }

    def find_release_by_tag(self, repository: str, tag: str) -> dict | None:
        if self.release_exists:
            return {"id": 321, "tag_name": tag}
        return None

    def upload_asset(self, upload_url: str, source: Path, asset_name: str) -> dict:
        self.uploaded.append(asset_name)
        return {
            "id": 654,
            "name": asset_name,
            "state": "uploaded",
            "size": source.stat().st_size,
            "digest": f"sha256:{hashlib.sha256(source.read_bytes()).hexdigest()}",
        }

    def get_release(self, repository: str, release_id: int) -> dict:
        return {
            "id": release_id,
            "tag_name": "aads-adapters-v1.0.0",
            "draft": not self.immutable,
            "immutable": self.immutable,
            "assets": [
                {
                    "id": 654,
                    "name": "tomato__leaf--adapter_meta.json",
                    "state": "uploaded",
                    "size": len(self.content),
                    "digest": f"sha256:{hashlib.sha256(self.content).hexdigest()}",
                }
            ],
        }

    def download_asset(self, repository: str, asset_id: int, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.content)


def _set_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_RELEASE_REPOSITORY", "example/private-repo")


def _write_approval(path: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    release = manifest["github_release"]
    path.write_text(
        json.dumps(
            {
                "schema_version": APPROVAL_SCHEMA,
                "approver_identity": "release-owner",
                "approved_at": "2026-07-13T10:00:00Z",
                "release_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "repository": release["repository"],
                "release_tag": release["tag"],
                "release_id": release["release_id"],
                "tag_commit_sha": release["tag_commit_sha"],
                "asset_ids": [record["asset_id"] for record in manifest["files"]],
                "redistribution_approved": True,
                "license_reviewer_identity": "license-owner",
            }
        ),
        encoding="utf-8",
    )


def test_manifest_rejects_unexpected_release_file() -> None:
    payload = _manifest(b"ok")
    payload["files"][0]["asset_name"] = "tomato__leaf--evil.pkl"
    with pytest.raises(ValueError, match="asset name"):
        validate_release_manifest(payload)


def test_fetch_uses_fixed_immutable_release_and_never_returns_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_repository(monkeypatch)
    content = b"metadata"
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(_manifest(content, state="immutable")), encoding="utf-8")
    client = FakeGitHub(content, immutable=True)
    receipt = fetch_adapter_release(
        manifest_path,
        tmp_path / "models" / "adapters",
        token="secret",
        client=client,
    )
    assert receipt["release_tag"] == "aads-adapters-v1.0.0"
    assert "token" not in receipt
    assert receipt["verified"] is True
    assert client.immutable_setting_checks == 0


def test_public_immutable_release_fetches_anonymously(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_repository(monkeypatch)
    for name in adapter_release.READ_TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    content = b"metadata"
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(
        json.dumps(_manifest(content, state="immutable", access_mode="public")),
        encoding="utf-8",
    )
    constructed_tokens: list[str | None] = []
    client = FakeGitHub(content, immutable=True, private=False)

    def _client_factory(token: str | None = None) -> FakeGitHub:
        constructed_tokens.append(token)
        return client

    monkeypatch.setattr(adapter_release, "GitHubReleaseClient", _client_factory)
    receipt = fetch_adapter_release(manifest_path, tmp_path / "models" / "adapters")

    assert receipt["verified"] is True
    assert constructed_tokens == [None]


def test_release_visibility_must_match_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_repository(monkeypatch)
    content = b"metadata"
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(
        json.dumps(_manifest(content, state="immutable", access_mode="public")),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="visibility"):
        fetch_adapter_release(
            manifest_path,
            tmp_path / "models" / "adapters",
            client=FakeGitHub(content, immutable=True, private=True),
        )


def test_draft_upload_and_record_bind_every_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_repository(monkeypatch)
    content = b"metadata"
    source = tmp_path / "source" / "adapter_meta.json"
    source.parent.mkdir()
    source.write_bytes(content)
    payload = _manifest(content)
    payload["files"][0]["source_path"] = str(source)
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt = create_adapter_draft(manifest_path, token="secret", client=FakeGitHub(content))
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    recorded = record_draft_release(manifest_path, receipt_path)
    assert recorded["release_id"] == 321
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["files"][0]["asset_id"] == 654


def test_verify_draft_checks_remote_asset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_repository(monkeypatch)
    content = b"metadata"
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(_manifest(content, state="draft")), encoding="utf-8")
    result = verify_adapter_draft(manifest_path, token="secret", client=FakeGitHub(content, release_exists=True))
    assert result["asset_count"] == 1
    assert result["verified"] is True


def test_promotion_requires_exact_release_identity_license_and_asset_ids(tmp_path: Path) -> None:
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(_manifest(b"ok", state="draft")), encoding="utf-8")
    approval_path = tmp_path / "approval.json"
    _write_approval(approval_path, manifest_path)
    assert validate_promotion_approval(approval_path, manifest_path)["approver_identity"] == "release-owner"
    payload = json.loads(approval_path.read_text(encoding="utf-8"))
    payload["asset_ids"] = [999]
    approval_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="asset IDs"):
        validate_promotion_approval(approval_path, manifest_path)


def test_notebook_keeps_legacy_root_without_reading_unpromoted_manifest(tmp_path: Path) -> None:
    result = prepare_notebook_adapter_root(
        {"inference": {"adapter_root": "runs"}},
        tmp_path,
        manifest_path=tmp_path / "missing.json",
    )
    assert result == {"adapter_root": "runs", "deployment_release": False, "verified": False}


def test_promote_pointer_requires_immutable_manifest_and_verified_cache(tmp_path: Path) -> None:
    content = b"metadata"
    manifest = _manifest(content, state="immutable")
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    approval_path = tmp_path / "approval.json"
    _write_approval(approval_path, manifest_path)
    cache_root = tmp_path / "models" / "adapters"
    local_file = cache_root / "tomato/leaf/continual_sd_lora_adapter/adapter_meta.json"
    local_file.parent.mkdir(parents=True)
    local_file.write_bytes(content)
    config_path = tmp_path / "base.json"
    config_path.write_text('{"config_schema_version":2,"inference":{"adapter_root":"runs"}}', encoding="utf-8")
    result = promote_adapter_pointer(manifest_path, approval_path, cache_root, [config_path])
    assert result["promoted"] is True
    assert json.loads(config_path.read_text(encoding="utf-8"))["inference"]["adapter_root"] == "models/adapters"


def test_receipt_redacts_token_and_authorization_fields(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    write_release_receipt(path, {"token": "secret", "Authorization": "Bearer secret", "verified": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"verified": True}
