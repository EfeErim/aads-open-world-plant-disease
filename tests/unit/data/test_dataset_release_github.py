from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.dataset_release_github import (
    fetch_dataset_release,
    materialize_dataset_release,
    preflight_dataset_release,
    validate_dataset_github_manifest,
)

# Synthetic one-image release (manifest, shard, GitHub release record) produced by the packaging
# and publishing code that was removed once the dataset release was final.
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "dataset_release"


class FakeGitHub:
    def __init__(self, fixture: str = "standard") -> None:
        root = FIXTURES / fixture
        self.release = json.loads((root / "remote_release.json").read_text(encoding="utf-8"))
        self.contents = {int(asset["id"]): (root / asset["name"]).read_bytes() for asset in self.release["assets"]}

    def get_repository(self, repository: str) -> dict:
        return {
            "id": 99,
            "full_name": repository,
            "private": True,
            "permissions": {"pull": True, "push": False},
        }

    def find_release_by_tag(self, repository: str, tag: str) -> dict | None:
        return self.release if self.release["tag_name"] == tag else None

    def get_release(self, repository: str, release_id: int) -> dict:
        assert self.release["id"] == release_id
        return self.release

    def download_asset(self, repository: str, asset_id: int, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.contents[asset_id])


def _manifest_copy(tmp_path: Path, fixture: str = "standard") -> Path:
    destination = tmp_path / "manifest.json"
    shutil.copyfile(FIXTURES / fixture / "manifest.json", destination)
    return destination


@pytest.fixture(autouse=True)
def _repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_RELEASE_REPOSITORY", "example/private-repo")


def test_published_fixture_manifest_is_valid() -> None:
    manifest = json.loads((FIXTURES / "standard" / "manifest.json").read_text(encoding="utf-8"))
    validate_dataset_github_manifest(manifest, require_immutable=True)


def test_long_member_path_materializes_on_windows(tmp_path: Path) -> None:
    manifest_path = _manifest_copy(tmp_path, "long_member")
    cache = tmp_path / "cache"
    fetch_dataset_release(manifest_path, cache, client=FakeGitHub("long_member"))

    result = materialize_dataset_release(manifest_path, cache, tmp_path / "materialized")

    assert result["verified"] is True
    assert result["file_count"] == 1


def test_fetch_and_materialize_are_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = _manifest_copy(tmp_path)
    client = FakeGitHub()
    cache = tmp_path / "cache"
    cache.mkdir()
    asset_name = json.loads(manifest_path.read_text(encoding="utf-8"))["assets"][0]["asset_name"]
    (cache / f"{asset_name}.partial").write_bytes(b"interrupted")
    monkeypatch.delenv("GITHUB_RELEASE_REPOSITORY")
    assert fetch_dataset_release(
        manifest_path,
        cache,
        repository="example/private-repo",
        client=client,
    )["downloaded_count"] == 1
    monkeypatch.setenv("GITHUB_RELEASE_REPOSITORY", "example/private-repo")
    assert fetch_dataset_release(manifest_path, cache, client=client)["reused_count"] == 1
    destination = tmp_path / "materialized"
    assert materialize_dataset_release(manifest_path, cache, destination)["verified"] is True
    assert (destination / "tomato/leaf/id_train/healthy/image.jpg").read_bytes() == b"jpeg-data"
    (destination / ".gitkeep").touch()
    assert materialize_dataset_release(manifest_path, cache, destination)["idempotent"] is True


def test_fetch_rejects_truncated_asset(tmp_path: Path) -> None:
    manifest_path = _manifest_copy(tmp_path)
    client = FakeGitHub()
    asset_id = next(iter(client.contents))
    client.contents[asset_id] = client.contents[asset_id][:-1]
    with pytest.raises(Exception, match="failed verification"):
        fetch_dataset_release(manifest_path, tmp_path / "bad-cache", client=client)
    assert not any((tmp_path / "bad-cache").iterdir())


def test_preflight_rejects_insufficient_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = _manifest_copy(tmp_path)
    monkeypatch.setattr(
        "src.data.dataset_release_github.shutil.disk_usage",
        lambda path: SimpleNamespace(total=100, used=99, free=1),
    )
    with pytest.raises(RuntimeError, match="workspace capacity"):
        preflight_dataset_release(manifest_path, tmp_path / "workspace", client=FakeGitHub())


def test_preflight_rejects_repository_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = _manifest_copy(tmp_path)
    monkeypatch.setenv("GITHUB_RELEASE_REPOSITORY", "example/other-repo")
    with pytest.raises(RuntimeError, match="must exactly match"):
        preflight_dataset_release(manifest_path, tmp_path / "workspace", client=FakeGitHub())


def test_preflight_accepts_fine_grained_read_token_without_classic_permissions_field(tmp_path: Path) -> None:
    manifest_path = _manifest_copy(tmp_path)

    class FineGrainedReadGitHub(FakeGitHub):
        def get_repository(self, repository: str) -> dict:
            result = super().get_repository(repository)
            result.pop("permissions")
            return result

    result = preflight_dataset_release(manifest_path, tmp_path / "workspace", client=FineGrainedReadGitHub())

    assert result["private"] is True
    assert result["read_access"] is True
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["peak_workspace_bytes"] == max(asset["size_bytes"] for asset in manifest["assets"])


def test_preflight_uses_successful_private_repo_lookup_as_read_proof(tmp_path: Path) -> None:
    manifest_path = _manifest_copy(tmp_path)

    class ActionsTokenGitHub(FakeGitHub):
        def get_repository(self, repository: str) -> dict:
            result = super().get_repository(repository)
            result["permissions"] = {"pull": False, "push": False, "admin": False}
            return result

    result = preflight_dataset_release(manifest_path, tmp_path / "workspace", client=ActionsTokenGitHub())

    assert result["private"] is True
    assert result["read_access"] is True
