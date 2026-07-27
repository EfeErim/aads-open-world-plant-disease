from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.dataset_release import _io_path, build_shard_plan, write_json
from src.data.dataset_release_github import (
    DATASET_APPROVAL_SCHEMA,
    dry_run_dataset_publish,
    fetch_dataset_release,
    materialize_dataset_release,
    package_dataset_shards,
    preflight_dataset_release,
    promote_dataset_pointer,
    publish_dataset_release,
    upload_dataset_draft,
    validate_dataset_github_manifest,
    verify_dataset_release,
)


def _candidate(content: bytes, *, distributed_path: str = "tomato/leaf/id_train/healthy/image.jpg") -> dict:
    digest = hashlib.sha256(content).hexdigest()
    return {
        "schema_version": "v1_dataset_github_release_candidate",
        "dataset_version": "1.0.0",
        "dataset_tag": "aads-dataset-v1.0.0",
        "staging_snapshot_id": "dataset-snapshot-" + "a" * 20,
        "snapshot_manifest_sha256": "a" * 64,
        "inventory_cutoff": "2026-07-13T00:00:00Z",
        "release_state": "candidate",
        "revocation_reason": "",
        "tombstones": [],
        "files": [
            {
                "sample_id": "sample-1",
                "distributed_path": distributed_path,
                "distributed_content_sha256": digest,
                "source_content_sha256": "b" * 64,
                "size_bytes": len(content),
                "target": "tomato__leaf",
                "class_name": "healthy",
                "evidence_role": "id_train",
                "split": "train",
                "evaluation_cohort_id": "cohort-1",
                "comparability_key": "key-1",
            }
        ],
    }


def _package(
    tmp_path: Path, content: bytes = b"jpeg-data", *, distributed_path: str = "tomato/leaf/id_train/healthy/image.jpg"
) -> tuple[Path, Path, dict]:
    candidate = _candidate(content, distributed_path=distributed_path)
    plan = build_shard_plan(candidate)
    candidate_path = tmp_path / "candidate.json"
    plan_path = tmp_path / "plan.json"
    staging_root = tmp_path / "stage"
    source = staging_root / candidate["files"][0]["distributed_path"]
    _io_path(source.parent).mkdir(parents=True)
    _io_path(source).write_bytes(content)
    write_json(candidate_path, candidate)
    write_json(plan_path, plan)
    manifest = package_dataset_shards(
        candidate_path,
        plan_path,
        staging_root,
        tmp_path / "package",
        repository="example/private-repo",
        tag_commit_sha="c" * 40,
        publisher="release-owner",
    )
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, staging_root, manifest


class FakeGitHub:
    def __init__(self) -> None:
        self.release: dict | None = None
        self.contents: dict[int, bytes] = {}
        self.next_asset_id = 100
        self.upload_count = 0
        self.publish_count = 0

    def get_repository(self, repository: str) -> dict:
        return {
            "id": 99,
            "full_name": repository,
            "private": True,
            "permissions": {"pull": True, "push": True},
        }

    def immutable_releases_enabled(self, repository: str) -> bool:
        return True

    def find_release_by_tag(self, repository: str, tag: str) -> dict | None:
        return self.release if self.release and self.release["tag_name"] == tag else None

    def create_draft(self, repository: str, tag: str, commit: str, *, name: str, body: str) -> dict:
        self.release = {
            "id": 77,
            "tag_name": tag,
            "draft": True,
            "immutable": False,
            "upload_url": "https://uploads.example/assets{?name}",
            "assets": [],
        }
        return self.release

    def upload_asset(self, upload_url: str, source: Path, asset_name: str) -> dict:
        content = source.read_bytes()
        asset = {
            "id": self.next_asset_id,
            "name": asset_name,
            "size": len(content),
            "state": "uploaded",
            "digest": f"sha256:{hashlib.sha256(content).hexdigest()}",
        }
        self.next_asset_id += 1
        self.upload_count += 1
        assert self.release is not None
        self.release["assets"].append(asset)
        self.contents[asset["id"]] = content
        return asset

    def get_release(self, repository: str, release_id: int) -> dict:
        assert self.release is not None and self.release["id"] == release_id
        return self.release

    def download_asset(self, repository: str, asset_id: int, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.contents[asset_id])

    def publish(self, repository: str, release_id: int) -> dict:
        assert self.release is not None
        self.release["draft"] = False
        self.release["immutable"] = True
        self.release["published_at"] = "2026-07-13T12:00:00Z"
        self.publish_count += 1
        return self.release

    def tag_commit_sha(self, repository: str, tag: str) -> str:
        return "c" * 40


def _set_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_RELEASE_REPOSITORY", "example/private-repo")


def test_long_member_path_packages_and_materializes_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative_path = "tomato/leaf/id_train/healthy/" + "/".join(("a" * 100, "b" * 100, "c" * 100)) + "/image.jpg"
    manifest_path, _, _ = _package(tmp_path, distributed_path=relative_path)
    _set_repository(monkeypatch)
    client = FakeGitHub()
    upload_dataset_draft(manifest_path, client=client, max_upload_bytes=10_000_000)
    verify_dataset_release(manifest_path, client=client, record=True)
    approval_path = tmp_path / "approval.json"
    _approval(approval_path, manifest_path)
    publish_dataset_release(manifest_path, approval_path, client=client, max_upload_bytes=10_000_000)
    cache = tmp_path / "cache"
    fetch_dataset_release(manifest_path, cache, client=client)

    result = materialize_dataset_release(manifest_path, cache, tmp_path / "materialized")

    assert result["verified"] is True
    assert result["file_count"] == 1


def _approval(path: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    write_json(
        path,
        {
            "schema_version": DATASET_APPROVAL_SCHEMA,
            "approver_identity": "release-owner",
            "approved_at": "2026-07-13T12:00:00Z",
            "release_manifest_sha256": manifest["release_manifest_sha256"],
            "repository": manifest["repository"],
            "release_tag": manifest["release_tag"],
            "release_id": manifest["release_id"],
            "tag_commit_sha": manifest["tag_commit_sha"],
            "asset_ids": [asset["asset_id"] for asset in manifest["assets"]],
            "redistribution_approved": True,
            "privacy_review_approved": True,
        },
    )


def test_zero_file_candidate_dry_run_blocks_without_external_mutation(tmp_path: Path) -> None:
    candidate = _candidate(b"x")
    candidate["files"] = []
    candidate_path = tmp_path / "candidate.json"
    plan_path = tmp_path / "plan.json"
    write_json(candidate_path, candidate)
    write_json(plan_path, build_shard_plan(candidate))
    result = dry_run_dataset_publish(
        candidate_path,
        plan_path,
        repository="example/private-repo",
        tag_commit_sha="c" * 40,
    )
    assert result["status"] == "blocked"
    assert result["external_mutation"] is False
    assert result["blocker"] == "zero_file_candidate_requires_reviewed_per_image_metadata"


def test_packaging_is_deterministic_stored_and_manifest_bound(tmp_path: Path) -> None:
    first_path, _, first = _package(tmp_path / "first")
    second_path, _, second = _package(tmp_path / "second")
    assert first["assets"][0]["sha256"] == second["assets"][0]["sha256"]
    assert first["release_manifest_sha256"] == second["release_manifest_sha256"]
    validate_dataset_github_manifest(json.loads(first_path.read_text(encoding="utf-8")))


def test_upload_retry_verify_fetch_and_materialize_are_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_repository(monkeypatch)
    manifest_path, _, _ = _package(tmp_path)
    client = FakeGitHub()
    first = upload_dataset_draft(manifest_path, client=client, max_upload_bytes=10_000_000)
    second = upload_dataset_draft(manifest_path, client=client, max_upload_bytes=10_000_000)
    assert first["uploaded_count"] == 1
    assert second["uploaded_count"] == 0
    assert second["reused_count"] == 1
    verified = verify_dataset_release(manifest_path, client=client, record=True)
    assert verified["verified"] is True
    approval_path = tmp_path / "approval.json"
    _approval(approval_path, manifest_path)
    assert publish_dataset_release(
        manifest_path, approval_path, client=client, max_upload_bytes=10_000_000
    )["immutable"] is True
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


def test_fetch_rejects_truncated_asset_and_pointer_rejects_partial_or_concurrent_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_repository(monkeypatch)
    manifest_path, _, _ = _package(tmp_path)
    client = FakeGitHub()
    upload_dataset_draft(manifest_path, client=client, max_upload_bytes=10_000_000)
    verify_dataset_release(manifest_path, client=client, record=True)
    approval_path = tmp_path / "approval.json"
    _approval(approval_path, manifest_path)
    publish_dataset_release(manifest_path, approval_path, client=client, max_upload_bytes=10_000_000)
    asset_id = next(iter(client.contents))
    client.contents[asset_id] = client.contents[asset_id][:-1]
    with pytest.raises(Exception, match="failed verification"):
        fetch_dataset_release(manifest_path, tmp_path / "bad-cache", client=client)

    client.contents[asset_id] += b"\x00"  # still wrong content, restore from local package instead
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    client.contents[asset_id] = Path(manifest["assets"][0]["local_path"]).read_bytes()
    cache = tmp_path / "cache"
    fetch_dataset_release(manifest_path, cache, client=client)
    materialized = tmp_path / "materialized"
    materialize_dataset_release(manifest_path, cache, materialized)
    pointer = tmp_path / "current.json"
    lock = pointer.with_suffix(".json.lock")
    lock.write_text("busy", encoding="utf-8")
    with pytest.raises(Exception, match="already in progress"):
        promote_dataset_pointer(manifest_path, cache, materialized, pointer)
    lock.unlink()
    assert promote_dataset_pointer(manifest_path, cache, materialized, pointer)["promoted"] is True
    assert promote_dataset_pointer(manifest_path, cache, materialized, pointer)["idempotent"] is True


def test_preflight_rejects_quota_and_shared_read_write_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_repository(monkeypatch)
    manifest_path, _, manifest = _package(tmp_path)
    client = FakeGitHub()
    with pytest.raises(RuntimeError, match="storage quota"):
        preflight_dataset_release(
            manifest_path,
            tmp_path / "workspace",
            write=True,
            client=client,
            max_upload_bytes=manifest["assets"][0]["size_bytes"] - 1,
        )
    monkeypatch.setenv("AADS_GITHUB_RELEASE_WRITE_TOKEN", "same-secret")
    monkeypatch.setenv("AADS_GITHUB_RELEASE_READ_TOKEN", "same-secret")
    with pytest.raises(RuntimeError, match="must be distinct"):
        preflight_dataset_release(
            manifest_path,
            tmp_path / "workspace",
            write=True,
            client=client,
            max_upload_bytes=10_000_000,
        )


def test_preflight_rejects_permissions_and_capacity_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_repository(monkeypatch)
    manifest_path, _, _ = _package(tmp_path)

    class ReadOnlyGitHub(FakeGitHub):
        def get_repository(self, repository: str) -> dict:
            result = super().get_repository(repository)
            result["permissions"] = {"pull": True, "push": False, "admin": False}
            return result

    with pytest.raises(RuntimeError, match="write access"):
        preflight_dataset_release(
            manifest_path,
            tmp_path / "workspace",
            write=True,
            client=ReadOnlyGitHub(),
            max_upload_bytes=10_000_000,
        )

    monkeypatch.setattr(
        "src.data.dataset_release_github.shutil.disk_usage",
        lambda path: SimpleNamespace(total=100, used=99, free=1),
    )
    with pytest.raises(RuntimeError, match="workspace capacity"):
        preflight_dataset_release(
            manifest_path,
            tmp_path / "workspace",
            write=True,
            client=FakeGitHub(),
            max_upload_bytes=10_000_000,
        )


def test_preflight_accepts_fine_grained_read_token_without_classic_permissions_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_repository(monkeypatch)
    manifest_path, _, _ = _package(tmp_path)

    class FineGrainedReadGitHub(FakeGitHub):
        def get_repository(self, repository: str) -> dict:
            result = super().get_repository(repository)
            result.pop("permissions")
            return result

    result = preflight_dataset_release(
        manifest_path,
        tmp_path / "workspace",
        write=False,
        client=FineGrainedReadGitHub(),
    )

    assert result["private"] is True
    assert result["read_access"] is True
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["peak_workspace_bytes"] == max(asset["size_bytes"] for asset in manifest["assets"])


def test_preflight_uses_successful_private_repo_lookup_as_read_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_repository(monkeypatch)
    manifest_path, _, _ = _package(tmp_path)

    class ActionsTokenGitHub(FakeGitHub):
        def get_repository(self, repository: str) -> dict:
            result = super().get_repository(repository)
            result["permissions"] = {"pull": False, "push": False, "admin": False}
            return result

    result = preflight_dataset_release(
        manifest_path,
        tmp_path / "workspace",
        write=False,
        client=ActionsTokenGitHub(),
    )

    assert result["private"] is True
    assert result["read_access"] is True
