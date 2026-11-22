"""Download and materialize the published, immutable AADS dataset release.

Fetching and materialization are separate fail-closed steps: every shard is checked
against the release manifest before anything is extracted.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from src.data.dataset_release import (
    DatasetContractError,
    DatasetResourceLimits,
    _io_path,
    canonical_sha256,
    read_json_dict,
    sha256_file,
)
from src.shared.github_releases import (
    COMMIT_SHA_PATTERN,
    READ_TOKEN_NAMES,
    REPOSITORY_ENV,
    REPOSITORY_PATTERN,
    GitHubReleaseClient,
    resolve_token,
)

DATASET_GITHUB_RELEASE_SCHEMA = "aads.dataset_release.v1"
DATASET_TAG_PATTERN = re.compile(r"^aads-dataset-v[0-9]+\.[0-9]+\.[0-9]+(?:[-.][A-Za-z0-9.-]+)?$")
DATASET_RELEASE_STATES = frozenset({"staged", "verified", "current", "revoked"})
TRANSFER_RETRY_ENV = "AADS_GITHUB_RELEASE_TRANSFER_RETRIES"
def _safe_relative_path(value: object, *, field: str) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    path = PurePosixPath(raw)
    if not raw or raw.startswith("/") or ".." in path.parts or ":" in path.parts[0]:
        raise DatasetContractError(f"{field} must be a safe relative path: {raw!r}")
    return path.as_posix()


def _release_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(payload)
    identity.pop("release_manifest_sha256", None)
    identity["assets"] = [
        {key: value for key, value in dict(asset).items() if key != "local_path"}
        for asset in payload.get("assets") or []
    ]
    return identity


def _validate_hash(value: object, *, field: str) -> str:
    digest = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise DatasetContractError(f"{field} must be a lowercase SHA-256")
    return digest


def validate_dataset_github_manifest(
    payload: Mapping[str, Any], *, require_draft: bool = False, require_immutable: bool = False
) -> None:
    if payload.get("schema_version") != DATASET_GITHUB_RELEASE_SCHEMA:
        raise DatasetContractError("Unsupported dataset GitHub Release manifest schema")
    if payload.get("release_state") not in DATASET_RELEASE_STATES:
        raise DatasetContractError("Dataset release_state is invalid")
    repository = str(payload.get("repository") or "")
    tag = str(payload.get("release_tag") or "")
    commit = str(payload.get("tag_commit_sha") or "")
    if not REPOSITORY_PATTERN.fullmatch(repository):
        raise DatasetContractError("Dataset repository must use explicit owner/repository form")
    if not DATASET_TAG_PATTERN.fullmatch(tag):
        raise DatasetContractError("Dataset release tag must use aads-dataset-v<version>")
    if not COMMIT_SHA_PATTERN.fullmatch(commit):
        raise DatasetContractError("Dataset tag commit must be an exact commit SHA")
    _validate_hash(payload.get("snapshot_manifest_sha256"), field="snapshot_manifest_sha256")
    recorded = _validate_hash(payload.get("release_manifest_sha256"), field="release_manifest_sha256")
    if recorded != canonical_sha256(_release_identity(payload)):
        raise DatasetContractError("Dataset release manifest SHA-256 does not match its canonical content")

    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise DatasetContractError("A publishable dataset release must contain at least one shard")
    seen_names: set[str] = set()
    seen_members: set[str] = set()
    for asset in assets:
        if not isinstance(asset, Mapping):
            raise DatasetContractError("Every dataset release asset must be an object")
        name = _safe_relative_path(asset.get("asset_name"), field="asset_name")
        if "/" in name or not name.endswith(".zip") or name in seen_names:
            raise DatasetContractError(f"Invalid or duplicate dataset shard asset: {name}")
        size = asset.get("size_bytes")
        if not isinstance(size, int) or size <= 0 or size >= DatasetResourceLimits().max_shard_bytes:
            raise DatasetContractError(f"Dataset shard is empty or exceeds the 1.9 GiB limit: {name}")
        _validate_hash(asset.get("sha256"), field=f"{name}.sha256")
        members = asset.get("members")
        if not isinstance(members, list) or not members:
            raise DatasetContractError(f"Dataset shard has no approved members: {name}")
        for member in members:
            member_path = _safe_relative_path(member.get("path"), field="member.path")
            _validate_hash(member.get("sha256"), field=f"{member_path}.sha256")
            if not isinstance(member.get("size_bytes"), int) or int(member["size_bytes"]) <= 0:
                raise DatasetContractError(f"Invalid dataset member size: {member_path}")
            if member_path in seen_members:
                raise DatasetContractError(f"Duplicate dataset member across shards: {member_path}")
            seen_members.add(member_path)
        if require_draft or require_immutable:
            if not isinstance(asset.get("asset_id"), int) or int(asset["asset_id"]) <= 0:
                raise DatasetContractError(f"GitHub asset_id is required for {name}")
        seen_names.add(name)

    release_id = payload.get("release_id")
    if (require_draft or require_immutable) and (not isinstance(release_id, int) or release_id <= 0):
        raise DatasetContractError("GitHub release_id is required")
    if require_draft and (payload.get("draft") is not True or payload.get("immutable") is not False):
        raise DatasetContractError("A mutable dataset draft is required")
    if require_immutable and (payload.get("draft") is not False or payload.get("immutable") is not True):
        raise DatasetContractError("A published immutable dataset release is required")
    if payload.get("release_state") == "current" and not require_immutable:
        if payload.get("immutable") is not True:
            raise DatasetContractError("Only an immutable dataset release may be current")
    if payload.get("release_state") == "revoked" and not str(payload.get("revocation_reason") or "").strip():
        raise DatasetContractError("A revoked dataset release requires a reason")
    if require_immutable and payload.get("release_state") == "revoked":
        raise DatasetContractError("A revoked dataset release cannot be fetched or promoted")


def _client(token: str | None, *, client: Any) -> Any:
    if client is not None:
        return client
    resolved = token or resolve_token()
    if not resolved:
        raise RuntimeError(f"One of {', '.join(READ_TOKEN_NAMES)} is required")
    return GitHubReleaseClient(resolved)


def preflight_dataset_release(
    manifest_path: Path,
    workspace_root: Path,
    *,
    repository: str | None = None,
    token: str | None = None,
    client: Any = None,
) -> dict[str, Any]:
    manifest = read_json_dict(manifest_path)
    validate_dataset_github_manifest(manifest)
    configured = str(repository or os.environ.get(REPOSITORY_ENV) or "")
    if configured != manifest["repository"]:
        raise RuntimeError(f"{REPOSITORY_ENV} must exactly match the dataset release manifest repository")
    api = _client(token, client=client)
    repository_info = api.get_repository(manifest["repository"])
    if str(repository_info.get("full_name") or "").lower() != str(manifest["repository"]).lower():
        raise RuntimeError("GitHub repository identity does not match the dataset manifest")
    if repository_info.get("private") is not True:
        raise RuntimeError("Dataset assets may only be stored in the configured private repository")
    retry_limit = int(str(os.environ.get(TRANSFER_RETRY_ENV) or "3"))
    if retry_limit < 1 or retry_limit > 10:
        raise RuntimeError(f"{TRANSFER_RETRY_ENV} must be between 1 and 10")
    workspace_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(workspace_root).free
    materialized_bytes = sum(
        int(member["size_bytes"]) for asset in manifest["assets"] for member in asset.get("members") or []
    )
    # Every transfer is streamed; keep one largest-shard margin for a retried partial download.
    peak_bytes = max(int(asset["size_bytes"]) for asset in manifest["assets"])
    if free_bytes < peak_bytes:
        raise RuntimeError(f"Insufficient dataset workspace capacity: required={peak_bytes}, available={free_bytes}")
    existing = api.find_release_by_tag(manifest["repository"], manifest["release_tag"])
    recorded_id = manifest.get("release_id")
    if existing is not None and recorded_id is None:
        raise RuntimeError("Dataset release tag already exists without a recorded release ID")
    if existing is None and recorded_id is not None:
        raise RuntimeError("Recorded dataset release ID no longer exists")
    if existing is not None and int(existing.get("id") or 0) != int(recorded_id or 0):
        raise RuntimeError("Dataset release tag resolves to a different release ID")
    return {
        "repository": manifest["repository"],
        "repository_id": repository_info.get("id"),
        "private": True,
        "read_access": True,
        "download_bytes": sum(int(asset["size_bytes"]) for asset in manifest["assets"]),
        "materialized_bytes": materialized_bytes,
        "peak_workspace_bytes": peak_bytes,
        "available_bytes": free_bytes,
        "transfer_retry_limit": retry_limit,
    }


def _verify_remote(manifest: Mapping[str, Any], remote: Mapping[str, Any], *, immutable: bool) -> None:
    if int(remote.get("id") or 0) != int(manifest.get("release_id") or 0):
        raise DatasetContractError("GitHub dataset release ID mismatch")
    if str(remote.get("tag_name") or "") != manifest["release_tag"]:
        raise DatasetContractError("GitHub dataset release tag mismatch")
    if remote.get("draft") is immutable or remote.get("immutable") is not immutable:
        raise DatasetContractError("GitHub dataset release mutability mismatch")
    expected = {str(asset["asset_name"]): asset for asset in manifest["assets"]}
    remote_assets = list(remote.get("assets") or [])
    actual = {str(asset.get("name") or ""): asset for asset in remote_assets}
    if len(actual) != len(remote_assets):
        raise DatasetContractError("GitHub dataset release contains duplicate asset names")
    if set(actual) != set(expected):
        raise DatasetContractError("GitHub dataset release asset allowlist mismatch")
    for name, record in expected.items():
        asset = actual[name]
        digest = str(asset.get("digest") or "")
        if asset.get("state") != "uploaded":
            raise DatasetContractError(f"GitHub dataset asset is not uploaded: {name}")
        if int(asset.get("id") or 0) != int(record.get("asset_id") or 0):
            raise DatasetContractError(f"GitHub dataset asset ID mismatch: {name}")
        if int(asset.get("size") or -1) != int(record["size_bytes"]):
            raise DatasetContractError(f"GitHub dataset asset size mismatch: {name}")
        if digest and digest != f"sha256:{record['sha256']}":
            raise DatasetContractError(f"GitHub dataset asset digest mismatch: {name}")


def verify_cached_shards(cache_root: Path, manifest: Mapping[str, Any]) -> None:
    validate_dataset_github_manifest(manifest, require_immutable=True)
    expected = {str(asset["asset_name"]): asset for asset in manifest["assets"]}
    actual = {path.name for path in cache_root.iterdir() if path.is_file()} if cache_root.is_dir() else set()
    if actual != set(expected):
        raise DatasetContractError(
            f"Dataset shard cache allowlist mismatch; missing={sorted(set(expected) - actual)}, "
            f"unexpected={sorted(actual - set(expected))}"
        )
    for name, asset in expected.items():
        path = cache_root / name
        if path.stat().st_size != asset["size_bytes"] or sha256_file(path) != asset["sha256"]:
            raise DatasetContractError(f"Dataset shard cache checksum mismatch: {name}")


def fetch_dataset_release(
    manifest_path: Path,
    cache_root: Path,
    *,
    repository: str | None = None,
    token: str | None = None,
    client: Any = None,
) -> dict[str, Any]:
    manifest = read_json_dict(manifest_path)
    validate_dataset_github_manifest(manifest, require_immutable=True)
    api = _client(token, client=client)
    preflight_dataset_release(
        manifest_path,
        cache_root.parent,
        repository=repository,
        client=api,
    )
    remote = api.get_release(manifest["repository"], int(manifest["release_id"]))
    _verify_remote(manifest, remote, immutable=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    expected_names = {str(asset["asset_name"]) for asset in manifest["assets"]}
    allowed_partial_names = {name + ".partial" for name in expected_names}
    unexpected = [
        path
        for path in cache_root.iterdir()
        if path.is_file() and path.name not in expected_names and path.name not in allowed_partial_names
    ]
    if unexpected:
        raise DatasetContractError(f"Unexpected dataset shard cache files: {[path.name for path in unexpected]}")
    downloaded = 0
    reused = 0
    for asset in manifest["assets"]:
        destination = cache_root / str(asset["asset_name"])
        if destination.is_file() and destination.stat().st_size == asset["size_bytes"] and sha256_file(destination) == asset[
            "sha256"
        ]:
            reused += 1
            continue
        partial = destination.with_suffix(destination.suffix + ".partial")
        partial.unlink(missing_ok=True)
        api.download_asset(manifest["repository"], int(asset["asset_id"]), partial)
        if partial.stat().st_size != asset["size_bytes"] or sha256_file(partial) != asset["sha256"]:
            partial.unlink(missing_ok=True)
            raise DatasetContractError(f"Downloaded dataset shard failed verification: {asset['asset_name']}")
        os.replace(partial, destination)
        downloaded += 1
    verify_cached_shards(cache_root, manifest)
    return {
        "repository": manifest["repository"],
        "release_tag": manifest["release_tag"],
        "release_id": manifest["release_id"],
        "cache_root": str(cache_root.resolve()),
        "downloaded_count": downloaded,
        "reused_count": reused,
        "verified": True,
    }


def _verify_materialized(root: Path, manifest: Mapping[str, Any]) -> None:
    expected = {
        str(member["path"]): member for asset in manifest["assets"] for member in asset.get("members") or []
    }
    io_root = _io_path(root)
    actual = {
        path.relative_to(io_root).as_posix()
        for path in io_root.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    }
    if actual != set(expected):
        raise DatasetContractError("Materialized dataset file allowlist mismatch")
    for relative_path, member in expected.items():
        path = _io_path(root / Path(*PurePosixPath(relative_path).parts))
        if path.stat().st_size != member["size_bytes"] or sha256_file(path) != member["sha256"]:
            raise DatasetContractError(f"Materialized dataset checksum mismatch: {relative_path}")


def materialize_dataset_release(
    manifest_path: Path, cache_root: Path, destination: Path, *, allow_replace: bool = False
) -> dict[str, Any]:
    manifest = read_json_dict(manifest_path)
    validate_dataset_github_manifest(manifest, require_immutable=True)
    verify_cached_shards(cache_root, manifest)
    destination = destination.resolve()
    if destination == destination.parent or (destination / ".git").exists():
        raise DatasetContractError("Refusing to materialize over a filesystem or repository root")
    if destination.exists():
        try:
            _verify_materialized(destination, manifest)
        except DatasetContractError:
            if not allow_replace:
                raise DatasetContractError(
                    "Existing materialization differs; explicit allow_replace is required for atomic replacement"
                ) from None
        else:
            return {
                "release_tag": manifest["release_tag"],
                "destination": str(destination),
                "file_count": manifest["file_count"],
                "verified": True,
                "idempotent": True,
            }
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = sum(int(member["size_bytes"]) for asset in manifest["assets"] for member in asset["members"])
    if shutil.disk_usage(destination.parent).free < required + max(int(asset["size_bytes"]) for asset in manifest["assets"]):
        raise RuntimeError("Insufficient capacity for dataset materialization")
    with tempfile.TemporaryDirectory(prefix="aads-dataset-materialize-", dir=destination.parent) as temp_dir:
        payload_root = Path(temp_dir) / "payload"
        payload_root.mkdir()
        for asset in manifest["assets"]:
            approved = {str(member["path"]): member for member in asset["members"]}
            with zipfile.ZipFile(cache_root / str(asset["asset_name"])) as archive:
                if archive.testzip() is not None:
                    raise DatasetContractError(f"Corrupt dataset shard: {asset['asset_name']}")
                archive_files = [info for info in archive.infolist() if not info.is_dir()]
                actual = {info.filename for info in archive_files}
                if actual != set(approved) or len(actual) != len(archive_files):
                    raise DatasetContractError(f"Dataset shard member allowlist mismatch: {asset['asset_name']}")
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    relative_path = _safe_relative_path(info.filename, field="archive member")
                    target = payload_root / Path(*PurePosixPath(relative_path).parts)
                    _io_path(target.parent).mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    size = 0
                    with archive.open(info) as source, _io_path(target).open("wb") as output:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            size += len(chunk)
                            if size > int(approved[relative_path]["size_bytes"]):
                                raise DatasetContractError(f"Dataset member exceeds approved size: {relative_path}")
                            digest.update(chunk)
                            output.write(chunk)
                    if size != approved[relative_path]["size_bytes"] or digest.hexdigest() != approved[relative_path][
                        "sha256"
                    ]:
                        raise DatasetContractError(f"Dataset member verification failed: {relative_path}")
        _verify_materialized(payload_root, manifest)
        backup = destination.with_name(destination.name + ".previous")
        if backup.exists():
            raise DatasetContractError(f"Stale materialization backup requires review: {backup}")
        if destination.exists():
            os.replace(_io_path(destination), _io_path(backup))
        try:
            os.replace(_io_path(payload_root), _io_path(destination))
        except BaseException:
            if backup.exists() and not destination.exists():
                os.replace(_io_path(backup), _io_path(destination))
            raise
        if backup.exists():
            shutil.rmtree(_io_path(backup))
    return {
        "release_tag": manifest["release_tag"],
        "destination": str(destination),
        "file_count": manifest["file_count"],
        "verified": True,
    }


