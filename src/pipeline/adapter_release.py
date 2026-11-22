"""GitHub Release helpers for immutable deployment adapters."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from src.shared.github_releases import (
    COMMIT_SHA_PATTERN,
    READ_TOKEN_NAMES,
    REPOSITORY_ENV,
    REPOSITORY_PATTERN,
    GitHubReleaseClient,
    resolve_token,
)
from src.shared.hash_utils import sha256_file
from src.shared.json_utils import read_json_dict

RELEASE_SCHEMA = "aads.adapter_release.v2"
MAX_RELEASE_ASSET_BYTES = 2 * 1024**3
EXPECTED_BUNDLE_FILES = frozenset(
    {
        "DINOV3_LICENSE.md",
        "README.md",
        "adapter_config.json",
        "adapter_meta.json",
        "adapter_model.safetensors",
        "classifier.pth",
        "fusion.pth",
        "production_readiness.json",
    }
)
TAG_PATTERN = re.compile(
    r"^(?:aads-adapters|aads-public-demo)-v[0-9]+\.[0-9]+\.[0-9]+(?:[-.][A-Za-z0-9.-]+)?$"
)
RELEASE_ACCESS_MODES = frozenset({"private", "public"})


def _validate_relative_path(value: object, *, field: str) -> Path:
    raw = str(value or "").replace("\\", "/").strip()
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path, got {raw!r}.")
    return path


def _github_release(manifest: dict[str, Any]) -> dict[str, Any]:
    release = manifest.get("github_release")
    if not isinstance(release, dict):
        raise ValueError("github_release must be an object.")
    repository = str(release.get("repository") or "")
    tag = str(release.get("tag") or "")
    commit = str(release.get("tag_commit_sha") or "")
    if not REPOSITORY_PATTERN.fullmatch(repository):
        raise ValueError("GitHub repository must use the explicit owner/repository form.")
    if not TAG_PATTERN.fullmatch(tag):
        raise ValueError("Adapter release tag must use an approved AADS adapter release namespace.")
    if not COMMIT_SHA_PATTERN.fullmatch(commit):
        raise ValueError("GitHub Release tag commit must be an exact 40-character commit SHA.")
    return release


def release_access_mode(manifest: dict[str, Any]) -> str:
    """Return the explicit release access mode, preserving private v2 compatibility."""
    release = _github_release(manifest)
    access_mode = str(release.get("access_mode") or "private").strip().lower()
    if access_mode not in RELEASE_ACCESS_MODES:
        raise ValueError(f"Unsupported GitHub Release access_mode: {access_mode!r}.")
    return access_mode


def validate_release_manifest(
    payload: dict[str, Any],
    *,
    require_draft: bool = False,
    require_immutable: bool = False,
) -> None:
    if payload.get("schema_version") != RELEASE_SCHEMA:
        raise ValueError(f"Unsupported adapter release schema: {payload.get('schema_version')!r}.")
    release = _github_release(payload)
    release_id = release.get("release_id")
    if (require_draft or require_immutable) and (not isinstance(release_id, int) or release_id <= 0):
        raise ValueError("GitHub release_id is required.")
    if require_draft and (release.get("draft") is not True or release.get("immutable") is not False):
        raise ValueError("A verified mutable draft release is required.")
    if require_immutable and (release.get("draft") is not False or release.get("immutable") is not True):
        raise ValueError("A published immutable GitHub Release is required.")

    release_files = payload.get("files")
    if not isinstance(release_files, list) or not release_files:
        raise ValueError("Adapter release manifest must contain files.")
    seen_assets: set[str] = set()
    targets: set[str] = set()
    for record in release_files:
        if not isinstance(record, dict):
            raise ValueError("Every adapter release file record must be an object.")
        target = str(record.get("target_id") or "")
        local_path = _validate_relative_path(record.get("local_path"), field="local_path").as_posix()
        asset_name = str(record.get("asset_name") or "")
        if Path(local_path).name not in EXPECTED_BUNDLE_FILES:
            raise ValueError(f"Unexpected adapter release file: {local_path}.")
        if not local_path.startswith(f"{target.replace('__', '/')}/"):
            raise ValueError(f"local_path does not match target {target}: {local_path}.")
        if asset_name != f"{target}--{Path(local_path).name}" or "/" in asset_name or "\\" in asset_name:
            raise ValueError(f"Invalid GitHub Release asset name: {asset_name!r}.")
        checksum = str(record.get("sha256") or "")
        size_bytes = record.get("size_bytes")
        if not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"Invalid SHA-256 for {asset_name}.")
        if not isinstance(size_bytes, int) or size_bytes < 0 or size_bytes >= MAX_RELEASE_ASSET_BYTES:
            raise ValueError(f"Invalid or oversized release asset: {asset_name}.")
        if asset_name in seen_assets:
            raise ValueError(f"Duplicate GitHub Release asset: {asset_name}.")
        if require_draft or require_immutable:
            if not isinstance(record.get("asset_id"), int) or int(record["asset_id"]) <= 0:
                raise ValueError(f"GitHub asset_id is required for {asset_name}.")
        seen_assets.add(asset_name)
        targets.add(target)
    if targets != set(payload.get("targets") or []):
        raise ValueError("Release target list does not match file records.")


def verify_release_files(root: Path, manifest: dict[str, Any]) -> None:
    validate_release_manifest(manifest)
    records = {str(record["local_path"]): record for record in manifest["files"]}
    expected = {relative_path: str(record["sha256"]) for relative_path, record in records.items()}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    }
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        raise ValueError(f"Release file allowlist mismatch; missing={missing}, unexpected={unexpected}.")
    for relative_path, checksum in expected.items():
        artifact_path = root / relative_path
        if sha256_file(artifact_path) != checksum:
            raise ValueError(f"Checksum mismatch: {relative_path}.")
        if artifact_path.name == "production_readiness.json":
            readiness = read_json_dict(artifact_path)
            context = readiness.get("context")
            if not isinstance(context, dict):
                raise ValueError(f"Production readiness context is missing: {relative_path}.")
            internal_target = (
                f"{str(context.get('crop_name') or '').strip().lower()}__"
                f"{str(context.get('part_name') or '').strip().lower()}"
            )
            expected_target = str(records[relative_path]["target_id"])
            if internal_target != expected_target:
                raise ValueError(
                    "Production readiness target mismatch: "
                    f"expected={expected_target}, actual={internal_target}, path={relative_path}."
                )


def _client(token: str | None, *, client: Any, allow_anonymous: bool = False) -> Any:
    if client is not None:
        return client
    resolved = token or resolve_token()
    if not resolved and not allow_anonymous:
        raise RuntimeError(f"One of {', '.join(READ_TOKEN_NAMES)} is required.")
    return GitHubReleaseClient(resolved)


def preflight_adapter_release(
    manifest_path: Path,
    cache_root: Path,
    *,
    token: str | None = None,
    client: Any = None,
) -> dict[str, Any]:
    manifest = read_json_dict(manifest_path)
    validate_release_manifest(manifest)
    release = _github_release(manifest)
    access_mode = release_access_mode(manifest)
    configured_repository = str(os.environ.get(REPOSITORY_ENV) or "")
    if configured_repository != release["repository"]:
        raise RuntimeError(f"{REPOSITORY_ENV} must exactly match the release manifest repository.")
    api = _client(token, client=client, allow_anonymous=access_mode == "public")
    repository = api.get_repository(release["repository"])
    if str(repository.get("full_name") or "").lower() != str(release["repository"]).lower():
        raise RuntimeError("GitHub API repository identity does not match the manifest.")
    repository_is_private = repository.get("private") is True
    if repository_is_private != (access_mode == "private"):
        raise RuntimeError(
            "GitHub repository visibility does not match the adapter release access_mode."
        )
    existing_release = api.find_release_by_tag(release["repository"], release["tag"])
    recorded_release_id = release.get("release_id")
    if existing_release is not None and recorded_release_id is None:
        raise RuntimeError("Release tag already has an unrecorded GitHub Release; refusing to create a duplicate.")
    if existing_release is None and recorded_release_id is not None:
        raise RuntimeError("Recorded GitHub Release ID/tag no longer exists.")
    if existing_release is not None and int(existing_release.get("id") or 0) != int(recorded_release_id or 0):
        raise RuntimeError("Recorded GitHub Release ID does not match the release tag.")
    total_bytes = sum(int(record["size_bytes"]) for record in manifest["files"])
    cache_parent = cache_root.resolve().parent
    cache_parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(cache_parent).free
    required_bytes = total_bytes + max(int(record["size_bytes"]) for record in manifest["files"]) + 64 * 1024**2
    if free_bytes < required_bytes:
        raise RuntimeError(f"Insufficient cache space: required={required_bytes}, available={free_bytes}.")
    return {
        "repository": release["repository"],
        "repository_id": repository.get("id"),
        "access_mode": access_mode,
        "private": repository_is_private,
        "total_bytes": total_bytes,
        "available_bytes": free_bytes,
    }


def _verify_remote_release(manifest: dict[str, Any], remote: dict[str, Any], *, immutable: bool) -> None:
    release = _github_release(manifest)
    if int(remote.get("id") or 0) != int(release.get("release_id") or 0):
        raise ValueError("GitHub release ID mismatch.")
    if str(remote.get("tag_name") or "") != release["tag"]:
        raise ValueError("GitHub release tag mismatch.")
    if remote.get("draft") is immutable or remote.get("immutable") is not immutable:
        raise ValueError("GitHub release draft/immutable state mismatch.")
    expected = {str(record["asset_name"]): record for record in manifest["files"]}
    actual = {str(asset.get("name") or ""): asset for asset in remote.get("assets") or []}
    if set(actual) != set(expected):
        raise ValueError("GitHub release asset allowlist mismatch.")
    for name, record in expected.items():
        asset = actual[name]
        digest = str(asset.get("digest") or "")
        if int(asset.get("id") or 0) != int(record.get("asset_id") or 0) or asset.get("size") != record["size_bytes"]:
            raise ValueError(f"GitHub release asset identity mismatch: {name}.")
        if digest and digest != f"sha256:{record['sha256']}":
            raise ValueError(f"GitHub release asset digest mismatch: {name}.")


def fetch_adapter_release(
    manifest_path: Path,
    cache_root: Path,
    *,
    token: str | None = None,
    client: Any = None,
) -> dict[str, Any]:
    manifest = read_json_dict(manifest_path)
    validate_release_manifest(manifest, require_immutable=True)
    release = _github_release(manifest)
    api = _client(token, client=client, allow_anonymous=release_access_mode(manifest) == "public")
    preflight_adapter_release(manifest_path, cache_root, client=api)
    remote = api.get_release(release["repository"], int(release["release_id"]))
    _verify_remote_release(manifest, remote, immutable=True)
    cache_root = cache_root.resolve()
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aads-adapter-release-", dir=cache_root.parent) as temp_dir:
        payload_root = Path(temp_dir) / "payload"
        for record in manifest["files"]:
            destination = payload_root / str(record["local_path"])
            api.download_asset(release["repository"], int(record["asset_id"]), destination)
        verify_release_files(payload_root, manifest)
        if cache_root.exists():
            shutil.rmtree(cache_root)
        shutil.move(str(payload_root), str(cache_root))
    return {
        "repository": release["repository"],
        "release_tag": release["tag"],
        "release_id": release["release_id"],
        "tag_commit_sha": release["tag_commit_sha"],
        "cache_root": str(cache_root),
        "verified": True,
    }


def prepare_notebook_adapter_root(
    config: dict[str, Any],
    repo_root: Path,
    *,
    manifest_path: Path,
    fetch_fn: Callable[..., dict[str, Any]] = fetch_adapter_release,
) -> dict[str, Any]:
    configured = Path(str((config.get("inference") or {}).get("adapter_root") or "models/adapters"))
    absolute_root = configured if configured.is_absolute() else (repo_root / configured).resolve()
    deployment_root = (repo_root / "models" / "adapters").resolve()
    if absolute_root != deployment_root:
        return {"adapter_root": str(configured), "deployment_release": False, "verified": False}
    manifest = read_json_dict(manifest_path)
    validate_release_manifest(manifest, require_immutable=True)
    release = _github_release(manifest)
    try:
        verify_release_files(absolute_root, manifest)
        fetched = False
        verified = True
    except (FileNotFoundError, ValueError):
        receipt = fetch_fn(manifest_path, absolute_root)
        fetched = True
        verified = bool(receipt.get("verified"))
    return {
        "adapter_root": str(configured),
        "deployment_release": True,
        "verified": verified,
        "release_tag": release["tag"],
        "release_id": release["release_id"],
        "tag_commit_sha": release["tag_commit_sha"],
        "fetched": fetched,
    }


