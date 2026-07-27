"""GitHub Release lifecycle for immutable AADS dataset shards.

The module consumes the Phase 4 snapshot/release contracts and reuses the
Phase 3 GitHub REST client.  It deliberately keeps publication, download,
materialization, and current-pointer promotion as separate fail-closed steps.
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
    validate_release_candidate,
    write_json,
)
from src.pipeline.adapter_release import (
    COMMIT_SHA_PATTERN,
    READ_TOKEN_NAMES,
    REPOSITORY_ENV,
    REPOSITORY_PATTERN,
    WRITE_TOKEN_NAMES,
    GitHubReleaseClient,
    resolve_token,
)

DATASET_GITHUB_RELEASE_SCHEMA = "aads.dataset_release.v1"
DATASET_APPROVAL_SCHEMA = "aads.dataset_release_approval.v1"
DATASET_POINTER_SCHEMA = "aads.dataset_release_pointer.v1"
DATASET_TAG_PATTERN = re.compile(r"^aads-dataset-v[0-9]+\.[0-9]+\.[0-9]+(?:[-.][A-Za-z0-9.-]+)?$")
DATASET_RELEASE_STATES = frozenset({"staged", "verified", "current", "revoked"})
MAX_DATASET_BYTES_ENV = "AADS_GITHUB_RELEASE_MAX_DATASET_BYTES"
TRANSFER_RETRY_ENV = "AADS_GITHUB_RELEASE_TRANSFER_RETRIES"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


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


def _refresh_manifest_hash(payload: dict[str, Any]) -> None:
    payload["release_manifest_sha256"] = canonical_sha256(_release_identity(payload))


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


def _validate_shard_plan(candidate: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    validate_release_candidate(candidate)
    plan_identity = dict(plan)
    recorded_plan_hash = _validate_hash(plan_identity.pop("shard_plan_sha256", None), field="shard_plan_sha256")
    if recorded_plan_hash != canonical_sha256(plan_identity):
        raise DatasetContractError("Shard plan SHA-256 does not match its canonical content")
    if plan.get("dataset_tag") != candidate.get("dataset_tag"):
        raise DatasetContractError("Shard plan tag does not match release candidate")
    for field in ("staging_snapshot_id", "snapshot_manifest_sha256"):
        if plan.get(field) != candidate.get(field):
            raise DatasetContractError(f"Shard plan {field} does not match release candidate")
    planned_members = [member for shard in plan.get("shards") or [] for member in shard.get("members") or []]
    candidate_files = {str(row["distributed_path"]): row for row in candidate.get("files") or []}
    if {str(row.get("path")) for row in planned_members} != set(candidate_files):
        raise DatasetContractError("Shard plan member allowlist does not match release candidate")
    if len(planned_members) != len(candidate_files):
        raise DatasetContractError("Shard plan contains duplicate members")
    if int(plan.get("max_shard_bytes") or 0) > DatasetResourceLimits().max_shard_bytes:
        raise DatasetContractError("Shard plan exceeds the maintained 1.9 GiB limit")
    shard_names = [str(shard.get("asset_name") or "") for shard in plan.get("shards") or []]
    if len(shard_names) != len(set(shard_names)):
        raise DatasetContractError("Shard plan contains duplicate asset names")
    for member in planned_members:
        record = candidate_files[str(member["path"])]
        if (
            member.get("sha256") != record.get("distributed_content_sha256")
            or member.get("size_bytes") != record.get("size_bytes")
            or member.get("sample_id") != record.get("sample_id")
        ):
            raise DatasetContractError(f"Shard plan member identity mismatch: {member.get('path')}")


def package_dataset_shards(
    candidate_path: Path,
    shard_plan_path: Path,
    staging_root: Path,
    package_root: Path,
    *,
    repository: str,
    tag_commit_sha: str,
    previous_release_tag: str | None = None,
    publisher: str | None = None,
) -> dict[str, Any]:
    """Create deterministic stored ZIP shards and their publish manifest."""
    candidate = read_json_dict(candidate_path)
    plan = read_json_dict(shard_plan_path)
    _validate_shard_plan(candidate, plan)
    if not candidate.get("files"):
        raise DatasetContractError("The zero-file Phase 4 candidate cannot be packaged or published")
    if not REPOSITORY_PATTERN.fullmatch(repository):
        raise DatasetContractError("repository must use explicit owner/repository form")
    if not COMMIT_SHA_PATTERN.fullmatch(tag_commit_sha):
        raise DatasetContractError("tag_commit_sha must be an exact commit SHA")
    package_root = package_root.resolve()
    staging_root = staging_root.resolve()
    package_root.mkdir(parents=True, exist_ok=True)
    allowed_files = {str(row["distributed_path"]): row for row in candidate["files"]}
    assets: list[dict[str, Any]] = []
    for shard in plan["shards"]:
        asset_name = _safe_relative_path(shard["asset_name"], field="asset_name")
        archive_path = package_root / asset_name
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for member in shard["members"]:
                relative_path = _safe_relative_path(member["path"], field="member.path")
                record = allowed_files[relative_path]
                logical_source = (staging_root / Path(*PurePosixPath(relative_path).parts)).resolve()
                if staging_root not in logical_source.parents:
                    raise DatasetContractError(f"Dataset member escapes staging root: {relative_path}")
                source = _io_path(logical_source)
                if not source.is_file():
                    raise DatasetContractError(f"Dataset member is missing after audit: {relative_path}")
                if source.stat().st_size != int(record["size_bytes"]) or sha256_file(source) != record[
                    "distributed_content_sha256"
                ]:
                    raise DatasetContractError(f"Dataset member changed after audit: {relative_path}")
                info = zipfile.ZipInfo(relative_path, date_time=FIXED_ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                with source.open("rb") as source_handle, archive.open(info, "w", force_zip64=True) as target_handle:
                    shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
        archive_size = archive_path.stat().st_size
        if archive_size >= DatasetResourceLimits().max_shard_bytes:
            raise DatasetContractError(f"Packaged shard exceeds the 1.9 GiB limit: {asset_name}")
        assets.append(
            {
                "asset_name": asset_name,
                "local_path": str(archive_path),
                "size_bytes": archive_size,
                "sha256": sha256_file(archive_path),
                "asset_id": None,
                "github_digest": None,
                "members": [
                    {
                        "path": member["path"],
                        "sha256": member["sha256"],
                        "size_bytes": member["size_bytes"],
                        "sample_id": member["sample_id"],
                    }
                    for member in shard["members"]
                ],
            }
        )
    payload: dict[str, Any] = {
        "schema_version": DATASET_GITHUB_RELEASE_SCHEMA,
        "dataset_version": candidate["dataset_version"],
        "staging_snapshot_id": candidate["staging_snapshot_id"],
        "snapshot_manifest_sha256": candidate["snapshot_manifest_sha256"],
        "repository": repository,
        "release_tag": candidate["dataset_tag"],
        "release_id": None,
        "tag_commit_sha": tag_commit_sha,
        "previous_release_tag": previous_release_tag,
        "publisher": publisher,
        "audit_outcome": "approved_manifest_only",
        "release_state": "staged",
        "revocation_reason": "",
        "draft": True,
        "immutable": False,
        "assets": assets,
        "file_count": len(candidate["files"]),
        "shard_count": len(assets),
        "distributed_bytes": sum(int(row["size_bytes"]) for row in candidate["files"]),
    }
    _refresh_manifest_hash(payload)
    validate_dataset_github_manifest(payload)
    return payload


def dry_run_dataset_publish(
    candidate_path: Path, shard_plan_path: Path, *, repository: str, tag_commit_sha: str
) -> dict[str, Any]:
    """Validate publication inputs without network or filesystem mutation."""
    candidate = read_json_dict(candidate_path)
    plan = read_json_dict(shard_plan_path)
    _validate_shard_plan(candidate, plan)
    if not REPOSITORY_PATTERN.fullmatch(repository):
        raise DatasetContractError("repository must use explicit owner/repository form")
    if not COMMIT_SHA_PATTERN.fullmatch(tag_commit_sha):
        raise DatasetContractError("tag_commit_sha must be an exact commit SHA")
    file_count = len(candidate.get("files") or [])
    return {
        "status": "ready" if file_count else "blocked",
        "dry_run": True,
        "external_mutation": False,
        "repository": repository,
        "release_tag": candidate["dataset_tag"],
        "tag_commit_sha": tag_commit_sha,
        "staging_snapshot_id": candidate["staging_snapshot_id"],
        "snapshot_manifest_sha256": candidate["snapshot_manifest_sha256"],
        "file_count": file_count,
        "shard_count": len(plan.get("shards") or []),
        "blocker": None if file_count else "zero_file_candidate_requires_reviewed_per_image_metadata",
    }


def public_github_contract_summary() -> dict[str, Any]:
    return {
        "github_release_schema": DATASET_GITHUB_RELEASE_SCHEMA,
        "approval_schema": DATASET_APPROVAL_SCHEMA,
        "current_pointer_schema": DATASET_POINTER_SCHEMA,
        "operations": ["audit", "diff", "publish", "fetch", "materialize", "verify"],
        "release_states": sorted(DATASET_RELEASE_STATES),
        "repository_env": REPOSITORY_ENV,
        "read_token_envs": list(READ_TOKEN_NAMES),
        "write_token_envs": list(WRITE_TOKEN_NAMES),
        "quota_env": MAX_DATASET_BYTES_ENV,
        "transfer_retry_env": TRANSFER_RETRY_ENV,
        "pull_request_network_access": False,
        "protected_smoke_workflow": ".github/workflows/dataset_release_smoke.yml",
    }


def _client(token: str | None, *, write: bool, client: Any) -> Any:
    if client is not None:
        return client
    resolved = token or resolve_token(write=write)
    if not resolved:
        names = WRITE_TOKEN_NAMES if write else READ_TOKEN_NAMES
        raise RuntimeError(f"One of {', '.join(names)} is required")
    return GitHubReleaseClient(resolved)


def _ensure_credential_separation() -> None:
    write_value = os.environ.get(WRITE_TOKEN_NAMES[0])
    for read_name in READ_TOKEN_NAMES:
        if write_value and os.environ.get(read_name) == write_value:
            raise RuntimeError(f"{WRITE_TOKEN_NAMES[0]} must be distinct from {read_name}")


def preflight_dataset_release(
    manifest_path: Path,
    workspace_root: Path,
    *,
    write: bool,
    repository: str | None = None,
    token: str | None = None,
    client: Any = None,
    max_upload_bytes: int | None = None,
) -> dict[str, Any]:
    manifest = read_json_dict(manifest_path)
    validate_dataset_github_manifest(manifest)
    configured = str(repository or os.environ.get(REPOSITORY_ENV) or "")
    if configured != manifest["repository"]:
        raise RuntimeError(f"{REPOSITORY_ENV} must exactly match the dataset release manifest repository")
    _ensure_credential_separation()
    api = _client(token, write=write, client=client)
    repository_info = api.get_repository(manifest["repository"])
    if str(repository_info.get("full_name") or "").lower() != str(manifest["repository"]).lower():
        raise RuntimeError("GitHub repository identity does not match the dataset manifest")
    if repository_info.get("private") is not True:
        raise RuntimeError("Dataset assets may only be stored in the configured private repository")
    permissions = dict(repository_info.get("permissions") or {})
    if write and not (permissions.get("push") or permissions.get("admin")):
        raise RuntimeError("Publisher credential does not prove Contents: write access")
    if write and not api.immutable_releases_enabled(manifest["repository"]):
        raise RuntimeError("GitHub immutable releases must be enabled before dataset draft creation")
    upload_bytes = sum(int(asset["size_bytes"]) for asset in manifest["assets"])
    ceiling = max_upload_bytes
    if ceiling is None and os.environ.get(MAX_DATASET_BYTES_ENV):
        ceiling = int(str(os.environ[MAX_DATASET_BYTES_ENV]))
    if write and ceiling is None:
        raise RuntimeError(f"{MAX_DATASET_BYTES_ENV} must define the approved release storage quota")
    if ceiling is not None and upload_bytes > ceiling:
        raise RuntimeError(f"Dataset release exceeds approved storage quota: required={upload_bytes}, quota={ceiling}")
    retry_limit = int(str(os.environ.get(TRANSFER_RETRY_ENV) or "3"))
    if retry_limit < 1 or retry_limit > 10:
        raise RuntimeError(f"{TRANSFER_RETRY_ENV} must be between 1 and 10")
    workspace_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(workspace_root).free
    materialized_bytes = sum(
        int(member["size_bytes"]) for asset in manifest["assets"] for member in asset.get("members") or []
    )
    # Preflight operates on already packaged shards and every transfer is streamed.
    # Requiring space for a second full upload plus a full materialization rejects
    # valid large releases even though neither operation allocates those together.
    # Keep one largest-shard margin for retry/partial-transfer safety.
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
        "write": write,
        "write_access": bool(permissions.get("push") or permissions.get("admin")),
        "read_access": True,
        "upload_bytes": upload_bytes,
        "materialized_bytes": materialized_bytes,
        "peak_workspace_bytes": peak_bytes,
        "available_bytes": free_bytes,
        "quota_bytes": ceiling,
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


def upload_dataset_draft(
    manifest_path: Path, *, token: str | None = None, client: Any = None, max_upload_bytes: int | None = None
) -> dict[str, Any]:
    """Create or resume a draft and upload only missing verified shards."""
    manifest = read_json_dict(manifest_path)
    validate_dataset_github_manifest(manifest)
    api = _client(token, write=True, client=client)
    preflight_dataset_release(
        manifest_path, manifest_path.resolve().parent, write=True, client=api, max_upload_bytes=max_upload_bytes
    )
    if manifest.get("release_id"):
        remote = api.get_release(manifest["repository"], int(manifest["release_id"]))
        if remote.get("draft") is not True or remote.get("immutable") is not False:
            raise DatasetContractError("Recorded dataset release is not a mutable draft")
    else:
        remote = api.create_draft(
            manifest["repository"],
            manifest["release_tag"],
            manifest["tag_commit_sha"],
            name=manifest["release_tag"],
            body="AADS v6 versioned dataset shards.",
        )
        manifest["release_id"] = int(remote["id"])
    remote_asset_rows = list(remote.get("assets") or [])
    remote_assets = {str(asset.get("name") or ""): asset for asset in remote_asset_rows}
    if len(remote_assets) != len(remote_asset_rows):
        raise DatasetContractError("Dataset draft contains duplicate asset names")
    uploaded_count = 0
    reused_count = 0
    for record in manifest["assets"]:
        name = str(record["asset_name"])
        existing = remote_assets.get(name)
        if existing is not None:
            digest = str(existing.get("digest") or "")
            if existing.get("state") != "uploaded" or int(existing.get("id") or 0) <= 0:
                raise DatasetContractError(f"Existing dataset shard is incomplete: {name}")
            if int(existing.get("size") or -1) != int(record["size_bytes"]) or (
                digest and digest != f"sha256:{record['sha256']}"
            ):
                raise DatasetContractError(f"Existing dataset shard conflicts with approved manifest: {name}")
            uploaded = existing
            reused_count += 1
        else:
            source = Path(str(record["local_path"]))
            if not source.is_file() or source.stat().st_size != record["size_bytes"] or sha256_file(source) != record[
                "sha256"
            ]:
                raise DatasetContractError(f"Dataset shard is missing or changed: {name}")
            uploaded = api.upload_asset(str(remote["upload_url"]), source, name)
            uploaded_count += 1
        uploaded_digest = str(uploaded.get("digest") or "")
        if (
            uploaded.get("state") != "uploaded"
            or str(uploaded.get("name") or "") != name
            or int(uploaded.get("id") or 0) <= 0
            or int(uploaded.get("size") or -1) != int(record["size_bytes"])
            or (uploaded_digest and uploaded_digest != f"sha256:{record['sha256']}")
        ):
            raise DatasetContractError(f"Uploaded dataset shard did not verify: {name}")
        record["asset_id"] = int(uploaded["id"])
        record["github_digest"] = uploaded_digest or None
    manifest["draft"] = True
    manifest["immutable"] = False
    manifest["release_state"] = "staged"
    _refresh_manifest_hash(manifest)
    validate_dataset_github_manifest(manifest, require_draft=True)
    write_json(manifest_path, manifest)
    return {
        "repository": manifest["repository"],
        "release_tag": manifest["release_tag"],
        "release_id": manifest["release_id"],
        "tag_commit_sha": manifest["tag_commit_sha"],
        "release_manifest_sha256": manifest["release_manifest_sha256"],
        "uploaded_count": uploaded_count,
        "reused_count": reused_count,
        "draft": True,
    }


def verify_dataset_release(
    manifest_path: Path, *, token: str | None = None, client: Any = None, record: bool = False
) -> dict[str, Any]:
    manifest = read_json_dict(manifest_path)
    validate_dataset_github_manifest(manifest, require_draft=manifest.get("immutable") is not True)
    api = _client(token, write=False, client=client)
    preflight_dataset_release(manifest_path, manifest_path.resolve().parent, write=False, client=api)
    remote = api.get_release(manifest["repository"], int(manifest["release_id"]))
    immutable = manifest.get("immutable") is True
    _verify_remote(manifest, remote, immutable=immutable)
    if record and not immutable:
        manifest["release_state"] = "verified"
        _refresh_manifest_hash(manifest)
        write_json(manifest_path, manifest)
    return {
        "repository": manifest["repository"],
        "release_tag": manifest["release_tag"],
        "release_id": manifest["release_id"],
        "tag_commit_sha": manifest["tag_commit_sha"],
        "release_manifest_sha256": manifest["release_manifest_sha256"],
        "asset_count": len(manifest["assets"]),
        "immutable": immutable,
        "verified": True,
    }


def validate_dataset_approval(approval_path: Path, manifest_path: Path) -> dict[str, Any]:
    approval = read_json_dict(approval_path)
    manifest = read_json_dict(manifest_path)
    validate_dataset_github_manifest(manifest, require_draft=True)
    if manifest.get("release_state") != "verified":
        raise DatasetContractError("Dataset draft must be independently verified before approval")
    if approval.get("schema_version") != DATASET_APPROVAL_SCHEMA:
        raise DatasetContractError("Dataset publication approval schema is invalid")
    required = ("approver_identity", "approved_at", "release_manifest_sha256", "repository", "release_tag")
    if any(not str(approval.get(field) or "").strip() for field in required):
        raise DatasetContractError("Dataset publication approval is incomplete")
    expected_identity = (
        manifest["repository"],
        manifest["release_tag"],
        int(manifest["release_id"]),
        manifest["tag_commit_sha"],
        manifest["release_manifest_sha256"],
    )
    actual_identity = (
        approval["repository"],
        approval["release_tag"],
        int(approval.get("release_id") or 0),
        approval.get("tag_commit_sha"),
        approval["release_manifest_sha256"],
    )
    if actual_identity != expected_identity:
        raise DatasetContractError("Dataset publication approval does not match the exact release identity")
    expected_ids = sorted(int(asset["asset_id"]) for asset in manifest["assets"])
    if sorted(int(value) for value in approval.get("asset_ids") or []) != expected_ids:
        raise DatasetContractError("Dataset publication approval does not match all asset IDs")
    if approval.get("redistribution_approved") is not True or approval.get("privacy_review_approved") is not True:
        raise DatasetContractError("Dataset redistribution and privacy approvals are required")
    return approval


def publish_dataset_release(
    manifest_path: Path,
    approval_path: Path,
    *,
    token: str | None = None,
    client: Any = None,
    max_upload_bytes: int | None = None,
) -> dict[str, Any]:
    manifest = read_json_dict(manifest_path)
    approval = validate_dataset_approval(approval_path, manifest_path)
    api = _client(token, write=True, client=client)
    preflight_dataset_release(
        manifest_path, manifest_path.resolve().parent, write=True, client=api, max_upload_bytes=max_upload_bytes
    )
    remote = api.get_release(manifest["repository"], int(manifest["release_id"]))
    _verify_remote(manifest, remote, immutable=False)
    api.publish(manifest["repository"], int(manifest["release_id"]))
    published = api.get_release(manifest["repository"], int(manifest["release_id"]))
    _verify_remote(manifest, published, immutable=True)
    if api.tag_commit_sha(manifest["repository"], manifest["release_tag"]) != manifest["tag_commit_sha"]:
        raise DatasetContractError("Published dataset tag commit differs from the approved commit")
    manifest["draft"] = False
    manifest["immutable"] = True
    manifest["release_state"] = "verified"
    manifest["approved_manifest_sha256"] = approval["release_manifest_sha256"]
    manifest["published_at"] = published.get("published_at")
    _refresh_manifest_hash(manifest)
    validate_dataset_github_manifest(manifest, require_immutable=True)
    write_json(manifest_path, manifest)
    return {
        "repository": manifest["repository"],
        "release_tag": manifest["release_tag"],
        "release_id": manifest["release_id"],
        "tag_commit_sha": manifest["tag_commit_sha"],
        "immutable": True,
        "published": True,
    }


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
    api = _client(token, write=False, client=client)
    preflight_dataset_release(
        manifest_path,
        cache_root.parent,
        write=False,
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


def promote_dataset_pointer(
    manifest_path: Path, cache_root: Path, materialized_root: Path, pointer_path: Path
) -> dict[str, Any]:
    manifest = read_json_dict(manifest_path)
    validate_dataset_github_manifest(manifest, require_immutable=True)
    if manifest.get("release_state") not in {"verified", "current"}:
        raise DatasetContractError("Only a verified immutable dataset release may become current")
    verify_cached_shards(cache_root, manifest)
    _verify_materialized(materialized_root, manifest)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = pointer_path.with_suffix(pointer_path.suffix + ".lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise DatasetContractError("Dataset current-pointer promotion is already in progress") from error
    os.close(descriptor)
    try:
        if pointer_path.exists():
            current = read_json_dict(pointer_path)
            if current.get("release_tag") == manifest["release_tag"]:
                return {"promoted": False, "idempotent": True, "release_tag": manifest["release_tag"]}
            if current.get("release_tag") != manifest.get("previous_release_tag"):
                raise DatasetContractError("Current dataset pointer does not match previous_release_tag")
        manifest["release_state"] = "current"
        _refresh_manifest_hash(manifest)
        pointer = {
            "schema_version": DATASET_POINTER_SCHEMA,
            "release_state": "current",
            "repository": manifest["repository"],
            "release_tag": manifest["release_tag"],
            "release_id": manifest["release_id"],
            "tag_commit_sha": manifest["tag_commit_sha"],
            "release_manifest_sha256": manifest["release_manifest_sha256"],
            "snapshot_manifest_sha256": manifest["snapshot_manifest_sha256"],
            "cache_root": str(cache_root.resolve()),
            "materialized_root": str(materialized_root.resolve()),
        }
        temporary = pointer_path.with_suffix(pointer_path.suffix + ".tmp")
        write_json(temporary, pointer)
        os.replace(temporary, pointer_path)
        write_json(manifest_path, manifest)
        return {"promoted": True, "idempotent": False, "release_tag": manifest["release_tag"]}
    finally:
        lock_path.unlink(missing_ok=True)
