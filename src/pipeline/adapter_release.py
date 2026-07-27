"""GitHub Release helpers for immutable deployment adapters."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from src.shared.json_utils import read_json_dict, write_json

RELEASE_SCHEMA = "aads.adapter_release.v2"
APPROVAL_SCHEMA = "aads.adapter_release_approval.v2"
GITHUB_API_VERSION = "2026-03-10"
GITHUB_API_ROOT = "https://api.github.com"
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
READ_TOKEN_NAMES = ("AADS_GITHUB_RELEASE_READ_TOKEN", "AADS_GITHUB_RELEASE_CI_READ_TOKEN")
WRITE_TOKEN_NAMES = ("AADS_GITHUB_RELEASE_WRITE_TOKEN",)
REPOSITORY_ENV = "GITHUB_RELEASE_REPOSITORY"
COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
TAG_PATTERN = re.compile(
    r"^(?:aads-adapters|aads-public-demo)-v[0-9]+\.[0-9]+\.[0-9]+(?:[-.][A-Za-z0-9.-]+)?$"
)
RELEASE_ACCESS_MODES = frozenset({"private", "public"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_token(*, write: bool) -> str | None:
    for name in WRITE_TOKEN_NAMES if write else READ_TOKEN_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    return None


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


class GitHubReleaseClient:
    """Minimal secret-redacting GitHub REST client for release operations."""

    def __init__(self, token: str | None = None) -> None:
        import requests

        self._requests = requests
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        headers = dict(self._headers)
        headers.update(kwargs.pop("headers", {}))
        response = self._requests.request(method, url, headers=headers, timeout=120, **kwargs)
        if not response.ok:
            try:
                message = str(response.json().get("message") or "request failed")
            except (TypeError, ValueError):
                message = "request failed"
            raise RuntimeError(f"GitHub API {method} {url.split('?')[0]} failed ({response.status_code}): {message}")
        return response

    def get_repository(self, repository: str) -> dict[str, Any]:
        return dict(self._request("GET", f"{GITHUB_API_ROOT}/repos/{repository}").json())

    def immutable_releases_enabled(self, repository: str) -> bool:
        response = self._request("GET", f"{GITHUB_API_ROOT}/repos/{repository}/immutable-releases")
        return bool(response.json().get("enabled"))

    def create_draft(self, repository: str, tag: str, commit: str, *, name: str, body: str) -> dict[str, Any]:
        payload = {
            "tag_name": tag,
            "target_commitish": commit,
            "name": name,
            "body": body,
            "draft": True,
            "prerelease": False,
            "make_latest": "false",
        }
        return dict(self._request("POST", f"{GITHUB_API_ROOT}/repos/{repository}/releases", json=payload).json())

    def get_release(self, repository: str, release_id: int) -> dict[str, Any]:
        return dict(self._request("GET", f"{GITHUB_API_ROOT}/repos/{repository}/releases/{release_id}").json())

    def find_release_by_tag(self, repository: str, tag: str) -> dict[str, Any] | None:
        response = self._request("GET", f"{GITHUB_API_ROOT}/repos/{repository}/releases", params={"per_page": 100})
        for release in response.json():
            if str(release.get("tag_name") or "") == tag:
                return dict(release)
        return None

    def upload_asset(self, upload_url: str, source: Path, asset_name: str) -> dict[str, Any]:
        url = upload_url.split("{")[0]
        with source.open("rb") as handle:
            response = self._request(
                "POST",
                url,
                params={"name": asset_name},
                headers={"Content-Type": "application/octet-stream"},
                data=handle,
            )
        return dict(response.json())

    def download_asset(self, repository: str, asset_id: int, destination: Path) -> None:
        response = self._request(
            "GET",
            f"{GITHUB_API_ROOT}/repos/{repository}/releases/assets/{asset_id}",
            headers={"Accept": "application/octet-stream"},
            stream=True,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    def publish(self, repository: str, release_id: int) -> dict[str, Any]:
        return dict(
            self._request(
                "PATCH",
                f"{GITHUB_API_ROOT}/repos/{repository}/releases/{release_id}",
                json={"draft": False, "make_latest": "false"},
            ).json()
        )

    def tag_commit_sha(self, repository: str, tag: str) -> str:
        response = self._request("GET", f"{GITHUB_API_ROOT}/repos/{repository}/git/ref/tags/{quote(tag, safe='')}")
        commit = str((response.json().get("object") or {}).get("sha") or "")
        if not COMMIT_SHA_PATTERN.fullmatch(commit):
            raise RuntimeError("Published GitHub Release tag did not resolve to a commit SHA.")
        return commit


def _client(token: str | None, *, write: bool, client: Any, allow_anonymous: bool = False) -> Any:
    if client is not None:
        return client
    resolved = token or resolve_token(write=write)
    if not resolved and (write or not allow_anonymous):
        names = WRITE_TOKEN_NAMES if write else READ_TOKEN_NAMES
        raise RuntimeError(f"One of {', '.join(names)} is required.")
    return GitHubReleaseClient(resolved)


def preflight_adapter_release(
    manifest_path: Path,
    cache_root: Path,
    *,
    write: bool,
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
    api = _client(
        token,
        write=write,
        client=client,
        allow_anonymous=not write and access_mode == "public",
    )
    repository = api.get_repository(release["repository"])
    if str(repository.get("full_name") or "").lower() != str(release["repository"]).lower():
        raise RuntimeError("GitHub API repository identity does not match the manifest.")
    repository_is_private = repository.get("private") is True
    if repository_is_private != (access_mode == "private"):
        raise RuntimeError(
            "GitHub repository visibility does not match the adapter release access_mode."
        )
    if write and not api.immutable_releases_enabled(release["repository"]):
        raise RuntimeError("GitHub immutable releases must be enabled before draft creation.")
    permissions = dict(repository.get("permissions") or {})
    if write and not (permissions.get("push") or permissions.get("admin")):
        raise RuntimeError("Publisher token does not prove Contents: write access.")
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
        "immutable_releases_enabled": True if write else None,
        "immutable_release_setting_checked": write,
        "total_bytes": total_bytes,
        "available_bytes": free_bytes,
        "write_access": bool(permissions.get("push") or permissions.get("admin")),
    }


def create_adapter_draft(
    manifest_path: Path,
    *,
    token: str | None = None,
    client: Any = None,
) -> dict[str, Any]:
    manifest = read_json_dict(manifest_path)
    validate_release_manifest(manifest)
    release = _github_release(manifest)
    api = _client(token, write=True, client=client)
    preflight = preflight_adapter_release(manifest_path, Path("models/adapters"), write=True, client=api)
    created = api.create_draft(
        release["repository"],
        release["tag"],
        release["tag_commit_sha"],
        name=str(release.get("name") or release["tag"]),
        body=str(release.get("body") or "AADS v6 controlled-demo deployment adapters."),
    )
    if created.get("draft") is not True or created.get("immutable") is not False:
        raise RuntimeError("GitHub did not create a mutable draft release.")
    if str(created.get("tag_name") or "") != release["tag"]:
        raise RuntimeError("GitHub draft tag does not match the manifest.")
    assets: list[dict[str, Any]] = []
    for record in manifest["files"]:
        source = Path(str(record.get("source_path") or ""))
        if not source.is_file() or source.stat().st_size != record["size_bytes"] or sha256_file(source) != record["sha256"]:
            raise ValueError(f"Source file missing or changed: {source}.")
        uploaded = api.upload_asset(str(created["upload_url"]), source, str(record["asset_name"]))
        digest = str(uploaded.get("digest") or "")
        if (
            uploaded.get("state") != "uploaded"
            or uploaded.get("name") != record["asset_name"]
            or uploaded.get("size") != record["size_bytes"]
            or (digest and digest != f"sha256:{record['sha256']}")
        ):
            raise RuntimeError(f"Uploaded GitHub asset did not verify: {record['asset_name']}.")
        assets.append(
            {
                "asset_name": record["asset_name"],
                "asset_id": int(uploaded["id"]),
                "size_bytes": int(uploaded["size"]),
                "sha256": record["sha256"],
                "github_digest": digest or None,
            }
        )
    return {
        "repository": release["repository"],
        "release_id": int(created["id"]),
        "release_tag": release["tag"],
        "tag_commit_sha": release["tag_commit_sha"],
        "draft": True,
        "immutable": False,
        "access_mode": release_access_mode(manifest),
        "private": bool(preflight["private"]),
        "html_url": created.get("html_url"),
        "assets": assets,
    }


def record_draft_release(manifest_path: Path, receipt_path: Path) -> dict[str, Any]:
    manifest = read_json_dict(manifest_path)
    receipt = read_json_dict(receipt_path)
    release = _github_release(manifest)
    expected_private = release_access_mode(manifest) == "private"
    if (
        receipt.get("private") is not expected_private
        or receipt.get("draft") is not True
        or receipt.get("immutable") is not False
    ):
        raise ValueError("Draft receipt must prove the configured mutable GitHub Release draft.")
    identity = (receipt.get("repository"), receipt.get("release_tag"), receipt.get("tag_commit_sha"))
    if identity != (release["repository"], release["tag"], release["tag_commit_sha"]):
        raise ValueError("Draft receipt identity does not match the manifest.")
    receipt_assets = {str(asset["asset_name"]): asset for asset in receipt.get("assets") or []}
    if set(receipt_assets) != {str(record["asset_name"]) for record in manifest["files"]}:
        raise ValueError("Draft receipt asset allowlist does not match the manifest.")
    for record in manifest["files"]:
        asset = receipt_assets[str(record["asset_name"])]
        if asset.get("size_bytes") != record["size_bytes"] or asset.get("sha256") != record["sha256"]:
            raise ValueError(f"Draft receipt checksum/size mismatch: {record['asset_name']}.")
        record["asset_id"] = int(asset["asset_id"])
        record["github_digest"] = asset.get("github_digest")
    release.update(
        {
            "release_id": int(receipt["release_id"]),
            "draft": True,
            "immutable": False,
            "html_url": receipt.get("html_url"),
        }
    )
    validate_release_manifest(manifest, require_draft=True)
    write_json(manifest_path, manifest)
    return {
        "repository": release["repository"],
        "release_tag": release["tag"],
        "release_id": release["release_id"],
        "tag_commit_sha": release["tag_commit_sha"],
        "manifest_sha256": sha256_file(manifest_path),
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


def verify_adapter_draft(
    manifest_path: Path,
    *,
    token: str | None = None,
    client: Any = None,
) -> dict[str, Any]:
    manifest = read_json_dict(manifest_path)
    validate_release_manifest(manifest, require_draft=True)
    release = _github_release(manifest)
    # Draft releases are never public GitHub assets, even when the final
    # repository visibility is public.
    api = _client(token, write=False, client=client)
    preflight_adapter_release(manifest_path, Path("models/adapters"), write=False, client=api)
    remote = api.get_release(release["repository"], int(release["release_id"]))
    _verify_remote_release(manifest, remote, immutable=False)
    return {
        "repository": release["repository"],
        "release_tag": release["tag"],
        "release_id": release["release_id"],
        "tag_commit_sha": release["tag_commit_sha"],
        "draft": True,
        "immutable": False,
        "asset_count": len(manifest["files"]),
        "verified": True,
    }


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
    api = _client(
        token,
        write=False,
        client=client,
        allow_anonymous=release_access_mode(manifest) == "public",
    )
    preflight_adapter_release(manifest_path, cache_root, write=False, client=api)
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


def validate_promotion_approval(approval_path: Path, manifest_path: Path) -> dict[str, Any]:
    approval = read_json_dict(approval_path)
    manifest = read_json_dict(manifest_path)
    validate_release_manifest(manifest, require_draft=not bool((manifest.get("github_release") or {}).get("immutable")))
    release = _github_release(manifest)
    if approval.get("schema_version") != APPROVAL_SCHEMA:
        raise ValueError("Missing or invalid release promotion approval schema.")
    required = (
        "approver_identity",
        "approved_at",
        "release_manifest_sha256",
        "repository",
        "release_tag",
        "release_id",
        "tag_commit_sha",
        "license_reviewer_identity",
    )
    if any(not str(approval.get(field) or "").strip() for field in required):
        raise ValueError("Promotion approval is missing required human approval fields.")
    approved_hash = str(release.get("approved_manifest_sha256") or sha256_file(manifest_path))
    if approval["release_manifest_sha256"] != approved_hash:
        raise ValueError("Promotion approval does not match the release manifest SHA-256.")
    identity = (approval["repository"], approval["release_tag"], int(approval["release_id"]), approval["tag_commit_sha"])
    expected = (release["repository"], release["tag"], int(release["release_id"]), release["tag_commit_sha"])
    if identity != expected:
        raise ValueError("Promotion approval does not match the GitHub Release identity.")
    if approval.get("redistribution_approved") is not True:
        raise ValueError("Redistribution-license approval is required.")
    expected_asset_ids = sorted(int(record["asset_id"]) for record in manifest["files"])
    if sorted(int(value) for value in approval.get("asset_ids") or []) != expected_asset_ids:
        raise ValueError("Promotion approval does not match the GitHub Release asset IDs.")
    return approval


def publish_adapter_release(
    manifest_path: Path,
    approval_path: Path,
    *,
    token: str | None = None,
    client: Any = None,
) -> dict[str, Any]:
    manifest = read_json_dict(manifest_path)
    approval = validate_promotion_approval(approval_path, manifest_path)
    release = _github_release(manifest)
    api = _client(token, write=True, client=client)
    preflight_adapter_release(manifest_path, Path("models/adapters"), write=True, client=api)
    remote = api.get_release(release["repository"], int(release["release_id"]))
    _verify_remote_release(manifest, remote, immutable=False)
    api.publish(release["repository"], int(release["release_id"]))
    published = api.get_release(release["repository"], int(release["release_id"]))
    _verify_remote_release(manifest, published, immutable=True)
    if api.tag_commit_sha(release["repository"], release["tag"]) != release["tag_commit_sha"]:
        raise RuntimeError("Published GitHub Release tag commit does not match the approved commit SHA.")
    release.update(
        {
            "draft": False,
            "immutable": True,
            "approved_manifest_sha256": approval["release_manifest_sha256"],
            "published_at": published.get("published_at"),
        }
    )
    manifest["license_review"] = {
        **dict(manifest.get("license_review") or {}),
        "status": "approved",
        "reviewed_by": approval["license_reviewer_identity"],
        "reviewed_at": approval["approved_at"],
        "redistribution_approved": True,
    }
    validate_release_manifest(manifest, require_immutable=True)
    write_json(manifest_path, manifest)
    return {
        "repository": release["repository"],
        "release_tag": release["tag"],
        "release_id": release["release_id"],
        "tag_commit_sha": release["tag_commit_sha"],
        "immutable": True,
        "published": True,
    }


def promote_adapter_pointer(
    manifest_path: Path,
    approval_path: Path,
    cache_root: Path,
    config_paths: list[Path],
) -> dict[str, Any]:
    approval = validate_promotion_approval(approval_path, manifest_path)
    manifest = read_json_dict(manifest_path)
    validate_release_manifest(manifest, require_immutable=True)
    release = _github_release(manifest)
    verify_release_files(cache_root, manifest)
    for config_path in config_paths:
        config = read_json_dict(config_path)
        inference = config.setdefault("inference", {})
        if not isinstance(inference, dict):
            raise ValueError(f"Config inference field must be an object: {config_path}")
        inference["adapter_root"] = "models/adapters"
        write_json(config_path, config)
    return {
        "promoted": True,
        "approver_identity": str(approval["approver_identity"]),
        "release_tag": release["tag"],
        "release_id": release["release_id"],
        "adapter_root": "models/adapters",
    }


def write_release_receipt(path: Path, receipt: dict[str, Any]) -> None:
    safe = {key: value for key, value in receipt.items() if "token" not in key.lower() and "authorization" not in key.lower()}
    write_json(path, safe)
