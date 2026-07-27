"""Release-aware dataset lineage contracts for training and readiness artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.data.dataset_release import canonical_sha256, read_json_dict, sha256_file

GITHUB_RELEASE_SOURCE = "github_release"
LOCAL_LEGACY_SOURCE = "local_legacy"
PUBLIC_SAMPLE_SOURCE = "public_sample"
LINEAGE_SCHEMA = "aads.dataset_lineage.v1"


class DatasetLineageError(ValueError):
    """Raised when a dataset lineage identity is incomplete or inconsistent."""


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = str(payload.get(field) or "").strip()
    if not value:
        raise DatasetLineageError(f"Dataset lineage requires {field}")
    return value


def _required_positive_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DatasetLineageError(f"Dataset lineage requires positive integer {field}")
    return value


def _asset_inventory(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise DatasetLineageError("Dataset lineage requires a non-empty GitHub asset inventory")
    inventory: list[dict[str, Any]] = []
    for raw_asset in assets:
        if not isinstance(raw_asset, Mapping):
            raise DatasetLineageError("Dataset release asset entries must be objects")
        inventory.append(
            {
                "asset_id": _required_positive_int(raw_asset, "asset_id"),
                "asset_name": _required_text(raw_asset, "asset_name"),
                "sha256": _required_text(raw_asset, "sha256"),
                "size_bytes": _required_positive_int(raw_asset, "size_bytes"),
            }
        )
    return sorted(inventory, key=lambda item: (item["asset_name"], item["asset_id"]))


def _github_release_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not bool(manifest.get("immutable")):
        raise DatasetLineageError("Dataset lineage requires immutable=true")
    if str(manifest.get("release_state") or "") not in {"verified", "current"}:
        raise DatasetLineageError("Dataset lineage requires a verified or current release")
    inventory = _asset_inventory(manifest)
    identity = {
        "repository": _required_text(manifest, "repository"),
        "release_tag": _required_text(manifest, "release_tag"),
        "release_id": _required_positive_int(manifest, "release_id"),
        "tag_commit_sha": _required_text(manifest, "tag_commit_sha"),
        "release_manifest_sha256": _required_text(manifest, "release_manifest_sha256"),
        "snapshot_manifest_sha256": _required_text(manifest, "snapshot_manifest_sha256"),
        "asset_inventory": inventory,
        "asset_inventory_sha256": canonical_sha256(inventory),
    }
    identity["release_lineage_key"] = f"github_release::{canonical_sha256(identity)}"
    return identity


def build_github_release_lineage(
    *,
    release_manifest: Mapping[str, Any],
    dataset_key: str,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    """Build deterministic lineage for one target in an immutable dataset release."""

    release = _github_release_identity(release_manifest)
    target = str(dataset_key or "").strip().lower()
    split_sha = str(split_manifest_sha256 or "").strip().lower()
    if not target:
        raise DatasetLineageError("Dataset lineage requires dataset_key")
    if len(split_sha) != 64:
        raise DatasetLineageError("Dataset lineage requires a split_manifest_sha256")
    target_identity = {
        "release_lineage_key": release["release_lineage_key"],
        "dataset_key": target,
        "split_manifest_sha256": split_sha,
    }
    return {
        "schema_version": LINEAGE_SCHEMA,
        "source_kind": GITHUB_RELEASE_SOURCE,
        "production_eligible": True,
        **release,
        "dataset_key": target,
        "split_manifest_sha256": split_sha,
        "dataset_lineage_key": f"github_release_target::{canonical_sha256(target_identity)}",
    }


def verify_release_target(
    *, release_manifest: Mapping[str, Any], dataset_key: str, target_root: str | Path
) -> None:
    """Prove that a materialized target exactly matches the release member inventory."""

    target = str(dataset_key or "").strip().lower()
    prefix = f"{target}/"
    expected: dict[str, str] = {}
    for asset in release_manifest.get("assets", []):
        if not isinstance(asset, Mapping):
            continue
        for member in asset.get("members", []):
            if not isinstance(member, Mapping):
                continue
            member_path = str(member.get("path") or "").replace("\\", "/")
            if member_path.startswith(prefix):
                expected[member_path[len(prefix) :]] = _required_text(member, "sha256")
    if not expected:
        raise DatasetLineageError(f"Release asset inventory does not contain target {target!r}")
    root = Path(target_root)
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)[:5]
        unexpected = sorted(actual_paths - set(expected))[:5]
        raise DatasetLineageError(
            f"Materialized target differs from release inventory; missing={missing} unexpected={unexpected}"
        )
    for relative_path, expected_sha in expected.items():
        if sha256_file(root / relative_path) != expected_sha:
            raise DatasetLineageError(f"Materialized target hash mismatch: {relative_path}")


def build_local_legacy_lineage(
    *,
    dataset_key: str,
    split_manifest_sha256: str,
    compatibility_reason: str,
) -> dict[str, Any]:
    """Build explicitly non-production lineage for the bounded Notebook 17 bridge."""

    target = str(dataset_key or "").strip().lower()
    split_sha = str(split_manifest_sha256 or "").strip().lower()
    reason = str(compatibility_reason or "").strip()
    if not target or len(split_sha) != 64 or not reason:
        raise DatasetLineageError(
            "local_legacy lineage requires dataset_key, split_manifest_sha256, and compatibility_reason"
        )
    return {
        "schema_version": LINEAGE_SCHEMA,
        "source_kind": LOCAL_LEGACY_SOURCE,
        "production_eligible": False,
        "dataset_key": target,
        "split_manifest_sha256": split_sha,
        "compatibility_reason": reason,
        "dataset_lineage_key": f"local_legacy::{target}::{split_sha}",
    }


def build_public_sample_lineage(*, dataset_key: str, split_manifest_sha256: str) -> dict[str, Any]:
    """Build explicit non-production lineage for the deterministic public sample."""

    target = str(dataset_key or "").strip().lower()
    split_sha = str(split_manifest_sha256 or "").strip().lower()
    if not target or len(split_sha) != 64:
        raise DatasetLineageError("public_sample lineage requires dataset_key and split_manifest_sha256")
    return {
        "schema_version": LINEAGE_SCHEMA,
        "source_kind": PUBLIC_SAMPLE_SOURCE,
        "production_eligible": False,
        "dataset_key": target,
        "split_manifest_sha256": split_sha,
        "compatibility_reason": "deterministic_synthetic_smoke_dataset",
        "dataset_lineage_key": f"public_sample::{target}::{split_sha}",
    }


def resolve_dataset_lineage(
    *,
    source_kind: str,
    dataset_key: str,
    split_manifest_path: str | Path,
    release_manifest_path: str | Path | None = None,
    allow_local_legacy: bool = False,
    compatibility_reason: str = "",
    materialized_target_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve a release or explicitly approved legacy lineage from durable manifests."""

    split_path = Path(split_manifest_path)
    if not split_path.is_file():
        raise DatasetLineageError(f"Dataset split manifest is missing: {split_path}")
    split_sha = sha256_file(split_path)
    kind = str(source_kind or "").strip().lower()
    if kind == GITHUB_RELEASE_SOURCE:
        if release_manifest_path is None:
            raise DatasetLineageError("github_release lineage requires release_manifest_path")
        release_path = Path(release_manifest_path)
        manifest = read_json_dict(release_path)
        if materialized_target_root is None:
            raise DatasetLineageError("github_release lineage requires materialized_target_root verification")
        verify_release_target(
            release_manifest=manifest,
            dataset_key=dataset_key,
            target_root=materialized_target_root,
        )
        lineage = build_github_release_lineage(
            release_manifest=manifest,
            dataset_key=dataset_key,
            split_manifest_sha256=split_sha,
        )
        lineage["release_manifest_path"] = str(release_path)
        return lineage
    if kind == LOCAL_LEGACY_SOURCE and allow_local_legacy:
        return build_local_legacy_lineage(
            dataset_key=dataset_key,
            split_manifest_sha256=split_sha,
            compatibility_reason=compatibility_reason,
        )
    if kind == LOCAL_LEGACY_SOURCE:
        raise DatasetLineageError("local_legacy lineage is unavailable to new customer training")
    if kind == PUBLIC_SAMPLE_SOURCE:
        return build_public_sample_lineage(
            dataset_key=dataset_key,
            split_manifest_sha256=split_sha,
        )
    raise DatasetLineageError(f"Unsupported dataset lineage source_kind={source_kind!r}")


def immutable_lineage_blockers(lineage: Mapping[str, Any] | None) -> list[str]:
    """Return production blockers for absent or non-immutable dataset lineage."""

    payload = dict(lineage or {})
    if payload.get("source_kind") != GITHUB_RELEASE_SOURCE or not bool(payload.get("production_eligible")):
        return ["immutable_dataset_release_lineage"]
    required = (
        "repository",
        "release_tag",
        "release_id",
        "tag_commit_sha",
        "asset_inventory_sha256",
        "release_manifest_sha256",
        "dataset_lineage_key",
    )
    return [f"dataset_lineage.{field}" for field in required if payload.get(field) in (None, "")]


def lineages_directly_comparable(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Only exact immutable release-target lineage is comparable by default."""

    return bool(
        not immutable_lineage_blockers(left)
        and not immutable_lineage_blockers(right)
        and str(left.get("dataset_lineage_key") or "") == str(right.get("dataset_lineage_key") or "")
    )
