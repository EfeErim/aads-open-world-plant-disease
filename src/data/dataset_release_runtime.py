"""Resolve, fetch, and materialize immutable dataset releases for runtime consumers."""

from __future__ import annotations

import re
from pathlib import Path

from src.data.dataset_release import DatasetContractError, read_json_dict
from src.data.dataset_release_github import fetch_dataset_release, materialize_dataset_release
from src.pipeline.adapter_release import resolve_token

DEFAULT_DATASET_RELEASE_REPOSITORY = "EfeErim/aads-open-world-plant-disease"
DEFAULT_DATASET_RELEASE_TAG = "aads-dataset-v1.0.0"
DEFAULT_DATASET_RELEASE_CACHE_ROOT = Path(".runtime_tmp/dataset_release_cache")
DEFAULT_MATERIALIZED_DATASET_ROOT = (
    DEFAULT_DATASET_RELEASE_CACHE_ROOT / DEFAULT_DATASET_RELEASE_TAG / "materialized"
)


class DatasetReleaseAccessBlocker(RuntimeError):
    """A private-release credential or network blocker, not a dataset-quality verdict."""


def resolve_dataset_release_manifest(repo_root: Path, *, repository: str, release_tag: str) -> Path:
    manifest_root = repo_root / "docs" / "evidence" / "current" / "dataset_release"
    matches: list[Path] = []
    for candidate in sorted(manifest_root.glob("*github_release_manifest.json")):
        payload = read_json_dict(candidate)
        if payload.get("repository") == repository and payload.get("release_tag") == release_tag:
            matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(
            "Exactly one tracked immutable dataset-release manifest must match "
            f"repository={repository!r} release_tag={release_tag!r}; found={len(matches)}"
        )
    return matches[0]


def fetch_materialize_dataset_release(
    *,
    root: Path,
    repository: str,
    release_tag: str,
    targets: list[str] | tuple[str, ...] | None = None,
    target: str = "",
    cache_root: str | Path,
    token: str | None = None,
    client: object | None = None,
) -> dict:
    """Fetch and verify one immutable release, then expose requested runtime targets."""
    repository = str(repository or "").strip()
    release_tag = str(release_tag or "").strip()
    requested_targets = list(targets or ())
    if target:
        requested_targets.append(target)
    requested_targets = list(dict.fromkeys(str(item or "").strip() for item in requested_targets if str(item).strip()))
    if not repository or not release_tag or not requested_targets:
        raise ValueError("dataset release repository, tag, and at least one target are required")
    invalid = [item for item in requested_targets if not re.fullmatch(r"[a-z0-9]+__[a-z0-9_]+", item)]
    if invalid:
        raise ValueError(f"Invalid dataset release target(s): {invalid!r}")

    resolved_token = str(token or resolve_token(write=False) or "").strip()
    if client is None and not resolved_token:
        raise DatasetReleaseAccessBlocker(
            "DATASET_RELEASE_ACCESS_BLOCKER: AADS_GITHUB_RELEASE_READ_TOKEN is required for the private release"
        )

    manifest_path = resolve_dataset_release_manifest(root, repository=repository, release_tag=release_tag)
    manifest = read_json_dict(manifest_path)
    cache_base = Path(cache_root).expanduser()
    if not cache_base.is_absolute():
        cache_base = (root / cache_base).resolve()
    release_cache = cache_base / release_tag
    shard_cache = release_cache / "shards"
    materialized_root = release_cache / "materialized"

    try:
        fetch_report = fetch_dataset_release(
            manifest_path,
            shard_cache,
            repository=repository,
            token=resolved_token or None,
            client=client,
        )
    except DatasetContractError:
        raise
    except (OSError, TimeoutError) as exc:
        raise DatasetReleaseAccessBlocker(f"DATASET_RELEASE_ACCESS_BLOCKER: {exc}") from exc
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("Insufficient dataset workspace capacity"):
            raise
        raise DatasetReleaseAccessBlocker(f"DATASET_RELEASE_ACCESS_BLOCKER: {message}") from exc

    materialize_report = materialize_dataset_release(manifest_path, shard_cache, materialized_root)
    selected_roots: dict[str, str] = {}
    for requested_target in requested_targets:
        target_root = materialized_root / requested_target
        if not target_root.is_dir():
            raise DatasetContractError(f"Dataset release does not contain requested target: {requested_target}")
        missing_splits = [name for name in ("continual", "val", "test") if not (target_root / name).is_dir()]
        if missing_splits:
            raise DatasetContractError(
                f"Materialized release target {requested_target} is missing runtime split folder(s): {missing_splits}"
            )
        selected_roots[requested_target] = str(target_root.resolve())

    result = {
        "repository": repository,
        "release_tag": release_tag,
        "release_id": manifest["release_id"],
        "release_manifest_sha256": manifest["release_manifest_sha256"],
        "manifest_path": str(manifest_path.resolve()),
        "cache_root": str(release_cache.resolve()),
        "runtime_dataset_root": str(materialized_root.resolve()),
        "selected_dataset_roots": selected_roots,
        "targets": requested_targets,
        "fetch": fetch_report,
        "materialization": materialize_report,
        "verified": True,
        "read_only": True,
    }
    if len(requested_targets) == 1:
        result["target"] = requested_targets[0]
        result["selected_dataset_root"] = selected_roots[requested_targets[0]]
    return result


__all__ = [
    "DEFAULT_DATASET_RELEASE_CACHE_ROOT",
    "DEFAULT_DATASET_RELEASE_REPOSITORY",
    "DEFAULT_DATASET_RELEASE_TAG",
    "DEFAULT_MATERIALIZED_DATASET_ROOT",
    "DatasetReleaseAccessBlocker",
    "fetch_materialize_dataset_release",
    "resolve_dataset_release_manifest",
]
