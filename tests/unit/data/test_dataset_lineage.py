import json
from pathlib import Path

import pytest

from src.data.dataset_lineage import (
    DatasetLineageError,
    build_github_release_lineage,
    immutable_lineage_blockers,
    lineages_directly_comparable,
    resolve_dataset_lineage,
)
from src.data.dataset_release import canonical_sha256, sha256_file


def _release_manifest(target_root: Path, *, release_id: int = 101, release_tag: str = "aads-dataset-v1") -> dict:
    members = [
        {
            "path": f"tomato__leaf/{path.relative_to(target_root).as_posix()}",
            "sha256": sha256_file(path),
        }
        for path in sorted(target_root.rglob("*"))
        if path.is_file()
    ]
    asset = {
        "asset_id": 202,
        "asset_name": "dataset.zip",
        "sha256": "a" * 64,
        "size_bytes": 123,
        "members": members,
    }
    manifest = {
        "immutable": True,
        "release_state": "current",
        "repository": "owner/repo",
        "release_tag": release_tag,
        "release_id": release_id,
        "tag_commit_sha": "b" * 40,
        "snapshot_manifest_sha256": "c" * 64,
        "assets": [asset],
    }
    identity = dict(manifest)
    identity["release_manifest_sha256"] = "d" * 64
    return identity


def _target(tmp_path: Path) -> Path:
    root = tmp_path / "tomato__leaf"
    (root / "continual" / "healthy").mkdir(parents=True)
    (root / "continual" / "healthy" / "sample.jpg").write_bytes(b"image")
    (root / "split_manifest.json").write_text(
        json.dumps({"dataset_key": "tomato__leaf"}), encoding="utf-8"
    )
    return root


def test_same_immutable_release_target_yields_same_lineage(tmp_path: Path) -> None:
    target = _target(tmp_path)
    release = _release_manifest(target)
    first = build_github_release_lineage(
        release_manifest=release,
        dataset_key="tomato__leaf",
        split_manifest_sha256=sha256_file(target / "split_manifest.json"),
    )
    second = build_github_release_lineage(
        release_manifest=dict(release),
        dataset_key="tomato__leaf",
        split_manifest_sha256=sha256_file(target / "split_manifest.json"),
    )

    assert first["dataset_lineage_key"] == second["dataset_lineage_key"]
    assert first["asset_inventory_sha256"] == canonical_sha256(first["asset_inventory"])
    assert lineages_directly_comparable(first, second) is True


def test_different_releases_are_not_directly_comparable(tmp_path: Path) -> None:
    target = _target(tmp_path)
    split_sha = sha256_file(target / "split_manifest.json")
    first = build_github_release_lineage(
        release_manifest=_release_manifest(target, release_id=101, release_tag="aads-dataset-v1"),
        dataset_key="tomato__leaf",
        split_manifest_sha256=split_sha,
    )
    second = build_github_release_lineage(
        release_manifest=_release_manifest(target, release_id=102, release_tag="aads-dataset-v2"),
        dataset_key="tomato__leaf",
        split_manifest_sha256=split_sha,
    )

    assert first["dataset_lineage_key"] != second["dataset_lineage_key"]
    assert lineages_directly_comparable(first, second) is False


def test_release_lineage_requires_exact_materialized_target(tmp_path: Path) -> None:
    target = _target(tmp_path)
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(_release_manifest(target)), encoding="utf-8")
    (target / "unexpected.txt").write_text("drift", encoding="utf-8")

    with pytest.raises(DatasetLineageError, match="differs from release inventory"):
        resolve_dataset_lineage(
            source_kind="github_release",
            dataset_key="tomato__leaf",
            split_manifest_path=target / "split_manifest.json",
            release_manifest_path=release_path,
            materialized_target_root=target,
        )


def test_local_legacy_is_explicitly_non_production(tmp_path: Path) -> None:
    target = _target(tmp_path)
    lineage = resolve_dataset_lineage(
        source_kind="local_legacy",
        dataset_key="tomato__leaf",
        split_manifest_path=target / "split_manifest.json",
        allow_local_legacy=True,
        compatibility_reason="notebook17_active_recovery_campaign_until_phase9",
    )

    assert lineage["production_eligible"] is False
    assert immutable_lineage_blockers(lineage) == ["immutable_dataset_release_lineage"]


def test_local_legacy_is_unavailable_without_explicit_compatibility(tmp_path: Path) -> None:
    target = _target(tmp_path)
    with pytest.raises(DatasetLineageError, match="unavailable to new customer training"):
        resolve_dataset_lineage(
            source_kind="local_legacy",
            dataset_key="tomato__leaf",
            split_manifest_path=target / "split_manifest.json",
        )


def test_public_sample_lineage_is_available_but_never_production_eligible(tmp_path: Path) -> None:
    target = _target(tmp_path)
    lineage = resolve_dataset_lineage(
        source_kind="public_sample",
        dataset_key="tomato__leaf",
        split_manifest_path=target / "split_manifest.json",
    )

    assert lineage["source_kind"] == "public_sample"
    assert lineage["production_eligible"] is False
    assert lineage["compatibility_reason"] == "deterministic_synthetic_smoke_dataset"
    assert immutable_lineage_blockers(lineage) == ["immutable_dataset_release_lineage"]
