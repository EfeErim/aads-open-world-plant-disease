from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from src.data.dataset_release import (
    DATASET_MANIFEST_SCHEMA,
    DatasetContractError,
    DatasetResourceLimits,
    bind_snapshot_operation,
    build_dataset_snapshot,
    build_runtime_parity_candidate,
    build_shard_plan,
    build_snapshot_reports,
    diff_snapshot_inventory,
    inspect_image,
    sanitize_source_uri,
    stage_sanitized_image,
    validate_archive,
    validate_frozen_evaluation_assignments,
    validate_release_candidate,
    validate_snapshot_manifest,
    validate_version_change,
    verify_snapshot_files,
)


def _image(path: Path, *, exif: bool = False, color: str = "green") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (12, 8), color=color)
    if exif:
        metadata = Image.Exif()
        metadata[0x010E] = "private note"
        image.save(path, exif=metadata)
    else:
        image.save(path)


def test_stage_sanitized_image_supports_long_destination_paths(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    _image(source, exif=True)
    destination = tmp_path / "stage" / ("a" * 100) / ("b" * 100) / ("c" * 100) / "sample.jpg"

    inspection = stage_sanitized_image(source, destination)

    assert len(str(destination.resolve())) > 260
    assert inspection["format"] == "JPEG"
    assert inspection["exif_keys"] == []
    assert inspection["size_bytes"] > 0


def test_snapshot_normalizes_decodable_extension_mismatch_without_weakening_direct_inspection(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = repo / "data/prepared_runtime_datasets/tomato__leaf"
    relative = "test/healthy/mislabeled.jpg"
    source = root / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color="green").save(source, format="PNG")
    with pytest.raises(DatasetContractError, match="Extension/MIME mismatch"):
        inspect_image(source)

    staging = repo / ".runtime_tmp/stage"
    snapshot = build_dataset_snapshot(
        [root],
        repo_root=repo,
        metadata=_approved_metadata("tomato__leaf", relative),
        staging_root=staging,
        dataset_version="1.0.0",
        inventory_cutoff="2026-07-13T00:00:00Z",
    )

    record = snapshot["records"][0]
    assert record["disposition"] == "uploadable"
    assert record["source_normalization_required"] is True
    assert record["source_normalization_reason"] == "extension_mime_mismatch"
    assert inspect_image(staging / str(record["distributed_path"]))["format"] == "JPEG"


def _approved_metadata(target: str, relative_path: str, **updates: object) -> dict[str, dict[str, object]]:
    value: dict[str, object] = {
        "source_asset_id": "asset-1",
        "source_uri": "https://user:secret@example.test/image.jpg?token=secret#fragment",
        "license": "CC-BY-4.0",
        "redistribution_allowed": True,
        "commercial_use_allowed": True,
        "license_evidence_uri": "https://example.test/license?token=secret",
        "license_reviewed_by": "reviewer",
        "license_reviewed_at": "2026-07-13T00:00:00Z",
        "privacy_review_status": "approved",
        "provenance": "unit fixture",
        "review_status": "approved",
        "evaluation_cohort_id": "cohort-v1",
        "comparability_key": "comparison-v1",
    }
    value.update(updates)
    return {f"{target}/{relative_path}": value}


def _snapshot(tmp_path: Path, *, metadata: dict | None = None) -> tuple[dict, Path, Path]:
    repo = tmp_path / "repo"
    root = repo / "data" / "prepared_runtime_datasets" / "tomato__leaf"
    relative = "test/healthy/sample.jpg"
    _image(root / relative, exif=True)
    staging = repo / ".runtime_tmp" / "dataset_stage"
    snapshot = build_dataset_snapshot(
        [root],
        repo_root=repo,
        metadata=metadata,
        staging_root=staging,
        dataset_version="1.0.0",
        inventory_cutoff="2026-07-13T00:00:00Z",
    )
    return snapshot, repo, staging


def test_approved_image_is_staged_without_exif_and_urls_are_sanitized(tmp_path: Path) -> None:
    metadata = _approved_metadata("tomato__leaf", "test/healthy/sample.jpg")
    snapshot, repo, staging = _snapshot(tmp_path, metadata=metadata)
    record = snapshot["records"][0]

    assert snapshot["schema_version"] == DATASET_MANIFEST_SCHEMA
    assert record["disposition"] == "uploadable"
    assert record["source_uri"] == "https://example.test/image.jpg"
    assert record["license_evidence_uri"] == "https://example.test/license"
    assert record["source_content_sha256"] != record["distributed_content_sha256"]
    assert inspect_image(staging / record["distributed_path"])["exif_keys"] == []
    assert verify_snapshot_files(snapshot, repo_root=repo, staging_root=staging)["verified"] is True


def test_missing_license_provenance_and_privacy_review_are_manifest_quarantined(tmp_path: Path) -> None:
    snapshot, _, _ = _snapshot(tmp_path)
    reports = build_snapshot_reports(snapshot, dataset_tag="aads-dataset-v1.0.0")

    record = snapshot["records"][0]
    assert record["disposition"] == "quarantine"
    assert "missing_license" in record["quarantine_reasons"]
    assert "missing_provenance" in record["quarantine_reasons"]
    assert "privacy_review_incomplete" in record["quarantine_reasons"]
    assert reports["quarantine"]["physical_action"] == "none"
    assert reports["release"]["files"] == []


def test_snapshot_identity_detects_tampering_and_operations_bind_exact_identity(tmp_path: Path) -> None:
    snapshot, _, _ = _snapshot(tmp_path)
    validate_snapshot_manifest(snapshot)
    identities = {
        (binding["staging_snapshot_id"], binding["manifest_sha256"])
        for binding in (bind_snapshot_operation(name, snapshot) for name in ("audit", "diff", "publish", "verify"))
    }
    assert len(identities) == 1
    snapshot["records"][0]["class_name"] = "tampered"
    with pytest.raises(DatasetContractError, match="manifest SHA-256"):
        validate_snapshot_manifest(snapshot)


def test_post_cutoff_arrival_is_visible_without_changing_snapshot(tmp_path: Path) -> None:
    snapshot, repo, _ = _snapshot(tmp_path)
    frozen_identity = (snapshot["staging_snapshot_id"], snapshot["manifest_sha256"])
    _image(repo / "data/prepared_runtime_datasets/tomato__leaf/test/healthy/new.jpg", color="red")

    result = diff_snapshot_inventory(snapshot, repo_root=repo)

    assert result["added"] == ["data/prepared_runtime_datasets/tomato__leaf/test/healthy/new.jpg"]
    assert (snapshot["staging_snapshot_id"], snapshot["manifest_sha256"]) == frozen_identity


def test_hash_overlap_is_reported_across_id_oe_and_ood_roles(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = repo / "data/prepared_runtime_datasets/tomato__leaf"
    for relative in ("test/healthy/a.png", "oe/a.png", "ood/a.png"):
        _image(root / relative)
    snapshot = build_dataset_snapshot(
        [root],
        repo_root=repo,
        metadata={},
        staging_root=repo / ".runtime_tmp/stage",
        dataset_version="1.0.0",
        inventory_cutoff="2026-07-13T00:00:00Z",
    )
    report = build_snapshot_reports(snapshot, dataset_tag="aads-dataset-v1.0.0")["audit"]

    assert len(report["duplicate_source_hashes"]) == 1
    assert report["evidence_role_overlaps"][0]["groups"] == ["id", "oe", "ood_test"]
    assert all(record["disposition"] == "quarantine" for record in snapshot["records"])
    assert all("evidence_role_hash_overlap" in record["quarantine_reasons"] for record in snapshot["records"])


def test_approved_duplicate_content_never_enters_release_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = repo / "data/prepared_runtime_datasets/tomato__leaf"
    for relative in ("test/healthy/a.png", "test/healthy/b.png"):
        _image(root / relative)
    metadata = {}
    metadata.update(_approved_metadata("tomato__leaf", "test/healthy/a.png", source_asset_id="asset-a"))
    metadata.update(_approved_metadata("tomato__leaf", "test/healthy/b.png", source_asset_id="asset-b"))
    snapshot = build_dataset_snapshot(
        [root],
        repo_root=repo,
        metadata=metadata,
        staging_root=repo / ".runtime_tmp/stage",
        dataset_version="1.0.0",
        inventory_cutoff="2026-07-13T00:00:00Z",
    )

    reports = build_snapshot_reports(snapshot, dataset_tag="aads-dataset-v1.0.0")
    assert reports["release"]["files"] == []
    assert all("duplicate_content_hash" in record["quarantine_reasons"] for record in snapshot["records"])


def test_images_that_match_after_metadata_removal_are_quarantined(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = repo / "data/prepared_runtime_datasets/tomato__leaf"
    first = "test/healthy/with-exif.jpg"
    second = "test/healthy/without-exif.jpg"
    _image(root / first, exif=True)
    _image(root / second, exif=False)
    metadata = {}
    metadata.update(_approved_metadata("tomato__leaf", first, source_asset_id="asset-a"))
    metadata.update(_approved_metadata("tomato__leaf", second, source_asset_id="asset-b"))

    snapshot = build_dataset_snapshot(
        [root],
        repo_root=repo,
        metadata=metadata,
        staging_root=repo / ".runtime_tmp/stage",
        dataset_version="1.0.0",
        inventory_cutoff="2026-07-13T00:00:00Z",
    )

    reports = build_snapshot_reports(snapshot, dataset_tag="aads-dataset-v1.0.0")
    assert reports["release"]["files"] == []
    assert len({record["source_content_sha256"] for record in snapshot["records"]}) == 2
    assert all(
        "duplicate_distributed_content_hash" in record["quarantine_reasons"] for record in snapshot["records"]
    )


def test_path_archive_mime_and_resource_bounds_fail_closed(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.jpg", b"not-an-image")
    with pytest.raises(DatasetContractError, match="Unsafe relative path"):
        validate_archive(archive)

    fake = tmp_path / "fake.jpg"
    fake.write_text("not an image", encoding="utf-8")
    with pytest.raises(DatasetContractError, match="Unreadable or unsafe"):
        inspect_image(fake)

    huge = tmp_path / "huge.png"
    _image(huge)
    with pytest.raises(DatasetContractError, match="configured bounds"):
        inspect_image(huge, DatasetResourceLimits(max_file_bytes=1))

    repo = tmp_path / "repo"
    root = repo / "data/prepared_runtime_datasets/tomato__leaf"
    _image(root / "test/healthy/a.jpg")
    with pytest.raises(DatasetContractError, match="outside the repository"):
        build_dataset_snapshot(
            [root],
            repo_root=repo,
            metadata={},
            staging_root=tmp_path / "external-stage",
            dataset_version="1.0.0",
            inventory_cutoff="2026-07-13T00:00:00Z",
        )


def test_runtime_parity_candidate_preserves_duplicate_paths_and_sidecars(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    (root / "tomato__leaf/train/healthy").mkdir(parents=True)
    content = b"same-runtime-bytes"
    (root / "tomato__leaf/train/healthy/a.jpg").write_bytes(content)
    (root / "tomato__leaf/train/healthy/b.jpg").write_bytes(content)
    (root / "adapter_ood_oe_evidence_manifest.csv").write_text("path,role\na.jpg,id_train\n", encoding="utf-8")
    (root / "tomato__leaf/split_manifest.json").write_text('{"schema_version":"v1"}', encoding="utf-8")

    candidate, plan = build_runtime_parity_candidate(
        root,
        dataset_version="1.0.0",
        dataset_tag="aads-dataset-v1.0.0",
        inventory_cutoff="2026-07-18T00:00:00Z",
    )

    assert candidate["candidate_profile"] == "runtime_parity"
    assert candidate["file_count"] == 4
    assert len(plan["shards"]) == 1
    assert {row["distributed_path"] for row in candidate["files"]} == {
        "adapter_ood_oe_evidence_manifest.csv",
        "tomato__leaf/split_manifest.json",
        "tomato__leaf/train/healthy/a.jpg",
        "tomato__leaf/train/healthy/b.jpg",
    }

    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("adapter_ood_oe_evidence_manifest.csv", b"path,role\n")
        handle.writestr("tomato__leaf/split_manifest.json", b"{}")
    assert validate_archive(archive)["member_count"] == 2


def test_version_tombstone_cohort_and_revocation_contracts() -> None:
    validate_version_change("1.2.3", "1.2.4", "patch")
    validate_version_change("1.2.3", "1.3.0", "minor")
    validate_version_change("1.2.3", "2.0.0", "major")
    with pytest.raises(DatasetContractError, match="minor version bump"):
        validate_version_change("1.2.3", "1.2.4", "minor")

    old = [{"sample_id": "a", "evidence_role": "id_test", "split": "test", "class_name": "healthy",
            "evaluation_cohort_id": "c1", "comparability_key": "k1"}]
    changed = [{**old[0], "class_name": "disease", "evaluation_cohort_id": "c2", "comparability_key": "k2"}]
    validate_frozen_evaluation_assignments(old, changed)
    with pytest.raises(DatasetContractError, match="new non-comparable cohort"):
        validate_frozen_evaluation_assignments(old, [{**old[0], "class_name": "disease"}])

    release = {
        "schema_version": "v1_dataset_github_release_candidate",
        "release_state": "revoked",
        "revocation_reason": "privacy withdrawal",
        "dataset_tag": "aads-dataset-v1.0.0",
        "staging_snapshot_id": "dataset-snapshot-1234",
        "snapshot_manifest_sha256": "a" * 64,
        "files": [],
    }
    with pytest.raises(DatasetContractError, match="Revoked"):
        build_shard_plan(release)


def test_shard_plan_is_deterministic_bounded_and_complete() -> None:
    release = {
        "schema_version": "v1_dataset_github_release_candidate",
        "release_state": "candidate",
        "dataset_tag": "aads-dataset-v1.0.0",
        "staging_snapshot_id": "dataset-snapshot-1234",
        "snapshot_manifest_sha256": "a" * 64,
        "files": [
            {
                "sample_id": f"s{i}",
                "distributed_path": f"data/{i}.jpg",
                "distributed_content_sha256": f"{i:064x}",
                "source_content_sha256": f"{i + 10:064x}",
                "size_bytes": 6,
            }
            for i in range(3)
        ],
    }
    limits = DatasetResourceLimits(max_shard_bytes=600)
    first = build_shard_plan(release, limits)
    second = build_shard_plan(json.loads(json.dumps(release)), limits)

    assert first == second
    assert [shard["member_count"] for shard in first["shards"]] == [1, 1, 1]
    assert all(shard["size_bytes"] <= 600 for shard in first["shards"])
    assert all(shard["content_bytes"] == 6 for shard in first["shards"])


def test_release_candidate_rejects_unsafe_duplicate_and_wrong_tag_identity() -> None:
    record = {
        "sample_id": "sample-a",
        "distributed_path": "target/test/a.jpg",
        "distributed_content_sha256": "a" * 64,
        "source_content_sha256": "b" * 64,
        "size_bytes": 10,
    }
    release = {
        "schema_version": "v1_dataset_github_release_candidate",
        "release_state": "candidate",
        "dataset_tag": "aads-dataset-v1.0.0",
        "staging_snapshot_id": "dataset-snapshot-1234",
        "snapshot_manifest_sha256": "c" * 64,
        "files": [record],
    }
    validate_release_candidate(release)
    with pytest.raises(DatasetContractError, match="duplicate"):
        validate_release_candidate({**release, "files": [record, record]})
    with pytest.raises(DatasetContractError, match="namespace"):
        validate_release_candidate({**release, "dataset_tag": "latest"})
    with pytest.raises(DatasetContractError, match="Unsafe relative path"):
        validate_release_candidate({**release, "files": [{**record, "distributed_path": "../escape.jpg"}]})


def test_source_url_sanitizer_strips_credentials_queries_and_fragments() -> None:
    assert sanitize_source_uri("https://user:secret@example.test/a.jpg?token=abc#gps") == (
        "https://example.test/a.jpg"
    )
