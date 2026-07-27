import json
from pathlib import Path

import pytest

from scripts.apply_internet_candidate_manifests import apply_candidate_placements


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_apply_candidate_placements_copies_targets_and_writes_summary(tmp_path: Path):
    repo_root = tmp_path / "repo"
    manifest_root = repo_root / "data" / "internet_image_candidates" / "20260510" / "phase1_ood_candidates"
    source_path = manifest_root / "strawberry__fruit" / "fruit_specific_unknowns" / "sample.jpg"
    target_path = repo_root / "data" / "prepared_runtime_datasets" / "strawberry__fruit" / "ood" / "fruit_specific_unknowns" / "sample.jpg"
    manifest_path = manifest_root / "curated_phase1_ood_placement_manifest.json"

    _write_bytes(source_path, b"candidate-image")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "placements": [
                    {
                        "candidate_path": str(source_path.relative_to(repo_root)).replace("/", "\\"),
                        "target_path": str(target_path.relative_to(repo_root)).replace("/", "\\"),
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = apply_candidate_placements(manifest_root.parent.parent, repo_root=repo_root)

    assert target_path.is_file()
    assert target_path.read_bytes() == b"candidate-image"
    assert summary["placement_count"] == 1
    assert summary["copied_count"] == 1
    assert summary["skipped_existing_count"] == 0


def test_apply_candidate_placements_rejects_ood_oe_hash_overlap(tmp_path: Path):
    repo_root = tmp_path / "repo"
    manifest_root = repo_root / "data" / "internet_image_candidates" / "20260510" / "phase1_ood_candidates"
    ood_source = manifest_root / "strawberry__fruit" / "fruit_specific_unknowns" / "shared.jpg"
    oe_source = manifest_root / "strawberry__fruit" / "oe_candidates" / "shared.jpg"
    ood_target = repo_root / "data" / "prepared_runtime_datasets" / "strawberry__fruit" / "ood" / "fruit_specific_unknowns" / "shared.jpg"
    oe_target = repo_root / "data" / "prepared_runtime_datasets" / "strawberry__fruit" / "oe" / "same_crop_fruit_negatives" / "shared.jpg"
    shared_bytes = b"shared-candidate"

    _write_bytes(ood_source, shared_bytes)
    _write_bytes(oe_source, shared_bytes)
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "curated_phase1_ood_placement_manifest.json").write_text(
        json.dumps(
            {
                "placements": [
                    {
                        "candidate_path": str(ood_source.relative_to(repo_root)).replace("/", "\\"),
                        "target_path": str(ood_target.relative_to(repo_root)).replace("/", "\\"),
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (manifest_root / "oe_reusable_placement_manifest.json").write_text(
        json.dumps(
            {
                "placements": [
                    {
                        "source_path": str(oe_source.relative_to(repo_root)).replace("/", "\\"),
                        "target_path": str(oe_target.relative_to(repo_root)).replace("/", "\\"),
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="both OOD and OE"):
        apply_candidate_placements(manifest_root.parent.parent, repo_root=repo_root)