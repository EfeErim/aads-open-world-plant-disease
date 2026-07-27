from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.data.adapter_ood_oe_materialization import build_materialization_plan, materialize_plan
from src.data.ood_evidence_manifest import read_manifest


def _write_bytes(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_materializer_deduplicates_identical_placements_and_preserves_derived_family(tmp_path: Path) -> None:
    manifest_root = tmp_path / "manifests"
    evidence_root = tmp_path / "evidence"
    prepared_root = tmp_path / "prepared"
    original_rel = "apricot__fruit/ood_dev/unknown/original.jpg"
    derived_rel = "apricot__fruit/ood_dev/unknown/derived.jpg"
    original_hash = _write_bytes(evidence_root / original_rel, b"original")
    derived_hash = _write_bytes(evidence_root / derived_rel, b"derived")
    fields = [
        "target",
        "disease_id",
        "role",
        "ood_type",
        "relative_path",
        "sha256",
        "source",
        "original_name",
        "source_url",
        "license",
        "review_status",
    ]
    original = {
        "target": "apricot__fruit",
        "disease_id": "unknown",
        "role": "ood_dev",
        "ood_type": "same_crop_unsupported_disease",
        "relative_path": original_rel,
        "sha256": original_hash,
        "source": "reviewed-source",
        "original_name": "original.jpg",
        "source_url": "https://example.test/original",
        "license": "test",
        "review_status": "source_label_and_visual_review_accepted",
    }
    _write_csv(manifest_root / "original_manifest.csv", fields, [original])
    _write_csv(manifest_root / "duplicate_manifest.csv", fields, [original])
    derived = {**original, "relative_path": derived_rel, "sha256": derived_hash, "original_name": "derived.jpg"}
    _write_csv(manifest_root / "derived_manifest.csv", fields, [derived])
    _write_csv(
        manifest_root / "derived_candidates.csv",
        ["filename", "sha256", "parent_relative_path"],
        [{"filename": "derived.jpg", "sha256": derived_hash, "parent_relative_path": original_rel}],
    )

    plan = build_materialization_plan(
        repo_root=tmp_path,
        manifest_root=manifest_root,
        evidence_root=evidence_root,
        prepared_root=prepared_root,
        fail_on_missing_source=True,
    )

    assert plan["copy_operation_count"] == 2
    rows = plan["manifest_rows"]
    original_row = next(row for row in rows if row["is_derived"] == "false")
    derived_row = next(row for row in rows if row["is_derived"] == "true")
    assert derived_row["evidence_family_id"] == original_row["evidence_family_id"]
    assert derived_row["parent_relative_path"] == original_row["relative_path"]

    manifest_path = prepared_root / "adapter_ood_oe_evidence_manifest.csv"
    result = materialize_plan(plan, prepared_root=prepared_root, manifest_path=manifest_path)
    assert result["copied"] == 2
    assert len(read_manifest(manifest_path)) == 2
    assert (prepared_root / "apricot__fruit" / "ood" / "ood_split_manifest.json").is_file()
    assert (
        prepared_root
        / "apricot__fruit"
        / "ood"
        / "same_crop_unsupported_disease"
        / "unknown"
        / "original.jpg"
    ).is_file()


def test_materializer_fails_closed_for_missing_reviewed_source(tmp_path: Path) -> None:
    manifest_root = tmp_path / "manifests"
    fields = [
        "target",
        "disease_id",
        "role",
        "ood_type",
        "relative_path",
        "sha256",
        "source",
        "review_status",
    ]
    _write_csv(
        manifest_root / "missing_manifest.csv",
        fields,
        [
            {
                "target": "apricot__fruit",
                "disease_id": "unknown",
                "role": "oe_train",
                "ood_type": "same_crop_unsupported_disease",
                "relative_path": "apricot__fruit/oe_train/unknown/missing.jpg",
                "sha256": "0" * 64,
                "source": "reviewed-source",
                "review_status": "accepted",
            }
        ],
    )

    with pytest.raises(FileNotFoundError):
        build_materialization_plan(
            repo_root=tmp_path,
            manifest_root=manifest_root,
            evidence_root=tmp_path / "evidence",
            prepared_root=tmp_path / "prepared",
            fail_on_missing_source=True,
        )


def test_materializer_places_reviewed_id_recovery_rows(tmp_path: Path) -> None:
    manifest_root = tmp_path / "manifests"
    evidence_root = tmp_path / "evidence"
    prepared_root = tmp_path / "prepared"
    relative = "strawberry__fruit/id_train/strawberry_anthracnose_fruit/new.jpg"
    digest = _write_bytes(evidence_root / relative, b"new-independent-id-image")
    _write_csv(
        manifest_root / "strawberry_id_manifest.csv",
        [
            "target",
            "disease_id",
            "role",
            "ood_type",
            "relative_path",
            "sha256",
            "source",
            "original_name",
            "source_url",
            "review_status",
        ],
        [
            {
                "target": "strawberry__fruit",
                "disease_id": "strawberry_anthracnose_fruit",
                "role": "id_train",
                "ood_type": "",
                "relative_path": relative,
                "sha256": digest,
                "source": "reviewed-public-dataset",
                "original_name": "new.jpg",
                "source_url": "https://example.test/dataset",
                "review_status": "accepted",
            }
        ],
    )

    plan = build_materialization_plan(
        repo_root=tmp_path,
        manifest_root=manifest_root,
        evidence_root=evidence_root,
        prepared_root=prepared_root,
        fail_on_missing_source=True,
    )
    row = next(row for row in plan["manifest_rows"] if row["sha256"] == digest)
    assert row["role"] == "id_train"
    assert row["relative_path"].endswith(
        "strawberry__fruit/continual/strawberry_anthracnose_fruit/new.jpg"
    )
    materialize_plan(
        plan,
        prepared_root=prepared_root,
        manifest_path=prepared_root / "adapter_ood_oe_evidence_manifest.csv",
    )
    assert (prepared_root / row["relative_path"].split("prepared/", maxsplit=1)[-1]).is_file()


def test_materializer_preserves_grouped_id_family_identity_from_split_manifest(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    manifest_root = tmp_path / "manifests"
    evidence_root = tmp_path / "evidence"
    target_root = prepared_root / "apricot__fruit"
    first = target_root / "continual" / "monilia" / "original.jpg"
    second = target_root / "continual" / "monilia" / "derived.jpg"
    _write_bytes(first, b"original")
    _write_bytes(second, b"derived")
    split_manifest = target_root / "split_manifest.json"
    split_manifest.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "runtime_relative_path": "continual/monilia/original.jpg",
                        "family_bundle_key": "family:source/monilia-001.jpg",
                    },
                    {
                        "runtime_relative_path": "continual/monilia/derived.jpg",
                        "family_bundle_key": "family:source/monilia-001.jpg",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    oe_relative = "apricot__fruit/oe_train/unknown/oe.jpg"
    oe_hash = _write_bytes(evidence_root / oe_relative, b"oe")
    _write_csv(
        manifest_root / "oe_manifest.csv",
        ["target", "disease_id", "role", "relative_path", "sha256", "source", "review_status"],
        [
            {
                "target": "apricot__fruit",
                "disease_id": "unknown",
                "role": "oe_train",
                "relative_path": oe_relative,
                "sha256": oe_hash,
                "source": "test",
                "review_status": "accepted",
            }
        ],
    )

    plan = build_materialization_plan(
        repo_root=tmp_path,
        manifest_root=manifest_root,
        evidence_root=evidence_root,
        prepared_root=prepared_root,
    )

    id_rows = [row for row in plan["manifest_rows"] if row["role"] == "id_train"]
    assert len(id_rows) == 2
    assert len({row["evidence_family_id"] for row in id_rows}) == 1


def test_materializer_fails_closed_when_no_reviewed_rows_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No reviewed OOD/OE evidence rows"):
        build_materialization_plan(
            repo_root=tmp_path,
            manifest_root=tmp_path / "missing-manifests",
            evidence_root=tmp_path / "missing-evidence",
            prepared_root=tmp_path / "prepared",
        )


def test_materializer_upgrades_legacy_reviewed_manifest_and_resolves_derived_parent(tmp_path: Path) -> None:
    evidence_root = tmp_path / "data" / "adapter_ood_oe_evidence"
    prepared_root = tmp_path / "data" / "prepared_runtime_datasets"
    original_rel = Path("apricot__fruit/ood_dev/unsupported/original.jpg")
    derived_rel = Path("apricot__fruit/ood_dev/unsupported/derived.jpg")
    original_hash = _write_bytes(evidence_root / original_rel, b"legacy-original")
    derived_hash = _write_bytes(evidence_root / derived_rel, b"legacy-derived")
    legacy_path = prepared_root / "adapter_ood_oe_evidence_manifest.csv"
    fields = [
        "target",
        "disease_id",
        "role",
        "ood_type",
        "destination_relative_path",
        "sha256",
        "source",
        "original_name",
        "source_url",
        "review_status",
    ]
    _write_csv(
        legacy_path,
        fields,
        [
            {
                "target": "apricot__fruit",
                "disease_id": "unsupported",
                "role": "ood_dev",
                "ood_type": "same_crop_unsupported_disease",
                "destination_relative_path": "apricot__fruit/ood/unsupported/original.jpg",
                "sha256": original_hash,
                "source": "reviewed-source",
                "original_name": "original.jpg",
                "source_url": "https://example.test/original",
                "review_status": "source_label_and_visual_review_accepted",
            },
            {
                "target": "apricot__fruit",
                "disease_id": "unsupported",
                "role": "ood_dev",
                "ood_type": "same_crop_unsupported_disease",
                "destination_relative_path": "apricot__fruit/ood/unsupported/derived.jpg",
                "sha256": derived_hash,
                "source": "derived_existing_good_tier_ood_dev",
                "original_name": f"01_{original_hash[:13]}_crop.jpg",
                "source_url": "https://example.test/original",
                "review_status": "source_label_and_visual_review_accepted",
            },
        ],
    )

    plan = build_materialization_plan(
        repo_root=tmp_path,
        manifest_root=tmp_path / "missing-manifests",
        evidence_root=evidence_root,
        prepared_root=prepared_root,
        legacy_manifest_path=legacy_path,
        fail_on_missing_source=True,
    )

    assert plan["reviewed_row_count"] == 2
    original = next(row for row in plan["manifest_rows"] if row["is_derived"] == "false")
    derived = next(row for row in plan["manifest_rows"] if row["is_derived"] == "true")
    assert derived["parent_relative_path"] == original["relative_path"]
    assert derived["evidence_family_id"] == original["evidence_family_id"]


def test_materializer_preserves_all_existing_frozen_ood_slices(tmp_path: Path) -> None:
    prepared_root = tmp_path / "data" / "prepared_runtime_datasets"
    ood_root = prepared_root / "apricot__fruit" / "ood"
    same_crop = ood_root / "same_crop_unsupported_disease" / "apricot_rust" / "same.jpg"
    wrong_part = ood_root / "wrong_part" / "apricot_leaf" / "wrong.jpg"
    generic = ood_root / "non_plant_misc" / "generic.jpg"
    same_hash = _write_bytes(same_crop, b"same-crop")
    wrong_hash = _write_bytes(wrong_part, b"wrong-part")
    generic_hash = _write_bytes(generic, b"generic")
    split_path = ood_root / "ood_split_manifest.json"
    split_path.write_text(
        "{\n"
        '  "entries": {\n'
        '    "same_crop_unsupported_disease/apricot_rust/same.jpg": '
        f'{{"sha256": "{same_hash}", "slice": "same_crop_unsupported_disease", "split": "test"}},\n'
        '    "wrong_part/apricot_leaf/wrong.jpg": '
        f'{{"sha256": "{wrong_hash}", "slice": "wrong_part", "split": "dev"}},\n'
        '    "non_plant_misc/generic.jpg": '
        f'{{"sha256": "{generic_hash}", "slice": "non_plant_misc", "split": "test"}}\n'
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    plan = build_materialization_plan(
        repo_root=tmp_path,
        manifest_root=tmp_path / "missing-manifests",
        evidence_root=tmp_path / "missing-evidence",
        prepared_root=prepared_root,
        fail_on_missing_source=True,
    )

    ood_rows = [row for row in plan["manifest_rows"] if row["role"].startswith("ood_")]
    assert {(row["ood_type"], row["disease_id"]) for row in ood_rows} == {
        ("same_crop_unsupported_disease", "apricot_rust"),
    }
    materialize_plan(
        plan,
        prepared_root=prepared_root,
        manifest_path=prepared_root / "adapter_ood_oe_evidence_manifest.csv",
    )
    assert same_crop.is_file()
    assert wrong_part.is_file()
    assert generic.is_file()
    preserved = json.loads(split_path.read_text(encoding="utf-8"))["entries"]
    assert "wrong_part/apricot_leaf/wrong.jpg" in preserved
    assert "non_plant_misc/generic.jpg" in preserved
