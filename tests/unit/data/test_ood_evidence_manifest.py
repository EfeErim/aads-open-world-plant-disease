from __future__ import annotations

import hashlib

from src.data.ood_evidence_manifest import REQUIRED_OOD_TYPES, validate_manifest_rows
from src.ood.recovery import TARGET_ADAPTERS


def _digest(index: int) -> str:
    return f"{index:064x}"


def _complete_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    index = 1
    for target in TARGET_ADAPTERS:
        for role, count in (("id_test", 30), ("id_train", 100), ("id_val", 15), ("oe_train", 1), ("ood_dev", 5)):
            for item in range(count):
                rows.append(
                    {
                        "target": target,
                        "role": role,
                        "relative_path": f"{target}/{role}/{item}.jpg",
                        "sha256": _digest(index),
                        "source": "unit-fixture",
                        "ood_type": "same_crop_unsupported_disease" if role == "ood_dev" else "",
                        "disease_id": "supported_class" if role.startswith("id_") else "unsupported_disease",
                        "evidence_family_id": f"family-{index}",
                        "source_manifest": "unit-fixture.csv",
                        "parent_relative_path": "",
                        "is_derived": "false",
                    }
                )
                index += 1
        for ood_type in REQUIRED_OOD_TYPES:
            for item in range(30):
                rows.append(
                    {
                        "target": target,
                        "role": "ood_test",
                        "relative_path": f"{target}/ood_test/{ood_type}/{item}.jpg",
                        "sha256": _digest(index),
                        "source": "unit-fixture",
                        "ood_type": ood_type,
                        "disease_id": "unsupported_disease",
                        "evidence_family_id": f"family-{index}",
                        "source_manifest": "unit-fixture.csv",
                        "parent_relative_path": "",
                        "is_derived": "false",
                    }
                )
                index += 1
    return rows


def test_complete_disjoint_manifest_passes() -> None:
    report = validate_manifest_rows(_complete_rows(), verify_files=False)

    assert report["ok"] is True
    assert report["target_count"] == 8
    assert report["error_count"] == 0


def test_selected_target_validation_ignores_unselected_target_files_and_floors() -> None:
    rows = [row for row in _complete_rows() if row["target"] == "strawberry__leaf"]

    report = validate_manifest_rows(
        rows,
        required_targets=["strawberry__leaf"],
        verify_files=False,
    )

    assert report["ok"] is True
    assert report["target_count"] == 1
    assert report["required_targets"] == ["strawberry__leaf"]
    assert set(report["target_role_counts"]) == {"strawberry__leaf"}


def test_release_materialization_resolves_required_repo_logical_prefix(tmp_path) -> None:
    row = next(item for item in _complete_rows() if item["target"] == "apricot__fruit")
    relative_member = "apricot__fruit/test/healthy/example.jpg"
    member = tmp_path / relative_member
    member.parent.mkdir(parents=True)
    member.write_bytes(b"release-member")
    row["relative_path"] = f"data/prepared_runtime_datasets/{relative_member}"
    row["sha256"] = hashlib.sha256(member.read_bytes()).hexdigest()

    report = validate_manifest_rows(
        [row],
        required_targets=["apricot__fruit"],
        base_dir=tmp_path,
        file_path_prefix="data/prepared_runtime_datasets",
        min_id_train_per_class=0,
        min_id_val_per_class=0,
        min_id_test_per_class=0,
        min_id_test=0,
        min_ood_test=0,
        min_per_ood_type=0,
        min_per_ood_disease=0,
    )

    assert report["ok"] is True
    assert report["error_count"] == 0


def test_hash_overlap_is_a_hard_failure() -> None:
    rows = _complete_rows()
    original = next(row for row in rows if row["target"] == "apricot__fruit" and row["role"] == "id_train")
    duplicate = next(
        row
        for row in rows
        if row["target"] == "apricot__fruit" and row["role"] == "ood_test"
    )
    duplicate["sha256"] = original["sha256"]

    report = validate_manifest_rows(rows, verify_files=False)

    codes = {issue["code"] for issue in report["issues"]}
    assert report["ok"] is False
    assert "hash_overlap" in codes
    assert "id_unknown_leakage" in codes


def test_cross_target_hash_reuse_is_not_adapter_split_leakage() -> None:
    rows = _complete_rows()
    other_target = next(row for row in rows if row["target"] != rows[0]["target"])
    other_target["sha256"] = rows[0]["sha256"]

    report = validate_manifest_rows(rows, verify_files=False)

    assert report["ok"] is True


def test_missing_target_and_slice_floors_fail() -> None:
    rows = [row for row in _complete_rows() if row["target"] != "tomato__leaf"]
    rows = [
        row
        for row in rows
        if not (
            row["target"] == "grape__fruit"
            and row["role"] == "ood_test"
            and row["ood_type"] == "same_crop_unsupported_disease"
        )
    ]

    report = validate_manifest_rows(rows, verify_files=False)

    failures = {(issue["target"], issue["code"]) for issue in report["issues"]}
    assert ("tomato__leaf", "id_test_floor") in failures
    assert ("tomato__leaf", "ood_test_floor") in failures
    assert ("grape__fruit", "ood_type_floor") in failures


def test_id_class_floors_fail_for_under_supported_class() -> None:
    rows = [
        row
        for row in _complete_rows()
        if not (
            row["target"] == "apricot__fruit"
            and row["role"] == "id_train"
            and int(row["relative_path"].rsplit("/", 1)[1].split(".", 1)[0]) >= 99
        )
    ]

    report = validate_manifest_rows(rows, verify_files=False)

    failures = {(issue["target"], issue["code"]) for issue in report["issues"]}
    assert ("apricot__fruit", "id_train_class_floor") in failures


def test_derived_view_cannot_cross_roles_or_add_an_independent_family() -> None:
    rows = _complete_rows()
    parent = rows[0]
    rows.append(
        {
            **parent,
            "role": "ood_test",
            "relative_path": "apricot__fruit/ood_test/derived.jpg",
            "sha256": _digest(9999),
            "parent_relative_path": parent["relative_path"],
            "is_derived": "true",
        }
    )

    report = validate_manifest_rows(rows, verify_files=False)

    codes = {issue["code"] for issue in report["issues"]}
    family_overlap = next(issue for issue in report["issues"] if issue["code"] == "family_role_overlap")
    assert "family_role_overlap" in codes
    assert family_overlap["target"] == "apricot__fruit"
    assert "derived_parent_contract_mismatch" in codes
