"""Validation contract for adapter ID, OE, OOD-dev, and OOD-test evidence."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from src.ood.recovery import TARGET_ADAPTERS

EVIDENCE_ROLES = {"id_train", "id_val", "id_test", "oe_train", "ood_dev", "ood_test"}
REQUIRED_OOD_TYPES = (
    "same_crop_unsupported_disease",
)
HISTORICAL_MANIFEST_SCHEMA = "v1_adapter_ood_oe_evidence_manifest"
MANIFEST_SCHEMA = "v2_adapter_ood_oe_evidence_manifest"
REQUIRED_FIELDS = (
    "target",
    "role",
    "relative_path",
    "sha256",
    "source",
    "evidence_family_id",
    "source_manifest",
    "parent_relative_path",
    "is_derived",
)


@dataclass(frozen=True)
class EvidenceIssue:
    severity: str
    code: str
    message: str
    target: str = ""
    row: int = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path, *, require_v2: bool = True) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        required = REQUIRED_FIELDS if require_v2 else REQUIRED_FIELDS[:5]
        missing = [field for field in required if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        return [{key: str(value or "").strip() for key, value in row.items()} for row in reader]


def _issue(
    issues: list[EvidenceIssue],
    severity: str,
    code: str,
    message: str,
    *,
    target: str = "",
    row: int = 0,
) -> None:
    issues.append(EvidenceIssue(severity, code, message, target, row))


def validate_manifest_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    required_targets: Sequence[str] | None = None,
    base_dir: Path | None = None,
    file_path_prefix: str = "",
    verify_files: bool = True,
    min_id_train_per_class: int = 100,
    min_id_val_per_class: int = 15,
    min_id_test_per_class: int = 15,
    min_id_test: int = 30,
    min_ood_test: int = 30,
    min_per_ood_type: int = 5,
    min_per_ood_disease: int = 5,
) -> dict[str, Any]:
    """Validate evidence identity, provenance, role separation, and sample floors."""
    normalized = [{key: str(value or "").strip() for key, value in row.items()} for row in rows]
    selected_targets = tuple(dict.fromkeys(str(target).strip().lower() for target in (required_targets or TARGET_ADAPTERS)))
    unknown_targets = [target for target in selected_targets if target not in TARGET_ADAPTERS]
    if unknown_targets:
        raise ValueError(f"Unknown required targets: {', '.join(unknown_targets)}")
    if not selected_targets:
        raise ValueError("At least one required target must be selected.")
    if required_targets is not None:
        normalized = [row for row in normalized if row.get("target", "").lower() in selected_targets]
    issues: list[EvidenceIssue] = []
    by_hash: dict[tuple[str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    target_role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    target_ood_counts: dict[str, Counter[str]] = defaultdict(Counter)
    target_role_families: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    target_ood_families: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    target_role_disease_families: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    by_family: dict[tuple[str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    by_path: dict[str, tuple[int, dict[str, str]]] = {}
    normalized_file_prefix = file_path_prefix.strip().replace("\\", "/").strip("/")
    prefix_parts = PurePosixPath(normalized_file_prefix).parts if normalized_file_prefix else ()

    for index, row in enumerate(normalized, start=2):
        target = row.get("target", "").lower()
        role = row.get("role", "").lower()
        relative_path = row.get("relative_path", "")
        digest = row.get("sha256", "").lower()
        source = row.get("source", "")
        ood_type = row.get("ood_type", "").lower()
        family_id = row.get("evidence_family_id", "").strip()
        disease_id = row.get("disease_id", "").strip()
        source_manifest = row.get("source_manifest", "").strip()
        parent_relative_path = row.get("parent_relative_path", "").strip().replace("\\", "/")
        is_derived_text = row.get("is_derived", "").strip().lower()
        row.update(
            {
                "target": target,
                "role": role,
                "sha256": digest,
                "ood_type": ood_type,
                "evidence_family_id": family_id,
                "source_manifest": source_manifest,
                "parent_relative_path": parent_relative_path,
                "is_derived": is_derived_text,
            }
        )

        if target not in TARGET_ADAPTERS:
            _issue(issues, "error", "invalid_target", f"Unknown required target: {target or '<empty>'}", row=index)
        if role not in EVIDENCE_ROLES:
            _issue(issues, "error", "invalid_role", f"Unknown evidence role: {role or '<empty>'}", target=target, row=index)
        for field in (field for field in REQUIRED_FIELDS if field != "parent_relative_path"):
            if not row.get(field, ""):
                _issue(
                    issues,
                    "error",
                    f"missing_{field}",
                    f"Required field {field!r} is empty.",
                    target=target,
                    row=index,
                )
        if digest and (len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
            _issue(issues, "error", "invalid_sha256", "sha256 must be 64 lowercase hex characters.", target=target, row=index)
        if role in {"ood_dev", "ood_test"} and not ood_type:
            _issue(issues, "error", "missing_ood_type", "OOD evidence requires ood_type.", target=target, row=index)
        if role not in {"ood_dev", "ood_test"} and ood_type:
            _issue(
                issues,
                "warning",
                "unexpected_ood_type",
                f"{role} row carries an OOD type; it is ignored for readiness floors.",
                target=target,
                row=index,
            )
        if is_derived_text not in {"true", "false"}:
            _issue(
                issues,
                "error",
                "invalid_is_derived",
                "is_derived must be 'true' or 'false'.",
                target=target,
                row=index,
            )
        if is_derived_text == "true" and not parent_relative_path:
            _issue(
                issues,
                "error",
                "derived_parent_missing",
                "Derived evidence must identify parent_relative_path.",
                target=target,
                row=index,
            )
        if is_derived_text == "false" and parent_relative_path:
            _issue(
                issues,
                "error",
                "unexpected_parent",
                "Original evidence must not identify parent_relative_path.",
                target=target,
                row=index,
            )

        if target in TARGET_ADAPTERS and role in EVIDENCE_ROLES:
            target_role_counts[target][role] += 1
            if family_id:
                target_role_families[target][role].add(family_id)
                if disease_id:
                    target_role_disease_families[target][role][disease_id].add(family_id)
            if role == "ood_test" and ood_type:
                target_ood_counts[target][ood_type] += 1
                if family_id:
                    target_ood_families[target][ood_type].add(family_id)
        if digest:
            by_hash[(target, digest)].append((index, row))
        if family_id:
            by_family[(target, family_id)].append((index, row))
        if relative_path:
            normalized_path = relative_path.replace("\\", "/").casefold()
            prior = by_path.get(normalized_path)
            if prior is not None:
                _issue(
                    issues,
                    "error",
                    "duplicate_relative_path",
                    f"Evidence path is repeated at rows {prior[0]} and {index}: {relative_path}",
                    target=target,
                    row=index,
                )
            else:
                by_path[normalized_path] = (index, row)

        if verify_files and base_dir is not None and relative_path:
            manifest_path = PurePosixPath(relative_path.replace("\\", "/"))
            if manifest_path.is_absolute() or any(part in {"", ".", ".."} for part in manifest_path.parts):
                _issue(
                    issues,
                    "error",
                    "unsafe_relative_path",
                    f"Evidence path is not a safe relative path: {relative_path}",
                    target=target,
                    row=index,
                )
                continue
            if prefix_parts and manifest_path.parts[: len(prefix_parts)] != prefix_parts:
                _issue(
                    issues,
                    "error",
                    "file_path_prefix_mismatch",
                    f"Evidence path does not start with required file prefix {normalized_file_prefix!r}: {relative_path}",
                    target=target,
                    row=index,
                )
                continue
            resolved_parts = manifest_path.parts[len(prefix_parts) :]
            if not resolved_parts:
                _issue(
                    issues,
                    "error",
                    "file_path_prefix_mismatch",
                    f"Evidence path has no file member after prefix {normalized_file_prefix!r}: {relative_path}",
                    target=target,
                    row=index,
                )
                continue
            path = base_dir.joinpath(*resolved_parts)
            if not path.is_file():
                _issue(issues, "error", "file_missing", f"Evidence file does not exist: {path}", target=target, row=index)
            elif digest and sha256_file(path) != digest:
                _issue(issues, "error", "hash_mismatch", f"sha256 does not match {path}", target=target, row=index)
        if not source:
            _issue(issues, "error", "provenance_missing", "Evidence source/provenance is required.", target=target, row=index)

    for (target, digest), occurrences in by_hash.items():
        roles = {row["role"] for _, row in occurrences}
        if len(occurrences) > 1:
            preview = ", ".join(f"row {index} ({row['target']}:{row['role']})" for index, row in occurrences[:5])
            _issue(
                issues,
                "error",
                "hash_overlap",
                f"Image hash {digest} appears more than once across evidence: {preview}",
                target=target,
            )
        if "oe_train" in roles and roles & {"ood_dev", "ood_test"}:
            _issue(issues, "error", "oe_ood_leakage", f"OE/OOD hash leakage detected for {digest}.")
        if roles & {"id_train", "id_val", "id_test"} and roles & {"oe_train", "ood_dev", "ood_test"}:
            _issue(issues, "error", "id_unknown_leakage", f"ID/unknown evidence hash leakage detected for {digest}.")

    for (target, family_id), occurrences in by_family.items():
        roles = {row["role"] for _, row in occurrences}
        if len(roles) > 1:
            preview = ", ".join(f"row {index} ({row['target']}:{row['role']})" for index, row in occurrences[:5])
            _issue(
                issues,
                "error",
                "family_role_overlap",
                f"Evidence family {family_id!r} crosses roles: {preview}",
                target=target,
            )

    for index, row in enumerate(normalized, start=2):
        if row.get("is_derived") != "true":
            continue
        parent_key = row.get("parent_relative_path", "").replace("\\", "/").casefold()
        parent_entry = by_path.get(parent_key)
        if parent_entry is None:
            _issue(
                issues,
                "error",
                "derived_parent_not_in_manifest",
                f"Derived evidence parent is not present in the combined manifest: {row.get('parent_relative_path', '')}",
                target=row.get("target", ""),
                row=index,
            )
            continue
        _, parent = parent_entry
        if parent.get("role") != row.get("role") or parent.get("evidence_family_id") != row.get("evidence_family_id"):
            _issue(
                issues,
                "error",
                "derived_parent_contract_mismatch",
                "Derived evidence must retain its parent's role and evidence_family_id.",
                target=row.get("target", ""),
                row=index,
            )

    for target in selected_targets:
        family_counts = target_role_families[target]
        id_classes = set(target_role_disease_families[target]["id_train"])
        id_classes.update(target_role_disease_families[target]["id_val"])
        id_classes.update(target_role_disease_families[target]["id_test"])
        for disease_id in sorted(id_classes):
            role_floors = (
                ("id_train", int(min_id_train_per_class), "id_train_class_floor"),
                ("id_val", int(min_id_val_per_class), "id_val_class_floor"),
                ("id_test", int(min_id_test_per_class), "id_test_class_floor"),
            )
            for role, floor, code in role_floors:
                count = len(target_role_disease_families[target][role][disease_id])
                if count < floor:
                    _issue(
                        issues,
                        "error",
                        code,
                        f"{role} class {disease_id!r} has {count} independent families; requires at least {floor}.",
                        target=target,
                    )
        if len(family_counts["id_test"]) < int(min_id_test):
            _issue(
                issues,
                "error",
                "id_test_floor",
                f"id_test has {len(family_counts['id_test'])} independent families; requires at least {min_id_test}.",
                target=target,
            )
        if len(family_counts["ood_test"]) < int(min_ood_test):
            _issue(
                issues,
                "error",
                "ood_test_floor",
                f"ood_test has {len(family_counts['ood_test'])} independent families; requires at least {min_ood_test}.",
                target=target,
            )
        for ood_type in REQUIRED_OOD_TYPES:
            count = len(target_ood_families[target][ood_type])
            if count < int(min_per_ood_type):
                _issue(
                    issues,
                    "error",
                    "ood_type_floor",
                    f"{ood_type} has {count} independent ood_test families; requires at least {min_per_ood_type}.",
                    target=target,
                )
        for disease_id, families in sorted(target_role_disease_families[target]["ood_test"].items()):
            if len(families) < int(min_per_ood_disease):
                _issue(
                    issues,
                    "error",
                    "ood_disease_floor",
                    f"OOD test disease {disease_id!r} has {len(families)} independent families; "
                    f"requires at least {min_per_ood_disease}.",
                    target=target,
                )

    errors = [issue for issue in issues if issue.severity == "error"]
    return {
        "schema_version": MANIFEST_SCHEMA,
        "ok": not errors,
        "row_count": len(normalized),
        "target_count": len([target for target in selected_targets if target_role_counts[target]]),
        "required_targets": list(selected_targets),
        "thresholds": {
            "min_id_train_per_class": int(min_id_train_per_class),
            "min_id_val_per_class": int(min_id_val_per_class),
            "min_id_test_per_class": int(min_id_test_per_class),
            "min_id_test": int(min_id_test),
            "min_ood_test": int(min_ood_test),
            "min_per_ood_type": int(min_per_ood_type),
            "min_per_ood_disease": int(min_per_ood_disease),
            "required_ood_types": list(REQUIRED_OOD_TYPES),
        },
        "target_role_counts": {target: dict(target_role_counts[target]) for target in selected_targets},
        "target_role_family_counts": {
            target: {role: len(families) for role, families in target_role_families[target].items()}
            for target in selected_targets
        },
        "target_ood_test_type_counts": {target: dict(target_ood_counts[target]) for target in TARGET_ADAPTERS},
        "target_ood_test_type_family_counts": {
            target: {ood_type: len(families) for ood_type, families in target_ood_families[target].items()}
            for target in TARGET_ADAPTERS
        },
        "target_role_disease_family_counts": {
            target: {
                role: {disease: len(families) for disease, families in sorted(diseases.items())}
                for role, diseases in target_role_disease_families[target].items()
            }
            for target in TARGET_ADAPTERS
        },
        "error_count": len(errors),
        "warning_count": sum(issue.severity == "warning" for issue in issues),
        "issues": [asdict(issue) for issue in issues],
    }
