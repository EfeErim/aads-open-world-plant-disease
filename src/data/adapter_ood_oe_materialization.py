"""Deterministic, family-aware materialization for adapter OOD/OE recovery evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.ood_evidence_manifest import EVIDENCE_ROLES, MANIFEST_SCHEMA, sha256_file
from src.ood.recovery import TARGET_ADAPTERS

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
OUTPUT_FIELDS = (
    "target",
    "role",
    "relative_path",
    "sha256",
    "source",
    "ood_type",
    "disease_id",
    "evidence_family_id",
    "source_manifest",
    "parent_relative_path",
    "is_derived",
)
ACCEPTED_REVIEW_STATUSES = {
    "accepted",
    "reviewed",
    "source_label_and_visual_review_accepted",
}
LEGACY_REQUIRED_FIELDS = {
    "target",
    "disease_id",
    "role",
    "destination_relative_path",
    "sha256",
    "source",
    "review_status",
}
SAME_CROP_UNSUPPORTED_SLICE = "same_crop_unsupported_disease"


@dataclass(frozen=True)
class CopyOperation:
    source: Path
    destination: Path
    row: dict[str, str]


def _stable_family_id(target: str, source_identity: str) -> str:
    normalized_identity = source_identity.replace("\\", "/").casefold()
    identity = f"{target}|{normalized_identity}"
    return f"family_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [{key: str(value or "").strip() for key, value in row.items()} for row in reader]
    return fields, rows


def _candidate_parent_map(manifest_path: Path) -> dict[str, str]:
    candidate_path = manifest_path.with_name(manifest_path.name.replace("_manifest.csv", "_candidates.csv"))
    if not candidate_path.is_file():
        return {}
    _, rows = _read_csv(candidate_path)
    return {
        row.get("sha256", "").lower(): row.get("parent_relative_path", "").replace("\\", "/")
        for row in rows
        if row.get("sha256") and row.get("parent_relative_path")
    }


def discover_reviewed_manifest_rows(manifest_root: Path, *, repo_root: Path | None = None) -> list[dict[str, str]]:
    """Read reviewed placement manifests while ignoring catalog/candidate CSVs."""

    accepted: list[dict[str, str]] = []
    required = {"target", "disease_id", "role", "relative_path", "sha256", "source", "review_status"}
    for path in sorted(manifest_root.glob("*_manifest.csv")):
        fields, rows = _read_csv(path)
        if not required.issubset(fields):
            continue
        parent_by_hash = _candidate_parent_map(path)
        for row in rows:
            if row.get("review_status", "").strip().lower() not in ACCEPTED_REVIEW_STATUSES:
                continue
            role = row.get("role", "").strip().lower()
            if role not in EVIDENCE_ROLES:
                continue
            row["target"] = row.get("target", "").strip().lower()
            row["role"] = role
            row["sha256"] = row.get("sha256", "").strip().lower()
            row["relative_path"] = row.get("relative_path", "").replace("\\", "/")
            if repo_root is not None:
                try:
                    source_manifest = path.resolve().relative_to(repo_root.resolve()).as_posix()
                except ValueError:
                    source_manifest = path.name
            else:
                source_manifest = path.name
            row["source_manifest"] = source_manifest
            row["parent_relative_path"] = parent_by_hash.get(row["sha256"], "")
            row["is_derived"] = "true" if row["parent_relative_path"] else "false"
            accepted.append(row)
    return accepted


def discover_legacy_manifest_rows(path: Path, *, repo_root: Path, evidence_root: Path) -> list[dict[str, str]]:
    """Upgrade the maintained reviewed v1 placement manifest into materializer input rows."""

    if not path.is_file():
        return []
    fields, source_rows = _read_csv(path)
    if not LEGACY_REQUIRED_FIELDS.issubset(fields):
        return []
    originals: dict[tuple[str, str, str, str], str] = {}
    for row in source_rows:
        if row.get("source", "").lower().startswith("derived"):
            continue
        source_relative = _legacy_source_relative_path(row, repo_root=repo_root, evidence_root=evidence_root)
        originals[(row["target"], row["role"], row["disease_id"], row["sha256"].lower())] = source_relative

    accepted: list[dict[str, str]] = []
    source_manifest = _repo_relative(path, repo_root)
    for source_row in source_rows:
        if source_row.get("review_status", "").lower() not in ACCEPTED_REVIEW_STATUSES:
            continue
        role = source_row.get("role", "").lower()
        if role not in {"oe_train", "ood_dev", "ood_test"}:
            continue
        row = dict(source_row)
        row["target"] = row.get("target", "").lower()
        row["role"] = role
        row["sha256"] = row.get("sha256", "").lower()
        row["relative_path"] = _legacy_source_relative_path(
            row,
            repo_root=repo_root,
            evidence_root=evidence_root,
        )
        row["source_manifest"] = source_manifest
        row["parent_relative_path"] = ""
        row["is_derived"] = "false"
        if row.get("source", "").lower().startswith("derived"):
            match = re.search(r"[0-9a-f]{12,16}", row.get("original_name", "").lower())
            candidates = [
                relative
                for (target, parent_role, disease_id, digest), relative in originals.items()
                if target == row["target"]
                and parent_role == row["role"]
                and disease_id == row.get("disease_id", "")
                and match is not None
                and digest.startswith(match.group(0))
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Derived legacy evidence must resolve to exactly one reviewed parent: "
                    f"{row.get('destination_relative_path', '')}"
                )
            row["parent_relative_path"] = candidates[0]
            row["is_derived"] = "true"
        accepted.append(row)
    return accepted


def _legacy_source_relative_path(row: Mapping[str, str], *, repo_root: Path, evidence_root: Path) -> str:
    filename = Path(row.get("destination_relative_path", "")).name
    source = evidence_root / row.get("target", "") / row.get("role", "") / row.get("disease_id", "") / filename
    return _repo_relative(source, repo_root)


def _prepared_ood_rows(prepared_root: Path, repo_root: Path) -> list[dict[str, str]]:
    """Upgrade the strict same-crop OOD gate from current frozen manifests.

    Non-gating smoke slices are retained in ``materialize_plan`` without being
    promoted into the strict combined evidence manifest.
    """

    rows: list[dict[str, str]] = []
    for target in TARGET_ADAPTERS:
        ood_root = prepared_root / target / "ood"
        split_path = ood_root / "ood_split_manifest.json"
        if not split_path.is_file():
            continue
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        entries = payload.get("entries") or {}
        if not isinstance(entries, Mapping):
            raise ValueError(f"{split_path} entries must be a JSON object")
        for relative, entry in sorted(entries.items()):
            if not isinstance(entry, Mapping):
                continue
            normalized = str(relative).replace("\\", "/")
            parts = Path(normalized).parts
            if len(parts) < 2:
                continue
            ood_type = str(entry.get("ood_type") or entry.get("slice") or parts[0]).lower()
            if ood_type != SAME_CROP_UNSUPPORTED_SLICE:
                continue
            disease_id = parts[1] if len(parts) >= 3 else ood_type
            split = str(entry.get("split") or entry.get("assignment") or "").lower()
            if split not in {"dev", "test"}:
                continue
            path = ood_root / Path(normalized)
            if not path.is_file():
                raise FileNotFoundError(f"Frozen prepared OOD evidence is missing: {path}")
            digest = str(entry.get("sha256") or "").lower() or sha256_file(path)
            rows.append(
                {
                    "target": target,
                    "role": "ood_dev" if split == "dev" else "ood_test",
                    "relative_path": _repo_relative(path, repo_root),
                    "sha256": digest,
                    "source": "reviewed_frozen_prepared_ood",
                    "ood_type": ood_type,
                    "disease_id": disease_id,
                    "source_manifest": _repo_relative(split_path, repo_root),
                    "parent_relative_path": "",
                    "is_derived": "false",
                    "original_name": path.name,
                    "source_url": "",
                }
            )
    return rows


def _destination_for(row: Mapping[str, str], prepared_root: Path) -> Path:
    target = row["target"]
    disease_id = row.get("disease_id", "unlabeled") or "unlabeled"
    filename = Path(row["relative_path"]).name
    id_split = {"id_train": "continual", "id_val": "val", "id_test": "test"}.get(row["role"])
    if id_split is not None:
        return prepared_root / target / id_split / disease_id / filename
    if row["role"] == "oe_train":
        return prepared_root / target / "oe" / disease_id / filename
    ood_type = row.get("ood_type", "") or "untyped_ood"
    if disease_id == ood_type:
        return prepared_root / target / "ood" / ood_type / filename
    return prepared_root / target / "ood" / ood_type / disease_id / filename


def _resolve_family_ids(rows: list[dict[str, str]]) -> None:
    family_by_source_path: dict[str, str] = {}
    for row in rows:
        if row["is_derived"] == "false":
            source_identity = "|".join(
                (
                    row.get("source_url", ""),
                    row.get("original_name", ""),
                    row.get("source", ""),
                )
            ).strip("|") or row["relative_path"]
            family_by_source_path[row["relative_path"].casefold()] = _stable_family_id(
                row["target"], source_identity
            )
    for row in rows:
        parent = row["parent_relative_path"]
        family_key = parent if parent else row["relative_path"]
        family = family_by_source_path.get(family_key.casefold())
        row["evidence_family_id"] = family or _stable_family_id(row["target"], parent or row["relative_path"])


def _deduplicate_rows(rows: Iterable[dict[str, str]], prepared_root: Path) -> list[dict[str, str]]:
    deduplicated: dict[str, dict[str, str]] = {}
    for row in rows:
        destination_key = _destination_for(row, prepared_root).as_posix().casefold()
        prior = deduplicated.get(destination_key)
        if prior is None:
            deduplicated[destination_key] = row
            continue
        # The maintained reviewed manifest carries richer source-family lineage
        # than a legacy frozen split entry.  When both describe the same exact
        # placement, retain the first reviewed row instead of rejecting the
        # harmless family-id representation difference.
        comparable = ("target", "sha256", "disease_id", "ood_type")
        if any(prior.get(key, "") != row.get(key, "") for key in comparable):
            raise ValueError(f"Conflicting reviewed rows target the same destination: {destination_key}")
        if prior.get("role") != row.get("role"):
            if row.get("source") != "reviewed_frozen_prepared_ood":
                raise ValueError(f"Conflicting reviewed rows target the same destination: {destination_key}")
            # The reviewed placement owns family-coherent dev/test lineage.
            # A legacy v1 split may have assigned derived siblings
            # independently; v2 repairs that leakage by retaining the reviewed
            # family assignment and rewriting the frozen split below.
        manifests = sorted(set(prior["source_manifest"].split(";") + row["source_manifest"].split(";")))
        prior["source_manifest"] = ";".join(item for item in manifests if item)
    return list(deduplicated.values())


def _repo_relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _source_path(row: Mapping[str, str], *, repo_root: Path, evidence_root: Path) -> Path:
    relative = Path(row["relative_path"])
    repo_candidate = repo_root / relative
    if relative.parts and relative.parts[0].lower() == "data":
        return repo_candidate
    return evidence_root / relative


def _id_rows(prepared_root: Path, repo_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    split_roles = {"continual": "id_train", "val": "id_val", "test": "id_test"}
    for target in TARGET_ADAPTERS:
        target_manifest = prepared_root / target / "split_manifest.json"
        family_by_runtime_path: dict[str, str] = {}
        if target_manifest.is_file():
            payload = json.loads(target_manifest.read_text(encoding="utf-8"))
            manifest_rows = payload.get("rows") or []
            if not isinstance(manifest_rows, list):
                raise ValueError(f"{target_manifest} rows must be a JSON array")
            for manifest_row in manifest_rows:
                if not isinstance(manifest_row, Mapping):
                    continue
                runtime_path = str(manifest_row.get("runtime_relative_path") or "").replace("\\", "/")
                if not runtime_path:
                    continue
                family_identity = str(
                    manifest_row.get("family_bundle_key")
                    or manifest_row.get("family_id")
                    or manifest_row.get("family_canonical_relative_path")
                    or runtime_path
                )
                family_by_runtime_path[runtime_path.casefold()] = family_identity
        for split, role in split_roles.items():
            split_root = prepared_root / target / split
            if not split_root.is_dir():
                continue
            for path in sorted(item for item in split_root.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES):
                relative_path = _repo_relative(path, repo_root)
                runtime_path = path.relative_to(prepared_root / target).as_posix()
                family_identity = family_by_runtime_path.get(runtime_path.casefold(), relative_path)
                rows.append(
                    {
                        "target": target,
                        "role": role,
                        "relative_path": relative_path,
                        "sha256": sha256_file(path),
                        "source": "prepared_runtime_dataset",
                        "ood_type": "",
                        "disease_id": path.parent.name,
                        "evidence_family_id": _stable_family_id(target, f"split_manifest:{family_identity}"),
                        "source_manifest": _repo_relative(target_manifest, repo_root)
                        if target_manifest.is_file()
                        else "prepared_runtime_dataset_layout",
                        "parent_relative_path": "",
                        "is_derived": "false",
                    }
                )
    return rows


def build_materialization_plan(
    *,
    repo_root: Path,
    manifest_root: Path,
    evidence_root: Path,
    prepared_root: Path,
    legacy_manifest_path: Path | None = None,
    fail_on_missing_source: bool = False,
) -> dict[str, Any]:
    rows = discover_reviewed_manifest_rows(manifest_root, repo_root=repo_root)
    if legacy_manifest_path is not None:
        rows.extend(
            discover_legacy_manifest_rows(
                legacy_manifest_path,
                repo_root=repo_root,
                evidence_root=evidence_root,
            )
        )
    rows.extend(_prepared_ood_rows(prepared_root, repo_root))
    if not rows:
        raise ValueError(
            "No reviewed OOD/OE evidence rows were discovered; materialization cannot pass with an empty source set."
        )
    _resolve_family_ids(rows)
    rows = _deduplicate_rows(rows, prepared_root)
    operations: list[CopyOperation] = []
    missing_sources: list[str] = []
    for row in rows:
        source = _source_path(row, repo_root=repo_root, evidence_root=evidence_root)
        destination = _destination_for(row, prepared_root)
        if not source.is_file():
            missing_sources.append(source.as_posix())
        elif sha256_file(source) != row["sha256"]:
            raise ValueError(f"Source hash does not match reviewed manifest: {source}")
        manifest_row = {
            key: row.get(key, "")
            for key in OUTPUT_FIELDS
        }
        manifest_row["relative_path"] = _repo_relative(destination, repo_root)
        if manifest_row["parent_relative_path"]:
            parent_source = _source_path(
                {"relative_path": manifest_row["parent_relative_path"]},
                repo_root=repo_root,
                evidence_root=evidence_root,
            )
            parent_row_path = _destination_for(
                {
                    **row,
                    "relative_path": manifest_row["parent_relative_path"],
                    "is_derived": "false",
                },
                prepared_root,
            )
            manifest_row["parent_relative_path"] = _repo_relative(parent_row_path, repo_root)
            if not parent_source.is_file() and fail_on_missing_source:
                missing_sources.append(parent_source.as_posix())
        operations.append(CopyOperation(source=source, destination=destination, row=manifest_row))
    if missing_sources and fail_on_missing_source:
        preview = ", ".join(sorted(set(missing_sources))[:5])
        raise FileNotFoundError(f"Missing {len(set(missing_sources))} reviewed evidence sources; first: {preview}")
    operation_rows = [operation.row for operation in operations]
    reviewed_id_paths = {
        row["relative_path"].replace("\\", "/").casefold()
        for row in operation_rows
        if row["role"] in {"id_train", "id_val", "id_test"}
    }
    id_rows = [
        row
        for row in _id_rows(prepared_root, repo_root)
        if row["relative_path"].replace("\\", "/").casefold() not in reviewed_id_paths
    ]
    manifest_rows = id_rows + operation_rows
    manifest_rows.sort(key=lambda row: (row["target"], row["role"], row["relative_path"]))
    return {
        "schema_version": MANIFEST_SCHEMA,
        "reviewed_row_count": len(rows),
        "copy_operation_count": len(operations),
        "missing_source_count": len(set(missing_sources)),
        "missing_sources": sorted(set(missing_sources)),
        "role_counts": dict(Counter(row["role"] for row in manifest_rows)),
        "manifest_rows": manifest_rows,
        "operations": operations,
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in OUTPUT_FIELDS} for row in rows)
    temporary.replace(path)


def materialize_plan(plan: Mapping[str, Any], *, prepared_root: Path, manifest_path: Path) -> dict[str, Any]:
    if int(plan.get("missing_source_count", 0)):
        raise FileNotFoundError("Materialization cannot write while reviewed evidence sources are missing.")
    operations = list(plan.get("operations") or [])
    copied = 0
    reused = 0
    split_entries: dict[str, dict[str, dict[str, Any]]] = {}
    for target in TARGET_ADAPTERS:
        split_path = prepared_root / target / "ood" / "ood_split_manifest.json"
        if not split_path.is_file():
            continue
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        entries = payload.get("entries") or {}
        if not isinstance(entries, Mapping):
            raise ValueError(f"{split_path} entries must be a JSON object")
        split_entries[target] = {
            str(relative).replace("\\", "/"): dict(entry)
            for relative, entry in entries.items()
            if isinstance(entry, Mapping)
        }
    for operation in operations:
        if not isinstance(operation, CopyOperation):
            raise TypeError("Materialization plan contains an invalid copy operation.")
        operation.destination.parent.mkdir(parents=True, exist_ok=True)
        if operation.destination.is_file() and sha256_file(operation.destination) == operation.row["sha256"]:
            reused += 1
        else:
            shutil.copy2(operation.source, operation.destination)
            if sha256_file(operation.destination) != operation.row["sha256"]:
                raise OSError(f"Copied evidence failed hash verification: {operation.destination}")
            copied += 1
        if operation.row["role"] in {"ood_dev", "ood_test"}:
            target = operation.row["target"]
            relative = operation.destination.relative_to(prepared_root / target / "ood").as_posix()
            split_entries.setdefault(target, {})[relative] = {
                "sha256": operation.row["sha256"],
                "slice": operation.row["disease_id"],
                "ood_type": operation.row["ood_type"],
                "evidence_family_id": operation.row["evidence_family_id"],
                "split": "dev" if operation.row["role"] == "ood_dev" else "test",
            }
    for target, entries in split_entries.items():
        split_path = prepared_root / target / "ood" / "ood_split_manifest.json"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "v2_family_frozen_ood_split_manifest",
            "assignment_policy": "reviewed_explicit_family_assignment",
            "entries": dict(sorted(entries.items())),
        }
        temporary = split_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(split_path)
    _write_csv(manifest_path, list(plan.get("manifest_rows") or []))
    return {
        "schema_version": MANIFEST_SCHEMA,
        "copied": copied,
        "reused": reused,
        "manifest_path": manifest_path.as_posix(),
        "manifest_row_count": len(list(plan.get("manifest_rows") or [])),
    }
