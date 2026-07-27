"""Content-addressed dataset staging contracts for immutable GitHub Releases.

Phase 4 intentionally stops before network publication.  This module owns the
local identity, audit, quarantine, lineage, safety, snapshot, and shard-plan
contracts that the Phase 5 publisher must consume without rebuilding them.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import stat
import warnings
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from PIL import Image, ImageOps, UnidentifiedImageError

DATASET_MANIFEST_SCHEMA = "v1_dataset_staging_snapshot"
QUARANTINE_MANIFEST_SCHEMA = "v1_dataset_quarantine_manifest"
RELEASE_MANIFEST_SCHEMA = "v1_dataset_github_release_candidate"
SHARD_PLAN_SCHEMA = "v1_dataset_zip_shard_plan"
OPERATION_BINDING_SCHEMA = "v1_dataset_snapshot_operation_binding"

IMAGE_IDENTITY_FIELDS = (
    "sample_id",
    "source_content_sha256",
    "distributed_content_sha256",
    "source_asset_id",
    "derived_from_sample_id",
    "target",
    "class_name",
    "evidence_role",
    "split",
    "source_uri",
    "license",
    "redistribution_allowed",
    "commercial_use_allowed",
    "license_evidence_uri",
    "license_reviewed_by",
    "license_reviewed_at",
    "privacy_review_status",
    "provenance",
    "review_status",
    "added_in_version",
    "removed_in_version",
    "removal_reason",
    "evaluation_cohort_id",
    "comparability_key",
)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_DATASET_MEMBER_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | {".csv", ".json"}
EXTENSION_FORMATS = {
    ".jpg": {"JPEG"},
    ".jpeg": {"JPEG"},
    ".png": {"PNG"},
    ".webp": {"WEBP"},
}
FORMAT_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
EVIDENCE_ROLES = {"id_train", "id_val", "id_test", "oe_train", "ood_dev", "ood_test"}
FROZEN_EVALUATION_ROLES = {"id_val", "id_test", "ood_dev", "ood_test"}
SPLIT_TO_ROLE = {
    "continual": "id_train",
    "train": "id_train",
    "val": "id_val",
    "test": "id_test",
    "oe": "oe_train",
    "ood_dev": "ood_dev",
    "ood_test": "ood_test",
    "ood": "ood_test",
}
UPLOADABLE_PRIVACY_STATES = {"approved", "complete", "no_private_metadata", "passed"}
UPLOADABLE_REVIEW_STATES = {"approved", "accepted"}
VERSION_CHANGE_LEVELS = {"patch": 0, "minor": 1, "major": 2}
PRIVATE_METADATA_KEYS = {"comment", "exif", "gps", "icc_profile", "photoshop", "xml", "xmp"}


@dataclass(frozen=True)
class DatasetResourceLimits:
    max_file_bytes: int = 100 * 1024 * 1024
    max_total_bytes: int = 20 * 1024 * 1024 * 1024
    max_width: int = 20_000
    max_height: int = 20_000
    max_pixels: int = 100_000_000
    max_file_count: int = 100_000
    max_archive_expansion_ratio: float = 100.0
    max_shard_bytes: int = 1_900_000_000


class DatasetContractError(ValueError):
    """Raised when a dataset manifest or input violates a fail-closed contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _io_path(path: Path) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _io_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DatasetContractError(f"Expected a JSON object: {path}")
    return payload


def sanitize_source_uri(value: str) -> str:
    """Remove credentials, query strings, and fragments before persistence/logging."""
    value = str(value or "").strip()
    if not value:
        return ""
    parts = urlsplit(value)
    if not parts.scheme or not parts.netloc:
        return value.split("?", 1)[0].split("#", 1)[0]
    hostname = parts.hostname or ""
    if parts.port:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme.lower(), hostname, parts.path, "", ""))


def _normalized_relative_path(value: str) -> str:
    raw = str(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DatasetContractError(f"Unsafe relative path: {value!r}")
    if ":" in path.parts[0]:
        raise DatasetContractError(f"Drive-qualified path is not allowed: {value!r}")
    return path.as_posix()


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def resolve_contained_file(root: Path, relative_path: str) -> Path:
    relative_path = _normalized_relative_path(relative_path)
    root = root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    current = root
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        if current.exists() and _is_reparse_point(current):
            raise DatasetContractError(f"Symlink/reparse-point path is not allowed: {relative_path}")
    resolved = candidate.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise DatasetContractError(f"Path escapes the intended root: {relative_path}")
    if not resolved.is_file():
        raise DatasetContractError(f"Expected a regular file: {relative_path}")
    return resolved


def validate_archive(path: Path, limits: DatasetResourceLimits | None = None) -> dict[str, Any]:
    """Validate ZIP members without extracting attacker-controlled content."""
    limits = limits or DatasetResourceLimits()
    compressed_total = 0
    expanded_total = 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > limits.max_file_count:
            raise DatasetContractError("Archive file count exceeds the configured limit")
        seen: set[str] = set()
        for member in members:
            member_path = _normalized_relative_path(member.filename.rstrip("/"))
            if member_path in seen:
                raise DatasetContractError(f"Duplicate archive member: {member_path}")
            seen.add(member_path)
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise DatasetContractError(f"Archive symlink member is not allowed: {member_path}")
            if member.is_dir():
                continue
            if Path(member_path).suffix.lower() not in ALLOWED_DATASET_MEMBER_EXTENSIONS:
                raise DatasetContractError(f"Archive member extension is not allowed: {member_path}")
            if member.file_size > limits.max_file_bytes:
                raise DatasetContractError(f"Archive member exceeds the file-size limit: {member_path}")
            compressed_total += member.compress_size
            expanded_total += member.file_size
            if expanded_total > limits.max_total_bytes:
                raise DatasetContractError("Archive expanded size exceeds the configured limit")
        ratio = expanded_total / max(compressed_total, 1)
        if ratio > limits.max_archive_expansion_ratio:
            raise DatasetContractError("Archive expansion ratio exceeds the configured limit")
    return {"member_count": len(members), "expanded_bytes": expanded_total, "expansion_ratio": ratio}


def inspect_image(
    path: Path,
    limits: DatasetResourceLimits | None = None,
    *,
    allow_extension_mismatch: bool = False,
) -> dict[str, Any]:
    limits = limits or DatasetResourceLimits()
    extension = path.suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise DatasetContractError(f"Image extension is not allowed: {extension}")
    io_path = _io_path(path)
    size_bytes = io_path.stat().st_size
    if size_bytes <= 0 or size_bytes > limits.max_file_bytes:
        raise DatasetContractError(f"Image size is outside the configured bounds: {size_bytes}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io_path) as image:
                image.verify()
            with Image.open(io_path) as image:
                image.load()
                image_format = str(image.format or "").upper()
                width, height = image.size
                metadata_keys = sorted(str(key) for key in image.info)
                exif = image.getexif()
                exif_keys = sorted(str(key) for key in exif.keys()) if exif else []
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, UnidentifiedImageError, OSError) as exc:
        raise DatasetContractError(f"Unreadable or unsafe image {path.name}: {exc}") from exc
    normalized_format = "JPEG" if image_format == "MPO" else image_format
    if normalized_format not in FORMAT_MIME:
        raise DatasetContractError(f"Unsupported decoded image format: {image_format}")
    if normalized_format not in EXTENSION_FORMATS[extension] and not allow_extension_mismatch:
        raise DatasetContractError(f"Extension/MIME mismatch for {path.name}: {extension} vs {image_format}")
    if width <= 0 or height <= 0 or width > limits.max_width or height > limits.max_height:
        raise DatasetContractError(f"Image dimensions exceed the configured bounds: {width}x{height}")
    if width * height > limits.max_pixels:
        raise DatasetContractError(f"Image pixel count exceeds the configured limit: {width * height}")
    guessed_mime = mimetypes.guess_type(path.name)[0]
    expected_mime = FORMAT_MIME[normalized_format]
    if guessed_mime and guessed_mime != expected_mime and not allow_extension_mismatch:
        raise DatasetContractError(f"Filename MIME does not match decoded image: {guessed_mime} vs {expected_mime}")
    return {
        "size_bytes": size_bytes,
        "width": width,
        "height": height,
        "pixel_count": width * height,
        "format": image_format,
        "normalized_format": normalized_format,
        "mime_type": expected_mime,
        "metadata_keys": metadata_keys,
        "exif_keys": exif_keys,
    }


def stage_sanitized_image(source: Path, destination: Path) -> dict[str, Any]:
    """Write a deterministic metadata-free distributed image into a staging root."""
    _io_path(destination.parent).mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(source) as image:
            clean = ImageOps.exif_transpose(image)
            clean.load()
            clean.info.clear()
            target_format = {
                ".jpg": "JPEG",
                ".jpeg": "JPEG",
                ".png": "PNG",
                ".webp": "WEBP",
            }.get(destination.suffix.lower())
            if not target_format:
                raise DatasetContractError(f"Unsupported distributed image extension: {destination.suffix}")
            save_options: dict[str, Any] = {}
            if target_format == "JPEG":
                clean = clean.convert("RGB")
                save_options = {"quality": 95, "subsampling": 0, "optimize": False, "progressive": False}
            elif target_format == "PNG":
                save_options = {"compress_level": 6, "optimize": False}
            elif target_format == "WEBP":
                save_options = {"quality": 95, "method": 6, "exact": True}
            clean.save(_io_path(destination), format=target_format, **save_options)
    inspected = inspect_image(destination)
    private_keys = {key.lower() for key in inspected["metadata_keys"]} & PRIVATE_METADATA_KEYS
    if private_keys or inspected["exif_keys"]:
        raise DatasetContractError(f"Distributed image still contains private metadata: {destination}")
    inspected["sha256"] = sha256_file(destination)
    return inspected


def derive_sample_id(source_asset_id: str, source_content_sha256: str) -> str:
    if not source_asset_id or len(source_content_sha256) != 64:
        raise DatasetContractError("sample_id requires source_asset_id and a valid source SHA-256")
    return "sample-" + canonical_sha256({"source_asset_id": source_asset_id, "sha256": source_content_sha256})[:32]


def required_version_bump(change_type: str) -> str:
    if change_type not in VERSION_CHANGE_LEVELS:
        raise DatasetContractError(f"Unknown dataset change type: {change_type}")
    return change_type


def validate_version_change(previous: str, current: str, change_type: str) -> None:
    def parse(value: str) -> tuple[int, int, int]:
        pieces = value.removeprefix("v").split(".")
        if len(pieces) != 3 or any(not piece.isdigit() for piece in pieces):
            raise DatasetContractError(f"Dataset version must be semantic x.y.z: {value}")
        return tuple(int(piece) for piece in pieces)  # type: ignore[return-value]

    old = parse(previous)
    new = parse(current)
    required = VERSION_CHANGE_LEVELS[required_version_bump(change_type)]
    actual = 2 if new[0] > old[0] else 1 if new[:2] > old[:2] else 0 if new > old else -1
    if actual < required:
        raise DatasetContractError(f"{change_type} changes require at least a {change_type} version bump")


def validate_frozen_evaluation_assignments(
    previous_records: Iterable[Mapping[str, Any]], current_records: Iterable[Mapping[str, Any]]
) -> None:
    previous = {str(row.get("sample_id")): row for row in previous_records if row.get("sample_id")}
    current = {str(row.get("sample_id")): row for row in current_records if row.get("sample_id")}
    for sample_id, old in previous.items():
        if old.get("evidence_role") not in FROZEN_EVALUATION_ROLES:
            continue
        new = current.get(sample_id)
        changed = new is None or any(new.get(field) != old.get(field) for field in ("evidence_role", "split", "class_name"))
        if not changed:
            continue
        if new is None:
            if not old.get("removed_in_version") or not old.get("removal_reason"):
                raise DatasetContractError(f"Frozen evaluation sample removal lacks a tombstone: {sample_id}")
            continue
        if (
            not new.get("evaluation_cohort_id")
            or not new.get("comparability_key")
            or new.get("evaluation_cohort_id") == old.get("evaluation_cohort_id")
            or new.get("comparability_key") == old.get("comparability_key")
        ):
            raise DatasetContractError(f"Frozen evaluation change requires a new non-comparable cohort: {sample_id}")


def _metadata_for(metadata: Mapping[str, Any], relative_path: str) -> dict[str, Any]:
    value = metadata.get(relative_path, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _record_quarantine_reasons(meta: Mapping[str, Any], inspection_error: str) -> list[str]:
    reasons: list[str] = []
    if inspection_error:
        reasons.append("image_validation_failed")
    for field in ("source_uri", "license", "license_evidence_uri", "license_reviewed_by", "license_reviewed_at", "provenance"):
        if not str(meta.get(field) or "").strip():
            reasons.append(f"missing_{field}")
    if not bool(meta.get("redistribution_allowed")):
        reasons.append("redistribution_not_approved")
    if not bool(meta.get("commercial_use_allowed")):
        reasons.append("commercial_use_not_approved")
    if str(meta.get("privacy_review_status") or "").lower() not in UPLOADABLE_PRIVACY_STATES:
        reasons.append("privacy_review_incomplete")
    if str(meta.get("review_status") or "").lower() not in UPLOADABLE_REVIEW_STATES:
        reasons.append("record_review_incomplete")
    return sorted(set(reasons))


def _infer_dataset_fields(relative_path: str) -> tuple[str, str, str]:
    parts = PurePosixPath(relative_path).parts
    split = parts[0].lower() if parts else ""
    role = SPLIT_TO_ROLE.get(split, "")
    class_name = parts[1] if len(parts) > 2 and split not in {"ood", "ood_dev", "ood_test", "oe"} else ""
    return split, role, class_name


def build_dataset_snapshot(
    roots: Sequence[Path],
    *,
    repo_root: Path,
    metadata: Mapping[str, Any] | None,
    staging_root: Path,
    dataset_version: str,
    inventory_cutoff: str,
    limits: DatasetResourceLimits | None = None,
) -> dict[str, Any]:
    """Audit immutable input inventory and stage only explicitly approved images."""
    limits = limits or DatasetResourceLimits()
    repo_root = repo_root.resolve(strict=True)
    staging_root = staging_root.resolve()
    if staging_root != repo_root and repo_root not in staging_root.parents:
        raise DatasetContractError(f"Staging root is outside the repository: {staging_root}")
    if staging_root.exists() and _is_reparse_point(staging_root):
        raise DatasetContractError(f"Staging root cannot be a symlink/reparse point: {staging_root}")
    metadata = metadata or {}
    records: list[dict[str, Any]] = []
    total_bytes = 0
    discovered: list[tuple[str, str, Path]] = []
    for root_value in roots:
        root = root_value.resolve(strict=True)
        if root != repo_root and repo_root not in root.parents:
            raise DatasetContractError(f"Dataset root is outside the repository: {root}")
        if _is_reparse_point(root):
            raise DatasetContractError(f"Dataset root cannot be a symlink/reparse point: {root}")
        target = root.name
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                continue
            relative = path.relative_to(root).as_posix()
            resolve_contained_file(root, relative)
            discovered.append((target, relative, path))
    if len(discovered) > limits.max_file_count:
        raise DatasetContractError("Dataset file count exceeds the configured limit")

    for target, relative, source in discovered:
        total_bytes += source.stat().st_size
        if total_bytes > limits.max_total_bytes:
            raise DatasetContractError("Dataset total bytes exceed the configured limit")
        metadata_key = f"{target}/{relative}"
        meta = _metadata_for(metadata, metadata_key)
        inspection_error = ""
        inspection: dict[str, Any] = {}
        normalization_reason = ""
        try:
            inspection = inspect_image(source, limits)
        except DatasetContractError as exc:
            if str(exc).startswith("Extension/MIME mismatch") or str(exc).startswith("Filename MIME does not match"):
                try:
                    inspection = inspect_image(source, limits, allow_extension_mismatch=True)
                    normalization_reason = "extension_mime_mismatch"
                except DatasetContractError as normalized_exc:
                    inspection_error = str(normalized_exc)
            else:
                inspection_error = str(exc)
        source_hash = sha256_file(source)
        source_asset_id = str(meta.get("source_asset_id") or f"repo:{metadata_key}")
        split, role, inferred_class = _infer_dataset_fields(relative)
        reasons = _record_quarantine_reasons(meta, inspection_error)
        sample_id = derive_sample_id(source_asset_id, source_hash)
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "source_content_sha256": source_hash,
            "distributed_content_sha256": None,
            "source_asset_id": source_asset_id,
            "derived_from_sample_id": str(meta.get("derived_from_sample_id") or ""),
            "target": target,
            "class_name": str(meta.get("class_name") or inferred_class),
            "evidence_role": str(meta.get("evidence_role") or role),
            "split": str(meta.get("split") or split),
            "source_uri": sanitize_source_uri(str(meta.get("source_uri") or "")),
            "license": str(meta.get("license") or ""),
            "redistribution_allowed": bool(meta.get("redistribution_allowed")),
            "commercial_use_allowed": bool(meta.get("commercial_use_allowed")),
            "license_evidence_uri": sanitize_source_uri(str(meta.get("license_evidence_uri") or "")),
            "license_reviewed_by": str(meta.get("license_reviewed_by") or ""),
            "license_reviewed_at": str(meta.get("license_reviewed_at") or ""),
            "privacy_review_status": str(meta.get("privacy_review_status") or ""),
            "provenance": str(meta.get("provenance") or ""),
            "review_status": str(meta.get("review_status") or ""),
            "added_in_version": str(meta.get("added_in_version") or dataset_version),
            "removed_in_version": str(meta.get("removed_in_version") or ""),
            "removal_reason": str(meta.get("removal_reason") or ""),
            "evaluation_cohort_id": str(meta.get("evaluation_cohort_id") or ""),
            "comparability_key": str(meta.get("comparability_key") or ""),
            "source_path": source.relative_to(repo_root).as_posix(),
            "distributed_path": None,
            "candidate_distributed_path": metadata_key,
            "source_size_bytes": source.stat().st_size,
            "distributed_size_bytes": None,
            "image": inspection,
            "inspection_error": inspection_error,
            "source_normalization_required": bool(normalization_reason),
            "source_normalization_reason": normalization_reason,
            "disposition": "quarantine" if reasons else "uploadable",
            "quarantine_reasons": reasons,
        }
        missing_identity = [field for field in IMAGE_IDENTITY_FIELDS if field not in record]
        if missing_identity:
            raise AssertionError(f"Internal record schema omission: {missing_identity}")
        records.append(record)

    records.sort(key=lambda row: (row["target"], row["source_path"], row["sample_id"]))
    records_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_hash[str(record["source_content_sha256"])].append(record)
    for same_hash in records_by_hash.values():
        if len(same_hash) < 2:
            continue
        roles = {str(record.get("evidence_role") or "") for record in same_hash}
        groups = set()
        if roles & {"id_train", "id_val", "id_test"}:
            groups.add("id")
        if "oe_train" in roles:
            groups.add("oe")
        if "ood_dev" in roles:
            groups.add("ood_dev")
        if "ood_test" in roles:
            groups.add("ood_test")
        for record in same_hash:
            record["quarantine_reasons"] = sorted(set(record["quarantine_reasons"]) | {"duplicate_content_hash"})
            if len(groups) > 1:
                record["quarantine_reasons"] = sorted(
                    set(record["quarantine_reasons"]) | {"evidence_role_hash_overlap"}
                )
            record["disposition"] = "quarantine"

    for record in records:
        candidate_path = str(record.pop("candidate_distributed_path"))
        if record["disposition"] != "uploadable":
            continue
        source = resolve_contained_file(repo_root, str(record["source_path"]))
        destination = staging_root / candidate_path
        distributed_inspection = stage_sanitized_image(source, destination)
        record["distributed_content_sha256"] = str(distributed_inspection["sha256"])
        record["distributed_path"] = destination.relative_to(staging_root).as_posix()
        record["distributed_size_bytes"] = distributed_inspection["size_bytes"]
    records_by_distributed_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        distributed_hash = str(record.get("distributed_content_sha256") or "")
        if record["disposition"] == "uploadable" and distributed_hash:
            records_by_distributed_hash[distributed_hash].append(record)
    for same_hash in records_by_distributed_hash.values():
        if len(same_hash) < 2:
            continue
        for record in same_hash:
            record["quarantine_reasons"] = sorted(
                set(record["quarantine_reasons"]) | {"duplicate_distributed_content_hash"}
            )
            record["disposition"] = "quarantine"
    identity_payload = {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "dataset_version": dataset_version,
        "inventory_cutoff": inventory_cutoff,
        "records": records,
    }
    manifest_sha = canonical_sha256(identity_payload)
    snapshot_id = f"dataset-snapshot-{manifest_sha[:20]}"
    return {
        **identity_payload,
        "staging_snapshot_id": snapshot_id,
        "manifest_sha256": manifest_sha,
        "created_at": utc_now(),
        "source_roots": [root.resolve().relative_to(repo_root).as_posix() for root in roots],
        "staging_root": staging_root.relative_to(repo_root).as_posix()
        if staging_root == repo_root or repo_root in staging_root.parents
        else str(staging_root),
        "record_count": len(records),
        "source_total_bytes": total_bytes,
    }


def validate_snapshot_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != DATASET_MANIFEST_SCHEMA:
        raise DatasetContractError("Unexpected dataset snapshot schema")
    records = payload.get("records")
    if not isinstance(records, list):
        raise DatasetContractError("Snapshot records must be a list")
    for record in records:
        if not isinstance(record, Mapping):
            raise DatasetContractError("Every snapshot record must be an object")
        missing = [field for field in IMAGE_IDENTITY_FIELDS if field not in record]
        if missing:
            raise DatasetContractError(f"Snapshot record is missing identity fields: {missing}")
        _normalized_relative_path(str(record.get("source_path") or ""))
        if record.get("distributed_path"):
            _normalized_relative_path(str(record["distributed_path"]))
        if record.get("evidence_role") and record.get("evidence_role") not in EVIDENCE_ROLES:
            raise DatasetContractError(f"Unknown evidence role: {record.get('evidence_role')}")
    identity = {
        "schema_version": payload["schema_version"],
        "dataset_version": payload.get("dataset_version"),
        "inventory_cutoff": payload.get("inventory_cutoff"),
        "records": records,
    }
    expected = canonical_sha256(identity)
    if payload.get("manifest_sha256") != expected:
        raise DatasetContractError("Snapshot manifest SHA-256 does not match its canonical content")
    if payload.get("staging_snapshot_id") != f"dataset-snapshot-{expected[:20]}":
        raise DatasetContractError("Snapshot ID is not bound to the manifest SHA-256")


def build_snapshot_reports(snapshot: Mapping[str, Any], *, dataset_tag: str) -> dict[str, dict[str, Any]]:
    validate_snapshot_manifest(snapshot)
    records = list(snapshot["records"])
    by_source_hash: dict[str, list[str]] = defaultdict(list)
    by_distributed_hash: dict[str, list[str]] = defaultdict(list)
    by_hash_roles: dict[str, set[str]] = defaultdict(set)
    for row in records:
        by_source_hash[str(row["source_content_sha256"])].append(str(row["sample_id"]))
        if row.get("distributed_content_sha256"):
            by_distributed_hash[str(row["distributed_content_sha256"])].append(str(row["sample_id"]))
        if row.get("evidence_role"):
            by_hash_roles[str(row["source_content_sha256"])].add(str(row["evidence_role"]))
    overlaps = []
    for digest, roles in sorted(by_hash_roles.items()):
        groups = []
        if roles & {"id_train", "id_val", "id_test"}:
            groups.append("id")
        if "oe_train" in roles:
            groups.append("oe")
        if "ood_dev" in roles:
            groups.append("ood_dev")
        if "ood_test" in roles:
            groups.append("ood_test")
        if len(groups) > 1:
            overlaps.append({"sha256": digest, "groups": groups, "roles": sorted(roles)})
    quarantined = [row for row in records if row.get("disposition") == "quarantine"]
    uploadable = [row for row in records if row.get("disposition") == "uploadable"]
    quarantine_manifest = {
        "schema_version": QUARANTINE_MANIFEST_SCHEMA,
        "staging_snapshot_id": snapshot["staging_snapshot_id"],
        "manifest_sha256": snapshot["manifest_sha256"],
        "physical_action": "none",
        "records": [
            {
                "sample_id": row["sample_id"],
                "source_path": row["source_path"],
                "source_content_sha256": row["source_content_sha256"],
                "reasons": row["quarantine_reasons"],
            }
            for row in quarantined
        ],
    }
    release_manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA,
        "dataset_version": snapshot["dataset_version"],
        "dataset_tag": dataset_tag,
        "staging_snapshot_id": snapshot["staging_snapshot_id"],
        "snapshot_manifest_sha256": snapshot["manifest_sha256"],
        "inventory_cutoff": snapshot["inventory_cutoff"],
        "release_state": "revoked" if snapshot.get("revoked") else "candidate",
        "revocation_reason": str(snapshot.get("revocation_reason") or ""),
        "tombstones": [
            {
                "sample_id": row["sample_id"],
                "removed_in_version": row["removed_in_version"],
                "removal_reason": row["removal_reason"],
            }
            for row in records
            if row.get("removed_in_version")
        ],
        "files": [
            {
                "sample_id": row["sample_id"],
                "distributed_path": row["distributed_path"],
                "distributed_content_sha256": row["distributed_content_sha256"],
                "size_bytes": row["distributed_size_bytes"],
                "source_content_sha256": row["source_content_sha256"],
                "target": row["target"],
                "class_name": row["class_name"],
                "evidence_role": row["evidence_role"],
                "split": row["split"],
                "evaluation_cohort_id": row["evaluation_cohort_id"],
                "comparability_key": row["comparability_key"],
            }
            for row in uploadable
        ],
    }
    audit_report = {
        "schema_version": "v1_dataset_snapshot_audit",
        "staging_snapshot_id": snapshot["staging_snapshot_id"],
        "manifest_sha256": snapshot["manifest_sha256"],
        "record_count": len(records),
        "uploadable_count": len(uploadable),
        "quarantined_count": len(quarantined),
        "duplicate_source_hashes": [
            {"sha256": digest, "sample_ids": ids}
            for digest, ids in sorted(by_source_hash.items())
            if len(ids) > 1
        ],
        "duplicate_distributed_hashes": [
            {"sha256": digest, "sample_ids": ids}
            for digest, ids in sorted(by_distributed_hash.items())
            if len(ids) > 1
        ],
        "evidence_role_overlaps": overlaps,
        "quarantine_reason_counts": _count_reasons(quarantined),
        "normalized_source_count": sum(bool(row.get("source_normalization_required")) for row in records),
        "source_total_bytes": snapshot.get("source_total_bytes", 0),
    }
    return {"audit": audit_report, "quarantine": quarantine_manifest, "release": release_manifest}


def _count_reasons(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in records:
        for reason in row.get("quarantine_reasons", []):
            counts[str(reason)] += 1
    return dict(sorted(counts.items()))


def build_shard_plan(
    release_manifest: Mapping[str, Any], limits: DatasetResourceLimits | None = None
) -> dict[str, Any]:
    limits = limits or DatasetResourceLimits()
    validate_release_candidate(release_manifest)
    files = release_manifest["files"]
    shards: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    ordered = sorted(files, key=lambda row: (str(row.get("distributed_path")), str(row.get("sample_id"))))
    for record in ordered:
        size = int(record.get("size_bytes") or 0)
        if size <= 0 or size > limits.max_shard_bytes:
            raise DatasetContractError(f"Invalid shard member size: {record.get('distributed_path')}")
        path_bytes = len(str(record.get("distributed_path") or "").encode("utf-8"))
        planned_bytes = size + 512 + 2 * path_bytes
        if planned_bytes > limits.max_shard_bytes:
            raise DatasetContractError(f"Shard member plus ZIP overhead exceeds the limit: {record.get('distributed_path')}")
        if current and current_bytes + planned_bytes > limits.max_shard_bytes:
            shards.append(_finish_shard(release_manifest, len(shards) + 1, current, current_bytes))
            current = []
            current_bytes = 0
        current.append(dict(record))
        current_bytes += planned_bytes
    if current:
        shards.append(_finish_shard(release_manifest, len(shards) + 1, current, current_bytes))
    plan = {
        "schema_version": SHARD_PLAN_SCHEMA,
        "dataset_tag": release_manifest.get("dataset_tag"),
        "staging_snapshot_id": release_manifest.get("staging_snapshot_id"),
        "snapshot_manifest_sha256": release_manifest.get("snapshot_manifest_sha256"),
        "archive_format": "zip",
        "compression": "stored",
        "max_shard_bytes": limits.max_shard_bytes,
        "shards": shards,
        "file_count": len(ordered),
        "total_bytes": sum(int(record.get("size_bytes") or 0) for record in ordered),
    }
    plan["shard_plan_sha256"] = canonical_sha256(plan)
    return plan


def build_runtime_parity_candidate(
    root: Path,
    *,
    dataset_version: str,
    dataset_tag: str,
    inventory_cutoff: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Inventory an exact runtime tree for immutable Release-backed parity."""
    root = root.resolve(strict=True)
    limits = DatasetResourceLimits()
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        relative_path = _normalized_relative_path(path.relative_to(root).as_posix())
        if path.suffix.lower() not in ALLOWED_DATASET_MEMBER_EXTENSIONS:
            raise DatasetContractError(f"Runtime parity member extension is not allowed: {relative_path}")
        resolved = resolve_contained_file(root, relative_path)
        size_bytes = resolved.stat().st_size
        if size_bytes <= 0 or size_bytes > limits.max_file_bytes:
            raise DatasetContractError(f"Runtime parity member size is outside the configured bounds: {relative_path}")
        digest = sha256_file(resolved)
        records.append(
            {
                "sample_id": "runtime-" + hashlib.sha256(relative_path.encode("utf-8")).hexdigest(),
                "distributed_path": relative_path,
                "distributed_content_sha256": digest,
                "source_content_sha256": digest,
                "size_bytes": size_bytes,
            }
        )
        total_bytes += size_bytes
        if len(records) > limits.max_file_count or total_bytes > limits.max_total_bytes:
            raise DatasetContractError("Runtime parity inventory exceeds the configured resource limits")
    if not records:
        raise DatasetContractError("Runtime parity candidate cannot be empty")
    inventory = {
        "candidate_profile": "runtime_parity",
        "dataset_version": dataset_version,
        "dataset_tag": dataset_tag,
        "inventory_cutoff": inventory_cutoff,
        "files": records,
    }
    snapshot_digest = canonical_sha256(inventory)
    candidate = {
        "schema_version": RELEASE_MANIFEST_SCHEMA,
        **inventory,
        "staging_snapshot_id": "dataset-snapshot-" + snapshot_digest[:20],
        "snapshot_manifest_sha256": snapshot_digest,
        "release_state": "candidate",
        "revocation_reason": "",
        "tombstones": [],
        "file_count": len(records),
        "distributed_bytes": total_bytes,
    }
    validate_release_candidate(candidate)
    return candidate, build_shard_plan(candidate)


def validate_release_candidate(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RELEASE_MANIFEST_SCHEMA:
        raise DatasetContractError("Invalid dataset release candidate manifest")
    if not re.fullmatch(r"aads-dataset-v\d+\.\d+\.\d+", str(payload.get("dataset_tag") or "")):
        raise DatasetContractError("Dataset release tag must use the aads-dataset-v<semver> namespace")
    snapshot_id = str(payload.get("staging_snapshot_id") or "")
    digest = str(payload.get("snapshot_manifest_sha256") or "")
    if not snapshot_id.startswith("dataset-snapshot-") or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise DatasetContractError("Release candidate is missing a valid snapshot identity")
    state = payload.get("release_state")
    if state not in {"candidate", "revoked"}:
        raise DatasetContractError("Dataset release state must be candidate or revoked")
    if state == "revoked":
        if not str(payload.get("revocation_reason") or "").strip():
            raise DatasetContractError("Revoked dataset release requires a reason")
        raise DatasetContractError("Revoked dataset releases cannot be sharded or published")
    files = payload.get("files")
    if not isinstance(files, list):
        raise DatasetContractError("Release candidate files must be a list")
    seen_sample_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for record in files:
        if not isinstance(record, Mapping):
            raise DatasetContractError("Every release file must be an object")
        sample_id = str(record.get("sample_id") or "")
        path = _normalized_relative_path(str(record.get("distributed_path") or ""))
        file_hash = str(record.get("distributed_content_sha256") or "")
        source_hash = str(record.get("source_content_sha256") or "")
        if not sample_id or len(file_hash) != 64 or len(source_hash) != 64:
            raise DatasetContractError(f"Release file identity is incomplete: {path}")
        if int(record.get("size_bytes") or 0) <= 0:
            raise DatasetContractError(f"Release file size must be positive: {path}")
        duplicate_content = file_hash in seen_hashes and payload.get("candidate_profile") != "runtime_parity"
        if sample_id in seen_sample_ids or path in seen_paths or duplicate_content:
            raise DatasetContractError(f"Release candidate contains duplicate identity/content: {path}")
        seen_sample_ids.add(sample_id)
        seen_paths.add(path)
        seen_hashes.add(file_hash)


def _finish_shard(
    release_manifest: Mapping[str, Any], index: int, records: list[dict[str, Any]], size_bytes: int
) -> dict[str, Any]:
    tag = str(release_manifest.get("dataset_tag") or "aads-dataset-candidate")
    members = [
        {
            "path": record["distributed_path"],
            "sha256": record["distributed_content_sha256"],
            "size_bytes": record["size_bytes"],
            "sample_id": record["sample_id"],
        }
        for record in records
    ]
    return {
        "asset_name": f"{tag}-shard-{index:04d}.zip",
        "size_bytes": size_bytes,
        "content_bytes": sum(int(record["size_bytes"]) for record in records),
        "member_count": len(members),
        "members": members,
        "members_sha256": canonical_sha256(members),
    }


def bind_snapshot_operation(operation: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    if operation not in {"audit", "diff", "publish", "verify"}:
        raise DatasetContractError(f"Unsupported snapshot operation: {operation}")
    validate_snapshot_manifest(snapshot)
    return {
        "schema_version": OPERATION_BINDING_SCHEMA,
        "operation": operation,
        "staging_snapshot_id": snapshot["staging_snapshot_id"],
        "manifest_sha256": snapshot["manifest_sha256"],
    }


def verify_snapshot_files(snapshot: Mapping[str, Any], *, repo_root: Path, staging_root: Path) -> dict[str, Any]:
    validate_snapshot_manifest(snapshot)
    checked_source = 0
    checked_distributed = 0
    for row in snapshot["records"]:
        source = resolve_contained_file(repo_root, str(row["source_path"]))
        if sha256_file(source) != row["source_content_sha256"]:
            raise DatasetContractError(f"Source changed after snapshot: {row['source_path']}")
        checked_source += 1
        if row.get("distributed_path"):
            distributed = resolve_contained_file(staging_root, str(row["distributed_path"]))
            if sha256_file(distributed) != row["distributed_content_sha256"]:
                raise DatasetContractError(f"Distributed image changed after snapshot: {row['distributed_path']}")
            inspection = inspect_image(distributed)
            private_keys = {key.lower() for key in inspection["metadata_keys"]} & PRIVATE_METADATA_KEYS
            if private_keys or inspection["exif_keys"]:
                raise DatasetContractError(f"Distributed image contains metadata: {row['distributed_path']}")
            checked_distributed += 1
    return {
        "verified": True,
        "staging_snapshot_id": snapshot["staging_snapshot_id"],
        "manifest_sha256": snapshot["manifest_sha256"],
        "source_files_checked": checked_source,
        "distributed_files_checked": checked_distributed,
    }


def diff_snapshot_inventory(snapshot: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Show post-cutoff arrivals/removals/changes without mutating the frozen snapshot."""
    validate_snapshot_manifest(snapshot)
    expected = {str(row["source_path"]): str(row["source_content_sha256"]) for row in snapshot["records"]}
    actual: dict[str, str] = {}
    for root_value in snapshot.get("source_roots", []):
        root_relative = _normalized_relative_path(str(root_value))
        root = resolve_contained_directory(repo_root, root_relative)
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                key = path.relative_to(repo_root).as_posix()
                actual[key] = sha256_file(path)
    return {
        "staging_snapshot_id": snapshot["staging_snapshot_id"],
        "manifest_sha256": snapshot["manifest_sha256"],
        "added": sorted(set(actual) - set(expected)),
        "removed": sorted(set(expected) - set(actual)),
        "changed": sorted(path for path in set(expected) & set(actual) if expected[path] != actual[path]),
    }


def resolve_contained_directory(root: Path, relative_path: str) -> Path:
    relative_path = _normalized_relative_path(relative_path)
    root = root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    current = root
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        if current.exists() and _is_reparse_point(current):
            raise DatasetContractError(f"Symlink/reparse-point path is not allowed: {relative_path}")
    resolved = candidate.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_dir():
        raise DatasetContractError(f"Directory escapes the intended root: {relative_path}")
    return resolved


def public_contract_summary() -> dict[str, Any]:
    return {
        "schemas": {
            "staging": DATASET_MANIFEST_SCHEMA,
            "quarantine": QUARANTINE_MANIFEST_SCHEMA,
            "release": RELEASE_MANIFEST_SCHEMA,
            "shard_plan": SHARD_PLAN_SCHEMA,
        },
        "required_image_identity": list(IMAGE_IDENTITY_FIELDS),
        "versioning": {
            "metadata_or_provenance_only": "patch",
            "image_label_split_or_evidence_role": "minor",
            "taxonomy_or_schema": "major",
        },
        "frozen_evaluation_roles": sorted(FROZEN_EVALUATION_ROLES),
        "selection_forbidden_roles": ["id_test", "ood_test"],
        "release_tag_namespace": "aads-dataset-v<major>.<minor>.<patch>",
        "publication_backend": "private immutable GitHub Releases",
        "git_lfs": False,
        "physical_quarantine_in_phase_4": False,
    }
