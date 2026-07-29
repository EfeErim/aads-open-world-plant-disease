"""Materialize an accepted grouped split into the runtime dataset layout."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.data.dataset_layout import IMAGE_EXTENSIONS, normalize_class_name
from src.guided_artifacts import refresh_prep_guided_artifacts
from src.shared.json_utils import read_json, write_json

DEFAULT_RUNTIME_ROOT = Path("data") / "prepared_runtime_datasets"


GROUPED_SPLIT_POLICY = "grouped_family_canonical_eval_60_20_20"


def build_prepared_dataset_key(crop_name: str, part_name: str = "unspecified") -> str:
    crop_key = normalize_class_name(crop_name) or "crop"
    part_key = normalize_class_name(part_name)
    if not part_key or part_key == "unspecified":
        return crop_key
    return f"{crop_key}__{part_key}"


def _fingerprint_paths(paths: Iterable[Path], *, root: Path) -> str:
    digest = hashlib.sha1()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def materialize_grouped_runtime_dataset(
    *,
    class_root: Path,
    crop_name: str,
    part_name: str = "unspecified",
    artifact_root: Path,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    ood_root: Optional[Path] = None,
    oe_root: Optional[Path] = None,
    materialization_strategy: str = "auto",
) -> Path:
    manifest = read_json(artifact_root / "proposed_split_manifest.json", default={}, expect_type=dict)
    if not isinstance(manifest, dict):
        raise RuntimeError("Grouped split manifest is missing or invalid.")
    if manifest.get("blocking_issues"):
        raise RuntimeError("Grouped split manifest contains blocking issues. Resolve them before materializing.")
    dataset_key = build_prepared_dataset_key(crop_name, part_name)
    crop_root = Path(runtime_root) / dataset_key
    resolved_ood_root = Path(ood_root) if ood_root is not None else None
    resolved_oe_root = Path(oe_root) if oe_root is not None else None
    ood_manifest: Optional[Dict[str, Any]] = None
    oe_manifest: Optional[Dict[str, Any]] = None
    ood_images: List[Path] = []
    oe_images: List[Path] = []
    if resolved_ood_root is not None:
        if not resolved_ood_root.exists():
            raise FileNotFoundError(f"OOD root not found: {resolved_ood_root}")
        if not resolved_ood_root.is_dir():
            raise NotADirectoryError(f"OOD root is not a directory: {resolved_ood_root}")
        ood_images = sorted(
            [
                path
                for path in resolved_ood_root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda path: str(path).lower(),
        )
        ood_manifest = {
            "source_root": str(resolved_ood_root.resolve()),
            "image_count": len(ood_images),
            "image_fingerprint": _fingerprint_paths(ood_images, root=resolved_ood_root),
        }
    if resolved_oe_root is not None:
        if not resolved_oe_root.exists():
            raise FileNotFoundError(f"OE root not found: {resolved_oe_root}")
        if not resolved_oe_root.is_dir():
            raise NotADirectoryError(f"OE root is not a directory: {resolved_oe_root}")
        oe_images = sorted(
            [
                path
                for path in resolved_oe_root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda path: str(path).lower(),
        )
        oe_manifest = {
            "source_root": str(resolved_oe_root.resolve()),
            "image_count": len(oe_images),
            "image_fingerprint": _fingerprint_paths(oe_images, root=resolved_oe_root),
        }
    if crop_root.exists():
        shutil.rmtree(crop_root)
    crop_root.mkdir(parents=True, exist_ok=True)
    rows = list(manifest.get("rows", []))
    for row in rows:
        split_name = str(row.get("split", "")).strip()
        if split_name not in {"continual", "val", "test"}:
            continue
        relative_path = Path(str(row.get("relative_path", "")))
        class_name = str(row.get("normalized_class_name", "")).strip()
        # Keep raw class token exactly as recorded in the manifest for path slicing.
        raw_class_name = str(row.get("raw_class_name", ""))
        source_path = Path(class_root) / relative_path
        destination_relative = relative_path.relative_to(raw_class_name)
        destination_path = crop_root / split_name / class_name / destination_relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source_path), str(destination_path))
        row["runtime_relative_path"] = destination_path.relative_to(crop_root).as_posix()

    if resolved_ood_root is not None:
        ood_dir = crop_root / "ood"
        ood_dir.mkdir(parents=True, exist_ok=True)
        for source_path in ood_images:
            destination_path = ood_dir / source_path.relative_to(resolved_ood_root)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source_path), str(destination_path))
    if resolved_oe_root is not None:
        oe_dir = crop_root / "oe"
        oe_dir.mkdir(parents=True, exist_ok=True)
        for source_path in oe_images:
            destination_path = oe_dir / source_path.relative_to(resolved_oe_root)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source_path), str(destination_path))
    split_manifest_path = write_json(
        crop_root / "split_manifest.json",
        {
            "schema_version": "v1_grouped_runtime_layout",
            "crop_name": str(crop_name),
            "part_name": str(part_name),
            "dataset_key": str(dataset_key),
            "source_root": str(class_root.resolve()),
            "artifact_root": str(artifact_root.resolve()),
            "split_policy": GROUPED_SPLIT_POLICY,
            "ood": ood_manifest,
            "oe": oe_manifest,
            "rows": rows,
        },
        ensure_ascii=False,
    )
    if ood_manifest is not None:
        ood_handoff_checklist = {
            "status": "materialized",
            "message": "Repo or explicit OOD tree was materialized into runtime_dataset/<dataset_key>/ood.",
            "source_root": str(ood_manifest.get("source_root", "")),
            "image_count": int(ood_manifest.get("image_count", 0)),
        }
        write_json(
            artifact_root / "ood_handoff_checklist.json",
            ood_handoff_checklist,
            ensure_ascii=False,
        )
        prep_summary = read_json(artifact_root / "prep_summary.json", default={}, expect_type=dict)
        if isinstance(prep_summary, dict):
            prep_summary["ood_handoff_checklist"] = dict(ood_handoff_checklist)
            write_json(artifact_root / "prep_summary.json", prep_summary, ensure_ascii=False)
    refresh_prep_guided_artifacts(
        artifact_root,
        overview_updates={
            "crop_name": str(crop_name),
            "part_name": str(part_name),
            "materialized_runtime_root": str(crop_root.resolve()),
            "split_manifest_path": str(split_manifest_path.resolve()),
            "ood_image_count": int((ood_manifest or {}).get("image_count", 0)),
        },
        extra_entries=[
            {
                "path": split_manifest_path,
                "category": "manifests",
                "priority": "high",
                "title_tr": "Materyalize edilmis runtime split manifesti",
                "description_tr": "Gercekten uretilen runtime dataset icindeki split manifest dosyasi.",
                "reader_goal": "Notebook 2'nin tuketecegi final runtime split yapisini gormek",
                "generated_by": "scripts.prepare_grouped_runtime_dataset",
                "decision_importance": "prep_gate",
                "read_order": 22,
            }
        ],
    )
    return Path(runtime_root)
