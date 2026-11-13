#!/usr/bin/env python3
"""Notebook 2 runtime dataset inspection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.shared.json_utils import read_json
from src.shared.string_utils import normalize_class_name
from src.training.services.class_balance import MIN_SUPPORTED_CLASS_SAMPLES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RUNTIME_SPLITS = ("continual", "val", "test")


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _count_split(root: Path, split_name: str) -> Dict[str, int]:
    split_root = root / split_name
    if not split_root.is_dir():
        return {}
    counts: Dict[str, int] = {}
    for class_dir in sorted((item for item in split_root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        class_name = normalize_class_name(class_dir.name)
        counts[class_name] = sum(1 for item in class_dir.rglob("*") if _is_image(item))
    return counts


def _summarize_manifest_rows(rows: list[Any]) -> Dict[str, Any]:
    synthetic_hint_count = 0
    eval_quality_risk_count = 0
    eval_ineligible_count = 0
    family_assignment_counts: Dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if bool(row.get("synthetic_hint")):
            synthetic_hint_count += 1
        if bool(row.get("eval_quality_risk")):
            eval_quality_risk_count += 1
        if row.get("family_eval_eligible") is False:
            eval_ineligible_count += 1
        assignment = str(row.get("family_assignment", row.get("split", "")) or "").strip().lower()
        if assignment:
            family_assignment_counts[assignment] = int(family_assignment_counts.get(assignment, 0)) + 1
    return {
        "row_count": len(rows),
        "synthetic_hint_count": synthetic_hint_count,
        "eval_quality_risk_count": eval_quality_risk_count,
        "eval_ineligible_count": eval_ineligible_count,
        "family_assignment_counts": family_assignment_counts,
    }


def _manifest_class_counts(payload: Dict[str, Any]) -> Dict[str, int]:
    classes = payload.get("classes", [])
    counts: Dict[str, int] = {}
    if isinstance(classes, list):
        for entry in classes:
            if not isinstance(entry, dict):
                continue
            class_name = normalize_class_name(entry.get("class_name", ""))
            if not class_name:
                continue
            try:
                counts[class_name] = int(entry.get("reference_image_count", entry.get("image_count")))
            except (TypeError, ValueError):
                continue
    if counts:
        return counts

    rows = payload.get("rows", [])
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            split_name = str(row.get("split", "")).strip().lower()
            if split_name not in {"continual", "val", "test"}:
                continue
            if bool(row.get("runtime_skipped")) or bool(row.get("generated_offline_augmentation")):
                continue
            if bool(row.get("synthetic_hint")):
                continue
            class_name = normalize_class_name(row.get("normalized_class_name", row.get("class_name", "")))
            if not class_name:
                continue
            counts[class_name] = int(counts.get(class_name, 0)) + 1
    return counts


def _dataset_scale_bucket(continual_images: int) -> str:
    total = int(max(0, continual_images))
    if total <= 400:
        return "tiny"
    if total <= 2000:
        return "small"
    if total <= 10000:
        return "medium"
    return "large"


def _coerce_like(reference: Any, value: Any) -> Any:
    if isinstance(reference, bool):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    if isinstance(reference, str):
        return str(value)
    return value


def inspect_runtime_dataset(dataset_root: Path, ood_root: Path | str | None = None) -> Dict[str, Any]:
    """Inspect the selected runtime dataset root for Notebook 2 validation."""

    resolved_root = Path(dataset_root).expanduser().resolve()
    split_class_counts = {split_name: _count_split(resolved_root, split_name) for split_name in RUNTIME_SPLITS}
    split_totals = {
        split_name: int(sum(class_counts.values()))
        for split_name, class_counts in split_class_counts.items()
    }
    resolved_ood_root = (
        Path(ood_root).expanduser().resolve()
        if ood_root is not None and str(ood_root).strip()
        else resolved_root / "ood"
    )
    ood_count = 0
    if resolved_ood_root.is_dir():
        ood_count = sum(1 for item in resolved_ood_root.rglob("*") if _is_image(item))
    manifest_path = resolved_root / "split_manifest.json"
    manifest_payload = (
        read_json(manifest_path, default={}, expect_type=dict)
        if manifest_path.exists()
        else {}
    )
    manifest_payload = dict(manifest_payload) if isinstance(manifest_payload, dict) else {}
    manifest_rows = list(manifest_payload.get("rows", [])) if isinstance(manifest_payload.get("rows", []), list) else []
    manifest_class_counts = _manifest_class_counts(manifest_payload)
    reference_class_counts = manifest_class_counts or dict(split_class_counts.get("continual", {}))
    class_names = sorted(
        {
            *split_class_counts.get("continual", {}).keys(),
            *split_class_counts.get("val", {}).keys(),
            *split_class_counts.get("test", {}).keys(),
        }
    )
    missing_required_splits = [split_name for split_name in RUNTIME_SPLITS if not (resolved_root / split_name).is_dir()]
    manifest_row_summary = _summarize_manifest_rows(manifest_rows)
    scale_bucket = _dataset_scale_bucket(split_totals.get("continual", 0))
    noisy_or_high_risk = bool(
        manifest_row_summary["synthetic_hint_count"] > 0 or manifest_row_summary["eval_quality_risk_count"] > 0
    )
    under_min_classes = sorted(
        class_name
        for class_name, count in reference_class_counts.items()
        if int(count) < int(MIN_SUPPORTED_CLASS_SAMPLES)
    )
    blockers: list[str] = []
    warnings: list[str] = []
    if missing_required_splits:
        blockers.append(f"Missing required split folders: {missing_required_splits}")
    if not class_names:
        blockers.append("No class directories were found in the runtime dataset.")
    if split_totals.get("val", 0) <= 0:
        blockers.append("Validation split has zero images.")
    if split_totals.get("test", 0) <= 0:
        blockers.append("Test split has zero images.")
    if under_min_classes:
        details = ", ".join(f"{name}={reference_class_counts[name]}" for name in under_min_classes)
        blockers.append(
            "Supported classes remain below the production minimum of "
            f"{MIN_SUPPORTED_CLASS_SAMPLES} images/class: {details}. "
            "Enable ALLOW_UNDER_MIN_TRAINING manually only for a research-only run."
        )
    if not manifest_payload:
        warnings.append("split_manifest.json was not found; dataset checks fall back to runtime split counts.")
    if ood_count <= 0:
        warnings.append("Real runtime OOD evidence is missing; Notebook 2 will rely on fallback held-out benchmarking.")
    if manifest_row_summary["synthetic_hint_count"] > 0:
        warnings.append(
            f"Manifest rows include {manifest_row_summary['synthetic_hint_count']} synthetic-hint item(s); keep main benchmark claims conservative."
        )
    if manifest_row_summary["eval_quality_risk_count"] > 0:
        warnings.append(
            f"Manifest rows include {manifest_row_summary['eval_quality_risk_count']} eval-quality-risk item(s); prefer cautious regularization."
        )
    return {
        "dataset_root": str(resolved_root),
        "manifest_path": str(manifest_path),
        "manifest_present": bool(manifest_payload),
        "split_presence": {
            **{split_name: (resolved_root / split_name).is_dir() for split_name in RUNTIME_SPLITS},
            "ood": resolved_ood_root.is_dir(),
        },
        "split_totals": {**split_totals, "ood": int(ood_count)},
        "ood_root": str(resolved_ood_root),
        "split_class_counts": split_class_counts,
        "manifest_class_counts": manifest_class_counts,
        "reference_class_counts": reference_class_counts,
        "class_names": class_names,
        "class_count": len(class_names),
        "dataset_scale_bucket": scale_bucket,
        "manifest_row_summary": manifest_row_summary,
        "real_ood_present": bool(ood_count > 0),
        "noisy_or_high_risk": noisy_or_high_risk,
        "under_min_classes": under_min_classes,
        "blockers": blockers,
        "warnings": warnings,
    }


def resolve_notebook_params(
    base_params: Dict[str, Any],
    manual_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve Notebook 2 parameters from visible values plus explicit manual overrides."""

    resolved = dict(base_params)
    for key, raw_value in dict(manual_overrides or {}).items():
        if key in resolved:
            resolved[key] = _coerce_like(resolved[key], raw_value)
        else:
            resolved[str(key)] = raw_value
    return resolved


__all__ = [
    "inspect_runtime_dataset",
    "resolve_notebook_params",
]
