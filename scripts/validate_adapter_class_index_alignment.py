#!/usr/bin/env python3
"""Validate adapter class-index metadata against prepared runtime datasets."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.colab_adapter_smoke_test import discover_adapter_candidates  # noqa: E402
from src.data.dataset_release_runtime import DEFAULT_MATERIALIZED_DATASET_ROOT  # noqa: E402
from src.shared.string_utils import normalize_notebook_identifier  # noqa: E402


def _class_names_from_dataset(dataset_root: Path) -> list[str]:
    class_root = dataset_root / "continual"
    if not class_root.is_dir():
        return []
    return sorted(path.name for path in class_root.iterdir() if path.is_dir())


def _normalized_collision_errors(class_names: list[str]) -> list[str]:
    normalized_to_raw: dict[str, str] = {}
    errors: list[str] = []
    for raw_name in class_names:
        normalized = normalize_notebook_identifier(raw_name)
        if not normalized:
            continue
        existing = normalized_to_raw.get(normalized)
        if existing is not None and existing != raw_name:
            errors.append(
                f"class names collapse to the same normalized key {normalized!r}: {existing!r}, {raw_name!r}"
            )
            continue
        normalized_to_raw[normalized] = raw_name
    return errors


def _candidate_dataset_key(candidate: dict[str, Any]) -> str:
    crop_name = str(candidate.get("crop_name") or "").strip().lower()
    part_name = str(candidate.get("part_name") or "").strip().lower()
    if not crop_name or not part_name:
        return ""
    return f"{crop_name}__{part_name}"


def _validate_candidate(candidate: dict[str, Any], *, dataset_root: Path) -> dict[str, Any]:
    adapter_dir = Path(str(candidate.get("adapter_dir") or ""))
    dataset_key = _candidate_dataset_key(candidate)
    expected_dataset_root = dataset_root / dataset_key if dataset_key else Path()
    errors: list[str] = []
    warnings: list[str] = []

    if not dataset_key:
        errors.append("adapter crop_name/part_name could not be resolved")
        expected_classes: list[str] = []
    elif not expected_dataset_root.is_dir():
        warnings.append(f"prepared runtime dataset not found: {expected_dataset_root}")
        expected_classes = []
    else:
        expected_classes = _class_names_from_dataset(expected_dataset_root)
        if not expected_classes:
            errors.append(f"prepared runtime dataset has no continual classes: {expected_dataset_root}")
        errors.extend(_normalized_collision_errors(expected_classes))

    adapter_classes = [str(item) for item in list(candidate.get("class_names") or [])]
    normalized_adapter_classes = [normalize_notebook_identifier(name) for name in adapter_classes]
    normalized_dataset_classes = [normalize_notebook_identifier(name) for name in expected_classes]
    mismatch_type = ""
    if expected_classes and adapter_classes != expected_classes:
        if normalized_adapter_classes == normalized_dataset_classes:
            mismatch_type = "raw_name_mismatch"
            warnings.append(
                "adapter class names differ from dataset class folder names but normalize to the same order"
            )
        else:
            mismatch_type = "index_semantic_mismatch"
            errors.append("adapter class order does not match prepared runtime dataset class order")

    status = "fail" if errors else "warn" if warnings else "pass"
    return {
        "status": status,
        "adapter_dir": str(adapter_dir),
        "run_id": str(candidate.get("run_id") or ""),
        "crop_name": str(candidate.get("crop_name") or ""),
        "part_name": str(candidate.get("part_name") or ""),
        "dataset_key": dataset_key,
        "dataset_root": str(expected_dataset_root) if dataset_key else "",
        "adapter_classes": adapter_classes,
        "dataset_classes": expected_classes,
        "normalized_adapter_classes": normalized_adapter_classes,
        "normalized_dataset_classes": normalized_dataset_classes,
        "mismatch_type": mismatch_type,
        "errors": errors,
        "warnings": warnings,
    }


def build_report(
    *,
    adapter_roots: list[Path],
    dataset_root: Path,
    crop_name: str | None = None,
) -> dict[str, Any]:
    candidates = discover_adapter_candidates(adapter_roots, crop_name=crop_name)
    adapters = [
        _validate_candidate(candidate, dataset_root=dataset_root)
        for candidate in candidates
    ]
    fail_count = sum(1 for item in adapters if item["status"] == "fail")
    warn_count = sum(1 for item in adapters if item["status"] == "warn")
    return {
        "status": "fail" if fail_count else "warn" if warn_count else "skipped" if not adapters else "pass",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "adapter_roots": [str(root) for root in adapter_roots],
        "dataset_root": str(dataset_root),
        "adapter_count": len(adapters),
        "fail_count": fail_count,
        "warn_count": warn_count,
        "adapters": adapters,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-root", action="append", default=[], help="Adapter search root. May repeat.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_MATERIALIZED_DATASET_ROOT)
    parser.add_argument("--crop-name", default=None, help="Optional crop filter.")
    parser.add_argument("--output", type=Path, default=Path(".runtime_tmp/adapter_class_index_alignment.json"))
    parser.add_argument("--strict", action="store_true", help="Return non-zero on warnings as well as errors.")
    parser.add_argument("--require-adapter", action="store_true", help="Return non-zero when no adapters are found.")
    args = parser.parse_args(argv)

    roots = [Path(item) for item in (args.adapter_root or ["models/adapters"])]
    report = build_report(adapter_roots=roots, dataset_root=Path(args.dataset_root), crop_name=args.crop_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(
        f"adapter_class_index_alignment status={report['status']} adapters={report['adapter_count']} "
        f"failures={report['fail_count']} warnings={report['warn_count']} output={args.output}"
    )
    if args.require_adapter and report["adapter_count"] == 0:
        return 1
    if report["fail_count"]:
        return 1
    if args.strict and report["warn_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
