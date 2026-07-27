#!/usr/bin/env python3
"""Validate exported adapter metadata completeness.

This guard checks deployed adapter bundles for the metadata and state files
needed to trace behavioral acceptance evidence through inference. Missing adapter roots are
reported as `skipped`; incomplete discovered adapters fail unless the issue is
only an optional calibration/readiness warning.
"""

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
from src.shared.json_utils import read_json  # noqa: E402


def _read_meta(adapter_dir: Path) -> dict[str, Any]:
    payload = read_json(adapter_dir / "adapter_meta.json", default={}, expect_type=dict)
    return dict(payload)


def _has_any(adapter_dir: Path, names: tuple[str, ...]) -> bool:
    return any((adapter_dir / name).exists() for name in names)


def _acceptance_candidates(adapter_dir: Path) -> list[Path]:
    return [
        adapter_dir / "adapter_behavioral_acceptance.json",
        adapter_dir.parent / "adapter_behavioral_acceptance.json",
        adapter_dir.parent.parent / "adapter_behavioral_acceptance.json",
        adapter_dir.parent / "artifacts" / "adapter_behavioral_acceptance.json",
        adapter_dir.parent.parent / "artifacts" / "adapter_behavioral_acceptance.json",
    ]


def validate_adapter(adapter_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        meta = _read_meta(adapter_dir)
    except Exception as exc:
        return {
            "adapter_dir": str(adapter_dir),
            "status": "fail",
            "errors": [f"Cannot read adapter_meta.json: {exc}"],
            "warnings": [],
        }

    required_meta = ("schema_version", "class_to_idx", "backbone")
    for key in required_meta:
        if key not in meta:
            errors.append(f"adapter_meta.{key} is required")

    if not isinstance(meta.get("class_to_idx"), dict) or not meta.get("class_to_idx"):
        errors.append("adapter_meta.class_to_idx must be a non-empty object")

    if not _has_any(adapter_dir, ("adapter_model.bin", "adapter_model.safetensors", "adapter_state.pt")):
        warnings.append("No LoRA adapter weight file found by known names")
    if not _has_any(adapter_dir, ("classifier.pth", "classifier_state.pt", "classifier_head.pt", "classifier.pt")):
        warnings.append("No classifier state file found by known names")

    ood_calibration = meta.get("ood_calibration")
    if not isinstance(ood_calibration, dict) or not ood_calibration:
        warnings.append("adapter_meta.ood_calibration is missing or empty")

    acceptance_path = next((path for path in _acceptance_candidates(adapter_dir) if path.exists()), None)
    acceptance_verdict = ""
    if acceptance_path is None:
        warnings.append("adapter_behavioral_acceptance.json was not found near adapter bundle")
    else:
        try:
            acceptance = read_json(acceptance_path, default={}, expect_type=dict)
            acceptance_verdict = str(acceptance.get("status") or "")
        except Exception as exc:
            warnings.append(f"Could not parse behavioral acceptance artifact at {acceptance_path}: {exc}")

    metadata_error = str(candidate.get("metadata_error") or "").strip()
    if metadata_error:
        warnings.append(f"discovery metadata warning: {metadata_error}")

    status = "fail" if errors else "warn" if warnings else "pass"
    return {
        "adapter_dir": str(adapter_dir),
        "crop_name": candidate.get("crop_name"),
        "part_name": candidate.get("part_name"),
        "run_id": candidate.get("run_id"),
        "status": status,
        "schema_version": str(meta.get("schema_version", "") or ""),
        "class_count": len(meta.get("class_to_idx", {})) if isinstance(meta.get("class_to_idx"), dict) else 0,
        "behavioral_acceptance_path": str(acceptance_path) if acceptance_path else "",
        "behavioral_acceptance_status": acceptance_verdict,
        "errors": errors,
        "warnings": warnings,
    }


def build_report(adapter_roots: list[Path]) -> dict[str, Any]:
    candidates = discover_adapter_candidates(adapter_roots)
    adapters = [
        validate_adapter(Path(str(candidate["adapter_dir"])), candidate)
        for candidate in candidates
    ]
    fail_count = sum(1 for item in adapters if item["status"] == "fail")
    warn_count = sum(1 for item in adapters if item["status"] == "warn")
    return {
        "status": "fail" if fail_count else "warn" if warn_count else "skipped" if not adapters else "pass",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "adapter_roots": [str(root) for root in adapter_roots],
        "adapter_count": len(adapters),
        "fail_count": fail_count,
        "warn_count": warn_count,
        "adapters": adapters,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-root", action="append", default=[], help="Adapter search root. May repeat.")
    parser.add_argument("--output", type=Path, default=Path(".runtime_tmp/adapter_metadata_completeness.json"))
    parser.add_argument("--strict", action="store_true", help="Return non-zero on warnings as well as errors.")
    args = parser.parse_args(argv)

    roots = [Path(item) for item in (args.adapter_root or ["models/adapters"])]
    report = build_report(roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"adapter_metadata status={report['status']} adapters={report['adapter_count']} "
        f"failures={report['fail_count']} warnings={report['warn_count']} output={args.output}"
    )
    if report["fail_count"]:
        return 1
    if args.strict and report["warn_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
