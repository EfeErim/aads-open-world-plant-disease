#!/usr/bin/env python3
"""Validate expected notebook output artifacts when generated outputs exist.

The check is intentionally non-invasive: source-only checkouts without local
Notebook 0/2 run outputs are reported as `skipped`. When outputs are present,
missing required manifest, guided, or adapter export artifacts are failures.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.reporting import write_json  # noqa: E402
from src.data.dataset_release_runtime import DEFAULT_MATERIALIZED_DATASET_ROOT  # noqa: E402


def _check_path(path: Path, label: str, errors: list[str], present: list[str]) -> None:
    if path.exists():
        present.append(label)
    else:
        errors.append(f"Missing {label}: {path}")


def _find_notebook2_roots(output_root: Path) -> list[Path]:
    if not output_root.exists():
        return []
    roots: set[Path] = set()
    for adapter_meta in output_root.rglob("continual_sd_lora_adapter/adapter_meta.json"):
        roots.add(adapter_meta.parent.parent)
    for overview in output_root.rglob("guided/01_run_overview.json"):
        roots.add(overview.parent.parent)
    return sorted(roots, key=lambda item: str(item).lower())


def _validate_notebook2_root(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    present: list[str] = []
    _check_path(root / "continual_sd_lora_adapter" / "adapter_meta.json", "adapter_meta", errors, present)
    _check_path(root / "guided" / "00_start_here.md", "guided_start", errors, present)
    _check_path(root / "guided" / "01_run_overview.json", "guided_overview", errors, present)
    _check_path(root / "guided" / "02_file_catalog.json", "guided_catalog", errors, present)
    return {
        "root": str(root),
        "notebook": "2_train_continual_sd_lora_adapter",
        "status": "fail" if errors else "pass",
        "present": present,
        "errors": errors,
    }


def _find_notebook0_roots(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        return []
    if (dataset_root / "split_manifest.json").exists():
        return [dataset_root]
    return sorted(
        child for child in dataset_root.iterdir() if child.is_dir() and (child / "split_manifest.json").exists()
    )


def _validate_notebook0_root(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    present: list[str] = []
    _check_path(root / "split_manifest.json", "split_manifest", errors, present)
    for split in ("continual", "val", "test"):
        _check_path(root / split, split, errors, present)
    return {
        "root": str(root),
        "notebook": "0_prepare_grouped_dataset_for_training",
        "status": "fail" if errors else "pass",
        "present": present,
        "errors": errors,
    }


def build_report(output_root: Path, dataset_root: Path) -> dict[str, Any]:
    outputs = [_validate_notebook2_root(root) for root in _find_notebook2_roots(output_root)]
    outputs.extend(_validate_notebook0_root(root) for root in _find_notebook0_roots(dataset_root))
    fail_count = sum(1 for item in outputs if item["status"] == "fail")
    return {
        "status": "fail" if fail_count else "skipped" if not outputs else "pass",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "dataset_root": str(dataset_root),
        "checked_count": len(outputs),
        "fail_count": fail_count,
        "outputs": outputs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/colab_notebook_training"))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_MATERIALIZED_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=Path(".runtime_tmp/notebook_output_validation.json"))
    args = parser.parse_args(argv)

    report = build_report(args.output_root, args.dataset_root)
    write_json(args.output, report)
    print(
        f"notebook_outputs status={report['status']} checked={report['checked_count']} "
        f"failures={report['fail_count']} output={args.output}"
    )
    return 1 if report["fail_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
