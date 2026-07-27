"""Compatibility wrapper and CLI for direct adapter smoke-test helpers.

The command-line path is contract-first: it discovers exported adapter bundles,
validates the metadata shape Notebook 3/4 and inference need, and optionally
runs image-folder predictions when a review image directory is provided. A
missing local adapter directory is reported as a skipped check so source-only CI
jobs can still run without generated model artifacts.
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

import src.pipeline.adapter_smoke as _impl  # noqa: E402

IMAGE_SUFFIXES = _impl.IMAGE_SUFFIXES
DEFAULT_ROBUST_VIEWS = _impl.DEFAULT_ROBUST_VIEWS
DEFAULT_EXPLANATION_METHOD = _impl.DEFAULT_EXPLANATION_METHOD
SUPPORTED_EXPLANATION_METHODS = _impl.SUPPORTED_EXPLANATION_METHODS
IMAGE_MEAN_PAD_RGB = _impl.IMAGE_MEAN_PAD_RGB
DEFAULT_DISCOVERY_ROOTS = _impl.DEFAULT_DISCOVERY_ROOTS
SKIP_DISCOVERY_DIR_NAMES = _impl.SKIP_DISCOVERY_DIR_NAMES
preprocess_image = _impl.preprocess_image
_build_adapter = _impl._build_adapter
_target_size = _impl._target_size
_prepare_view_tensor = _impl._prepare_view_tensor
_iter_adapter_meta_paths = _impl._iter_adapter_meta_paths


def _sync_impl() -> None:
    _impl.preprocess_image = preprocess_image
    _impl._build_adapter = _build_adapter
    _impl._target_size = _target_size
    _impl._prepare_view_tensor = _prepare_view_tensor
    _impl._iter_adapter_meta_paths = _iter_adapter_meta_paths


def discover_adapter_candidates(*args, **kwargs):
    _sync_impl()
    return _impl.discover_adapter_candidates(*args, **kwargs)


def load_adapter_summary(*args, **kwargs):
    _sync_impl()
    return _impl.load_adapter_summary(*args, **kwargs)


def predict_single_image(*args, **kwargs):
    _sync_impl()
    return _impl.predict_single_image(*args, **kwargs)


def predict_image_folder(*args, **kwargs):
    _sync_impl()
    return _impl.predict_image_folder(*args, **kwargs)


def build_prediction_visualization_images(*args, **kwargs):
    _sync_impl()
    return _impl.build_prediction_visualization_images(*args, **kwargs)


def _read_adapter_meta_for_cli(adapter_dir: Path) -> dict[str, Any]:
    meta_path = adapter_dir / "adapter_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing adapter_meta.json in {adapter_dir}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {meta_path}")
    return dict(payload)


def _validate_adapter_contract(adapter_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        meta = _read_adapter_meta_for_cli(adapter_dir)
    except Exception as exc:
        return {
            "adapter_dir": str(adapter_dir),
            "crop_name": candidate.get("crop_name"),
            "part_name": candidate.get("part_name"),
            "status": "fail",
            "errors": [str(exc)],
            "warnings": [],
        }

    class_to_idx = meta.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        errors.append("adapter_meta.class_to_idx must be a non-empty object")
    elif any(not isinstance(index, int) for index in class_to_idx.values()):
        errors.append("adapter_meta.class_to_idx values must be integer class ids")

    schema_version = str(meta.get("schema_version", "") or "").strip()
    if schema_version != "v6":
        errors.append("adapter_meta.schema_version must be 'v6'")

    engine = str(meta.get("engine", "") or "").strip()
    if engine and engine != "continual_sd_lora":
        errors.append("adapter_meta.engine must be 'continual_sd_lora' when present")
    if not engine:
        warnings.append("adapter_meta.engine is missing; loader will assume continual_sd_lora")

    backbone = meta.get("backbone")
    if not isinstance(backbone, dict) or not str(backbone.get("model_name", "") or "").strip():
        errors.append("adapter_meta.backbone.model_name is required")

    ood_calibration = meta.get("ood_calibration")
    if not isinstance(ood_calibration, dict) or not ood_calibration.get("version"):
        warnings.append("adapter_meta.ood_calibration.version is missing")

    metadata_error = str(candidate.get("metadata_error") or "").strip()
    if metadata_error:
        warnings.append(f"discovery metadata warning: {metadata_error}")

    status = "fail" if errors else "warn" if warnings else "pass"
    return {
        "adapter_dir": str(adapter_dir),
        "crop_name": candidate.get("crop_name"),
        "part_name": candidate.get("part_name"),
        "run_id": candidate.get("run_id"),
        "class_count": len(class_to_idx) if isinstance(class_to_idx, dict) else 0,
        "schema_version": schema_version,
        "engine": engine or "continual_sd_lora",
        "status": status,
        "errors": errors,
        "warnings": warnings,
    }


def _write_cli_report(output_path: Path, report: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate direct adapter smoke-test contracts and optionally run image predictions.",
    )
    parser.add_argument(
        "--adapter-root",
        action="append",
        default=[],
        help="Adapter search root. May be repeated. Defaults to models/adapters.",
    )
    parser.add_argument("--crop-name", default=None, help="Optional crop filter.")
    parser.add_argument("--part-name", default=None, help="Optional part filter.")
    parser.add_argument("--image-dir", default=None, help="Optional folder of smoke-test images.")
    parser.add_argument("--device", default="cpu", help="Device for optional image predictions.")
    parser.add_argument("--config-env", default="colab", help="Config environment used by adapter helpers.")
    parser.add_argument(
        "--output",
        default=".runtime_tmp/adapter_smoke_test.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero on warnings as well as errors.",
    )
    parser.add_argument(
        "--require-adapter",
        action="store_true",
        help="Return non-zero when no adapter bundles are discovered.",
    )
    parser.add_argument(
        "--show-mirror-adapters",
        action="store_true",
        help="Do not collapse duplicate telemetry/output/checkpoint mirrors from the same run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a metadata-first smoke check suitable for CI and scheduled automation."""
    args = _build_parser().parse_args(argv)
    _sync_impl()

    roots = [Path(root) for root in (args.adapter_root or ["models/adapters"])]
    candidates = discover_adapter_candidates(
        roots,
        crop_name=args.crop_name,
        collapse_run_mirrors=not args.show_mirror_adapters,
    )
    if args.part_name:
        requested_part = str(args.part_name).strip().lower()
        candidates = [
            candidate
            for candidate in candidates
            if str(candidate.get("part_name") or "unspecified").strip().lower() == requested_part
        ]

    rows = [
        _validate_adapter_contract(Path(str(candidate["adapter_dir"])), candidate)
        for candidate in candidates
    ]

    predictions: list[dict[str, Any]] = []
    prediction_errors: list[str] = []
    if args.image_dir and rows:
        for row in rows:
            if row["status"] == "fail":
                continue
            try:
                predictions.extend(
                    predict_image_folder(
                        args.image_dir,
                        row.get("crop_name"),
                        adapter_dir=row["adapter_dir"],
                        config_env=args.config_env,
                        device=args.device,
                        part_name=row.get("part_name"),
                    )
                )
            except Exception as exc:
                prediction_errors.append(f"{row['adapter_dir']}: {exc}")

    fail_count = sum(1 for row in rows if row["status"] == "fail") + len(prediction_errors)
    warn_count = sum(1 for row in rows if row["status"] == "warn")
    skipped = not rows
    if skipped and args.require_adapter:
        fail_count += 1

    report = {
        "status": "fail" if fail_count else "warn" if warn_count else "skipped" if skipped else "pass",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "adapter_roots": [str(root) for root in roots],
        "adapter_count": len(rows),
        "fail_count": fail_count,
        "warn_count": warn_count,
        "strict": bool(args.strict),
        "require_adapter": bool(args.require_adapter),
        "adapters": rows,
        "prediction_count": len(predictions),
        "prediction_errors": prediction_errors,
    }
    if predictions:
        report["predictions"] = predictions

    _write_cli_report(Path(args.output), report)

    print(
        f"adapter_smoke_contracts status={report['status']} "
        f"adapters={len(rows)} failures={fail_count} warnings={warn_count} output={args.output}"
    )
    if fail_count:
        return 1
    if args.strict and warn_count:
        return 1
    return 0


__all__ = [
    "DEFAULT_DISCOVERY_ROOTS",
    "DEFAULT_EXPLANATION_METHOD",
    "DEFAULT_ROBUST_VIEWS",
    "IMAGE_MEAN_PAD_RGB",
    "IMAGE_SUFFIXES",
    "SKIP_DISCOVERY_DIR_NAMES",
    "SUPPORTED_EXPLANATION_METHODS",
    "_build_adapter",
    "_iter_adapter_meta_paths",
    "_prepare_view_tensor",
    "_target_size",
    "build_prediction_visualization_images",
    "discover_adapter_candidates",
    "load_adapter_summary",
    "main",
    "predict_image_folder",
    "predict_single_image",
    "preprocess_image",
]


if __name__ == "__main__":
    raise SystemExit(main())
