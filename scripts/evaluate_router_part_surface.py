#!/usr/bin/env python3
"""Evaluate router part predictions and sweep abstention thresholds offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image

from src.core.config_manager import get_config
from src.router.label_normalization import normalize_part_label
from src.router.router_pipeline import RouterPipeline
from src.shared.contracts import RouterAnalysisResult

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def discover_eval_samples(root: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for crop_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        expected_crop = str(crop_dir.name).strip().lower()
        for part_dir in sorted(path for path in crop_dir.iterdir() if path.is_dir()):
            expected_part = normalize_part_label(part_dir.name)
            for image_path in sorted(path for path in part_dir.rglob("*") if _is_image_path(path)):
                samples.append(
                    {
                        "image_path": image_path,
                        "expected_crop": expected_crop,
                        "expected_part": expected_part or "unknown",
                    }
                )
    return samples


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        import logging
        logging.exception('Unhandled exception')
        raise
        return float(default)


def sample_from_analysis(
    *,
    image_path: Path,
    expected_crop: str,
    expected_part: str,
    analysis: RouterAnalysisResult,
    compatible_parts: List[str],
) -> Dict[str, Any]:
    detection = analysis.primary_detection
    detection_payload = {} if detection is None else detection.to_dict()
    predicted_crop = str(detection_payload.get("crop", "unknown") or "unknown").strip().lower()
    predicted_part = normalize_part_label(detection_payload.get("part", "unknown")) or "unknown"
    raw_part_label = normalize_part_label(detection_payload.get("raw_part_label", predicted_part)) or "unknown"
    raw_part_confidence = _coerce_float(detection_payload.get("raw_part_confidence"), _coerce_float(detection_payload.get("part_confidence"), 0.0))
    raw_part_second_confidence = _coerce_float(detection_payload.get("raw_part_second_confidence"), 0.0)
    raw_part_margin = _coerce_float(
        detection_payload.get("raw_part_margin"),
        raw_part_confidence - raw_part_second_confidence,
    )
    return {
        "image_path": str(image_path),
        "expected_crop": expected_crop,
        "expected_part": expected_part,
        "predicted_crop": predicted_crop,
        "predicted_part": predicted_part,
        "router_message": str(analysis.message or ""),
        "crop_correct": predicted_crop == expected_crop,
        "compatible_parts": [normalize_part_label(part) for part in compatible_parts if normalize_part_label(part)],
        "raw_part_label": raw_part_label,
        "raw_part_confidence": raw_part_confidence,
        "raw_part_second_confidence": raw_part_second_confidence,
        "part_unknown_confidence": _coerce_float(detection_payload.get("part_unknown_confidence"), 0.0),
        "raw_part_margin": raw_part_margin,
        "part_rejection_reason": str(detection_payload.get("part_rejection_reason", "") or ""),
    }


def simulate_thresholded_part_prediction(
    sample: Dict[str, Any],
    *,
    min_confidence: float,
    margin_threshold: float,
    unknown_label: str = "unknown",
) -> str:
    raw_part_label = normalize_part_label(sample.get("raw_part_label", unknown_label)) or unknown_label
    compatible_parts = {
        normalize_part_label(part)
        for part in sample.get("compatible_parts", [])
        if normalize_part_label(part)
    }
    if raw_part_label == unknown_label or not compatible_parts or raw_part_label not in compatible_parts:
        return unknown_label
    if _coerce_float(sample.get("part_unknown_confidence"), 0.0) >= _coerce_float(sample.get("raw_part_confidence"), 0.0):
        return unknown_label
    if _coerce_float(sample.get("raw_part_confidence"), 0.0) < float(min_confidence):
        return unknown_label
    if _coerce_float(sample.get("raw_part_margin"), 0.0) < float(margin_threshold):
        return unknown_label
    return raw_part_label


def summarize_predictions(samples: Iterable[Dict[str, Any]], *, predicted_key: str = "predicted_part") -> Dict[str, Any]:
    rows = list(samples)
    sample_count = len(rows)
    if sample_count <= 0:
        return {
            "sample_count": 0,
            "non_unknown_precision": 0.0,
            "non_unknown_recall": 0.0,
            "abstention_rate": 0.0,
            "unsupported_part_emission": 0,
            "confusion_matrix": {},
        }

    correct_non_unknown = 0
    predicted_non_unknown = 0
    unknown_count = 0
    unsupported_part_emission = 0
    confusion_matrix: Dict[str, Dict[str, int]] = {}
    for sample in rows:
        expected_part = str(sample["expected_part"])
        predicted_part = normalize_part_label(sample.get(predicted_key, "unknown")) or "unknown"
        if not bool(sample.get("crop_correct", False)):
            predicted_part = "crop_mismatch"
        confusion_matrix.setdefault(expected_part, {})
        confusion_matrix[expected_part][predicted_part] = confusion_matrix[expected_part].get(predicted_part, 0) + 1

        if predicted_part == "unknown":
            unknown_count += 1
            continue
        if predicted_part == "crop_mismatch":
            continue
        predicted_non_unknown += 1
        if predicted_part == expected_part:
            correct_non_unknown += 1
        compatible_parts = {
            normalize_part_label(part)
            for part in sample.get("compatible_parts", [])
            if normalize_part_label(part)
        }
        if predicted_part not in compatible_parts:
            unsupported_part_emission += 1

    return {
        "sample_count": sample_count,
        "non_unknown_precision": 0.0 if predicted_non_unknown <= 0 else round(correct_non_unknown / predicted_non_unknown, 4),
        "non_unknown_recall": round(correct_non_unknown / sample_count, 4),
        "abstention_rate": round(unknown_count / sample_count, 4),
        "unsupported_part_emission": unsupported_part_emission,
        "confusion_matrix": confusion_matrix,
    }


def sweep_part_thresholds(
    samples: List[Dict[str, Any]],
    *,
    min_confidence_grid: List[float],
    margin_grid: List[float],
    unknown_label: str = "unknown",
) -> Dict[str, Any]:
    crop_correct_samples = [sample for sample in samples if bool(sample.get("crop_correct", False))]
    rows: List[Dict[str, Any]] = []
    for min_confidence in min_confidence_grid:
        for margin_threshold in margin_grid:
            simulated_samples = []
            for sample in crop_correct_samples:
                simulated = dict(sample)
                simulated["simulated_part"] = simulate_thresholded_part_prediction(
                    sample,
                    min_confidence=min_confidence,
                    margin_threshold=margin_threshold,
                    unknown_label=unknown_label,
                )
                simulated_samples.append(simulated)
            metrics = summarize_predictions(simulated_samples, predicted_key="simulated_part")
            rows.append(
                {
                    "min_confidence": round(float(min_confidence), 4),
                    "margin": round(float(margin_threshold), 4),
                    **metrics,
                }
            )

    eligible_rows = [row for row in rows if int(row["unsupported_part_emission"]) == 0] or rows
    recommended = None
    if eligible_rows:
        recommended = sorted(
            eligible_rows,
            key=lambda row: (
                -float(row["non_unknown_precision"]),
                float(row["abstention_rate"]),
                -float(row["non_unknown_recall"]),
                float(row["min_confidence"]),
                float(row["margin"]),
            ),
        )[0]
    return {
        "sample_count": len(crop_correct_samples),
        "rows": rows,
        "recommended": recommended,
    }


def build_risk_coverage_report(samples: List[Dict[str, Any]], threshold_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize crop+part risk/coverage from the available router-part evidence."""
    sample_count = len(samples)
    crop_correct_count = sum(1 for sample in samples if bool(sample.get("crop_correct", False)))
    crop_mismatch_count = sample_count - crop_correct_count
    coverage_rows: List[Dict[str, Any]] = []
    for row in threshold_rows:
        accepted = int(round((1.0 - float(row.get("abstention_rate", 0.0))) * int(row.get("sample_count", 0))))
        precision = float(row.get("non_unknown_precision", 0.0))
        coverage = 0.0 if sample_count <= 0 else round(accepted / sample_count, 4)
        risk = round(1.0 - precision, 4) if accepted > 0 else 0.0
        coverage_rows.append(
            {
                "min_confidence": row.get("min_confidence"),
                "margin": row.get("margin"),
                "coverage": coverage,
                "accepted_count": accepted,
                "risk": risk,
                "part_non_unknown_precision": precision,
                "part_non_unknown_recall": row.get("non_unknown_recall"),
                "part_abstention_rate": row.get("abstention_rate"),
            }
        )
    recommended = None
    if coverage_rows:
        recommended = sorted(
            coverage_rows,
            key=lambda item: (
                float(item["risk"]),
                -float(item["coverage"]),
                float(item["min_confidence"]),
                float(item["margin"]),
            ),
        )[0]
    return {
        "sample_count": sample_count,
        "crop_accuracy": round(crop_correct_count / max(1, sample_count), 4),
        "crop_mismatch_count": crop_mismatch_count,
        "coverage_definition": "coverage is accepted part predictions over all evaluated samples",
        "risk_definition": "risk is 1 - non_unknown_precision among accepted crop-correct part predictions",
        "rows": coverage_rows,
        "recommended": recommended,
    }


def evaluate_router_part_surface(
    root: Path,
    *,
    config_env: str | None = "colab",
    device: str = "cuda",
    min_confidence_grid: List[float] | None = None,
    margin_grid: List[float] | None = None,
) -> Dict[str, Any]:
    dataset = discover_eval_samples(root)
    config = get_config(environment=config_env)
    router = RouterPipeline(config=config, device=device)
    router.load_models()
    if not router.is_ready():
        raise RuntimeError(
            "Router models failed to become ready for inference. "
            "Check router.vlm.enabled, model availability, and router dependency installation."
        )

    compatible_surface = getattr(router, "crop_part_compatibility", {})
    samples: List[Dict[str, Any]] = []
    for item in dataset:
        image = Image.open(item["image_path"]).convert("RGB")
        analysis = router.analyze_image_result(image)
        samples.append(
            sample_from_analysis(
                image_path=item["image_path"],
                expected_crop=item["expected_crop"],
                expected_part=item["expected_part"],
                analysis=analysis,
                compatible_parts=compatible_surface.get(item["expected_crop"], []),
            )
        )

    per_crop: Dict[str, Dict[str, Any]] = {}
    for crop_name in sorted({sample["expected_crop"] for sample in samples}):
        crop_samples = [sample for sample in samples if sample["expected_crop"] == crop_name]
        per_crop[crop_name] = {
            "sample_count": len(crop_samples),
            "crop_accuracy": round(
                sum(1 for sample in crop_samples if sample["crop_correct"]) / max(1, len(crop_samples)),
                4,
            ),
            **summarize_predictions(crop_samples),
        }

    min_confidence_grid = (
        [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        if min_confidence_grid is None
        else [float(value) for value in min_confidence_grid]
    )
    margin_grid = (
        [0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
        if margin_grid is None
        else [float(value) for value in margin_grid]
    )

    threshold_sweep = sweep_part_thresholds(
        samples,
        min_confidence_grid=min_confidence_grid,
        margin_grid=margin_grid,
        unknown_label="unknown",
    )

    return {
        "dataset_root": str(root),
        "sample_count": len(samples),
        "crop_accuracy": round(sum(1 for sample in samples if sample["crop_correct"]) / max(1, len(samples)), 4),
        "crops": per_crop,
        "threshold_sweep": threshold_sweep,
        "risk_coverage": build_risk_coverage_report(samples, list(threshold_sweep.get("rows", []))),
    }


def _parse_grid(raw_value: str) -> List[float]:
    return [float(item.strip()) for item in str(raw_value).split(",") if str(item).strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate router part predictions and sweep abstention thresholds.")
    parser.add_argument("--root", type=Path, required=True, help="Eval dataset root: data/router_part_eval/<crop>/<part>/*")
    parser.add_argument("--config-env", default="colab", help="Config environment override (default: colab)")
    parser.add_argument("--device", default="cuda", help="Torch device preference")
    parser.add_argument(
        "--min-confidence-grid",
        default="0.20,0.25,0.30,0.35,0.40,0.45,0.50",
        help="Comma-separated part min-confidence sweep values",
    )
    parser.add_argument(
        "--margin-grid",
        default="0.00,0.02,0.05,0.08,0.10,0.12,0.15",
        help="Comma-separated part margin sweep values",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    result = evaluate_router_part_surface(
        args.root,
        config_env=args.config_env,
        device=args.device,
        min_confidence_grid=_parse_grid(args.min_confidence_grid),
        margin_grid=_parse_grid(args.margin_grid),
    )
    body = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
