#!/usr/bin/env python3
"""Evaluate full router crop/part handoff behavior on an offline surface."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config_manager import get_config
from src.router.label_normalization import normalize_part_label
from src.router.router_pipeline import RouterPipeline
from src.shared.contracts import RouterAnalysisResult

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ABSTAIN_STATUSES = {"unknown_crop", "router_uncertain", "unknown", "unavailable", "error", "failed"}


def _is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _iter_image_samples(root: Path, parts: tuple[str, ...], payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    sample_root = root.joinpath(*parts)
    if not sample_root.exists():
        return []
    return [
        {"image_path": image_path, **payload}
        for image_path in sorted(path for path in sample_root.rglob("*") if _is_image_path(path))
    ]


def discover_eval_samples(root: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    id_root = root / "id"
    if id_root.exists():
        for crop_dir in sorted(path for path in id_root.iterdir() if path.is_dir()):
            expected_crop = str(crop_dir.name).strip().lower()
            for part_dir in sorted(path for path in crop_dir.iterdir() if path.is_dir()):
                expected_part = normalize_part_label(part_dir.name) or "unknown"
                samples.extend(
                    _iter_image_samples(
                        root,
                        ("id", expected_crop, part_dir.name),
                        {
                            "group": "id",
                            "expected_crop": expected_crop,
                            "expected_part": expected_part,
                            "expected_handoff": True,
                        },
                    )
                )

    for negative_kind in ("off_crop", "non_plant"):
        negative_root = root / "negatives" / negative_kind
        if negative_root.exists():
            for label_dir in sorted(path for path in negative_root.iterdir() if path.is_dir()):
                samples.extend(
                    _iter_image_samples(
                        root,
                        ("negatives", negative_kind, label_dir.name),
                        {
                            "group": negative_kind,
                            "expected_crop": "unknown",
                            "expected_part": "unknown",
                            "expected_handoff": False,
                            "negative_label": str(label_dir.name),
                        },
                    )
                )

    ambiguous_root = root / "ambiguous"
    if ambiguous_root.exists():
        for label_dir in sorted(path for path in ambiguous_root.iterdir() if path.is_dir()):
            samples.extend(
                _iter_image_samples(
                    root,
                    ("ambiguous", label_dir.name),
                    {
                        "group": "ambiguous",
                        "expected_crop": "unknown",
                        "expected_part": "unknown",
                        "expected_handoff": False,
                        "negative_label": str(label_dir.name),
                    },
                )
            )

    wrong_part_root = root / "wrong_part"
    if wrong_part_root.exists():
        for crop_dir in sorted(path for path in wrong_part_root.iterdir() if path.is_dir()):
            expected_crop = str(crop_dir.name).strip().lower()
            for part_dir in sorted(path for path in crop_dir.iterdir() if path.is_dir()):
                unsupported_part = normalize_part_label(part_dir.name) or str(part_dir.name).strip().lower()
                samples.extend(
                    _iter_image_samples(
                        root,
                        ("wrong_part", crop_dir.name, part_dir.name),
                        {
                            "group": "wrong_part",
                            "expected_crop": expected_crop,
                            "expected_part": "unknown",
                            "unsupported_part": unsupported_part,
                            "expected_handoff": True,
                        },
                    )
                )

    return samples


def _supported_crops(config: Dict[str, Any]) -> set[str]:
    router_cfg = config.get("router", {}) if isinstance(config.get("router"), dict) else {}
    crop_mapping = router_cfg.get("crop_mapping", {}) if isinstance(router_cfg.get("crop_mapping"), dict) else {}
    return {str(crop).strip().lower() for crop in crop_mapping.keys()}


def _routing_score(detection_payload: Dict[str, Any]) -> float:
    quality_score = detection_payload.get("quality_score")
    if quality_score is not None:
        return _coerce_float(quality_score, 0.0)
    return _coerce_float(detection_payload.get("crop_confidence"), 0.0)


def _top_crop_candidates(analysis: RouterAnalysisResult, *, limit: int = 5) -> List[Dict[str, Any]]:
    best_by_crop: Dict[str, Dict[str, Any]] = {}
    detections = list(analysis.detections or [])
    if not detections and analysis.primary_detection is not None:
        detections = [analysis.primary_detection]
    for detection in detections:
        payload = detection.to_dict()
        crop_name = str(payload.get("crop", "unknown") or "unknown").strip().lower()
        candidate = {
            "crop": crop_name,
            "part": normalize_part_label(payload.get("part", "unknown")) or "unknown",
            "crop_confidence": _coerce_float(payload.get("crop_confidence"), 0.0),
            "part_confidence": _coerce_float(payload.get("part_confidence"), 0.0),
            "quality_score": None if payload.get("quality_score") is None else _coerce_float(payload.get("quality_score")),
        }
        existing = best_by_crop.get(crop_name)
        if existing is None or _routing_score(candidate) > _routing_score(existing):
            best_by_crop[crop_name] = candidate

    ordered = sorted(best_by_crop.values(), key=lambda item: _routing_score(item), reverse=True)
    return ordered[: max(1, int(limit))]


def _routing_runner_up(
    analysis: RouterAnalysisResult,
    *,
    primary_crop: str,
) -> tuple[str, float | None]:
    runner_up_crop = ""
    runner_up_score: float | None = None
    for detection in list(analysis.detections or []):
        detection_payload = detection.to_dict()
        candidate_crop = str(detection_payload.get("crop", "") or "").strip().lower()
        if not candidate_crop or candidate_crop == primary_crop:
            continue
        candidate_score = _routing_score(detection_payload)
        if runner_up_score is None or candidate_score > runner_up_score:
            runner_up_crop = candidate_crop
            runner_up_score = candidate_score
    return runner_up_crop, runner_up_score


def _compatible_parts(config: Dict[str, Any], crop_name: str) -> set[str]:
    router_cfg = config.get("router", {}) if isinstance(config.get("router"), dict) else {}
    crop_mapping = router_cfg.get("crop_mapping", {}) if isinstance(router_cfg.get("crop_mapping"), dict) else {}
    payload = crop_mapping.get(crop_name, {}) if isinstance(crop_mapping.get(crop_name), dict) else {}
    return {
        normalize_part_label(part)
        for part in list(payload.get("parts", []))
        if normalize_part_label(part)
    }


def sample_from_analysis(
    *,
    item: Dict[str, Any],
    analysis: RouterAnalysisResult,
    latency_ms: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    detection = analysis.primary_detection
    detection_payload = {} if detection is None else detection.to_dict()
    top_crop_candidates = _top_crop_candidates(analysis)
    predicted_crop = str(detection_payload.get("crop", "unknown") or "unknown").strip().lower()
    predicted_part = normalize_part_label(detection_payload.get("part", "unknown")) or "unknown"
    raw_part_label = str(detection_payload.get("raw_part_label", predicted_part) or predicted_part).strip().lower()
    raw_part_confidence = _coerce_float(
        detection_payload.get("raw_part_confidence"),
        _coerce_float(detection_payload.get("part_confidence"), 0.0),
    )
    raw_part_second_confidence = _coerce_float(detection_payload.get("raw_part_second_confidence"), 0.0)
    raw_part_margin = _coerce_float(
        detection_payload.get("raw_part_margin"),
        raw_part_confidence - raw_part_second_confidence,
    )
    part_rejection_reason = str(detection_payload.get("part_rejection_reason", "") or "")
    confidence = _coerce_float(detection_payload.get("crop_confidence"), 0.0)
    status = str(analysis.status or "ok").strip().lower()
    supported_crops = _supported_crops(config)
    raw_handoff_crop = predicted_crop in supported_crops and predicted_crop != "unknown" and status not in ABSTAIN_STATUSES
    inference_cfg = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}
    min_confidence = _coerce_float(inference_cfg.get("router_min_confidence"), 0.0)
    min_margin = _coerce_float(inference_cfg.get("router_min_margin"), 0.0)
    primary_score = _routing_score(detection_payload)
    runner_up_crop, runner_up_score = _routing_runner_up(analysis, primary_crop=predicted_crop)
    routing_margin = None if runner_up_score is None else primary_score - runner_up_score
    gate_reasons: List[str] = []
    if raw_handoff_crop and min_confidence > 0.0 and confidence < min_confidence:
        gate_reasons.append("router_min_confidence")
    if raw_handoff_crop and min_margin > 0.0 and routing_margin is not None and routing_margin < min_margin:
        gate_reasons.append("router_min_margin")
    handoff_crop = raw_handoff_crop and not gate_reasons
    rejection_reasons = [
        str(reason)
        for reason in [str(analysis.message or ""), part_rejection_reason, *gate_reasons]
        if str(reason).strip()
    ]
    compatible_parts = _compatible_parts(config, predicted_crop)
    unsupported_part_emitted = bool(
        predicted_part not in {"", "unknown"}
        and compatible_parts
        and predicted_part not in compatible_parts
    )
    expected_crop = str(item.get("expected_crop", "unknown") or "unknown").strip().lower()
    expected_part = normalize_part_label(item.get("expected_part", "unknown")) or "unknown"
    group = str(item.get("group", "unknown"))

    return {
        "image_path": str(item["image_path"]),
        "group": group,
        "expected_crop": expected_crop,
        "expected_part": expected_part,
        "expected_handoff": bool(item.get("expected_handoff", False)),
        "predicted_crop": predicted_crop,
        "predicted_part": predicted_part,
        "router_status": status,
        "router_message": str(analysis.message or ""),
        "top_crop_candidates": top_crop_candidates,
        "crop_confidence_margin": (
            None
            if len(top_crop_candidates) < 2
            else round(
                _coerce_float(top_crop_candidates[0].get("crop_confidence"), 0.0)
                - _coerce_float(top_crop_candidates[1].get("crop_confidence"), 0.0),
                4,
            )
        ),
        "crop_confidence": confidence,
        "routing_score": primary_score,
        "runner_up_crop": runner_up_crop,
        "runner_up_score": runner_up_score,
        "routing_margin": routing_margin,
        "router_handoff_crop": bool(raw_handoff_crop),
        "runtime_gate_rejected": bool(gate_reasons),
        "runtime_gate_reasons": gate_reasons,
        "rejection_reason": "; ".join(rejection_reasons),
        "raw_part_label": raw_part_label,
        "raw_part_confidence": raw_part_confidence,
        "raw_part_second_confidence": raw_part_second_confidence,
        "raw_part_margin": raw_part_margin,
        "part_unknown_confidence": _coerce_float(detection_payload.get("part_unknown_confidence"), 0.0),
        "part_rejection_reason": part_rejection_reason,
        "part_open_set_rejection_reason": str(detection_payload.get("part_open_set_rejection_reason", "") or ""),
        "processing_time_ms": float(analysis.processing_time_ms or latency_ms),
        "latency_ms": float(latency_ms),
        "handoff_crop": bool(handoff_crop),
        "crop_correct": group in {"id", "wrong_part"} and predicted_crop == expected_crop,
        "part_correct": group == "id" and predicted_crop == expected_crop and predicted_part == expected_part,
        "part_abstained": predicted_part in {"", "unknown"},
        "unsupported_part_emitted": unsupported_part_emitted,
        "unsupported_part": str(item.get("unsupported_part", "")),
        "detection_count": int(analysis.detections_count or 0),
    }


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(float(numerator) / float(denominator), 4)


def summarize_predictions(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [dict(sample) for sample in samples]
    id_rows = [row for row in rows if row.get("group") == "id"]
    negative_rows = [row for row in rows if row.get("group") in {"off_crop", "non_plant", "ambiguous"}]
    wrong_part_rows = [row for row in rows if row.get("group") == "wrong_part"]
    non_unknown_id_parts = [row for row in id_rows if not bool(row.get("part_abstained", False))]
    correct_non_unknown_id_parts = [row for row in non_unknown_id_parts if bool(row.get("part_correct", False))]
    latencies = [float(row.get("latency_ms", 0.0)) for row in rows]
    sorted_latencies = sorted(latencies)
    p95_index = max(0, min(len(sorted_latencies) - 1, int(round((len(sorted_latencies) - 1) * 0.95)))) if sorted_latencies else 0

    return {
        "sample_count": len(rows),
        "id_sample_count": len(id_rows),
        "negative_sample_count": len(negative_rows),
        "wrong_part_sample_count": len(wrong_part_rows),
        "crop_accuracy": _rate(sum(1 for row in id_rows if bool(row.get("crop_correct", False))), len(id_rows)),
        "negative_false_accept_rate": _rate(
            sum(1 for row in negative_rows if bool(row.get("handoff_crop", False))),
            len(negative_rows),
        ),
        "abstention_rate": _rate(
            sum(1 for row in rows if not bool(row.get("handoff_crop", False))),
            len(rows),
        ),
        "part_non_unknown_precision": _rate(len(correct_non_unknown_id_parts), len(non_unknown_id_parts)),
        "part_recall": _rate(len(correct_non_unknown_id_parts), len(id_rows)),
        "part_abstention_rate": _rate(sum(1 for row in id_rows if bool(row.get("part_abstained", False))), len(id_rows)),
        "unsupported_part_emissions": sum(1 for row in rows if bool(row.get("unsupported_part_emitted", False))),
        "wrong_part_rejection_rate": _rate(
            sum(
                1
                for row in wrong_part_rows
                if bool(row.get("crop_correct", False)) and bool(row.get("part_abstained", False))
            ),
            len(wrong_part_rows),
        ),
        "mean_latency_ms": 0.0 if not latencies else round(statistics.fmean(latencies), 4),
        "p95_latency_ms": 0.0 if not sorted_latencies else round(sorted_latencies[p95_index], 4),
    }


def risk_coverage_curve(samples: List[Dict[str, Any]], *, thresholds: List[float]) -> List[Dict[str, Any]]:
    id_rows = [row for row in samples if row.get("group") == "id"]
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        accepted = [
            row
            for row in id_rows
            if bool(row.get("handoff_crop", False)) and float(row.get("crop_confidence", 0.0)) >= float(threshold)
        ]
        errors = sum(1 for row in accepted if not bool(row.get("crop_correct", False)))
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "coverage": _rate(len(accepted), len(id_rows)),
                "risk": _rate(errors, len(accepted)),
                "accepted": len(accepted),
                "errors": errors,
            }
        )
    return rows


def threshold_sweep(samples: List[Dict[str, Any]], *, thresholds: List[float]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        thresholded = []
        for sample in samples:
            row = dict(sample)
            raw_handoff = bool(row.get("router_handoff_crop", row.get("handoff_crop", False)))
            row["handoff_crop"] = raw_handoff
            if not raw_handoff or float(row.get("crop_confidence", 0.0)) < float(threshold):
                row["handoff_crop"] = False
                row["predicted_part"] = "unknown"
                row["part_abstained"] = True
            thresholded.append(row)
        rows.append({"threshold": round(float(threshold), 4), **summarize_predictions(thresholded)})
    eligible = [row for row in rows if float(row["negative_false_accept_rate"]) <= 0.05] or rows
    recommended = None
    if eligible:
        recommended = sorted(
            eligible,
            key=lambda row: (
                float(row["negative_false_accept_rate"]),
                -float(row["crop_accuracy"]),
                -float(row["part_non_unknown_precision"]),
                float(row["abstention_rate"]),
                float(row["threshold"]),
            ),
        )[0]
    return {"rows": rows, "recommended": recommended}


def evaluate_router_surface(
    root: Path,
    *,
    config_env: str | None = "colab",
    device: str = "cuda",
    threshold_grid: List[float] | None = None,
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

    samples: List[Dict[str, Any]] = []
    for item in dataset:
        image = Image.open(item["image_path"]).convert("RGB")
        started = time.perf_counter()
        analysis = router.analyze_image_result(image)
        latency_ms = (time.perf_counter() - started) * 1000.0
        samples.append(
            sample_from_analysis(
                item=item,
                analysis=analysis,
                latency_ms=latency_ms,
                config=config,
            )
        )

    thresholds = (
        [0.0, 0.25, 0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90]
        if threshold_grid is None
        else [float(value) for value in threshold_grid]
    )
    per_group = {
        group: summarize_predictions([sample for sample in samples if sample.get("group") == group])
        for group in sorted({str(sample.get("group", "")) for sample in samples})
    }
    return {
        "dataset_root": str(root),
        "sample_count": len(samples),
        "metrics": summarize_predictions(samples),
        "groups": per_group,
        "risk_coverage_curve": risk_coverage_curve(samples, thresholds=thresholds),
        "threshold_sweep": threshold_sweep(samples, thresholds=thresholds),
        "samples": samples,
    }


def _parse_grid(raw_value: str) -> List[float]:
    return [float(item.strip()) for item in str(raw_value).split(",") if str(item).strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate full router crop/part handoff behavior.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Eval root: data/router_eval/{id,negatives,ambiguous,wrong_part}/...",
    )
    parser.add_argument("--config-env", default="colab", help="Config environment override (default: colab)")
    parser.add_argument("--device", default="cuda", help="Torch device preference")
    parser.add_argument(
        "--threshold-grid",
        default="0.0,0.25,0.40,0.50,0.60,0.65,0.70,0.80,0.90",
        help="Comma-separated crop confidence threshold sweep values",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    result = evaluate_router_surface(
        args.root,
        config_env=args.config_env,
        device=args.device,
        threshold_grid=_parse_grid(args.threshold_grid),
    )
    body = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
