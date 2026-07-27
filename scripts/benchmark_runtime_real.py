#!/usr/bin/env python3
"""Capture opt-in real-runtime inference benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.workflows.inference import InferenceWorkflow


def _time_call(fn: Callable[[], Any]) -> tuple[float, Any]:
    started = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, result


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    rank = max(0.0, min(1.0, float(percentile) / 100.0)) * float(len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - float(lower)
    return float(ordered[lower] + ((ordered[upper] - ordered[lower]) * weight))


def _status_of(payload: Any) -> str:
    status = getattr(payload, "status", None)
    if status is not None:
        return str(status)
    if isinstance(payload, dict):
        return str(payload.get("status", ""))
    return type(payload).__name__


def _summarize_measurements(samples_ms: Sequence[float]) -> Dict[str, float | int | None]:
    if not samples_ms:
        return {
            "count": 0,
            "min_ms": None,
            "max_ms": None,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        }
    values = [float(sample) for sample in samples_ms]
    return {
        "count": int(len(values)),
        "min_ms": round(min(values), 4),
        "max_ms": round(max(values), 4),
        "mean_ms": round(sum(values) / float(len(values)), 4),
        "p50_ms": round(float(_percentile(values, 50.0) or 0.0), 4),
        "p95_ms": round(float(_percentile(values, 95.0) or 0.0), 4),
        "p99_ms": round(float(_percentile(values, 99.0) or 0.0), 4),
    }


def _benchmark_repeated(fn: Callable[[], Any], *, repeat: int, warmup: int) -> Dict[str, Any]:
    for _ in range(max(0, int(warmup))):
        fn()
    samples_ms = []
    statuses: list[str] = []
    for _ in range(max(1, int(repeat))):
        elapsed_ms, result = _time_call(fn)
        samples_ms.append(elapsed_ms)
        statuses.append(_status_of(result))
    status_counts = {str(name): int(count) for name, count in Counter(statuses).items()}
    return {
        "summary": _summarize_measurements(samples_ms),
        "status_counts": status_counts,
        "last_status": statuses[-1] if statuses else "",
    }


def _build_workflow(args: argparse.Namespace) -> InferenceWorkflow:
    return InferenceWorkflow(
        environment=args.environment,
        device=args.device,
        adapter_root=args.adapter_root,
    )


def collect_real_benchmarks(args: argparse.Namespace) -> Dict[str, Any]:
    image = Image.open(args.image).convert("RGB")

    cold_workflow = _build_workflow(args)
    cold_router_ms: float | None = None
    cold_router_error = ""
    try:
        cold_router_ms, _ = _time_call(cold_workflow.runtime.load_router)
    except Exception as exc:
        cold_router_error = str(exc)

    workflow = _build_workflow(args)
    router_ready = False
    router_error = ""
    try:
        workflow.runtime.load_router()
        router_ready = True
    except Exception as exc:
        router_error = str(exc)

    payload: Dict[str, Any] = {
        "script": "benchmark_runtime_real",
        "image_path": str(Path(args.image).resolve()),
        "device": str(args.device),
        "environment": str(args.environment or ""),
        "adapter_root": ("" if args.adapter_root is None else str(Path(args.adapter_root).resolve())),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "router_cold_load_ms": None if cold_router_ms is None else round(float(cold_router_ms), 4),
        "router_cold_load_error": str(cold_router_error),
    }

    if router_ready:
        payload["router_only"] = _benchmark_repeated(
            lambda: workflow.runtime._route(image),
            repeat=args.repeat,
            warmup=args.warmup,
        )
        payload["inference_warm_single"] = _benchmark_repeated(
            lambda: workflow.predict_result(image, return_ood=not args.disable_ood),
            repeat=args.repeat,
            warmup=args.warmup,
        )
    else:
        payload["router_only"] = {"error": str(router_error)}
        payload["inference_warm_single"] = {"error": str(router_error)}

    if args.crop_hint:
        payload["inference_crop_hint"] = _benchmark_repeated(
            lambda: workflow.predict_result(
                image,
                crop_hint=args.crop_hint,
                part_hint=args.part_hint,
                return_ood=not args.disable_ood,
                trust_crop_hint=bool(args.trust_crop_hint),
            ),
            repeat=args.repeat,
            warmup=args.warmup,
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture opt-in real runtime benchmarks.")
    parser.add_argument("--image", type=Path, required=True, help="Path to one input image.")
    parser.add_argument("--device", default="cuda", help="Runtime device passed to InferenceWorkflow.")
    parser.add_argument("--environment", default=None, help="Optional config environment.")
    parser.add_argument("--adapter-root", type=Path, default=None, help="Optional adapter root override.")
    parser.add_argument("--crop-hint", default=None, help="Optional crop hint for inference timing.")
    parser.add_argument("--part-hint", default=None, help="Optional part hint paired with --crop-hint.")
    parser.add_argument("--trust-crop-hint", action="store_true", help="Trust --crop-hint and skip router timing.")
    parser.add_argument("--repeat", type=int, default=5, help="Measured iterations per benchmark.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations before timing.")
    parser.add_argument("--disable-ood", action="store_true", help="Disable OOD fields in inference payloads.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Input image not found: {args.image}")

    benchmarks = collect_real_benchmarks(args)
    body = json.dumps(benchmarks, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
