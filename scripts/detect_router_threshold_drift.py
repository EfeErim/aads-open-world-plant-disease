#!/usr/bin/env python3
"""Detect router confidence drift from JSON/JSONL telemetry rows.

The script compares recent router confidence or margin distributions against a
baseline window using a lightweight Wasserstein-style distance. Missing logs are
reported as `skipped`; drift is reported as `warn` by default so scheduled jobs
surface recalibration needs without breaking source-only CI.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


def _iter_json_payloads(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    paths = [root] if root.is_file() else sorted(root.rglob("*.json")) + sorted(root.rglob("*.jsonl"))
    payloads: list[dict[str, Any]] = []
    for path in paths:
        try:
            if path.suffix.lower() == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        item = json.loads(line)
                        if isinstance(item, dict):
                            payloads.append(item)
            else:
                item = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(item, list):
                    payloads.extend(row for row in item if isinstance(row, dict))
                elif isinstance(item, dict):
                    payloads.append(item)
        except Exception:
            continue
    return payloads


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _extract_metric(payload: dict[str, Any], metric: str) -> float | None:
    candidates = [
        payload.get(metric),
        payload.get(f"router_{metric}"),
        payload.get(f"crop_{metric}"),
    ]
    detection = payload.get("primary_detection")
    if isinstance(detection, dict):
        candidates.extend([detection.get(metric), detection.get(f"crop_{metric}")])
    router = payload.get("router")
    if isinstance(router, dict):
        candidates.extend([router.get(metric), router.get(f"crop_{metric}")])
    for candidate in candidates:
        value = _coerce_float(candidate)
        if value is not None:
            return value
    return None


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def _distribution_distance(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    qs = [index / 20 for index in range(21)]
    return mean(abs(_quantile(left, q) - _quantile(right, q)) for q in qs)


def build_report(root: Path, *, metric: str, threshold: float, min_samples: int) -> dict[str, Any]:
    payloads = _iter_json_payloads(root)
    values = [value for payload in payloads if (value := _extract_metric(payload, metric)) is not None]
    if len(values) < min_samples * 2:
        return {
            "status": "skipped",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "root": str(root),
            "metric": metric,
            "sample_count": len(values),
            "reason": f"Need at least {min_samples * 2} metric values for baseline/recent comparison.",
        }

    midpoint = len(values) // 2
    baseline = values[:midpoint]
    recent = values[midpoint:]
    distance = _distribution_distance(baseline, recent)
    status = "warn" if distance > threshold else "pass"
    return {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "metric": metric,
        "sample_count": len(values),
        "baseline_count": len(baseline),
        "recent_count": len(recent),
        "distance": round(distance, 6),
        "threshold": float(threshold),
        "baseline_mean": round(mean(baseline), 6),
        "recent_mean": round(mean(recent), 6),
        "recommendation": "Re-run router calibration sweep." if status == "warn" else "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--metric", default="confidence")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path(".runtime_tmp/router_drift_report.json"))
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(args.root, metric=args.metric, threshold=args.threshold, min_samples=args.min_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"router_drift status={report['status']} samples={report.get('sample_count', 0)} output={args.output}"
    )
    if args.fail_on_drift and report["status"] == "warn":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
