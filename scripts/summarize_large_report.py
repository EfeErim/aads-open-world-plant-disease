"""Summarize large JSON/CSV reports without dumping full artifacts into Codex context."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

STATUS_KEY_PARTS = ("status", "outcome", "state", "error")
ROW_KEY_HINTS = ("row", "sample", "result", "prediction", "target")
METRIC_KEY_PARTS = (
    "accuracy",
    "auroc",
    "capture",
    "confidence",
    "coverage",
    "count",
    "error",
    "f1",
    "false_positive",
    "loss",
    "macro",
    "precision",
    "recall",
    "risk",
    "score",
    "support",
)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _shorten(value: Any, *, limit: int = 160) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[: limit - 3] + "..."
    return value


def _path_join(parent: str, key: str) -> str:
    return key if not parent else f"{parent}.{key}"


def _scalar_preview(payload: JsonDict, *, max_items: int) -> JsonDict:
    preview: JsonDict = {}
    for key, value in payload.items():
        if len(preview) >= max_items:
            break
        if _is_scalar(value):
            preview[key] = _shorten(value)
    return preview


def _container_preview(value: Any, *, max_items: int) -> JsonDict:
    if isinstance(value, dict):
        return {
            "type": "dict",
            "key_count": len(value),
            "keys": list(value.keys())[:max_items],
            "scalar_preview": _scalar_preview(value, max_items=max_items),
        }
    if isinstance(value, list):
        item_types = Counter(type(item).__name__ for item in value[: max_items * 4])
        preview: JsonDict = {
            "type": "list",
            "length": len(value),
            "sample_item_types": dict(item_types.most_common(max_items)),
        }
        first_dict = next((item for item in value if isinstance(item, dict)), None)
        if first_dict is not None:
            preview["sample_dict_keys"] = list(first_dict.keys())[:max_items]
            preview["sample_scalar_preview"] = _scalar_preview(first_dict, max_items=max_items)
        return preview
    return {"type": type(value).__name__, "value": _shorten(value)}


def _walk_json(value: Any, *, max_scan_items: int) -> list[tuple[str, Any]]:
    seen = 0
    stack: list[tuple[str, Any]] = [("", value)]
    visited: list[tuple[str, Any]] = []
    while stack and seen < max_scan_items:
        path, current = stack.pop()
        visited.append((path, current))
        seen += 1
        if isinstance(current, dict):
            for key, child in reversed(list(current.items())):
                stack.append((_path_join(path, str(key)), child))
        elif isinstance(current, list):
            for index, child in reversed(list(enumerate(current[:max_scan_items]))):
                stack.append((f"{path}[{index}]" if path else f"[{index}]", child))
    return visited


def _collect_numeric_metrics(items: list[tuple[str, Any]], *, max_items: int) -> list[JsonDict]:
    buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for path, value in items:
        if not _is_number(value):
            continue
        key = path.rsplit(".", 1)[-1].split("[", 1)[0]
        if any(part in key.lower() for part in METRIC_KEY_PARTS):
            buckets[key].append((path, float(value)))

    metrics: list[JsonDict] = []
    for key, values in buckets.items():
        numeric_values = [value for _, value in values]
        metrics.append(
            {
                "key": key,
                "count": len(values),
                "min": min(numeric_values),
                "max": max(numeric_values),
                "mean": sum(numeric_values) / len(numeric_values),
                "sample_paths": [path for path, _ in values[:max_items]],
            }
        )
    return sorted(metrics, key=lambda item: (-int(item["count"]), str(item["key"])))[: max_items * 4]


def _collect_status_counts(items: list[tuple[str, Any]], *, max_items: int) -> list[JsonDict]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for path, value in items:
        if not isinstance(value, str):
            continue
        key = path.rsplit(".", 1)[-1].split("[", 1)[0]
        if any(part in key.lower() for part in STATUS_KEY_PARTS):
            buckets[key][_shorten(value, limit=80)] += 1

    status_counts: list[JsonDict] = []
    for key, counts in buckets.items():
        status_counts.append(
            {
                "key": key,
                "unique_value_count": len(counts),
                "top_values": dict(counts.most_common(max_items)),
            }
        )
    return sorted(status_counts, key=lambda item: (-int(item["unique_value_count"]), str(item["key"])))[: max_items * 4]


def _list_score(path: str, value: list[Any]) -> int:
    if not value:
        return 0
    first_dicts = [item for item in value[:20] if isinstance(item, dict)]
    if not first_dicts:
        return 0
    keys = {str(key).lower() for item in first_dicts for key in item}
    score = len(first_dicts)
    score += sum(3 for hint in ROW_KEY_HINTS if hint in path.lower())
    score += sum(2 for key in keys if any(part in key for part in METRIC_KEY_PARTS))
    score += sum(2 for key in keys if any(part in key for part in STATUS_KEY_PARTS))
    score += sum(2 for key in keys if key in {"target", "target_id", "crop", "part", "expected_label", "diagnosis"})
    return score


def _extract_representative_rows(items: list[tuple[str, Any]], *, max_items: int) -> JsonDict | None:
    candidates = [(path, value, _list_score(path, value)) for path, value in items if isinstance(value, list)]
    candidates = [(path, value, score) for path, value, score in candidates if score > 0]
    if not candidates:
        return None
    path, rows, _score = max(candidates, key=lambda item: (item[2], len(item[1])))
    dict_rows = [row for row in rows if isinstance(row, dict)]
    columns = sorted({str(key) for row in dict_rows[: min(len(dict_rows), max_items * 10)] for key in row})
    samples = []
    preferred_keys = [
        "target_id",
        "crop",
        "part",
        "expected_label",
        "diagnosis",
        "accuracy",
        "macro_f1",
        "full_confidence",
        "requires_review",
        "roi_evidence_status",
        "status",
        "outcome",
    ]
    for row in dict_rows[:max_items]:
        sample: JsonDict = {}
        for key in preferred_keys:
            if key in row and _is_scalar(row[key]):
                sample[key] = _shorten(row[key])
        if len(sample) < max_items:
            for key, value in row.items():
                if len(sample) >= max_items:
                    break
                if key not in sample and _is_scalar(value):
                    sample[str(key)] = _shorten(value)
        samples.append(sample)
    return {"path": path, "row_count": len(dict_rows), "columns": columns[: max_items * 6], "sample_rows": samples}


def summarize_json(path: Path, *, max_items: int = 8, max_scan_items: int = 50_000) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    visited = _walk_json(payload, max_scan_items=max_scan_items)
    summary: JsonDict = {
        "path": str(path),
        "format": "json",
        "size_bytes": path.stat().st_size,
        "root": _container_preview(payload, max_items=max_items),
        "scanned_items": len(visited),
        "numeric_metrics": _collect_numeric_metrics(visited, max_items=max_items),
        "status_counts": _collect_status_counts(visited, max_items=max_items),
    }
    representative_rows = _extract_representative_rows(visited, max_items=max_items)
    if representative_rows:
        summary["representative_rows"] = representative_rows
    return summary


def summarize_csv(path: Path, *, max_items: int = 8) -> JsonDict:
    row_count = 0
    samples: list[JsonDict] = []
    numeric_values: dict[str, list[float]] = defaultdict(list)
    categorical_counts: dict[str, Counter[str]] = defaultdict(Counter)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            if len(samples) < max_items:
                samples.append({key: _shorten(value) for key, value in row.items() if value not in (None, "")})
            for key, raw_value in row.items():
                if raw_value in (None, ""):
                    continue
                try:
                    number = float(raw_value)
                except ValueError:
                    if len(categorical_counts[key]) <= max_items * 4:
                        categorical_counts[key][_shorten(raw_value, limit=80)] += 1
                else:
                    numeric_values[key].append(number)

    numeric_summary = []
    for key, values in numeric_values.items():
        if not values:
            continue
        numeric_summary.append(
            {
                "key": key,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
        )
    categorical_summary = [
        {"key": key, "top_values": dict(counts.most_common(max_items))}
        for key, counts in categorical_counts.items()
        if counts
    ]
    return {
        "path": str(path),
        "format": "csv",
        "size_bytes": path.stat().st_size,
        "columns": columns,
        "row_count": row_count,
        "sample_rows": samples,
        "numeric_columns": sorted(numeric_summary, key=lambda item: str(item["key"]))[: max_items * 4],
        "categorical_columns": sorted(categorical_summary, key=lambda item: str(item["key"]))[:max_items],
    }


def summarize_report(path: Path, *, max_items: int = 8, max_scan_items: int = 50_000) -> JsonDict:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return summarize_json(path, max_items=max_items, max_scan_items=max_scan_items)
    if suffix == ".csv":
        return summarize_csv(path, max_items=max_items)
    raise ValueError(f"Unsupported report format: {suffix or '<none>'}. Expected .json or .csv")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSON or CSV report to summarize.")
    parser.add_argument("--max-items", type=int, default=8, help="Maximum samples per summary section.")
    parser.add_argument("--max-scan-items", type=int, default=50_000, help="Maximum JSON nodes to scan.")
    args = parser.parse_args(argv)

    summary = summarize_report(args.path, max_items=args.max_items, max_scan_items=args.max_scan_items)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
