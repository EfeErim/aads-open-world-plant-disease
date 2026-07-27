"""Small training-related helper functions moved out of workflows for reuse and testing."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict

from src.shared.json_utils import deep_merge


def loader_size(loader: Any, *, loader_name: str = "loader") -> int:
    if loader is None:
        return 0
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        try:
            return int(len(dataset))
        except TypeError:
            pass
        except Exception as exc:
            raise RuntimeError(f"Failed to determine dataset size for {loader_name}.") from exc
    try:
        return int(len(loader))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"Failed to determine dataset size for {loader_name}.") from exc


def build_loader_sizes(loaders: Dict[str, Any]) -> Dict[str, int]:
    return {
        name: loader_size(loader, loader_name=f"{name} loader")
        for name, loader in loaders.items()
    }


def loader_batch_count(loader: Any, *, loader_name: str = "loader") -> int:
    if loader is None:
        return 0
    try:
        return int(len(loader))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"Failed to determine batch count for {loader_name}.") from exc


def stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stringify_paths(item) for item in value]
    return value


def _dataset_class_counts(loader: Any) -> Dict[str, int]:
    dataset = getattr(loader, "dataset", None)
    classes = [str(name) for name in getattr(dataset, "classes", [])]
    labels = getattr(dataset, "labels", None)
    if not classes or not isinstance(labels, list):
        return {}
    counts = Counter(int(label) for label in labels)
    return {
        class_name: int(counts.get(class_index, 0))
        for class_index, class_name in enumerate(classes)
    }


def _resolve_loader_sampler_runtime(loader: Any) -> Dict[str, Any]:
    return {
        "requested_sampler": str(getattr(loader, "_requested_sampler", "auto")),
        "resolved_sampler": str(getattr(loader, "_resolved_sampler", "auto")),
        "decision_reason": str(getattr(loader, "_sampler_decision_reason", "unknown")),
        "imbalance_ratio": float(getattr(loader, "_sampler_imbalance_ratio", 1.0)),
        "imbalance_ratio_threshold": float(getattr(loader, "_sampler_imbalance_ratio_threshold", 1.0)),
        "class_counts": dict(getattr(loader, "_sampler_class_counts", {})),
    }


def _inject_class_balance_runtime(config: Dict[str, Any], class_balance_runtime: Dict[str, Any]) -> Dict[str, Any]:
    return deep_merge(
        dict(config),
        {
            "training": {
                "continual": {
                    "class_balance": dict(class_balance_runtime),
                }
            }
        },
    )
