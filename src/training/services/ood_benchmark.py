"""Held-out OOD evidence helpers for production readiness."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, cast

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler

from src.shared.artifacts import ArtifactStore
from src.training.services.metrics import (
    compute_plan_metrics,
    load_plan_targets,
    validate_ood_metrics,
    write_plan_metric_artifact,
)
from src.training.services.ood_score_selection import (
    is_auto_primary_score_method,
    normalize_requested_primary_score_method,
    resolve_runtime_primary_score_method,
    select_best_ood_score_method,
)
from src.training.services.reporting import persist_ood_benchmark_artifacts
from src.training.validation import evaluate_model_with_artifact_metrics

_OOD_BENCHMARK_METRIC_NAMES = (
    "accuracy",
    "ood_auroc",
    "ood_false_positive_rate",
    "ood_samples",
    "in_distribution_samples",
    "sure_ds_f1",
    "conformal_empirical_coverage",
    "conformal_avg_set_size",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BenchmarkRunPreparation:
    resolved_classes: List[str]
    target_values: Dict[str, Any]
    artifact_root: Path
    base_context: Dict[str, Any]
    resumable_folds: Dict[str, Dict[str, Any]]
    train_loader: Any
    calibration_loader: Any
    eval_loader: Any
    calibration_split_name: str
    eval_split_name: str
    requested_primary_score_method: str
    primary_score_method: str


@dataclass(frozen=True)
class _BenchmarkDatasetStage:
    train_dataset: Dataset
    calibration_dataset: Dataset
    eval_dataset: Dataset
    class_to_idx: Dict[str, int]
    train_labels: List[int]
    calibration_labels: List[int]
    eval_labels: List[int]
    dataset_fingerprints: Dict[str, Any]



def _loader_size(loader: Any) -> int:
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        try:
            return int(len(dataset))
        except TypeError:
            pass
        except Exception as exc:
            raise RuntimeError("Failed to resolve dataset length for OOD benchmark loader.") from exc
    try:
        return int(len(loader))
    except TypeError:
        return 0
    except Exception as exc:
        raise RuntimeError("Failed to resolve loader length for OOD benchmark loader.") from exc


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _dataset_labels(dataset: Any) -> List[int]:
    labels = getattr(dataset, "labels", None)
    if isinstance(labels, list):
        return [int(label) for label in labels]
    resolved: List[int] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not isinstance(sample, tuple) or len(sample) < 2:
            raise ValueError("Dataset must return `(image, label)` tuples for OOD benchmarking.")
        resolved.append(int(sample[1]))
    return resolved


class FilteredLabelDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        *,
        classes: Sequence[str],
        label_map: Optional[Dict[int, int]] = None,
        constant_label: Optional[int] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = [int(index) for index in indices]
        self.classes = [str(name) for name in classes]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self._label_map = {int(key): int(value) for key, value in dict(label_map or {}).items()}
        self._constant_label = int(constant_label) if constant_label is not None else None
        base_labels = _dataset_labels(base_dataset)
        self.labels = [self._remap_label(base_labels[index]) for index in self.indices]

    def _remap_label(self, label: int) -> int:
        if self._constant_label is not None:
            return int(self._constant_label)
        if label not in self._label_map:
            raise KeyError(f"Label {label} is not present in the fold remapping.")
        return int(self._label_map[label])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[self.indices[int(index)]]
        del label
        return image, self.labels[int(index)]


def _build_weighted_sampler(labels: Sequence[int]) -> WeightedRandomSampler:
    counts = Counter(int(label) for label in labels)
    weights = [1.0 / float(counts[int(label)]) for label in labels]
    generator = torch.Generator()
    generator.manual_seed(42)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True, generator=generator)


def _clone_loader(source_loader: DataLoader, dataset: Dataset, *, training_like: bool) -> DataLoader:
    batch_size = int(getattr(source_loader, "batch_size", 1) or 1)
    num_workers = int(getattr(source_loader, "num_workers", 0))
    pin_memory = bool(getattr(source_loader, "pin_memory", False))
    persistent_workers = bool(getattr(source_loader, "persistent_workers", False)) if num_workers > 0 else False
    collate_fn = getattr(source_loader, "collate_fn", None)
    worker_init_fn = getattr(source_loader, "worker_init_fn", None)
    generator = getattr(source_loader, "generator", None)
    drop_last = bool(getattr(source_loader, "drop_last", False)) if training_like else False
    timeout = float(getattr(source_loader, "timeout", 0.0))
    prefetch_factor = getattr(source_loader, "prefetch_factor", None)

    sampler = None
    shuffle = False
    source_sampler = getattr(source_loader, "sampler", None)
    if training_like and isinstance(source_sampler, WeightedRandomSampler) and hasattr(dataset, "labels"):
        sampler = _build_weighted_sampler(getattr(dataset, "labels", []))
    elif training_like and isinstance(source_sampler, RandomSampler):
        shuffle = True

    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "collate_fn": collate_fn,
        "worker_init_fn": worker_init_fn,
        "generator": generator,
        "drop_last": drop_last,
        "timeout": timeout,
    }
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    cloned = DataLoader(dataset, **kwargs)
    if hasattr(source_loader, "_seed_base"):
        setattr(cloned, "_seed_base", getattr(source_loader, "_seed_base"))
    if hasattr(source_loader, "_sampler_seed_base"):
        setattr(cloned, "_sampler_seed_base", getattr(source_loader, "_sampler_seed_base"))
    return cloned


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _std(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean_value = _mean(values)
    assert mean_value is not None
    variance = sum((float(value) - mean_value) ** 2 for value in values) / float(len(values))
    return float(variance ** 0.5)


def _select_eval_loader(loaders: Dict[str, Any], *, calibration_split_name: str) -> tuple[str, Any]:
    test_loader = loaders.get("test")
    if test_loader is not None and _loader_size(test_loader) > 0:
        return "test", test_loader
    if calibration_split_name == "val":
        return "", None
    val_loader = loaders.get("val")
    if val_loader is not None and _loader_size(val_loader) > 0:
        return "val", val_loader
    return "", None


def _select_calibration_loader(loaders: Dict[str, Any]) -> tuple[str, Any]:
    val_loader = loaders.get("val")
    if val_loader is not None and _loader_size(val_loader) > 0:
        return "val", val_loader
    return "train", loaders.get("train")


def _empty_metric_map() -> Dict[str, None]:
    return {name: None for name in _OOD_BENCHMARK_METRIC_NAMES}


def _primary_score_method(config: Dict[str, Any]) -> str:
    return normalize_requested_primary_score_method(
        config.get("training", {})
        .get("continual", {})
        .get("ood", {})
        .get("primary_score_method", "auto")
    )


def _normalize_resume_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_resume_value(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_resume_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _resume_relevant_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    training_cfg = config.get("training", {}).get("continual", {})
    return dict(_normalize_resume_value(training_cfg))


def _hash_sequence(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _dataset_file_signature(dataset: Any) -> str:
    image_paths = getattr(dataset, "image_paths", None)
    if not isinstance(image_paths, list):
        return ""

    digest = hashlib.sha256()
    for raw_path in image_paths:
        path = Path(raw_path)
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        try:
            stat = path.stat()
            digest.update(str(int(stat.st_size)).encode("utf-8"))
            digest.update(b":")
            digest.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
        except Exception as exc:
            digest.update(f"missing:{exc.__class__.__name__}".encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _dataset_resume_fingerprint(dataset: Any) -> Dict[str, Any]:
    labels = _dataset_labels(dataset)
    label_counts = {
        str(label): int(count)
        for label, count in sorted(Counter(int(label) for label in labels).items(), key=lambda item: item[0])
    }
    fingerprint: Dict[str, Any] = {
        "dataset_type": type(dataset).__name__,
        "size": len(labels),
        "classes": [str(name) for name in getattr(dataset, "classes", [])],
        "label_counts": label_counts,
        "label_sequence_sha256": _hash_sequence([str(label) for label in labels]),
    }
    for attr_name in ("data_dir", "crop", "split"):
        attr_value = getattr(dataset, attr_name, None)
        if attr_value is not None:
            fingerprint[attr_name] = str(attr_value)
    path_signature = _dataset_file_signature(dataset)
    if path_signature:
        fingerprint["path_signature_sha256"] = path_signature
    return fingerprint


def _build_resume_key(
    *,
    crop_name: str,
    held_out_class: str,
    seen_classes: Sequence[str],
    sample_counts: Dict[str, int],
    dataset_fingerprints: Dict[str, Any],
    config: Dict[str, Any],
    device: str,
    num_epochs: Optional[int],
) -> str:
    payload = {
        "resume_key_schema": "v2",
        "crop_name": str(crop_name),
        "held_out_class": str(held_out_class),
        "seen_classes": [str(name) for name in seen_classes],
        "sample_counts": {str(key): int(value) for key, value in sample_counts.items()},
        "dataset_fingerprints": _normalize_resume_value(dataset_fingerprints),
        "training": _resume_relevant_training_config(config),
        "device": str(device),
        "num_epochs": None if num_epochs is None else int(num_epochs),
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _load_existing_benchmark_summary(artifact_root: Path) -> Dict[str, Any]:
    summary_path = Path(artifact_root) / "ood_benchmark" / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Ignoring invalid OOD benchmark summary at %s: %s", summary_path, exc)
        return {}
    except OSError as exc:
        logger.warning("Ignoring unreadable OOD benchmark summary at %s: %s", summary_path, exc)
        return {}


def _build_resumable_fold_index(summary_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    fold_index: Dict[str, Dict[str, Any]] = {}
    for fold in list(summary_payload.get("folds", [])) if isinstance(summary_payload, dict) else []:
        if not isinstance(fold, dict):
            continue
        if str(fold.get("status", "")) != "completed":
            continue
        held_out_class = str(fold.get("held_out_class", "")).strip()
        resume_key = str(fold.get("resume_key", "")).strip()
        metric_gate_path = str(dict(fold.get("paths", {})).get("metric_gate_json", "")).strip()
        if not held_out_class or not resume_key or not metric_gate_path:
            continue
        if not Path(metric_gate_path).exists():
            continue
        fold_index[f"{held_out_class}:{resume_key}"] = fold
    return fold_index


def _compute_method_metrics(evaluation: Any) -> Dict[str, Dict[str, Optional[float]]]:
    method_scores = dict(getattr(evaluation, "ood_scores_by_method", {}) or {})
    method_metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for method_name, scores in method_scores.items():
        method_metrics[str(method_name)] = compute_plan_metrics(
            y_true=evaluation.y_true,
            y_pred=evaluation.y_pred,
            ood_labels=evaluation.ood_labels,
            ood_scores=scores,
            sure_ds_f1=evaluation.sure_ds_f1,
            conformal_empirical_coverage=evaluation.conformal_empirical_coverage,
            conformal_avg_set_size=evaluation.conformal_avg_set_size,
        )
    return method_metrics


def _persist_benchmark_summary(
    *,
    artifact_root: Path,
    summary_payload: Dict[str, Any],
    telemetry: Any,
) -> Dict[str, Any]:
    artifacts = persist_ood_benchmark_artifacts(
        artifact_root=artifact_root,
        summary_payload=summary_payload,
        telemetry=telemetry,
    )
    summary_payload["paths"] = {key: str(path) for key, path in artifacts.items()}
    return summary_payload


def _emit_telemetry_log(telemetry: Any, message: str, *, level: str = "info") -> None:
    if telemetry is None or not hasattr(telemetry, "emit_log"):
        return
    telemetry.emit_log(message, phase="evaluation", level=level)


def _resource_snapshot(device: str) -> Dict[str, Any]:
    resolved_device = str(device or "")
    if not resolved_device.startswith("cuda") or not torch.cuda.is_available():
        return {}
    try:
        if ":" in resolved_device:
            device_index = int(resolved_device.split(":", 1)[1])
        else:
            device_index = torch.cuda.current_device()
        return {
            "device": resolved_device,
            "device_name": str(torch.cuda.get_device_name(device_index)),
            "memory_allocated_mb": round(float(torch.cuda.memory_allocated(device_index)) / (1024.0 * 1024.0), 3),
            "memory_reserved_mb": round(float(torch.cuda.memory_reserved(device_index)) / (1024.0 * 1024.0), 3),
            "max_memory_allocated_mb": round(
                float(torch.cuda.max_memory_allocated(device_index)) / (1024.0 * 1024.0),
                3,
            ),
            "max_memory_reserved_mb": round(
                float(torch.cuda.max_memory_reserved(device_index)) / (1024.0 * 1024.0),
                3,
            ),
        }
    except (RuntimeError, ValueError, TypeError, OSError):
        return {}


def _persist_benchmark_progress(
    *,
    artifact_root: Path,
    telemetry: Any,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    progress_payload = {
        "schema_version": "v6_ood_benchmark_progress",
        "ts": _utc_now_iso(),
        **dict(payload),
    }
    progress_dir = Path(artifact_root) / "ood_benchmark"
    progress_path = ArtifactStore(progress_dir).write_json("progress.json", progress_payload)
    if telemetry is not None and hasattr(telemetry, "copy_artifact_file"):
        telemetry.copy_artifact_file(progress_path, "ood_benchmark/progress.json")
    from src.shared.telemetry import update_latest as _update_latest
    _update_latest(
        telemetry,
        {
            "phase": "ood_benchmark",
            "status": progress_payload.get("status"),
            "stage": progress_payload.get("stage"),
            "fold_index": progress_payload.get("fold_index"),
            "fold_total": progress_payload.get("fold_total"),
            "held_out_class": progress_payload.get("held_out_class"),
            "completed_folds": progress_payload.get("completed_folds"),
            "failed_folds": progress_payload.get("failed_folds"),
            "last_error": progress_payload.get("last_error"),
        },
    )
    return progress_payload


def _persist_fold_failure_diagnostics(
    *,
    artifact_root: Path,
    telemetry: Any,
    held_out_class: str,
    failure_payload: Dict[str, Any],
    traceback_text: str,
) -> Dict[str, str]:
    fold_dir = Path(artifact_root) / "ood_benchmark" / "folds" / held_out_class
    store = ArtifactStore(fold_dir)
    failure_json = store.write_json("failure.json", failure_payload)
    failure_txt = store.write_text("failure_traceback.txt", traceback_text)
    if telemetry is not None and hasattr(telemetry, "copy_artifact_file"):
        telemetry.copy_artifact_file(failure_json, f"ood_benchmark/folds/{held_out_class}/failure.json")
        telemetry.copy_artifact_file(failure_txt, f"ood_benchmark/folds/{held_out_class}/failure_traceback.txt")
    return {
        "failure_json": str(failure_json),
        "failure_traceback_txt": str(failure_txt),
    }


def _persist_fold_metric_gate(
    *,
    artifact_root: Path,
    telemetry: Any,
    held_out_class: str,
    evaluation: Any,
    seen_classes: Sequence[str],
    target_values: Dict[str, float],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    fold_dir = Path(artifact_root) / "ood_benchmark" / "folds" / held_out_class
    primary_score_method = str(
        getattr(
            evaluation,
            "ood_primary_score_method",
            context.get("ood_primary_score_method", "ensemble"),
        )
        or "ensemble"
    )
    metrics = compute_plan_metrics(
        y_true=evaluation.y_true,
        y_pred=evaluation.y_pred,
        ood_labels=evaluation.ood_labels,
        ood_scores=evaluation.ood_scores,
        sure_ds_f1=evaluation.sure_ds_f1,
        conformal_empirical_coverage=evaluation.conformal_empirical_coverage,
        conformal_avg_set_size=evaluation.conformal_avg_set_size,
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    metric_gate_json = fold_dir / "metric_gate.json"
    metric_gate = write_plan_metric_artifact(
        output_path=metric_gate_json,
        metrics=metrics,
        targets=target_values,
        require_ood=True,
        context={
            "num_classes": len(list(seen_classes)),
            "ood_primary_score_method": primary_score_method,
            **dict(context),
        },
    )
    if telemetry is not None and hasattr(telemetry, "copy_artifact_file"):
        telemetry.copy_artifact_file(metric_gate_json, f"ood_benchmark/folds/{held_out_class}/metric_gate.json")
    return {
        "metric_gate": metric_gate,
        "primary_score_method": primary_score_method,
        "method_metrics": _compute_method_metrics(evaluation),
        "paths": {"metric_gate_json": str(metric_gate_json)},
    }


def _build_failed_summary(
    *,
    reason: str,
    base_context: Dict[str, Any],
    target_values: Dict[str, float],
    extra_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "v6_ood_benchmark",
        "status": "failed",
        "passed": False,
        "ood_evidence_source": "held_out_benchmark",
        "reason": str(reason),
        "metrics": _empty_metric_map(),
        "metric_std": {},
        "method_comparison": {},
        "method_comparison_metrics": {},
        "method_comparison_metric_std": {},
        "evaluation": validate_ood_metrics({}, target_values, require_ood=True),
        "successful_folds": 0,
        "failed_folds": 0,
        "folds": [],
        "targets": target_values,
        "context": {
            **base_context,
            **dict(extra_context or {}),
        },
    }


def _build_fold_sample_counts(
    train_indices: Sequence[int],
    calibration_indices: Sequence[int],
    eval_id_indices: Sequence[int],
    eval_ood_indices: Sequence[int],
) -> Dict[str, int]:
    return {
        "train_samples": len(train_indices),
        "calibration_samples": len(calibration_indices),
        "eval_in_distribution_samples": len(eval_id_indices),
        "eval_ood_samples": len(eval_ood_indices),
    }


def _build_fold_payload(
    *,
    held_out_class: str,
    status: str,
    reason: str,
    seen_classes: Sequence[str],
    sample_counts: Dict[str, int],
    primary_score_method: str = "ensemble",
    resume_key: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    method_metrics: Optional[Dict[str, Any]] = None,
    evaluation: Optional[Dict[str, Any]] = None,
    paths: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "held_out_class": held_out_class,
        "status": status,
        "reason": reason,
        "seen_classes": list(seen_classes),
        "primary_score_method": str(primary_score_method or "ensemble"),
        "resume_key": str(resume_key),
        "metrics": dict(metrics or {}),
        "method_metrics": dict(method_metrics or {}),
        "sample_counts": dict(sample_counts),
        "paths": dict(paths or {}),
        "diagnostics": dict(diagnostics or {}),
    }
    if evaluation is not None:
        payload["evaluation"] = dict(evaluation)
    return payload


def _resolve_fold_context(
    *,
    crop_name: str,
    held_out_class: str,
    resolved_classes: Sequence[str],
    class_to_idx: Dict[str, int],
    train_labels: Sequence[int],
    calibration_labels: Sequence[int],
    eval_labels: Sequence[int],
    dataset_fingerprints: Dict[str, Any],
    config: Dict[str, Any],
    device: str,
    num_epochs: Optional[int],
) -> Dict[str, Any]:
    seen_classes = [name for name in resolved_classes if name != held_out_class]
    held_out_label = int(class_to_idx[held_out_class])
    seen_label_map = {int(class_to_idx[name]): idx for idx, name in enumerate(seen_classes)}
    train_indices = [idx for idx, label in enumerate(train_labels) if label in seen_label_map]
    calibration_indices = [idx for idx, label in enumerate(calibration_labels) if label in seen_label_map]
    eval_id_indices = [idx for idx, label in enumerate(eval_labels) if label in seen_label_map]
    eval_ood_indices = [idx for idx, label in enumerate(eval_labels) if label == held_out_label]
    sample_counts = _build_fold_sample_counts(
        train_indices,
        calibration_indices,
        eval_id_indices,
        eval_ood_indices,
    )
    resume_key = _build_resume_key(
        crop_name=crop_name,
        held_out_class=held_out_class,
        seen_classes=seen_classes,
        sample_counts=sample_counts,
        dataset_fingerprints=dataset_fingerprints,
        config=config,
        device=device,
        num_epochs=num_epochs,
    )
    return {
        "held_out_class": held_out_class,
        "seen_classes": seen_classes,
        "seen_label_map": seen_label_map,
        "train_indices": train_indices,
        "calibration_indices": calibration_indices,
        "eval_id_indices": eval_id_indices,
        "eval_ood_indices": eval_ood_indices,
        "sample_counts": sample_counts,
        "resume_key": resume_key,
    }


def _record_fold_started(
    *,
    fold_index: int,
    fold_total: int,
    fold_context: Dict[str, Any],
    base_context: Dict[str, Any],
    artifact_root: Path,
    telemetry: Any,
    emit: Callable[[str, Dict[str, Any]], None],
    benchmark_started_at: float,
    completed_fold_count: int,
    failed_fold_count: int,
    device: str,
) -> None:
    held_out_class = str(fold_context["held_out_class"])
    seen_classes = list(fold_context["seen_classes"])
    sample_counts = dict(fold_context["sample_counts"])
    emit(
        "ood_benchmark_fold_started",
        {
            **base_context,
            "fold_index": fold_index,
            "fold_total": fold_total,
            "held_out_class": held_out_class,
            "seen_classes": seen_classes,
            "sample_counts": sample_counts,
        },
    )
    _emit_telemetry_log(
        telemetry,
        f"OOD benchmark fold {fold_index}/{fold_total} started for held_out_class={held_out_class}.",
    )
    _persist_benchmark_progress(
        artifact_root=artifact_root,
        telemetry=telemetry,
        payload={
            **base_context,
            "status": "running",
            "stage": "fold_started",
            "fold_index": fold_index,
            "fold_total": fold_total,
            "held_out_class": held_out_class,
            "seen_classes": seen_classes,
            "sample_counts": sample_counts,
            "completed_folds": completed_fold_count,
            "failed_folds": failed_fold_count,
            "elapsed_sec": round(time.time() - benchmark_started_at, 3),
            "resource_snapshot": _resource_snapshot(device),
        },
    )


def _has_fold_samples(sample_counts: Dict[str, int]) -> bool:
    return (
        sample_counts["train_samples"] > 0
        and sample_counts["calibration_samples"] > 0
        and sample_counts["eval_in_distribution_samples"] > 0
        and sample_counts["eval_ood_samples"] > 0
    )


def _build_missing_fold_samples_payload(
    *,
    fold_context: Dict[str, Any],
    primary_score_method: str,
) -> Dict[str, Any]:
    return _build_fold_payload(
        held_out_class=str(fold_context["held_out_class"]),
        status="failed",
        reason="missing_fold_samples",
        seen_classes=list(fold_context["seen_classes"]),
        sample_counts=dict(fold_context["sample_counts"]),
        primary_score_method=primary_score_method,
        resume_key=str(fold_context["resume_key"]),
        diagnostics={"failed_stage": "fold_started"},
    )


def _resume_completed_fold(
    *,
    resumable_folds: Dict[str, Dict[str, Any]],
    fold_context: Dict[str, Any],
    fold_index: int,
    fold_total: int,
    base_context: Dict[str, Any],
    artifact_root: Path,
    telemetry: Any,
    emit: Callable[[str, Dict[str, Any]], None],
    benchmark_started_at: float,
    completed_fold_count: int,
    failed_fold_count: int,
    device: str,
) -> Optional[Dict[str, Any]]:
    held_out_class = str(fold_context["held_out_class"])
    resume_key = str(fold_context["resume_key"])
    resumed_fold = resumable_folds.get(f"{held_out_class}:{resume_key}")
    if resumed_fold is None:
        return None

    fold_payload = dict(resumed_fold)
    diagnostics = dict(fold_payload.get("diagnostics", {}))
    diagnostics["resume_hit"] = True
    fold_payload["diagnostics"] = diagnostics
    _emit_telemetry_log(
        telemetry,
        (
            f"OOD benchmark fold {fold_index}/{fold_total} resumed from cached "
            f"artifacts for held_out_class={held_out_class}."
        ),
    )
    _persist_benchmark_progress(
        artifact_root=artifact_root,
        telemetry=telemetry,
        payload={
            **base_context,
            "status": "running",
            "stage": "fold_resumed",
            "fold_index": fold_index,
            "fold_total": fold_total,
            "held_out_class": held_out_class,
            "completed_folds": completed_fold_count,
            "failed_folds": failed_fold_count,
            "last_completed_fold": held_out_class,
            "elapsed_sec": round(time.time() - benchmark_started_at, 3),
            "resource_snapshot": _resource_snapshot(device),
        },
    )
    emit(
        "ood_benchmark_fold_completed",
        {
            **base_context,
            "fold_index": fold_index,
            "fold_total": fold_total,
            **fold_payload,
        },
    )
    return fold_payload


def _build_fold_loaders(
    *,
    train_loader: DataLoader,
    calibration_loader: DataLoader,
    eval_loader: DataLoader,
    train_dataset: Dataset,
    calibration_dataset: Dataset,
    eval_dataset: Dataset,
    fold_context: Dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    seen_classes = list(fold_context["seen_classes"])
    seen_label_map = dict(fold_context["seen_label_map"])
    fold_train_dataset = FilteredLabelDataset(
        train_dataset,
        fold_context["train_indices"],
        classes=seen_classes,
        label_map=seen_label_map,
    )
    fold_calibration_dataset = FilteredLabelDataset(
        calibration_dataset,
        fold_context["calibration_indices"],
        classes=seen_classes,
        label_map=seen_label_map,
    )
    fold_eval_dataset = FilteredLabelDataset(
        eval_dataset,
        fold_context["eval_id_indices"],
        classes=seen_classes,
        label_map=seen_label_map,
    )
    fold_ood_dataset = FilteredLabelDataset(
        eval_dataset,
        fold_context["eval_ood_indices"],
        classes=["__ood__"],
        constant_label=-1,
    )
    return (
        _clone_loader(train_loader, fold_train_dataset, training_like=True),
        _clone_loader(calibration_loader, fold_calibration_dataset, training_like=False),
        _clone_loader(eval_loader, fold_eval_dataset, training_like=False),
        _clone_loader(eval_loader, fold_ood_dataset, training_like=False),
    )


def _run_benchmark_fold(
    *,
    crop_name: str,
    fold_context: Dict[str, Any],
    train_loader: DataLoader,
    calibration_loader: DataLoader,
    eval_loader: DataLoader,
    train_dataset: Dataset,
    calibration_dataset: Dataset,
    eval_dataset: Dataset,
    config: Dict[str, Any],
    device: str,
    artifact_root: Path,
    adapter_factory: Callable[..., Any],
    run_id: str,
    num_epochs: Optional[int],
    telemetry: Any,
    target_values: Dict[str, Any],
    base_context: Dict[str, Any],
    primary_score_method: str,
) -> Dict[str, Any]:
    held_out_class = str(fold_context["held_out_class"])
    seen_classes = list(fold_context["seen_classes"])
    sample_counts = dict(fold_context["sample_counts"])
    resume_key = str(fold_context["resume_key"])
    fold_train_loader, fold_calibration_loader, fold_eval_loader, fold_ood_loader = _build_fold_loaders(
        train_loader=train_loader,
        calibration_loader=calibration_loader,
        eval_loader=eval_loader,
        train_dataset=train_dataset,
        calibration_dataset=calibration_dataset,
        eval_dataset=eval_dataset,
        fold_context=fold_context,
    )

    fold_started_at = time.time()
    current_stage = "adapter_initializing"
    try:
        adapter = adapter_factory(
            crop_name=str(crop_name),
            model_name=str(
                config.get("training", {})
                .get("continual", {})
                .get("backbone", {})
                .get("model_name", "facebook/dinov3-vitl16-pretrain-lvd1689m")
            ),
            device=str(device),
        )
        adapter.initialize_engine(class_names=seen_classes, config=config)
        current_stage = "fold_training"
        session = adapter.build_training_session(
            train_loader=fold_train_loader,
            num_epochs=num_epochs,
            val_loader=fold_calibration_loader,
            run_id=f"{run_id}_ood_fold_{held_out_class}",
            checkpoint_every_n_steps=0,
            checkpoint_on_exception=False,
        )
        session.run()
        current_stage = "fold_calibrating_ood"
        adapter.calibrate_ood(fold_calibration_loader)
        trainer = getattr(session, "trainer", getattr(adapter, "_trainer", None))
        current_stage = "fold_evaluating"
        evaluation = evaluate_model_with_artifact_metrics(trainer, fold_eval_loader, ood_loader=fold_ood_loader)
        if evaluation is None:
            raise RuntimeError("No evaluation samples were produced for the fold.")
        artifacts = _persist_fold_metric_gate(
            artifact_root=artifact_root,
            telemetry=telemetry,
            held_out_class=held_out_class,
            evaluation=evaluation,
            seen_classes=seen_classes,
            target_values=target_values,
            context={
                **base_context,
                "held_out_class": held_out_class,
                "seen_classes": seen_classes,
                "resume_key": resume_key,
                "ood_primary_score_method": str(
                    getattr(evaluation, "ood_primary_score_method", primary_score_method) or primary_score_method
                ),
                **sample_counts,
                **dict(evaluation.context),
            },
        )
        metric_gate = dict(artifacts.get("metric_gate", {}))
        return _build_fold_payload(
            held_out_class=held_out_class,
            status="completed",
            reason="",
            seen_classes=seen_classes,
            sample_counts=sample_counts,
            primary_score_method=str(artifacts.get("primary_score_method", primary_score_method)),
            resume_key=resume_key,
            metrics=dict(metric_gate.get("metrics", {})),
            method_metrics=dict(artifacts.get("method_metrics", {})),
            evaluation=dict(metric_gate.get("evaluation", {})),
            paths=dict(artifacts.get("paths", {})),
            diagnostics={
                "completed_stage": "fold_evaluating",
                "fold_duration_sec": round(time.time() - fold_started_at, 3),
                "resource_snapshot": _resource_snapshot(device),
            },
        )
    except Exception as exc:
        traceback_text = traceback.format_exc()
        failure_reason = f"{exc.__class__.__name__}: {exc}"
        failure_paths = _persist_fold_failure_diagnostics(
            artifact_root=artifact_root,
            telemetry=telemetry,
            held_out_class=held_out_class,
            failure_payload={
                "held_out_class": held_out_class,
                "failed_stage": current_stage,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
                "run_id": run_id,
                "crop_name": crop_name,
                "sample_counts": sample_counts,
                "ts": _utc_now_iso(),
            },
            traceback_text=traceback_text,
        )
        return _build_fold_payload(
            held_out_class=held_out_class,
            status="failed",
            reason=failure_reason,
            seen_classes=seen_classes,
            sample_counts=sample_counts,
            primary_score_method=primary_score_method,
            resume_key=resume_key,
            paths=failure_paths,
            diagnostics={
                "failed_stage": current_stage,
                "error_type": exc.__class__.__name__,
                "fold_duration_sec": round(time.time() - fold_started_at, 3),
                "resource_snapshot": _resource_snapshot(device),
            },
        )


def _record_fold_completion(
    *,
    fold_payload: Dict[str, Any],
    fold_index: int,
    fold_total: int,
    base_context: Dict[str, Any],
    artifact_root: Path,
    telemetry: Any,
    emit: Callable[[str, Dict[str, Any]], None],
    benchmark_started_at: float,
    completed_fold_count: int,
    failed_fold_count: int,
    device: str,
) -> None:
    status = str(fold_payload.get("status", "failed"))
    held_out_class = str(fold_payload.get("held_out_class", ""))
    _persist_benchmark_progress(
        artifact_root=artifact_root,
        telemetry=telemetry,
        payload={
            **base_context,
            "status": "running",
            "stage": "fold_completed" if status == "completed" else "fold_failed",
            "fold_index": fold_index,
            "fold_total": fold_total,
            "held_out_class": held_out_class,
            "completed_folds": completed_fold_count,
            "failed_folds": failed_fold_count,
            "last_completed_fold": held_out_class if status == "completed" else "",
            "last_error": fold_payload.get("reason") if status != "completed" else "",
            "elapsed_sec": round(time.time() - benchmark_started_at, 3),
            "resource_snapshot": _resource_snapshot(device),
        },
    )
    emit(
        "ood_benchmark_fold_completed",
        {
            **base_context,
            "fold_index": fold_index,
            "fold_total": fold_total,
            **fold_payload,
        },
    )


def _selected_fold_metrics(
    fold_payload: Dict[str, Any],
    *,
    selected_primary_score_method: str,
    requested_primary_score_method: str,
) -> Dict[str, Any]:
    if is_auto_primary_score_method(requested_primary_score_method):
        selected_method_metrics = dict(
            dict(fold_payload.get("method_metrics", {})).get(selected_primary_score_method, {})
        )
        if selected_method_metrics:
            return selected_method_metrics
    return dict(fold_payload.get("metrics", {}))


def _collect_fold_target_failures(
    folds: Sequence[Dict[str, Any]],
    *,
    selected_primary_score_method: str,
    requested_primary_score_method: str,
    target_values: Dict[str, Any],
) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for fold_payload in folds:
        metrics = _selected_fold_metrics(
            fold_payload,
            selected_primary_score_method=selected_primary_score_method,
            requested_primary_score_method=requested_primary_score_method,
        )
        evaluation = validate_ood_metrics(metrics, target_values, require_ood=True)
        missing_requirements = [
            metric_name
            for metric_name, detail in dict(evaluation.get("checks", {})).items()
            if not bool(detail.get("asserted", False)) or not bool(detail.get("passed", False))
        ]
        if not missing_requirements:
            continue
        failures.append(
            {
                "held_out_class": str(fold_payload.get("held_out_class", "")),
                "primary_score_method": str(selected_primary_score_method),
                "missing_requirements": missing_requirements,
                "metrics": metrics,
                "evaluation": evaluation,
            }
        )
    return failures


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_worst_fold_summary(folds: Sequence[Dict[str, Any]], method_name: str) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for fold_payload in folds:
        method_metrics = dict(fold_payload.get("method_metrics", {})).get(method_name, {})
        metrics = dict(method_metrics or {}) if isinstance(method_metrics, dict) else {}
        fpr = _coerce_float(metrics.get("ood_false_positive_rate"))
        auroc = _coerce_float(metrics.get("ood_auroc"))
        if fpr is None and auroc is None:
            continue
        candidates.append(
            {
                "held_out_class": str(fold_payload.get("held_out_class", "")),
                "ood_false_positive_rate": fpr,
                "ood_auroc": auroc,
                "metrics": metrics,
            }
        )
    if not candidates:
        return {}
    candidates.sort(
        key=lambda item: (
            bool(item.get("ood_false_positive_rate") is None),
            0.0 if item.get("ood_false_positive_rate") is None else -float(item["ood_false_positive_rate"]),
            float("inf") if item.get("ood_auroc") is None else float(item["ood_auroc"]),
            str(item.get("held_out_class", "")),
        )
    )
    return candidates[0]


def _build_method_comparison_payload(
    folds: Sequence[Dict[str, Any]],
    *,
    target_values: Dict[str, Any],
) -> Dict[str, Any]:
    aggregate_by_method, std_by_method = _aggregate_method_metric_stats(folds)
    methods: Dict[str, Any] = {}
    for method_name in sorted(aggregate_by_method.keys()):
        pooled_metrics = dict(aggregate_by_method.get(method_name, {}))
        metric_std = dict(std_by_method.get(method_name, {}))
        method_payload: Dict[str, Any] = {
            "pooled_metrics": pooled_metrics,
            "metric_std": metric_std,
            "pooled_gate_eligible": bool(
                validate_ood_metrics(pooled_metrics, target_values, require_ood=True).get("passed", False)
            ),
        }
        worst_fold = _build_worst_fold_summary(folds, method_name)
        if worst_fold:
            method_payload["worst_fold"] = worst_fold
        methods[method_name] = method_payload
    return {
        "schema_version": "v1_ood_method_comparison",
        "evidence_source": "held_out_benchmark",
        "methods": methods,
    }

def _build_benchmark_summary_payload(
    *,
    folds: List[Dict[str, Any]],
    primary_score_method: str,
    requested_primary_score_method: str,
    target_values: Dict[str, Any],
    base_context: Dict[str, Any],
) -> Dict[str, Any]:
    successful_folds = [fold for fold in folds if fold.get("status") == "completed"]
    failed_folds = [fold for fold in folds if fold.get("status") != "completed"]
    aggregate_metrics, metric_std = _aggregate_fold_metric_stats(successful_folds)
    method_comparison_metrics, method_comparison_metric_std = _aggregate_method_metric_stats(successful_folds)
    method_comparison = _build_method_comparison_payload(successful_folds, target_values=target_values)
    selected_primary_score_method = primary_score_method
    selected_metrics = dict(aggregate_metrics)
    selected_metric_std = dict(metric_std)
    selection_source = "configured"
    if is_auto_primary_score_method(requested_primary_score_method):
        comparison_for_selection: Dict[str, Any] = (
            dict(method_comparison) if dict(method_comparison.get("methods", {})) else dict(method_comparison_metrics)
        )
        selected_primary_score_method = select_best_ood_score_method(
            comparison_for_selection,
            fallback=primary_score_method,
        )
        if comparison_for_selection:
            selection_source = "held_out_benchmark"
            selected_method_payload = dict(method_comparison.get("methods", {})).get(selected_primary_score_method, {})
            if isinstance(selected_method_payload, dict) and selected_method_payload:
                selected_metrics = dict(selected_method_payload.get("pooled_metrics", aggregate_metrics))
                selected_metric_std = dict(selected_method_payload.get("metric_std", metric_std))
            else:
                selected_metrics = dict(method_comparison_metrics.get(selected_primary_score_method, aggregate_metrics))
                selected_metric_std = dict(method_comparison_metric_std.get(selected_primary_score_method, metric_std))

    method_comparison.update(
        {
            "requested_primary_score_method": requested_primary_score_method,
            "selected_primary_score_method": selected_primary_score_method,
            "selection_source": selection_source,
        }
    )
    ood_validation = validate_ood_metrics(selected_metrics, target_values, require_ood=True)
    fold_target_failures = _collect_fold_target_failures(
        successful_folds,
        selected_primary_score_method=selected_primary_score_method,
        requested_primary_score_method=requested_primary_score_method,
        target_values=target_values,
    )
    passed = bool(not failed_folds and not fold_target_failures and ood_validation["passed"])
    return {
        "schema_version": "v6_ood_benchmark",
        "status": "completed" if not failed_folds else "failed",
        "passed": passed,
        "ood_evidence_source": "held_out_benchmark",
        "requested_primary_score_method": requested_primary_score_method,
        "primary_score_method": selected_primary_score_method,
        "primary_score_selection_source": selection_source,
        "metrics": selected_metrics,
        "metric_std": selected_metric_std,
        "method_comparison": method_comparison,
        "method_comparison_metrics": method_comparison_metrics,
        "method_comparison_metric_std": method_comparison_metric_std,
        "evaluation": ood_validation,
        "fold_target_failures": fold_target_failures,
        "successful_folds": len(successful_folds),
        "failed_folds": len(failed_folds),
        "folds": folds,
        "targets": target_values,
        "context": base_context,
    }
def _aggregate_fold_metric_stats(
    folds: Sequence[Dict[str, Any]],
) -> tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    aggregate_metrics: Dict[str, Optional[float]] = {}
    metric_std: Dict[str, Optional[float]] = {}
    successful_folds = list(folds)
    for metric_name in _OOD_BENCHMARK_METRIC_NAMES:
        values = [
            float(fold["metrics"][metric_name])
            for fold in successful_folds
            if fold.get("metrics", {}).get(metric_name) is not None
        ]
        aggregate_metrics[metric_name] = _mean(values) if len(values) == len(successful_folds) else None
        metric_std[metric_name] = _std(values) if len(values) == len(successful_folds) else None
    return aggregate_metrics, metric_std


def _aggregate_method_metric_stats(
    folds: Sequence[Dict[str, Any]],
) -> tuple[Dict[str, Dict[str, Optional[float]]], Dict[str, Dict[str, Optional[float]]]]:
    aggregate_by_method: Dict[str, Dict[str, Optional[float]]] = {}
    std_by_method: Dict[str, Dict[str, Optional[float]]] = {}
    successful_folds = list(folds)
    method_names = sorted(
        {
            str(method_name)
            for fold in successful_folds
            for method_name in dict(fold.get("method_metrics", {})).keys()
        }
    )
    for method_name in method_names:
        method_metric_values: Dict[str, Optional[float]] = {}
        method_metric_std: Dict[str, Optional[float]] = {}
        for metric_name in _OOD_BENCHMARK_METRIC_NAMES:
            values = [
                float(fold["method_metrics"][method_name][metric_name])
                for fold in successful_folds
                if method_name in dict(fold.get("method_metrics", {}))
                and dict(fold["method_metrics"][method_name]).get(metric_name) is not None
            ]
            method_metric_values[metric_name] = _mean(values) if len(values) == len(successful_folds) else None
            method_metric_std[metric_name] = _std(values) if len(values) == len(successful_folds) else None
        aggregate_by_method[method_name] = method_metric_values
        std_by_method[method_name] = method_metric_std
    return aggregate_by_method, std_by_method


def run_leave_one_class_out_benchmark(
    *,
    crop_name: str,
    class_names: Sequence[str],
    loaders: Dict[str, Any],
    config: Dict[str, Any],
    device: str,
    artifact_root: Path,
    adapter_factory: Callable[..., Any],
    run_id: str,
    num_epochs: Optional[int] = None,
    telemetry: Any = None,
    emit_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    min_classes: int = 3,
) -> Dict[str, Any]:
    resolved_classes = [str(name) for name in class_names]
    target_values = load_plan_targets()
    artifact_root = Path(artifact_root)
    emit = emit_event or (lambda _event_type, _payload: None)
    requested_primary_score_method = _primary_score_method(config)
    primary_score_method = resolve_runtime_primary_score_method(requested_primary_score_method)
    train_loader = loaders.get("train")
    calibration_split_name, calibration_loader = _select_calibration_loader(loaders)
    eval_split_name, eval_loader = _select_eval_loader(loaders, calibration_split_name=calibration_split_name)

    base_context = {
        "crop_name": str(crop_name),
        "run_id": str(run_id),
        "eval_split_name": eval_split_name,
        "calibration_split_name": calibration_split_name,
        "class_count": len(resolved_classes),
        "estimated_fold_trainings": len(resolved_classes),
        "requested_primary_score_method": requested_primary_score_method,
        "primary_score_method": primary_score_method,
    }
    existing_summary = _load_existing_benchmark_summary(artifact_root)
    resumable_folds = _build_resumable_fold_index(existing_summary)
    run_stage = _BenchmarkRunPreparation(
        resolved_classes=resolved_classes,
        target_values=target_values,
        artifact_root=artifact_root,
        base_context=base_context,
        resumable_folds=resumable_folds,
        train_loader=train_loader,
        calibration_loader=calibration_loader,
        eval_loader=eval_loader,
        calibration_split_name=calibration_split_name,
        eval_split_name=eval_split_name,
        requested_primary_score_method=requested_primary_score_method,
        primary_score_method=primary_score_method,
    )

    if len(run_stage.resolved_classes) < int(min_classes):
        return _persist_benchmark_summary(
            artifact_root=run_stage.artifact_root,
            summary_payload=_build_failed_summary(
                reason="insufficient_classes_for_fallback",
                base_context=run_stage.base_context,
                target_values=run_stage.target_values,
                extra_context={"required_min_classes": int(min_classes)},
            ),
            telemetry=telemetry,
        )

    if run_stage.train_loader is None or _loader_size(run_stage.train_loader) <= 0:
        return _persist_benchmark_summary(
            artifact_root=run_stage.artifact_root,
            summary_payload=_build_failed_summary(
                reason="missing_train_loader",
                base_context=run_stage.base_context,
                target_values=run_stage.target_values,
            ),
            telemetry=telemetry,
        )

    train_dataset = run_stage.train_loader.dataset
    calibration_dataset = getattr(calibration_loader, "dataset", None)
    eval_dataset = getattr(eval_loader, "dataset", None)
    if calibration_dataset is None or eval_dataset is None:
        return _persist_benchmark_summary(
            artifact_root=run_stage.artifact_root,
            summary_payload=_build_failed_summary(
                reason="missing_isolated_eval_loader",
                base_context=run_stage.base_context,
                target_values=run_stage.target_values,
            ),
            telemetry=telemetry,
        )

    class_to_idx = dict(
        getattr(
            train_dataset,
            "class_to_idx",
            {name: idx for idx, name in enumerate(run_stage.resolved_classes)},
        )
    )
    train_labels = _dataset_labels(train_dataset)
    calibration_labels = _dataset_labels(calibration_dataset)
    eval_labels = _dataset_labels(eval_dataset)
    dataset_fingerprints = {
        "train": _dataset_resume_fingerprint(train_dataset),
        "calibration": {
            "split_name": run_stage.calibration_split_name,
            **_dataset_resume_fingerprint(calibration_dataset),
        },
        "evaluation": {
            "split_name": run_stage.eval_split_name,
            **_dataset_resume_fingerprint(eval_dataset),
        },
    }
    dataset_stage = _BenchmarkDatasetStage(
        train_dataset=train_dataset,
        calibration_dataset=calibration_dataset,
        eval_dataset=eval_dataset,
        class_to_idx=class_to_idx,
        train_labels=train_labels,
        calibration_labels=calibration_labels,
        eval_labels=eval_labels,
        dataset_fingerprints=dataset_fingerprints,
    )
    folds: List[Dict[str, Any]] = []
    completed_fold_count = 0
    failed_fold_count = 0
    benchmark_started_at = time.time()
    fold_total = len(run_stage.resolved_classes)

    _persist_benchmark_progress(
        artifact_root=run_stage.artifact_root,
        telemetry=telemetry,
        payload={
            **base_context,
            "status": "running",
            "stage": "benchmark_started",
            "fold_index": 0,
            "fold_total": fold_total,
            "held_out_class": "",
            "completed_folds": completed_fold_count,
            "failed_folds": failed_fold_count,
            "elapsed_sec": 0.0,
            "resource_snapshot": _resource_snapshot(device),
        },
    )

    emit(
        "ood_benchmark_started",
        {
            **base_context,
            "fold_count": fold_total,
        },
    )

    for fold_index, held_out_class in enumerate(run_stage.resolved_classes, start=1):
        fold_context = _resolve_fold_context(
            crop_name=crop_name,
            held_out_class=held_out_class,
            resolved_classes=run_stage.resolved_classes,
            class_to_idx=dataset_stage.class_to_idx,
            train_labels=dataset_stage.train_labels,
            calibration_labels=dataset_stage.calibration_labels,
            eval_labels=dataset_stage.eval_labels,
            dataset_fingerprints=dataset_stage.dataset_fingerprints,
            config=config,
            device=device,
            num_epochs=num_epochs,
        )
        _record_fold_started(
            fold_index=fold_index,
            fold_total=fold_total,
            fold_context=fold_context,
            base_context=run_stage.base_context,
            artifact_root=run_stage.artifact_root,
            telemetry=telemetry,
            emit=emit,
            benchmark_started_at=benchmark_started_at,
            completed_fold_count=completed_fold_count,
            failed_fold_count=failed_fold_count,
            device=device,
        )

        if not _has_fold_samples(dict(fold_context["sample_counts"])):
            fold_payload = _build_missing_fold_samples_payload(
                fold_context=fold_context,
                primary_score_method=primary_score_method,
            )
            folds.append(fold_payload)
            failed_fold_count += 1
            _record_fold_completion(
                fold_payload=fold_payload,
                fold_index=fold_index,
                fold_total=fold_total,
                base_context=run_stage.base_context,
                artifact_root=run_stage.artifact_root,
                telemetry=telemetry,
                emit=emit,
                benchmark_started_at=benchmark_started_at,
                completed_fold_count=completed_fold_count,
                failed_fold_count=failed_fold_count,
                device=device,
            )
            continue


        next_completed_fold_count = completed_fold_count + 1
        resumed_fold = _resume_completed_fold(
            resumable_folds=run_stage.resumable_folds,
            fold_context=fold_context,
            fold_index=fold_index,
            fold_total=fold_total,
            base_context=run_stage.base_context,
            artifact_root=run_stage.artifact_root,
            telemetry=telemetry,
            emit=emit,
            benchmark_started_at=benchmark_started_at,
            completed_fold_count=next_completed_fold_count,
            failed_fold_count=failed_fold_count,
            device=device,
        )
        if resumed_fold is not None:
            folds.append(resumed_fold)
            completed_fold_count = next_completed_fold_count
            continue

        fold_payload = _run_benchmark_fold(
            crop_name=crop_name,
            fold_context=fold_context,
            train_loader=cast(DataLoader[Any], train_loader),
            calibration_loader=calibration_loader,
            eval_loader=eval_loader,
            train_dataset=train_dataset,
            calibration_dataset=calibration_dataset,
            eval_dataset=eval_dataset,
            config=config,
            device=device,
            artifact_root=run_stage.artifact_root,
            adapter_factory=adapter_factory,
            run_id=run_id,
            num_epochs=num_epochs,
            telemetry=telemetry,
            target_values=run_stage.target_values,
            base_context=run_stage.base_context,
            primary_score_method=primary_score_method,
        )
        if fold_payload.get("status") != "completed":
            _emit_telemetry_log(
                telemetry,
                (
                    f"OOD benchmark fold {fold_index}/{fold_total} failed at stage="
                    f"{dict(fold_payload.get('diagnostics', {})).get('failed_stage', 'unknown')} "
                    f"for held_out_class={held_out_class}: {fold_payload.get('reason', '')}"
                ),
                level="error",
            )

        folds.append(fold_payload)
        if fold_payload.get("status") == "completed":
            completed_fold_count += 1
        else:
            failed_fold_count += 1
        _record_fold_completion(
            fold_payload=fold_payload,
            fold_index=fold_index,
            fold_total=fold_total,
            base_context=run_stage.base_context,
            artifact_root=run_stage.artifact_root,
            telemetry=telemetry,
            emit=emit,
            benchmark_started_at=benchmark_started_at,
            completed_fold_count=completed_fold_count,
            failed_fold_count=failed_fold_count,
            device=device,
        )

    summary_payload = _build_benchmark_summary_payload(
        folds=folds,
        primary_score_method=primary_score_method,
        requested_primary_score_method=requested_primary_score_method,
        target_values=run_stage.target_values,
        base_context=run_stage.base_context,
    )
    summary_payload = _persist_benchmark_summary(
        artifact_root=run_stage.artifact_root,
        summary_payload=summary_payload,
        telemetry=telemetry,
    )
    _persist_benchmark_progress(
        artifact_root=run_stage.artifact_root,
        telemetry=telemetry,
        payload={
            **base_context,
            "status": summary_payload["status"],
            "stage": "benchmark_completed",
            "fold_index": len(resolved_classes),
            "fold_total": len(resolved_classes),
            "held_out_class": "",
            "completed_folds": summary_payload["successful_folds"],
            "failed_folds": summary_payload["failed_folds"],
            "passed": summary_payload["passed"],
            "elapsed_sec": round(time.time() - benchmark_started_at, 3),
            "resource_snapshot": _resource_snapshot(device),
        },
    )
    emit(
        "ood_benchmark_completed",
        {
            **base_context,
            "passed": summary_payload["passed"],
            "status": summary_payload["status"],
            "successful_folds": summary_payload["successful_folds"],
            "failed_folds": summary_payload["failed_folds"],
            "paths": summary_payload["paths"],
        },
    )
    return summary_payload
