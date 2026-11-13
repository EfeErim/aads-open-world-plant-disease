"""Internal helpers for the public training workflow facade."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.training.services.class_balance import (
    MIN_SUPPORTED_CLASS_SAMPLES,
    build_class_balance_runtime,
    format_under_min_class_error,
)
from src.training.services.runtime_dataset import resolve_runtime_dataset
from src.utils.training_helpers import (
    _dataset_class_counts,
    _inject_class_balance_runtime,
    _resolve_loader_sampler_runtime,
    build_loader_sizes,
    loader_batch_count,
    stringify_paths,
)


@dataclass
class TrainingRunSetup:
    run_id: str
    training_cfg: Dict[str, Any]
    data_cfg: Dict[str, Any]
    colab_cfg: Dict[str, Any]
    loaders: Dict[str, Any]
    loader_sizes: Dict[str, int]
    loader_batch_counts: Dict[str, int]
    detected_classes: List[str]
    split_class_counts: Dict[str, Dict[str, int]]
    adapter: Any
    sampler_runtime: Dict[str, Any]
    class_balance_runtime: Dict[str, Any]
    runtime_dataset_key: str
    runtime_crop_root: Path
    runtime_dataset_resolution_source: str
    part_name: str


# Helpers moved to src.utils.training_helpers for reuse and testability.


def _validate_training_layout(
    *,
    crop_name: str,
    data_dir: str | Path,
    batch_size: int,
    loaders: Dict[str, Any],
    loader_sizes: Dict[str, int],
    loader_batch_counts: Dict[str, int],
) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    train_loader = loaders.get("train")
    if train_loader is None:
        raise ValueError("Training loader was not created.")

    detected_classes = [str(name) for name in getattr(train_loader.dataset, "classes", [])]
    if not detected_classes:
        raise ValueError(f"No classes found for crop '{crop_name}' in {data_dir}")

    train_samples = int(loader_sizes.get("train", 0))
    if train_samples <= 0:
        raise ValueError(
            f"Training split '{crop_name}/continual' is empty under {data_dir}. "
            "At least one in-distribution training sample is required."
        )

    train_batches = int(loader_batch_counts.get("train", 0))
    if train_batches <= 0:
        raise ValueError(
            "Training produced zero batches before the first optimizer step. "
            f"train_samples={train_samples}, batch_size={batch_size}. "
            "Reduce batch_size or increase the continual split size."
        )

    split_class_counts = {
        split_name: _dataset_class_counts(loader)
        for split_name, loader in loaders.items()
        if split_name in {"train", "val", "test"}
    }
    train_counts = dict(split_class_counts.get("train", {}))
    eval_only_classes: Dict[str, List[str]] = {}
    for split_name in ("val", "test"):
        split_counts = dict(split_class_counts.get(split_name, {}))
        leaked = [
            name
            for name, count in split_counts.items()
            if int(count) > 0 and int(train_counts.get(name, 0)) <= 0
        ]
        if leaked:
            eval_only_classes[split_name] = sorted(leaked)
    if eval_only_classes:
        details = "; ".join(f"{split}: {', '.join(names)}" for split, names in eval_only_classes.items())
        raise ValueError(
            "Validation/test splits contain classes that have no continual-train samples: "
            + details
        )

    if train_counts:
        missing_train_classes = [name for name in detected_classes if int(train_counts.get(name, 0)) <= 0]
        if missing_train_classes:
            raise ValueError(
                "Detected classes without training samples: "
                + ", ".join(sorted(missing_train_classes))
            )

    return detected_classes, split_class_counts


def select_calibration_source(loaders: Dict[str, Any], loader_sizes: Dict[str, int]) -> tuple[str, Any]:
    val_loader = loaders.get("val")
    if val_loader is not None and int(loader_sizes.get("val", 0)) > 0:
        return "val", val_loader
    return "train", loaders.get("train")


def prepare_training_run(
    *,
    config: Dict[str, Any],
    device: str,
    crop_name: str,
    part_name: Optional[str],
    data_dir: str | Path,
    class_names: Optional[Sequence[str]],
    num_workers: Optional[int],
    pin_memory: Optional[bool],
    use_cache: bool,
    sampler: Optional[str],
    error_policy: Optional[str],
    run_id: str,
    loader_factory: Callable[..., Dict[str, Any]],
    adapter_factory: Callable[..., Any],
) -> TrainingRunSetup:
    training_cfg = dict(config.get("training", {}).get("continual", {}))
    data_cfg = dict(training_cfg.get("data", {}))
    ood_cfg = dict(training_cfg.get("ood", {}))
    colab_cfg = dict(config.get("colab", {}).get("training", {}))
    resolved_run_id = str(run_id or f"{crop_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}")
    resolved_dataset = resolve_runtime_dataset(data_dir=data_dir, crop_name=crop_name)
    loader_data_dir = resolved_dataset.crop_root.parent

    loaders = loader_factory(
        data_dir=str(loader_data_dir),
        crop=resolved_dataset.dataset_key,
        class_names=class_names,
        batch_size=int(training_cfg.get("batch_size", 8)),
        num_workers=int(num_workers if num_workers is not None else colab_cfg.get("num_workers", 2)),
        use_cache=bool(use_cache),
        cache_size=int(data_cfg.get("cache_size", 1000)),
        cache_train_split=bool(data_cfg.get("cache_train_split", False)),
        target_size=int(data_cfg.get("target_size", 224)),
        error_policy=str(error_policy or data_cfg.get("loader_error_policy", "tolerant")),
        sampler=str(sampler or data_cfg.get("sampler", "auto")),
        seed=int(training_cfg.get("seed", 42)),
        validate_images_on_init=bool(data_cfg.get("validate_images_on_init", False)),
        augmentation_policy=str(data_cfg.get("augmentation_policy", "randaugment")),
        randaugment_num_ops=int(data_cfg.get("randaugment_num_ops", 2)),
        randaugment_magnitude=int(data_cfg.get("randaugment_magnitude", 7)),
        augmix_severity=int(data_cfg.get("augmix_severity", 3)),
        augmix_width=int(data_cfg.get("augmix_width", 3)),
        augmix_depth=int(data_cfg.get("augmix_depth", -1)),
        augmix_alpha=float(data_cfg.get("augmix_alpha", 1.0)),
        ood_root=(str(ood_cfg.get("ood_root", "") or "") or None),
        oe_root=(
            str(ood_cfg.get("oe_root", "") or "").strip()
            or None
        ),
        real_ood_split_enabled=bool(ood_cfg.get("real_split_enabled", True)),
        real_ood_split_dev_fraction=float(ood_cfg.get("real_split_dev_fraction", 0.4)),
        real_ood_split_min_per_slice=int(ood_cfg.get("real_split_min_per_slice", 2)),
        real_ood_split_min_total=int(ood_cfg.get("real_split_min_total", 30)),
        real_ood_split_manifest_name=str(
            ood_cfg.get("real_split_manifest_name", "ood_split_manifest.json") or "ood_split_manifest.json"
        ),
        real_ood_enforce_oe_disjoint=bool(ood_cfg.get("enforce_oe_disjoint", True)),
        pin_memory=bool(colab_cfg.get("pin_memory", True) if pin_memory is None else pin_memory),
    )
    loader_sizes = build_loader_sizes(loaders)
    loader_batch_counts = {
        name: loader_batch_count(loader, loader_name=f"{name} loader")
        for name, loader in loaders.items()
    }
    detected_classes, split_class_counts = _validate_training_layout(
        crop_name=crop_name,
        data_dir=data_dir,
        batch_size=int(training_cfg.get("batch_size", 8)),
        loaders=loaders,
        loader_sizes=loader_sizes,
        loader_batch_counts=loader_batch_counts,
    )

    sampler_runtime = _resolve_loader_sampler_runtime(loaders.get("train"))
    allow_under_min_training = bool(data_cfg.get("allow_under_min_training", False))
    class_balance_cfg = dict(training_cfg.get("class_balance", {}))
    class_balance_runtime = build_class_balance_runtime(
        crop_name=crop_name,
        data_dir=loader_data_dir,
        detected_classes=detected_classes,
        split_class_counts=split_class_counts,
        dataset_key=resolved_dataset.dataset_key,
        production_min_supported_samples=MIN_SUPPORTED_CLASS_SAMPLES,
        allow_under_min_training=allow_under_min_training,
        resolved_train_sampler=str(sampler_runtime.get("resolved_sampler", "")),
        allow_sampler_and_loss_rebalance=bool(
            class_balance_cfg.get("allow_sampler_and_loss_rebalance", False)
        ),
    )
    if class_balance_runtime.get("under_min_classes") and not allow_under_min_training:
        raise ValueError(format_under_min_class_error(class_balance_runtime))

    adapter_kwargs = {
        "crop_name": crop_name,
        "model_name": str(
            training_cfg.get(
                "backbone",
                {},
            ).get("model_name", "facebook/dinov3-vitl16-pretrain-lvd1689m")
        ),
        "device": device,
    }
    if part_name is not None:
        adapter_kwargs["part_name"] = part_name
    try:
        adapter = adapter_factory(**adapter_kwargs)
    except TypeError as exc:
        if part_name is None or "part_name" not in str(exc):
            raise
        adapter_kwargs.pop("part_name", None)
        adapter = adapter_factory(**adapter_kwargs)
    adapter.initialize_engine(
        class_names=detected_classes,
        config=_inject_class_balance_runtime(config, class_balance_runtime),
    )

    return TrainingRunSetup(
        run_id=resolved_run_id,
        training_cfg=training_cfg,
        data_cfg=data_cfg,
        colab_cfg=colab_cfg,
        loaders=loaders,
        loader_sizes=loader_sizes,
        loader_batch_counts=loader_batch_counts,
        detected_classes=detected_classes,
        split_class_counts=split_class_counts,
        adapter=adapter,
        sampler_runtime=sampler_runtime,
        class_balance_runtime=class_balance_runtime,
        runtime_dataset_key=resolved_dataset.dataset_key,
        runtime_crop_root=resolved_dataset.crop_root,
        runtime_dataset_resolution_source=resolved_dataset.resolution_source,
        part_name=str(part_name or "unspecified"),
    )

def build_artifact_payload(
    *,
    training_artifacts: Dict[str, Any],
    validation_artifacts: Dict[str, Any],
    test_artifacts: Dict[str, Any],
    ood_benchmark: Dict[str, Any],
    readiness_artifacts: Dict[str, Any],
    behavioral_artifacts: Dict[str, Any],
    summary_artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    return stringify_paths(
        {
            "training": training_artifacts,
            "validation": validation_artifacts.get("paths", {}),
            "test": test_artifacts.get("paths", {}),
            "ood_benchmark": dict(ood_benchmark.get("paths", {})),
            "adapter_behavioral_acceptance": {
                "acceptance_json": behavioral_artifacts.get("acceptance_json")
            },
            "production_readiness": {"readiness_json": readiness_artifacts.get("readiness_json")},
            "summary": summary_artifacts,
        }
    )
