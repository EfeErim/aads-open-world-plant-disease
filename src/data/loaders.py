"""DataLoader assembly helpers."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, TypedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .datasets import CropDataset, infer_crop_classes_from_layout
from .ood_splits import (
    ensure_ood_split_manifest,
    manifest_slice_map,
    select_manifest_paths,
    validate_ood_oe_disjoint,
)

VALID_SAMPLERS = {"auto", "shuffle", "weighted"}
AUTO_WEIGHTED_SAMPLER_IMBALANCE_RATIO = 1.5


class SamplerRuntime(TypedDict):
    requested_sampler: str
    resolved_sampler: str
    decision_reason: str
    imbalance_ratio: float
    imbalance_ratio_threshold: float
    class_counts: Dict[str, int]


def seed_worker_factory(base_seed: int) -> Any:
    def _seed_worker(worker_id: int) -> None:
        worker_seed = int(base_seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32 - 1))
        torch.manual_seed(worker_seed)

    return _seed_worker


def dict_collate_fn(batch: List[tuple[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
    if not batch:
        return {"images": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)}
    images, labels = zip(*batch)
    return {
        "images": torch.stack(list(images), dim=0),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_weighted_sampler(dataset: CropDataset, seed: int) -> WeightedRandomSampler:
    counts = Counter(dataset.labels)
    weights = [1.0 / float(counts[label]) for label in dataset.labels]
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def _dataset_class_count_map(dataset: CropDataset) -> Dict[str, int]:
    classes = [str(name) for name in getattr(dataset, "classes", [])]
    label_counts = Counter(int(label) for label in getattr(dataset, "labels", []) if int(label) >= 0)
    if classes:
        return {
            class_name: int(label_counts.get(class_index, 0))
            for class_index, class_name in enumerate(classes)
        }
    return {str(label): int(count) for label, count in sorted(label_counts.items(), key=lambda item: item[0])}


def resolve_train_sampler(
    dataset: CropDataset,
    *,
    requested_sampler: str,
    imbalance_ratio_threshold: float = AUTO_WEIGHTED_SAMPLER_IMBALANCE_RATIO,
) -> SamplerRuntime:
    requested = str(requested_sampler).strip().lower()
    class_counts = _dataset_class_count_map(dataset)
    nonzero_counts = [int(count) for count in class_counts.values() if int(count) > 0]
    imbalance_ratio = 1.0
    if nonzero_counts:
        imbalance_ratio = float(max(nonzero_counts)) / float(min(nonzero_counts))

    resolved = requested
    reason = "explicit_request"
    if requested == "auto":
        if len(nonzero_counts) <= 1:
            resolved = "shuffle"
            reason = "single_class_or_empty"
        elif imbalance_ratio >= float(imbalance_ratio_threshold):
            resolved = "weighted"
            reason = "imbalance_detected"
        else:
            resolved = "shuffle"
            reason = "balanced_enough"

    return {
        "requested_sampler": requested,
        "resolved_sampler": resolved,
        "decision_reason": reason,
        "imbalance_ratio": round(float(imbalance_ratio), 4),
        "imbalance_ratio_threshold": float(imbalance_ratio_threshold),
        "class_counts": class_counts,
    }


def create_training_loaders(
    data_dir: str,
    crop: str,
    class_names: List[str] | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    use_cache: bool = True,
    cache_size: int = 1000,
    cache_train_split: bool = False,
    target_size: int = 224,
    error_policy: str = "tolerant",
    sampler: str = "auto",
    seed: int = 42,
    validate_images_on_init: bool = False,
    augmentation_policy: str = "randaugment",
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 7,
    augmix_severity: int = 3,
    augmix_width: int = 3,
    augmix_depth: int = -1,
    augmix_alpha: float = 1.0,
    ood_root: str | Path | None = None,
    oe_root: str | Path | None = None,
    real_ood_split_enabled: bool = True,
    real_ood_split_dev_fraction: float = 0.4,
    real_ood_split_min_per_slice: int = 2,
    real_ood_split_min_total: int = 30,
    real_ood_split_manifest_name: str = "ood_split_manifest.json",
    real_ood_enforce_oe_disjoint: bool = True,
    *,
    dataset_cls: type[CropDataset] = CropDataset,
    infer_classes_fn: Callable[[str, str], List[str]] = infer_crop_classes_from_layout,
    collate_fn: Callable[[List[tuple[torch.Tensor, int]]], Dict[str, torch.Tensor]] = dict_collate_fn,
    sampler_builder: Callable[[CropDataset, int], WeightedRandomSampler] = build_weighted_sampler,
    worker_seed_factory: Callable[[int], Any] = seed_worker_factory,
    **dataloader_kwargs: Any,
) -> Dict[str, DataLoader]:
    sampler_name = str(sampler).strip().lower()
    if sampler_name not in VALID_SAMPLERS:
        raise ValueError(f"Unsupported sampler: {sampler}")

    inferred_classes = [str(name) for name in infer_classes_fn(data_dir, crop)]
    resolved_classes = [str(name) for name in class_names] if class_names is not None else inferred_classes
    if inferred_classes and class_names is not None and set(resolved_classes) != set(inferred_classes):
        missing = sorted(set(inferred_classes) - set(resolved_classes))
        extra = sorted(set(resolved_classes) - set(inferred_classes))
        details: List[str] = []
        if missing:
            details.append(f"missing from override: {', '.join(missing)}")
        if extra:
            details.append(f"not found on disk: {', '.join(extra)}")
        raise ValueError("Provided class_names do not match the dataset layout (" + "; ".join(details) + ").")

    pin_memory = bool(dataloader_kwargs.pop("pin_memory", True))
    persistent_workers = bool(dataloader_kwargs.pop("persistent_workers", num_workers > 0))
    prefetch_factor = dataloader_kwargs.pop("prefetch_factor", None)

    loaders: Dict[str, DataLoader] = {}
    worker_init_fn = worker_seed_factory(int(seed))
    for split in ("train", "val", "test"):
        split_seed_offset = 0 if split == "train" else 10 if split == "val" else 20
        sampler_seed_offset = 100 if split == "train" else 110 if split == "val" else 120
        dataset = dataset_cls(
            data_dir=data_dir,
            crop=crop,
            split=split,
            class_names=resolved_classes,
            transform=(split == "train"),
            target_size=target_size,
            use_cache=use_cache,
            cache_size=cache_size,
            cache_train_split=cache_train_split,
            error_policy=error_policy,
            validate_images_on_init=validate_images_on_init,
            augmentation_policy=augmentation_policy,
            randaugment_num_ops=randaugment_num_ops,
            randaugment_magnitude=randaugment_magnitude,
            augmix_severity=augmix_severity,
            augmix_width=augmix_width,
            augmix_depth=augmix_depth,
            augmix_alpha=augmix_alpha,
        )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(seed) + split_seed_offset)

        split_sampler = None
        shuffle = split == "train"
        drop_last = split == "train"
        sampler_runtime: SamplerRuntime = {
            "requested_sampler": sampler_name,
            "resolved_sampler": sampler_name,
            "decision_reason": "non_train_split",
            "imbalance_ratio": 1.0,
            "imbalance_ratio_threshold": float(AUTO_WEIGHTED_SAMPLER_IMBALANCE_RATIO),
            "class_counts": {},
        }
        if split == "train":
            sampler_runtime = resolve_train_sampler(
                dataset,
                requested_sampler=sampler_name,
                imbalance_ratio_threshold=AUTO_WEIGHTED_SAMPLER_IMBALANCE_RATIO,
            )
            if sampler_runtime["resolved_sampler"] == "weighted" and len(dataset) > 0:
                split_sampler = sampler_builder(dataset, int(seed))
                shuffle = False

        extra_kwargs = dict(dataloader_kwargs)
        if num_workers <= 0:
            extra_kwargs.pop("prefetch_factor", None)
            persistent_workers = False
        elif prefetch_factor is not None:
            extra_kwargs["prefetch_factor"] = int(prefetch_factor)

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if split_sampler is None else False,
            sampler=split_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            generator=loader_generator,
            drop_last=drop_last,
            **extra_kwargs,
        )
        setattr(loaders[split], "_seed_base", int(seed) + split_seed_offset)
        setattr(loaders[split], "_sampler_seed_base", int(seed) + sampler_seed_offset)
        setattr(loaders[split], "_requested_sampler", str(sampler_runtime["requested_sampler"]))
        setattr(loaders[split], "_resolved_sampler", str(sampler_runtime["resolved_sampler"]))
        setattr(loaders[split], "_sampler_decision_reason", str(sampler_runtime["decision_reason"]))
        setattr(loaders[split], "_sampler_imbalance_ratio", float(sampler_runtime["imbalance_ratio"]))
        setattr(
            loaders[split],
            "_sampler_imbalance_ratio_threshold",
            float(sampler_runtime["imbalance_ratio_threshold"]),
        )
        setattr(loaders[split], "_sampler_class_counts", dict(sampler_runtime["class_counts"]))

    def _build_ood_loader(
        *,
        source_root: Path,
        split_paths: List[Path] | None = None,
        split_name: str = "ood",
        seed_offset: int = 30,
    ) -> DataLoader:
        ood_dataset = dataset_cls(
            data_dir=data_dir,
            crop=crop,
            split="ood",
            split_root=source_root,
            class_names=resolved_classes,
            transform=False,
            target_size=target_size,
            use_cache=use_cache,
            cache_size=cache_size,
            cache_train_split=cache_train_split,
            error_policy=error_policy,
            validate_images_on_init=validate_images_on_init,
            augmentation_policy=augmentation_policy,
            randaugment_num_ops=randaugment_num_ops,
            randaugment_magnitude=randaugment_magnitude,
            augmix_severity=augmix_severity,
            augmix_width=augmix_width,
            augmix_depth=augmix_depth,
            augmix_alpha=augmix_alpha,
        )
        if split_paths is not None:
            allowed = {str(path.resolve(strict=False)) for path in split_paths}
            selected_paths: List[Path] = []
            selected_labels: List[int] = []
            selected_types: List[str] = []
            sample_types = list(getattr(ood_dataset, "ood_sample_types", []))
            for index, image_path in enumerate(list(getattr(ood_dataset, "image_paths", []))):
                if str(Path(image_path).resolve(strict=False)) not in allowed:
                    continue
                selected_paths.append(image_path)
                selected_labels.append(-1)
                if index < len(sample_types):
                    selected_types.append(str(sample_types[index]))
                else:
                    selected_types.append("unlabeled")
            ood_dataset.image_paths = selected_paths
            ood_dataset.labels = selected_labels
            ood_dataset.ood_sample_types = selected_types

        ood_generator = torch.Generator()
        ood_generator.manual_seed(int(seed) + seed_offset)
        ood_extra_kwargs = dict(dataloader_kwargs)
        if num_workers <= 0:
            ood_extra_kwargs.pop("prefetch_factor", None)
        elif prefetch_factor is not None:
            ood_extra_kwargs["prefetch_factor"] = int(prefetch_factor)

        loader = DataLoader(
            ood_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            generator=ood_generator,
            **ood_extra_kwargs,
        )
        setattr(loader, "_seed_base", int(seed) + seed_offset)
        setattr(loader, "_sampler_seed_base", int(seed) + 100 + seed_offset)
        setattr(loader, "_ood_split_name", split_name)
        return loader

    resolved_ood_root = Path(ood_root).expanduser() if ood_root is not None and str(ood_root).strip() else Path(data_dir) / crop / "ood"
    resolved_oe_root = (
        Path(oe_root).expanduser()
        if oe_root is not None and str(oe_root).strip()
        else Path(data_dir) / crop / "oe"
    )
    if resolved_ood_root.exists():
        if not resolved_ood_root.is_dir():
            raise NotADirectoryError(f"OOD root is not a directory: {resolved_ood_root}")
        if bool(real_ood_enforce_oe_disjoint) and resolved_oe_root.exists():
            if not resolved_oe_root.is_dir():
                raise NotADirectoryError(f"OE root is not a directory: {resolved_oe_root}")
            validate_ood_oe_disjoint(resolved_ood_root, resolved_oe_root)
        manifest = None
        if bool(real_ood_split_enabled):
            try:
                manifest = ensure_ood_split_manifest(
                    resolved_ood_root,
                    manifest_name=str(real_ood_split_manifest_name or "ood_split_manifest.json"),
                    seed=int(seed),
                    dev_fraction=float(real_ood_split_dev_fraction),
                    min_per_slice=int(real_ood_split_min_per_slice),
                    min_total_for_dev_test=int(real_ood_split_min_total),
                )
            except OSError:
                manifest = None
        if manifest is not None:
            dev_paths = select_manifest_paths(resolved_ood_root, manifest, "dev")
            test_paths = select_manifest_paths(resolved_ood_root, manifest, "test")
            if dev_paths and test_paths:
                slice_map = manifest_slice_map(resolved_ood_root, manifest)
                loaders["ood_dev"] = _build_ood_loader(
                    source_root=resolved_ood_root,
                    split_paths=dev_paths,
                    split_name="ood_dev",
                    seed_offset=31,
                )
                loaders["ood"] = _build_ood_loader(
                    source_root=resolved_ood_root,
                    split_paths=test_paths,
                    split_name="ood_test",
                    seed_offset=30,
                )
                for loader in (loaders["ood_dev"], loaders["ood"]):
                    ood_split_dataset: Any = getattr(loader, "dataset", None)
                    if ood_split_dataset is not None:
                        ood_split_dataset.ood_sample_types = [
                            slice_map.get(str(Path(path).resolve(strict=False)), "unlabeled")
                            for path in list(getattr(ood_split_dataset, "image_paths", []))
                        ]
                    setattr(loader, "_ood_split_manifest", manifest)
                    setattr(
                        loader,
                        "_ood_split_manifest_path",
                        str(resolved_ood_root / str(real_ood_split_manifest_name or "ood_split_manifest.json")),
                    )
            else:
                loaders["ood"] = _build_ood_loader(source_root=resolved_ood_root, split_name="ood", seed_offset=30)
        else:
            loaders["ood"] = _build_ood_loader(source_root=resolved_ood_root, split_name="ood", seed_offset=30)
    if resolved_oe_root.exists():
        if not resolved_oe_root.is_dir():
            raise NotADirectoryError(f"OE root is not a directory: {resolved_oe_root}")
        loaders["oe"] = _build_ood_loader(
            source_root=resolved_oe_root,
            split_name="oe",
            seed_offset=32,
        )
        setattr(loaders["oe"], "_ood_split_name", "oe")
    return loaders
