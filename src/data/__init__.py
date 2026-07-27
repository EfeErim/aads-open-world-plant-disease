"""Dataset and loader building blocks with lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CropDataset",
    "IMAGE_EXTENSIONS",
    "LRUCache",
    "VALID_ERROR_POLICIES",
    "VALID_SAMPLERS",
    "build_image_transform",
    "create_training_loaders",
    "dict_collate_fn",
    "infer_crop_classes_from_layout",
    "preprocess_image",
]

_EXPORTS = {
    "LRUCache": ("src.data.cache", "LRUCache"),
    "IMAGE_EXTENSIONS": ("src.data.datasets", "IMAGE_EXTENSIONS"),
    "VALID_ERROR_POLICIES": ("src.data.datasets", "VALID_ERROR_POLICIES"),
    "CropDataset": ("src.data.datasets", "CropDataset"),
    "infer_crop_classes_from_layout": ("src.data.datasets", "infer_crop_classes_from_layout"),
    "VALID_SAMPLERS": ("src.data.loaders", "VALID_SAMPLERS"),
    "create_training_loaders": ("src.data.loaders", "create_training_loaders"),
    "dict_collate_fn": ("src.data.loaders", "dict_collate_fn"),
    "build_image_transform": ("src.data.transforms", "build_image_transform"),
    "preprocess_image": ("src.data.transforms", "preprocess_image"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name), attribute_name)
