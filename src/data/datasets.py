from __future__ import annotations

import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset

from .ood_splits import infer_ood_slice
from .transforms import build_image_transform


class LRUCache:
    """Tiny TTL-capable cache for preprocessed validation images."""

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = int(max(1, capacity))
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.ttl_seconds: Optional[float] = None

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        if self._is_expired(key):
            self.__delitem__(key)
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def _is_expired(self, key: str) -> bool:
        """Return True if the given key has expired under the current TTL."""
        if self.ttl_seconds is None:
            return False
        age = time.time() - self.timestamps.get(key, 0.0)
        return age > self.ttl_seconds

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        self.timestamps[key] = time.time()
        while len(self.cache) > self.capacity:
            oldest, _ = self.cache.popitem(last=False)
            self.timestamps.pop(oldest, None)

    def clear(self) -> None:
        self.cache.clear()
        self.timestamps.clear()

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, key: str) -> Optional[Any]:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.put(key, value)

    def __delitem__(self, key: str) -> None:
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VALID_ERROR_POLICIES = {"tolerant", "strict"}


def normalize_split(split: str) -> str:
    if split == "train":
        return "continual"
    if split in {"val", "test", "continual", "ood"}:
        return split
    raise ValueError(f"Unsupported split: {split}")


def default_crop_classes(crop: str) -> List[str]:
    defaults = {
        "tomato": ["healthy", "early_blight", "late_blight", "septoria_leaf_spot", "bacterial_spot"],
        "pepper": ["healthy", "bell_pepper_bacterial_spot"],
        "corn": ["healthy", "common_rust", "northern_leaf_blight"],
    }
    return defaults.get(str(crop).strip().lower(), ["healthy"])


def infer_crop_classes_from_layout(data_dir: str, crop: str) -> List[str]:
    crop_root = Path(data_dir) / crop
    class_names: set[str] = set()
    for split in ("continual", "val", "test"):
        split_root = crop_root / split
        if not split_root.exists():
            continue
        for class_dir in split_root.iterdir():
            if class_dir.is_dir() and class_dir.name:
                class_names.add(class_dir.name)
    return sorted(class_names)


class CropDataset(Dataset):
    """Dataset over `data/<crop>/{continual,val,test}/<class>/<image>`."""

    def __init__(
        self,
        data_dir: str,
        crop: str,
        split: str = "train",
        class_names: Optional[Sequence[str]] = None,
        transform: bool = True,
        target_size: int = 224,
        use_cache: bool = True,
        cache_size: int = 1000,
        cache_train_split: bool = False,
        error_policy: str = "tolerant",
        validate_images_on_init: bool = False,
        augmentation_policy: str = "randaugment",
        randaugment_num_ops: int = 2,
        randaugment_magnitude: int = 7,
        augmix_severity: int = 3,
        augmix_width: int = 3,
        augmix_depth: int = -1,
        augmix_alpha: float = 1.0,
        split_root: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.crop = str(crop)
        self.split = normalize_split(split)
        self.split_root = Path(split_root).expanduser() if split_root is not None else None
        self.target_size = int(target_size)
        self.augmentation_policy = str(augmentation_policy)
        self.randaugment_num_ops = int(randaugment_num_ops)
        self.randaugment_magnitude = int(randaugment_magnitude)
        self.augmix_severity = int(augmix_severity)
        self.augmix_width = int(augmix_width)
        self.augmix_depth = int(augmix_depth)
        self.augmix_alpha = float(augmix_alpha)
        self.use_cache = bool(use_cache)
        self.cache = LRUCache(cache_size) if self.use_cache else None
        self.cache_train_split = bool(cache_train_split)
        self.error_policy = str(error_policy).strip().lower()
        if self.error_policy not in VALID_ERROR_POLICIES:
            raise ValueError(f"Unsupported error policy: {error_policy}")
        self.validate_images_on_init = bool(validate_images_on_init)
        self.load_errors: List[Dict[str, str]] = []
        self.skipped_files: List[str] = []
        self._retired_indices: set[int] = set()

        inferred = [str(name) for name in class_names] if class_names is not None else infer_crop_classes_from_layout(
            data_dir=str(self.data_dir),
            crop=self.crop,
        )
        self.classes = inferred or default_crop_classes(self.crop)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.ood_sample_types: List[str] = []
        self.image_paths, self.labels = self._load_data()
        if self.validate_images_on_init:
            self._validate_image_paths()
        self.transform = build_image_transform(
            self.target_size,
            training=bool(transform),
            augmentation_policy=self.augmentation_policy,
            randaugment_num_ops=self.randaugment_num_ops,
            randaugment_magnitude=self.randaugment_magnitude,
            augmix_severity=self.augmix_severity,
            augmix_width=self.augmix_width,
            augmix_depth=self.augmix_depth,
            augmix_alpha=self.augmix_alpha,
        )

    def _load_data(self) -> Tuple[List[Path], List[int]]:
        base_dir = self.split_root if self.split_root is not None else self.data_dir / self.crop / self.split
        if not base_dir.exists():
            return [], []

        if self.split == "ood":
            ood_paths = sorted(
                path
                for path in base_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            self.ood_sample_types = [infer_ood_slice(path, base_dir) for path in ood_paths]
            return ood_paths, [-1] * len(ood_paths)

        image_paths: List[Path] = []
        labels: List[int] = []
        for class_name in self.classes:
            class_dir = base_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                image_paths.append(image_path)
                labels.append(self.class_to_idx[class_name])
        return image_paths, labels

    def _record_error(self, img_path: Path, exc: Exception) -> None:
        payload = {"path": str(img_path), "error": str(exc)}
        self.load_errors.append(payload)
        self.skipped_files.append(str(img_path))
        logger.error("Failed to load %s: %s", img_path, exc)

    def _validate_image_paths(self) -> None:
        if not self.image_paths:
            return

        valid_paths: List[Path] = []
        valid_labels: List[int] = []
        failed: List[Dict[str, str]] = []
        for img_path, label in zip(self.image_paths, self.labels):
            try:
                self._verify_image_file(img_path)
                valid_paths.append(img_path)
                valid_labels.append(label)
            except (OSError, SyntaxError, ValueError) as exc:
                # PIL reports truncated or corrupt files with any of these types.
                failed.append({"path": str(img_path), "error": str(exc)})

        if failed and self.error_policy == "strict":
            first = failed[0]
            raise ValueError(f"Failed to validate dataset image {first['path']}: {first['error']}")
        self._apply_validation_results(failed, valid_paths, valid_labels)

    def _apply_validation_results(self, failed: List[Dict[str, str]], valid_paths: List[Path], valid_labels: List[int]) -> None:
        if failed and self.error_policy == "strict":
            first = failed[0]
            raise ValueError(f"Failed to validate dataset image {first['path']}: {first['error']}")

        if not failed:
            return

        self.load_errors.extend(failed)
        self.skipped_files.extend(row["path"] for row in failed)
        valid_path_set = {str(path) for path in valid_paths}
        if self.split == "ood" and self.ood_sample_types:
            self.ood_sample_types = [
                sample_type
                for path, sample_type in zip(self.image_paths, self.ood_sample_types)
                if str(path) in valid_path_set
            ]
        self.image_paths = valid_paths
        self.labels = valid_labels

    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def _decode_image(img_path: Path) -> Image.Image:
        raw = cv2.imread(str(img_path))
        if raw is None:
            raise ValueError(f"Failed to load image: {img_path}")
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def _verify_image_file(img_path: Path) -> None:
        with Image.open(img_path) as image:
            image.verify()

    def _load_image(self, img_path: Path) -> Image.Image:
        cache = self.cache
        cache_enabled_for_split = self.use_cache and cache is not None and (
            self.split != "continual" or self.cache_train_split
        )
        if cache_enabled_for_split and cache is not None:
            cached = cache.get(str(img_path))
            if cached is not None:
                return cached.copy()

        image = self._decode_image(img_path)

        if cache_enabled_for_split and cache is not None:
            cache.put(str(img_path), image.copy())
        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if not self.image_paths:
            raise IndexError("CropDataset is empty.")

        attempts = 0
        current_idx = int(idx % len(self.image_paths))
        while attempts < len(self.image_paths):
            if current_idx in self._retired_indices:
                attempts += 1
                current_idx = (current_idx + 1) % len(self.image_paths)
                continue
            img_path = self.image_paths[current_idx]
            label = self.labels[current_idx]
            try:
                image = self.transform(self._load_image(img_path))
                return image, label
            except Exception as exc:
                self._record_error(img_path, exc)
                if self.error_policy == "strict":
                    raise
                self._retired_indices.add(current_idx)
                attempts += 1
                current_idx = (current_idx + 1) % len(self.image_paths)

        raise RuntimeError("Unable to load any valid images from dataset.")

