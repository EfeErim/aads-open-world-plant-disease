#!/usr/bin/env python3
"""Duplicate-aware grouped dataset preparation for Colab and local workflows."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageOps

from src.data.dataset_layout import (
    IMAGE_EXTENSIONS,
    class_name_aliases,
    normalize_class_name,
)
from src.data.grouped_dataset_embeddings import (
    _compute_neighbor_pairs as _compute_neighbor_pairs,
)
from src.data.grouped_dataset_embeddings import (
    _encode_bioclip_with_components as _encode_bioclip_with_components,
)
from src.data.grouped_dataset_embeddings import (
    _encode_dinov3_with_components as _encode_dinov3_with_components,
)
from src.data.grouped_dataset_embeddings import _encode_with_oom_retries as _encode_with_oom_retries_impl
from src.data.grouped_dataset_embeddings import _is_cuda_oom as _is_cuda_oom
from src.data.grouped_dataset_embeddings import _load_bioclip_components as _load_bioclip_components
from src.data.grouped_dataset_embeddings import _load_dinov3_components as _load_dinov3_components
from src.data.grouped_dataset_embeddings import _progress as _progress
from src.data.grouped_dataset_embeddings import _resolve_amp_dtype as _resolve_amp_dtype
from src.data.grouped_dataset_embeddings import _resolve_embedding_device as _resolve_embedding_device
from src.data.grouped_dataset_embeddings import (
    resolve_safe_embedding_batch_size as resolve_safe_embedding_batch_size,
)
from src.data.grouped_dataset_materialization import DEFAULT_RUNTIME_ROOT as DEFAULT_RUNTIME_ROOT
from src.data.grouped_dataset_materialization import build_prepared_dataset_key as build_prepared_dataset_key
from src.data.grouped_dataset_materialization import (
    materialize_grouped_runtime_dataset as materialize_grouped_runtime_dataset,
)
from src.data.grouped_dataset_policy import (
    BIOCLIP_AUTO_MIN,
    BIOCLIP_REVIEW_MIN,
    DEFAULT_NEIGHBORS,
    DINO_AUTO_MIN,
    DINO_REVIEW_MIN,
    HUMAN_REVIEW_PACKET_FILENAME,
    LABEL_REVIEW_SUMMARY_FILENAME,
    PHASH_AUTO_MAX_DISTANCE,
    PHASH_REVIEW_MAX_DISTANCE,
    PHASH_SIZE,
)
from src.data.grouped_dataset_policy import (
    BIOCLIP_CROSS_CLASS_BLOCK_MIN as BIOCLIP_CROSS_CLASS_BLOCK_MIN,
)
from src.data.grouped_dataset_policy import (
    DINO_CROSS_CLASS_BLOCK_MIN as DINO_CROSS_CLASS_BLOCK_MIN,
)
from src.data.grouped_dataset_reporting import _skipped_class_entry as _skipped_class_entry
from src.data.grouped_dataset_reporting import build_human_review_packet as build_human_review_packet
from src.data.grouped_dataset_reporting import build_label_review_summary as build_label_review_summary
from src.data.grouped_dataset_reporting import format_human_review_packet as format_human_review_packet
from src.guided_artifacts import refresh_prep_guided_artifacts
from src.shared.csv_utils import write_csv_rows
from src.shared.json_utils import read_json, write_json

DEFAULT_DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_BIOCLIP_MODEL_ID = "imageomics/bioclip-2.5-vith14"
DEFAULT_ARTIFACT_ROOT = Path("outputs") / "colab_notebook_data_prep"
SOURCE_HINT_UNKNOWN = "unknown"
QUALITY_WARN_MIN_SIZE = 224
QUALITY_CRITICAL_MIN_SIZE = 160
SYNTHETIC_HINT_KEYWORDS = (
    "aug",
    "augment",
    "augmented",
    "roboflow",
    "gan",
    "synthetic",
    "flip",
    "rot",
    "rotate",
    "bright",
    "contrast",
    "noise",
    "pca",
)
EVAL_RISK_KEYWORDS = (
    "ekran",
    "ekran_goruntusu",
    "ekran_görüntüsü",
    "screenshot",
    "preview",
    "pngtree",
    "shutterstock",
    "gettyimages",
    "freepik",
    "alamy",
)
SOURCE_LIKE_WEBSITE_KEYWORDS = (
    "istockphoto",
    "pngtree",
    "shutterstock",
    "gettyimages",
    "freepik",
    "alamy",
)
SOURCE_LIKE_GROUP_UNKNOWN = "unknown"
SOURCE_STYLE_GROUP_UNKNOWN = "unknown"
STEM_NOISE_KEYWORDS = {
    "copy",
    "edited",
    "edit",
    "crop",
    "cropped",
    "flip",
    "flipped",
    "rotate",
    "rotated",
    "rot",
    "aug",
    "augmented",
    "augment",
    "preview",
    "small",
    "large",
    "final",
    "web",
}
SOURCE_STYLE_RISK_KEYWORDS = tuple(
    sorted(
        set(EVAL_RISK_KEYWORDS)
        | {
            "depositphotos",
            "dreamstime",
            "watermark",
            "watermarked",
            "sample",
            "preview",
            "download",
            "resized",
            "compressed",
            "whatsapp",
            "telegram",
            "facebook",
            "instagram",
        }
    )
)
WEB_EXPORT_KEYWORDS = (
    "download",
    "preview",
    "resized",
    "compressed",
    "web",
    "whatsapp",
    "telegram",
    "facebook",
    "instagram",
)
LABEL_RISK_CLEAR = "clear"
LABEL_RISK_TRAIN_ONLY = "train_only_risk"
LABEL_RISK_REVIEW = "review_candidate"
LABEL_RISK_BLOCKING = "blocking_conflict"


def _encode_with_oom_retries(
    encode_fn: Callable[..., np.ndarray],
    *,
    paths: Sequence[Path],
    batch_size: int,
    progress_fn: Optional[Callable[[str], None]],
    batch_progress_fn: Optional[Callable[[int, int], None]] = None,
    label: str,
    **kwargs: Any,
) -> np.ndarray:
    return _encode_with_oom_retries_impl(
        encode_fn,
        paths=paths,
        batch_size=batch_size,
        progress_fn=progress_fn,
        batch_progress_fn=batch_progress_fn,
        label=label,
        oom_predicate=_is_cuda_oom,
        **kwargs,
    )


@dataclass
class ImageRecord:
    relative_path: str
    absolute_path: str
    raw_class_name: str
    normalized_class_name: str
    source_hint: str
    source_like_group: str
    synthetic_hint: bool
    eval_quality_risk: bool
    readable: bool
    width: int
    height: int
    blur_score: float
    brightness_mean: float
    exact_hash: str
    phash_hex: str
    class_order_index: int
    excluded_reason: str = ""
    source_style_group: str = SOURCE_STYLE_GROUP_UNKNOWN
    source_style_risk: bool = False
    source_style_reason: str = ""
    aspect_ratio_bucket: str = "unknown"
    resolution_bucket: str = "unknown"
    border_layout: str = "unknown"
    compression_style: str = "unknown"


@dataclass
class LabelRiskRecord:
    relative_path: str
    normalized_class_name: str
    label_risk_level: str
    label_risk_reason: str
    label_risk_score: float
    train_only_routed: bool = False


@dataclass
class ReviewPair:
    pair_type: str
    class_a: str
    class_b: str
    path_a: str
    path_b: str
    exact_match: bool
    phash_distance: int
    dino_cosine: float
    bioclip_cosine: float
    adjacency_distance: Optional[int]
    review_rank: int
    decision: str
    reason: str
    cluster_id: str = ""
    triage_resolution: str = ""
    triage_reason: str = ""


@dataclass
class ReviewCluster:
    cluster_id: str
    normalized_class_name: str
    image_count: int
    pair_count: int
    source_hint_count: int
    source_hints: str
    synthetic_hint_count: int
    min_phash_distance: int
    max_phash_distance: int
    max_dino_cosine: float
    max_bioclip_cosine: float
    min_adjacency_distance: Optional[int]
    resolution: str
    reason: str
    relative_paths: str


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def _load_taxonomy_expected_classes(crop_name: str, taxonomy_path: Optional[Path]) -> set[str]:
    if taxonomy_path is None or not Path(taxonomy_path).exists():
        return set()
    payload = read_json(Path(taxonomy_path), default={}, expect_type=dict)
    if not isinstance(payload, dict):
        return set()
    crop_key = normalize_class_name(crop_name)
    crop_specific = payload.get("crop_specific_diseases", {})
    if not isinstance(crop_specific, dict):
        return set()
    expected = {normalize_class_name(item) for item in crop_specific.get(crop_key, []) if normalize_class_name(item)}
    expected.add("healthy")
    return expected


def normalize_prepared_class_name(raw_class_name: str, *, crop_name: str, expected_classes: set[str]) -> str:
    normalized = normalize_class_name(raw_class_name)
    if not expected_classes:
        return normalized
    aliases = class_name_aliases(normalized, crop_name=normalize_class_name(crop_name))
    matches = sorted(expected_classes & aliases)
    if len(matches) == 1:
        return matches[0]
    return normalized


def _normalize_relative_path(value: Any) -> str:
    normalized = str(value or "").strip().replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    normalized = normalized.lstrip("./")
    return normalized.strip("/")


def _infer_source_hint(class_dir: Path, image_path: Path) -> str:
    relative = image_path.relative_to(class_dir)
    if len(relative.parts) <= 1:
        return SOURCE_HINT_UNKNOWN
    hint = normalize_class_name(relative.parts[0])
    return hint or SOURCE_HINT_UNKNOWN


def _has_synthetic_hint(path_like: str) -> bool:
    return bool(_synthetic_hint_matches(path_like))


def _synthetic_hint_matches(path_like: str) -> List[str]:
    normalized_relative = _normalize_relative_path(path_like)
    parts = [normalize_class_name(part) for part in normalized_relative.split("/") if normalize_class_name(part)]
    if len(parts) > 1:
        # The first path component is the class folder. Do not let disease names such as
        # botrytis_bunch_rot trip rotation-augmentation detection.
        parts = parts[1:]
    tokens = [token for part in parts for token in normalize_class_name(Path(part).stem).split("_") if token]
    matches: List[str] = []
    for index, token in enumerate(tokens):
        if token in SYNTHETIC_HINT_KEYWORDS and token != "rot":
            matches.append(token)
            continue
        if re.fullmatch(r"rot(?:90|180|270)", token):
            matches.append(token)
            continue
        if token == "rot" and index + 1 < len(tokens) and tokens[index + 1] in {"90", "180", "270"}:
            matches.append(f"{token}_{tokens[index + 1]}")
    return matches


def _normalized_path_tokens(path_like: str) -> List[str]:
    normalized = normalize_class_name(path_like)
    return [token for token in normalized.split("_") if token]


def _stem_family_fingerprint(relative_path: str) -> str:
    stem = normalize_class_name(Path(str(relative_path or "")).stem)
    tokens = [token for token in stem.split("_") if token]
    if not tokens:
        return ""
    cleaned: List[str] = []
    for token in tokens:
        if token in STEM_NOISE_KEYWORDS:
            continue
        if token in {"ekran", "goruntusu"}:
            continue
        if token.isdigit() and len(token) <= 3:
            continue
        cleaned.append(token)
    if not cleaned:
        cleaned = tokens[:]
    fingerprint = "_".join(cleaned).strip("_")
    fingerprint = re.sub(r"(_+\d+)$", "", fingerprint).strip("_")
    return fingerprint or stem


def _infer_source_like_group(record: ImageRecord) -> str:
    normalized_path = normalize_class_name(record.relative_path)
    for keyword in SOURCE_LIKE_WEBSITE_KEYWORDS:
        if keyword not in normalized_path:
            continue
        match = re.search(rf"{keyword}_(\d+)", normalized_path)
        if match:
            return f"web:{keyword}:{match.group(1)}"
        return f"web:{keyword}"

    screenshot_match = re.search(
        r"(ekran_goruntusu|ekran_görüntüsü|screenshot)_(\d{4})_(\d{2})_(\d{2})",
        normalized_path,
    )
    if screenshot_match:
        return f"screenshot:{screenshot_match.group(2)}_{screenshot_match.group(3)}_{screenshot_match.group(4)}"
    if "ekran_goruntusu" in normalized_path or "ekran_görüntüsü" in normalized_path or "screenshot" in normalized_path:
        stem_fingerprint = _stem_family_fingerprint(record.relative_path)
        return f"screenshot:{stem_fingerprint or 'batch'}"

    if record.source_hint and record.source_hint != SOURCE_HINT_UNKNOWN:
        return f"hint:{normalize_class_name(record.source_hint)}"
    return SOURCE_LIKE_GROUP_UNKNOWN


def _has_eval_quality_risk(record: ImageRecord) -> bool:
    normalized_path = normalize_class_name(record.relative_path)
    if any(keyword in normalized_path for keyword in EVAL_RISK_KEYWORDS):
        return True
    if record.source_like_group.startswith("screenshot:") or record.source_like_group.startswith("web:"):
        return True
    return False


def _compute_exact_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compute_phash_hex(image: Image.Image) -> str:
    img = ImageOps.grayscale(image).resize((32, 32))
    pixels = np.asarray(img, dtype=np.float32)
    dct = np.fft.fft2(pixels)
    low_freq = np.abs(dct[: PHASH_SIZE + 1, : PHASH_SIZE + 1])
    med = np.median(low_freq[1:, 1:])
    bits = low_freq > med
    packed = 0
    for bit in bits.flatten()[: PHASH_SIZE * PHASH_SIZE]:
        packed = (packed << 1) | int(bool(bit))
    width = PHASH_SIZE * PHASH_SIZE // 4
    return f"{packed:0{width}x}"


def _phash_distance(a: str, b: str) -> int:
    return int((int(a, 16) ^ int(b, 16)).bit_count())


def _compute_blur_and_brightness(image: Image.Image) -> tuple[float, float]:
    grayscale = np.asarray(ImageOps.grayscale(image), dtype=np.float32)
    brightness_mean = float(grayscale.mean() / 255.0) if grayscale.size else 0.0
    if grayscale.shape[0] < 3 or grayscale.shape[1] < 3:
        return 0.0, brightness_mean
    center = grayscale[1:-1, 1:-1]
    lap = -4.0 * center + grayscale[:-2, 1:-1] + grayscale[2:, 1:-1] + grayscale[1:-1, :-2] + grayscale[1:-1, 2:]
    blur_score = float(lap.var())
    return blur_score, brightness_mean


def _aspect_ratio_bucket(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "unknown"
    ratio = float(width) / float(height)
    if ratio < 0.70:
        return "portrait_narrow"
    if ratio < 0.90:
        return "portrait"
    if ratio <= 1.10:
        return "square"
    if ratio <= 1.45:
        return "landscape"
    return "landscape_wide"


def _resolution_bucket(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "unknown"
    min_dim = min(int(width), int(height))
    max_dim = max(int(width), int(height))
    if min_dim < QUALITY_CRITICAL_MIN_SIZE:
        return "tiny"
    if min_dim < QUALITY_WARN_MIN_SIZE:
        return "small"
    if max_dim <= 768:
        return "web_small"
    if max_dim <= 1400:
        return "web_medium"
    return "large"


def _detect_border_layout(image: Image.Image) -> str:
    if image.width < 16 or image.height < 16:
        return "too_small"
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    band = max(2, min(image.width, image.height) // 32)
    border = np.concatenate(
        [
            pixels[:band, :, :].reshape(-1, 3),
            pixels[-band:, :, :].reshape(-1, 3),
            pixels[:, :band, :].reshape(-1, 3),
            pixels[:, -band:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    center = pixels[band:-band, band:-band, :].reshape(-1, 3)
    if center.size == 0:
        return "too_small"
    border_mean = float(border.mean())
    center_mean = float(center.mean())
    border_std = float(border.std())
    center_std = float(center.std())
    if border_std < 0.035 and abs(border_mean - center_mean) > 0.18:
        if border_mean > 0.82:
            return "light_frame"
        if border_mean < 0.18:
            return "dark_frame"
        return "flat_frame"
    if center_std > 0 and border_std / max(center_std, 1e-6) < 0.35 and abs(border_mean - center_mean) > 0.10:
        return "letterbox_or_margin"
    return "natural"


def _compression_style(path_like: str, width: int, height: int) -> str:
    normalized = normalize_class_name(path_like)
    suffix = Path(str(path_like or "")).suffix.lower().lstrip(".") or "unknown"
    if re.search(r"(?:^|_)(?:\d{2,5})x(?:\d{2,5})(?:_|$)", normalized):
        return f"web_export_{suffix}"
    if any(keyword in normalized for keyword in WEB_EXPORT_KEYWORDS):
        return f"web_export_{suffix}"
    if width > 0 and height > 0 and max(width, height) in {224, 256, 299, 320, 512, 612, 640, 768, 1024}:
        return f"common_res_{suffix}"
    return suffix


def _source_style_keyword_matches(path_like: str) -> List[str]:
    tokens = _normalized_path_tokens(path_like)
    normalized = "_".join(tokens)
    return [keyword for keyword in SOURCE_STYLE_RISK_KEYWORDS if keyword in normalized]


def _infer_source_style_group(record: ImageRecord) -> str:
    source_like_group = str(record.source_like_group or SOURCE_LIKE_GROUP_UNKNOWN).strip()
    if source_like_group != SOURCE_LIKE_GROUP_UNKNOWN:
        return source_like_group

    keyword_matches = _source_style_keyword_matches(record.relative_path)
    if keyword_matches:
        keyword = keyword_matches[0]
        return f"style:{keyword}:{record.aspect_ratio_bucket}:{record.resolution_bucket}:{record.compression_style}"

    if record.border_layout in {"light_frame", "dark_frame", "flat_frame", "letterbox_or_margin"}:
        return (
            f"layout:{record.border_layout}:{record.aspect_ratio_bucket}:"
            f"{record.resolution_bucket}:{record.compression_style}"
        )

    if record.source_hint and record.source_hint != SOURCE_HINT_UNKNOWN:
        return f"hint:{normalize_class_name(record.source_hint)}"
    return SOURCE_STYLE_GROUP_UNKNOWN


def _apply_source_style_risk(records: Sequence[ImageRecord]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str], List[ImageRecord]] = defaultdict(list)
    for record in records:
        group = str(record.source_style_group or SOURCE_STYLE_GROUP_UNKNOWN).strip()
        if group == SOURCE_STYLE_GROUP_UNKNOWN:
            continue
        groups[(record.normalized_class_name, group)].append(record)

    rows: List[Dict[str, Any]] = []
    for (class_name, group), group_records in sorted(groups.items(), key=lambda item: item[0]):
        keyword_hits = sorted(
            {hit for record in group_records for hit in _source_style_keyword_matches(record.relative_path)}
        )
        has_web_or_screenshot = group.startswith(("web:", "screenshot:"))
        has_style_group = group.startswith(("style:", "layout:"))
        has_eval_risk = any(record.eval_quality_risk for record in group_records)
        has_web_export = any(str(record.compression_style).startswith("web_export") for record in group_records)
        layout_hits = sorted(
            {
                record.border_layout
                for record in group_records
                if record.border_layout in {"light_frame", "dark_frame", "flat_frame", "letterbox_or_margin"}
            }
        )
        risk_reasons: List[str] = []
        if has_web_or_screenshot:
            risk_reasons.append("web_or_screenshot_source_style")
        if keyword_hits:
            risk_reasons.append("path_keyword:" + "|".join(keyword_hits[:5]))
        if has_eval_risk:
            risk_reasons.append("eval_quality_proxy")
        if has_web_export:
            risk_reasons.append("web_export_signature")
        if layout_hits:
            risk_reasons.append("layout:" + "|".join(layout_hits))
        if has_style_group:
            risk_reasons.append("weak_style_cluster")

        risk = bool(risk_reasons) and (
            len(group_records) >= 2 or has_web_or_screenshot or has_eval_risk or has_style_group
        )
        reason = "; ".join(risk_reasons) if risk_reasons else ""
        for record in group_records:
            record.source_style_risk = bool(risk)
            record.source_style_reason = reason if risk else ""
        rows.append(
            {
                "normalized_class_name": class_name,
                "source_style_group": group,
                "image_count": len(group_records),
                "source_style_risk": bool(risk),
                "source_style_reason": reason,
                "aspect_ratio_buckets": "|".join(sorted({record.aspect_ratio_bucket for record in group_records})),
                "resolution_buckets": "|".join(sorted({record.resolution_bucket for record in group_records})),
                "border_layouts": "|".join(sorted({record.border_layout for record in group_records})),
                "compression_styles": "|".join(sorted({record.compression_style for record in group_records})),
                "relative_paths": "|".join(
                    record.relative_path for record in sorted(group_records, key=lambda item: item.relative_path)
                ),
            }
        )
    return rows


def _path_keyword_penalty(path_like: str) -> int:
    if not str(path_like or "").strip():
        return len(SYNTHETIC_HINT_KEYWORDS)
    return len(_synthetic_hint_matches(path_like))


def _record_preference_key(record: ImageRecord) -> tuple[Any, ...]:
    pixels = int(record.width) * int(record.height)
    min_dim = min(int(record.width), int(record.height)) if pixels else 0
    resolution_penalty = 2 if pixels <= 0 else (1 if min_dim < QUALITY_WARN_MIN_SIZE else 0)
    brightness_penalty = abs(float(record.brightness_mean) - 0.5)
    return (
        1 if record.synthetic_hint else 0,
        _path_keyword_penalty(record.relative_path),
        resolution_penalty,
        -pixels,
        -float(record.blur_score),
        float(brightness_penalty),
        len(Path(record.relative_path).as_posix()),
        record.relative_path.lower(),
    )


def _select_canonical_record(records: Sequence[ImageRecord]) -> ImageRecord:
    if not records:
        raise ValueError("Cannot select a canonical record from an empty family.")
    return min(records, key=_record_preference_key)


def scan_class_root_dataset(
    *,
    class_root: Path,
    crop_name: str,
    taxonomy_path: Optional[Path] = None,
) -> tuple[List[ImageRecord], Dict[str, Any]]:
    expected_classes = _load_taxonomy_expected_classes(crop_name, taxonomy_path)
    records: List[ImageRecord] = []
    normalization_report: Dict[str, Any] = {
        "crop_name": str(crop_name),
        "expected_classes": sorted(expected_classes),
        "raw_to_normalized": {},
        "unmatched_raw_classes": [],
    }

    class_dirs = sorted([path for path in class_root.iterdir() if path.is_dir()], key=lambda path: path.name.lower())
    for class_dir in class_dirs:
        normalized_class_name = normalize_prepared_class_name(
            class_dir.name,
            crop_name=crop_name,
            expected_classes=expected_classes,
        )
        normalization_report["raw_to_normalized"][class_dir.name] = normalized_class_name
        if expected_classes and normalized_class_name not in expected_classes:
            normalization_report["unmatched_raw_classes"].append(class_dir.name)
        class_image_paths = sorted(class_dir.rglob("*"), key=lambda path: str(path).lower())
        class_order_index = 0
        for image_path in class_image_paths:
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative_path = image_path.relative_to(class_root).as_posix()
            source_hint = _infer_source_hint(class_dir, image_path)
            synthetic_hint = _has_synthetic_hint(relative_path)
            readable = False
            width = 0
            height = 0
            border_layout = "unknown"
            blur_score = 0.0
            brightness_mean = 0.0
            exact_hash = ""
            phash_hex = "0" * (PHASH_SIZE * PHASH_SIZE // 4)
            excluded_reason = ""
            try:
                with Image.open(image_path) as raw:
                    image = ImageOps.exif_transpose(raw.convert("RGB"))
                    width, height = image.size
                    border_layout = _detect_border_layout(image)
                    blur_score, brightness_mean = _compute_blur_and_brightness(image)
                    phash_hex = _compute_phash_hex(image)
                    readable = True
            except Exception:
                import logging

                logging.exception("Unhandled exception")
                raise
            try:
                exact_hash = _compute_exact_hash(image_path)
            except Exception:
                import logging

                logging.exception("Unhandled exception")
                raise
            records.append(
                ImageRecord(
                    relative_path=relative_path,
                    absolute_path=str(image_path.resolve()),
                    raw_class_name=class_dir.name,
                    normalized_class_name=normalized_class_name,
                    source_hint=source_hint,
                    source_like_group=SOURCE_LIKE_GROUP_UNKNOWN,
                    synthetic_hint=synthetic_hint,
                    eval_quality_risk=False,
                    readable=readable,
                    width=int(width),
                    height=int(height),
                    aspect_ratio_bucket=_aspect_ratio_bucket(width, height),
                    resolution_bucket=_resolution_bucket(width, height),
                    border_layout=border_layout,
                    compression_style=_compression_style(relative_path, width, height),
                    blur_score=float(blur_score),
                    brightness_mean=float(brightness_mean),
                    exact_hash=exact_hash,
                    phash_hex=phash_hex,
                    class_order_index=int(class_order_index),
                    excluded_reason=excluded_reason,
                )
            )
            class_order_index += 1
    normalization_report["unmatched_raw_classes"] = sorted(set(normalization_report["unmatched_raw_classes"]))
    for record in records:
        record.source_like_group = _infer_source_like_group(record)
        record.eval_quality_risk = _has_eval_quality_risk(record)
        record.source_style_group = _infer_source_style_group(record)
    return records, normalization_report


def _refresh_record_availability(records: Sequence[ImageRecord]) -> Dict[str, int]:
    excluded_counts = {
        "missing_after_scan": 0,
        "unreadable_after_scan": 0,
    }
    for record in records:
        if record.excluded_reason:
            continue
        image_path = Path(record.absolute_path)
        if not image_path.is_file():
            record.readable = False
            record.excluded_reason = "missing_after_scan"
            excluded_counts["missing_after_scan"] += 1
            continue
        try:
            with Image.open(image_path) as raw:
                _ = ImageOps.exif_transpose(raw.convert("RGB"))
        except Exception:
            import logging

            logging.exception("Unhandled exception")
            raise
    return excluded_counts


def _adjacency_distance(a: ImageRecord, b: ImageRecord) -> Optional[int]:
    if a.normalized_class_name != b.normalized_class_name:
        return None
    return abs(int(a.class_order_index) - int(b.class_order_index))


def _review_rank(a: ImageRecord, b: ImageRecord) -> int:
    distance = _adjacency_distance(a, b)
    if distance is None:
        return 10**9
    return int(distance)


def _classify_review_cluster(
    *,
    records: Sequence[ImageRecord],
    pairs: Sequence[ReviewPair],
) -> tuple[str, str]:
    cluster_size = len(records)
    source_hints = {
        record.source_hint for record in records if record.source_hint and record.source_hint != SOURCE_HINT_UNKNOWN
    }
    source_like_groups = {
        record.source_like_group
        for record in records
        if record.source_like_group and record.source_like_group != SOURCE_LIKE_GROUP_UNKNOWN
    }
    synthetic_count = sum(1 for record in records if record.synthetic_hint)
    min_adjacency = min(
        (pair.adjacency_distance for pair in pairs if pair.adjacency_distance is not None),
        default=None,
    )
    max_phash = max((int(pair.phash_distance) for pair in pairs), default=PHASH_REVIEW_MAX_DISTANCE + 1)
    max_dino = max((float(pair.dino_cosine) for pair in pairs), default=-1.0)
    max_bioclip = max((float(pair.bioclip_cosine) for pair in pairs), default=-1.0)

    if cluster_size > 6:
        return "manual_review", "cluster size exceeds conservative auto-resolve threshold"
    if len(source_hints) > 1:
        return "manual_review", "mixed source hints in same-class review cluster"
    if len(source_like_groups) > 1:
        return "manual_review", "mixed source-like groups in same-class review cluster"

    strong_similarity = max_dino >= DINO_AUTO_MIN and (max_bioclip < 0 or max_bioclip >= BIOCLIP_REVIEW_MIN)
    adjacency_supported = min_adjacency is not None and int(min_adjacency) <= 2 and cluster_size <= 3
    synthetic_supported = synthetic_count > 0 and cluster_size <= 4
    source_like_supported = len(source_like_groups) == 1 and cluster_size <= 4

    if synthetic_supported:
        return "auto_resolve", "small same-class cluster has synthetic lineage hints"
    if source_like_supported:
        return "auto_resolve", "small same-class cluster shares one source-like group"
    if adjacency_supported:
        return "auto_resolve", "very small same-class cluster is adjacency-backed"
    if max_phash > PHASH_REVIEW_MAX_DISTANCE:
        return "manual_review", "perceptual hash disagreement is too large"

    if strong_similarity and cluster_size <= 4:
        return "auto_resolve", "same-class low-risk cluster with strong lineage/similarity support"
    return "manual_review", "same-class cluster lacks strong heuristic support for auto-resolution"


def _triage_review_clusters(
    *,
    review_pairs: Sequence[ReviewPair],
    record_lookup: Dict[str, ImageRecord],
    uf: UnionFind,
) -> tuple[List[ReviewPair], List[ReviewCluster], List[ReviewCluster]]:
    if not review_pairs:
        return [], [], []

    cluster_uf = UnionFind()
    for pair in review_pairs:
        key_a = f"{pair.class_a}::{pair.path_a}"
        key_b = f"{pair.class_b}::{pair.path_b}"
        cluster_uf.union(key_a, key_b)

    cluster_pairs: Dict[str, List[ReviewPair]] = defaultdict(list)
    cluster_paths: Dict[str, set[str]] = defaultdict(set)
    for pair in review_pairs:
        root = cluster_uf.find(f"{pair.class_a}::{pair.path_a}")
        cluster_pairs[root].append(pair)
        cluster_paths[root].add(pair.path_a)
        cluster_paths[root].add(pair.path_b)

    cluster_rows: List[ReviewCluster] = []
    unresolved_pairs: List[ReviewPair] = []
    auto_resolved_clusters: List[ReviewCluster] = []

    for cluster_index, root in enumerate(sorted(cluster_pairs.keys()), start=1):
        pairs = cluster_pairs[root]
        class_name = pairs[0].class_a
        records = [record_lookup[path] for path in sorted(cluster_paths[root])]
        resolution, reason = _classify_review_cluster(records=records, pairs=pairs)
        cluster_id = f"{class_name}__review_cluster_{cluster_index:04d}"
        source_hints = sorted(
            {
                record.source_hint
                for record in records
                if record.source_hint and record.source_hint != SOURCE_HINT_UNKNOWN
            }
        )
        min_adjacency = min(
            (pair.adjacency_distance for pair in pairs if pair.adjacency_distance is not None),
            default=None,
        )
        cluster_row = ReviewCluster(
            cluster_id=cluster_id,
            normalized_class_name=class_name,
            image_count=len(records),
            pair_count=len(pairs),
            source_hint_count=len(source_hints),
            source_hints="|".join(source_hints),
            synthetic_hint_count=sum(1 for record in records if record.synthetic_hint),
            min_phash_distance=min(int(pair.phash_distance) for pair in pairs),
            max_phash_distance=max(int(pair.phash_distance) for pair in pairs),
            max_dino_cosine=max(float(pair.dino_cosine) for pair in pairs),
            max_bioclip_cosine=max(float(pair.bioclip_cosine) for pair in pairs),
            min_adjacency_distance=min_adjacency,
            resolution=resolution,
            reason=reason,
            relative_paths="|".join(record.relative_path for record in records),
        )
        cluster_rows.append(cluster_row)

        if resolution == "auto_resolve":
            base = records[0].relative_path
            for record in records[1:]:
                uf.union(base, record.relative_path)
            auto_resolved_clusters.append(cluster_row)
            for pair in pairs:
                pair.cluster_id = cluster_id
                pair.triage_resolution = resolution
                pair.triage_reason = reason
        else:
            for pair in pairs:
                pair.cluster_id = cluster_id
                pair.triage_resolution = resolution
                pair.triage_reason = reason
                unresolved_pairs.append(pair)

    return unresolved_pairs, cluster_rows, auto_resolved_clusters


def _risk_rank(level: str) -> int:
    return {
        LABEL_RISK_CLEAR: 0,
        LABEL_RISK_TRAIN_ONLY: 1,
        LABEL_RISK_REVIEW: 2,
        LABEL_RISK_BLOCKING: 3,
    }.get(str(level), 0)


def _clear_label_risk(record: ImageRecord) -> LabelRiskRecord:
    return LabelRiskRecord(
        relative_path=record.relative_path,
        normalized_class_name=record.normalized_class_name,
        label_risk_level=LABEL_RISK_CLEAR,
        label_risk_reason="",
        label_risk_score=0.0,
    )


def _merge_label_risk(
    label_risks: Dict[str, LabelRiskRecord],
    *,
    record: ImageRecord,
    level: str,
    reason: str,
    score: float,
) -> None:
    existing = label_risks.get(record.relative_path)
    if existing is not None and _risk_rank(existing.label_risk_level) > _risk_rank(level):
        return
    if existing is not None and _risk_rank(existing.label_risk_level) == _risk_rank(level):
        reasons = [item for item in [existing.label_risk_reason, reason] if item]
        existing.label_risk_reason = "; ".join(dict.fromkeys(reasons))
        existing.label_risk_score = max(float(existing.label_risk_score), float(score))
        return
    label_risks[record.relative_path] = LabelRiskRecord(
        relative_path=record.relative_path,
        normalized_class_name=record.normalized_class_name,
        label_risk_level=level,
        label_risk_reason=reason,
        label_risk_score=float(score),
    )


def _build_label_risk_audit(
    *,
    records: Sequence[ImageRecord],
    review_pairs: Sequence[ReviewPair],
    auto_resolved_review_clusters: Sequence[ReviewCluster],
    blocking_conflicts: Sequence[ReviewPair],
) -> tuple[Dict[str, LabelRiskRecord], List[Dict[str, Any]], Dict[str, Any]]:
    record_lookup = {record.relative_path: record for record in records}
    label_risks: Dict[str, LabelRiskRecord] = {record.relative_path: _clear_label_risk(record) for record in records}

    for pair in blocking_conflicts:
        for path in (pair.path_a, pair.path_b):
            record = record_lookup.get(path)
            if record is None:
                continue
            _merge_label_risk(
                label_risks,
                record=record,
                level=LABEL_RISK_BLOCKING,
                reason=pair.reason or "strong cross-class duplicate/conflict",
                score=1.0,
            )

    for pair in review_pairs:
        reason = pair.triage_reason or pair.reason or "borderline same-class review cluster"
        for path in (pair.path_a, pair.path_b):
            record = record_lookup.get(path)
            if record is None:
                continue
            _merge_label_risk(
                label_risks,
                record=record,
                level=LABEL_RISK_REVIEW,
                reason=reason,
                score=0.75,
            )

    for cluster in auto_resolved_review_clusters:
        paths = [path for path in str(cluster.relative_paths or "").split("|") if path]
        reason = cluster.reason or "auto-routed borderline same-class cluster"
        for path in paths:
            record = record_lookup.get(path)
            if record is None:
                continue
            _merge_label_risk(
                label_risks,
                record=record,
                level=LABEL_RISK_TRAIN_ONLY,
                reason=reason,
                score=0.45,
            )

    review_candidates: List[Dict[str, Any]] = []
    for risk in sorted(
        label_risks.values(),
        key=lambda item: (
            -_risk_rank(item.label_risk_level),
            -item.label_risk_score,
            item.normalized_class_name,
            item.relative_path,
        ),
    ):
        if risk.label_risk_level != LABEL_RISK_REVIEW:
            continue
        review_candidates.append(
            {
                "relative_path": risk.relative_path,
                "normalized_class_name": risk.normalized_class_name,
                "label_risk_level": risk.label_risk_level,
                "label_risk_reason": risk.label_risk_reason,
                "label_risk_score": round(float(risk.label_risk_score), 4),
            }
        )

    level_counts = {
        level: sum(1 for risk in label_risks.values() if risk.label_risk_level == level)
        for level in (LABEL_RISK_CLEAR, LABEL_RISK_TRAIN_ONLY, LABEL_RISK_REVIEW, LABEL_RISK_BLOCKING)
    }
    summary = {
        "schema_version": "v1_label_risk_summary",
        "total_images": len(records),
        "level_counts": level_counts,
        "review_candidate_count": level_counts[LABEL_RISK_REVIEW],
        "train_only_risk_count": level_counts[LABEL_RISK_TRAIN_ONLY],
        "blocking_conflict_count": level_counts[LABEL_RISK_BLOCKING],
        "signals": {
            "cross_class_conflicts": len(blocking_conflicts),
            "same_class_review_pairs": len(review_pairs),
            "auto_resolved_same_class_clusters": len(auto_resolved_review_clusters),
            "neighbor_disagreement_scope": "exact/phash conflicts plus grouped same-class representation review",
        },
        "policy": {
            "clear": "eligible for canonical eval if other split filters pass",
            "train_only_risk": "usable for continual only",
            "review_candidate": "kept out of canonical eval and emitted for manual review",
            "blocking_conflict": "materialization-blocking through cross-class conflict policy",
        },
    }
    return label_risks, review_candidates, summary


def _build_hash_groups(records: Sequence[ImageRecord], *, key_name: str) -> Dict[str, List[ImageRecord]]:
    groups: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in records:
        key = str(getattr(record, key_name, "") or "").strip()
        if key:
            groups[key].append(record)
    return groups


def _register_cross_class_conflicts(
    *,
    groups: Dict[str, List[ImageRecord]],
    conflict_reason: str,
    blocking_conflicts: List[ReviewPair],
    seen_pairs: set[tuple[str, str]],
) -> None:
    for _, values in groups.items():
        grouped_by_class: Dict[str, List[ImageRecord]] = defaultdict(list)
        for record in values:
            grouped_by_class[record.normalized_class_name].append(record)
        if len(grouped_by_class) < 2:
            continue
        for records_a, records_b in itertools.combinations(grouped_by_class.values(), 2):
            for record_a in records_a:
                for record_b in records_b:
                    pair = (
                        min(record_a.relative_path, record_b.relative_path),
                        max(record_a.relative_path, record_b.relative_path),
                    )
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    blocking_conflicts.append(
                        ReviewPair(
                            pair_type="cross_class_conflict",
                            class_a=record_a.normalized_class_name,
                            class_b=record_b.normalized_class_name,
                            path_a=record_a.relative_path,
                            path_b=record_b.relative_path,
                            exact_match=record_a.exact_hash == record_b.exact_hash,
                            phash_distance=_phash_distance(record_a.phash_hex, record_b.phash_hex),
                            dino_cosine=-1.0,
                            bioclip_cosine=-1.0,
                            adjacency_distance=None,
                            review_rank=10**9,
                            decision="block",
                            reason=conflict_reason,
                        )
                    )


def _compute_pair_similarity(
    embeddings: np.ndarray,
    *,
    path_keys: Sequence[str],
    pairs: Iterable[tuple[str, str]],
) -> Dict[tuple[str, str], float]:
    if embeddings.size == 0:
        return {}
    index_by_path = {path: idx for idx, path in enumerate(path_keys)}
    similarities: Dict[tuple[str, str], float] = {}
    for path_a, path_b in pairs:
        idx_a = index_by_path.get(path_a)
        idx_b = index_by_path.get(path_b)
        if idx_a is None or idx_b is None:
            continue
        score = float(np.dot(embeddings[idx_a], embeddings[idx_b]))
        similarities[(min(path_a, path_b), max(path_a, path_b))] = score
    return similarities


def _select_bioclip_candidate_pairs(
    dino_pairs: Dict[tuple[str, str], float],
    records_by_path: Dict[str, ImageRecord],
) -> List[tuple[str, str]]:
    return [
        pair
        for pair, dino_cosine in dino_pairs.items()
        if dino_cosine >= DINO_REVIEW_MIN
        and _phash_distance(records_by_path[pair[0]].phash_hex, records_by_path[pair[1]].phash_hex)
        > PHASH_AUTO_MAX_DISTANCE
    ]


def _family_targets(records: Sequence[ImageRecord]) -> Dict[str, tuple[int, int, int]]:
    totals: Dict[str, int] = defaultdict(int)
    for record in records:
        if not record.excluded_reason:
            totals[record.normalized_class_name] += 1
    return {class_name: _estimate_grouped_split_counts(total) for class_name, total in totals.items()}


def _estimate_grouped_split_counts(total: int) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total < 3:
        return total, 0, 0

    val_count = max(1, round(total * 0.2))
    test_count = max(1, round(total * 0.2))
    train_count = total - val_count - test_count
    if train_count < 1:
        train_count = 1
        overflow = train_count + val_count + test_count - total
        while overflow > 0 and (val_count > 1 or test_count > 1):
            if val_count >= test_count and val_count > 1:
                val_count -= 1
            elif test_count > 1:
                test_count -= 1
            overflow -= 1
    return train_count, val_count, test_count


def _auto_merge_same_class_source_like_families(
    *,
    records: Sequence[ImageRecord],
    uf: UnionFind,
) -> None:
    grouped: Dict[tuple[str, str, str], List[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.excluded_reason:
            continue
        source_like_group = str(record.source_like_group or SOURCE_LIKE_GROUP_UNKNOWN).strip()
        stem_fingerprint = _stem_family_fingerprint(record.relative_path)
        if source_like_group == SOURCE_LIKE_GROUP_UNKNOWN or not stem_fingerprint:
            continue
        grouped[(record.normalized_class_name, source_like_group, stem_fingerprint)].append(record)
    for _, values in grouped.items():
        if len(values) < 2:
            continue
        base = values[0].relative_path
        for other in values[1:]:
            uf.union(base, other.relative_path)


def _bundle_key_for_family(
    *,
    family_id: str,
    family_records: Sequence[ImageRecord],
) -> str:
    source_style_groups = {
        record.source_style_group
        for record in family_records
        if record.source_style_group and record.source_style_group != SOURCE_STYLE_GROUP_UNKNOWN
    }
    if len(source_style_groups) == 1:
        source_style_group = next(iter(source_style_groups))
        if source_style_group.startswith(("capture:", "web:", "screenshot:", "hint:", "style:", "layout:")):
            return source_style_group

    source_like_groups = {
        record.source_like_group
        for record in family_records
        if record.source_like_group and record.source_like_group != SOURCE_LIKE_GROUP_UNKNOWN
    }
    if len(source_like_groups) == 1:
        source_like_group = next(iter(source_like_groups))
        if source_like_group.startswith(("capture:", "web:", "screenshot:", "hint:")):
            return source_like_group
    return f"family:{family_id}"


def _assign_splits_for_units(
    *,
    units: List[tuple[str, List[ImageRecord]]],
    targets: tuple[int, int, int],
) -> Dict[str, str]:
    if len(units) < 3:
        raise ValueError("Need at least 3 units for grouped split assignment.")
    desired = {
        "continual": int(targets[0]),
        "val": int(targets[1]),
        "test": int(targets[2]),
    }
    used = {"continual": 0, "val": 0, "test": 0}
    assignments: Dict[str, str] = {}
    ordered = sorted(units, key=lambda item: (-len(item[1]), item[0]))
    # Seed one unit per split to avoid empty split artifacts.
    for split_name, (unit_id, unit_records) in zip(("continual", "val", "test"), ordered[:3]):
        assignments[unit_id] = split_name
        used[split_name] += len(unit_records)
    for unit_id, unit_records in ordered[3:]:
        size = len(unit_records)
        deficits = {split_name: desired[split_name] - used[split_name] for split_name in ("continual", "val", "test")}
        preferred = sorted(
            deficits.items(),
            key=lambda item: (item[1] < 0, -item[1], used[item[0]], item[0]),
        )[0][0]
        assignments[unit_id] = preferred
        used[preferred] += size
    return assignments


def build_grouped_dataset_plan(
    *,
    class_root: Path,
    crop_name: str,
    part_name: str = "unspecified",
    artifact_root: Path,
    taxonomy_path: Optional[Path] = None,
    dino_model_id: str = DEFAULT_DINOV3_MODEL_ID,
    bioclip_model_id: str = DEFAULT_BIOCLIP_MODEL_ID,
    device: str = "cpu",
    batch_size: int = 16,
    neighbors: int = DEFAULT_NEIGHBORS,
    under_min_eval_policy: str = "block",
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    import torch

    requested_device = str(device or "cpu").strip() or "cpu"
    device = _resolve_embedding_device(requested_device)
    if device != requested_device:
        _progress(
            progress_fn,
            f"Requested embedding device '{requested_device}' is unavailable; falling back to '{device}'.",
        )
    amp_dtype = _resolve_amp_dtype(device)
    effective_batch_size = resolve_safe_embedding_batch_size(device, requested_batch_size=batch_size)
    _progress(
        progress_fn,
        f"Embedding device={device} requested_device={requested_device} batch_size={effective_batch_size}.",
    )
    normalized_under_min_eval_policy = str(under_min_eval_policy or "block").strip().lower()
    if normalized_under_min_eval_policy not in {"block", "skip"}:
        raise ValueError("under_min_eval_policy must be either 'block' or 'skip'.")

    _progress(progress_fn, "Scanning class-root dataset and computing hashes.")
    records, normalization_report = scan_class_root_dataset(
        class_root=class_root,
        crop_name=crop_name,
        taxonomy_path=taxonomy_path,
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    availability_exclusions = _refresh_record_availability(records)
    if availability_exclusions["missing_after_scan"]:
        _progress(
            progress_fn,
            f"Excluded {availability_exclusions['missing_after_scan']} image(s) that disappeared after the initial scan.",
        )
    if availability_exclusions["unreadable_after_scan"]:
        _progress(
            progress_fn,
            f"Excluded {availability_exclusions['unreadable_after_scan']} image(s) that became unreadable after the initial scan.",
        )
    valid_records = [record for record in records if not record.excluded_reason]
    source_style_group_rows = _apply_source_style_risk(valid_records)
    class_to_records: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in valid_records:
        class_to_records[record.normalized_class_name].append(record)
    records_by_class_and_path = {
        name: {record.relative_path: record for record in items} for name, items in class_to_records.items()
    }

    dino_scores: Dict[tuple[str, str], float] = {}
    bioclip_scores: Dict[tuple[str, str], float] = {}
    review_pairs: List[ReviewPair] = []
    blocking_conflicts: List[ReviewPair] = []
    class_dino_pairs_by_class: Dict[str, Dict[tuple[str, str], float]] = {}
    seen_blocking_pairs: set[tuple[str, str]] = set()
    global_exact_hash_groups = _build_hash_groups(valid_records, key_name="exact_hash")
    global_phash_groups = _build_hash_groups(valid_records, key_name="phash_hex")
    _register_cross_class_conflicts(
        groups=global_exact_hash_groups,
        conflict_reason="cross-class exact duplicate",
        blocking_conflicts=blocking_conflicts,
        seen_pairs=seen_blocking_pairs,
    )
    _register_cross_class_conflicts(
        groups=global_phash_groups,
        conflict_reason="cross-class identical perceptual hash",
        blocking_conflicts=blocking_conflicts,
        seen_pairs=seen_blocking_pairs,
    )

    uf = UnionFind()
    for record in valid_records:
        uf.find(record.relative_path)

    by_hash: Dict[tuple[str, str], List[ImageRecord]] = defaultdict(list)
    for record in valid_records:
        by_hash[(record.normalized_class_name, record.exact_hash)].append(record)
    for (_, _), group in by_hash.items():
        if len(group) < 2:
            continue
        base = group[0].relative_path
        for other in group[1:]:
            uf.union(base, other.relative_path)

    _auto_merge_same_class_source_like_families(records=valid_records, uf=uf)

    _progress(progress_fn, "Loading DINOv3 once for the full audit run.")
    dino_processor, dino_model = _load_dinov3_components(dino_model_id, device=device)
    dino_model.to(device)
    for normalized_class_name, class_records in class_to_records.items():
        _progress(progress_fn, f"Processing class '{normalized_class_name}' ({len(class_records)} readable images).")
        paths = [Path(record.absolute_path) for record in class_records]
        path_keys = [record.relative_path for record in class_records]
        if len(paths) < 2:
            continue
        _progress(progress_fn, f"Encoding DINOv3 embeddings for class '{normalized_class_name}'.")
        dino_embeddings = _encode_with_oom_retries(
            _encode_dinov3_with_components,
            paths=paths,
            label=f"DINOv3 class '{normalized_class_name}'",
            processor=dino_processor,
            model=dino_model,
            batch_size=effective_batch_size,
            device=device,
            amp_dtype=amp_dtype,
            progress_fn=progress_fn,
            batch_progress_fn=lambda done, total: _progress(
                progress_fn,
                f"DINOv3 class '{normalized_class_name}' encoded {done}/{total} images.",
            ),
        )
        class_dino_pairs = _compute_neighbor_pairs(
            dino_embeddings,
            paths=path_keys,
            neighbors=neighbors,
        )
        dino_scores.update(class_dino_pairs)
        class_dino_pairs_by_class[normalized_class_name] = class_dino_pairs

    dino_model.to("cpu")
    del dino_model
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    bioclip_candidates_by_class = {
        class_name: _select_bioclip_candidate_pairs(
            class_dino_pairs,
            records_by_class_and_path[class_name],
        )
        for class_name, class_dino_pairs in class_dino_pairs_by_class.items()
    }

    bioclip_preprocess = None
    bioclip_model = None
    if any(bioclip_candidates_by_class.values()):
        _progress(progress_fn, "Loading BioCLIP once for candidate refinement.")
        bioclip_preprocess, bioclip_model = _load_bioclip_components(bioclip_model_id, device=device)
        bioclip_model.to(device)

    for normalized_class_name, class_dino_pairs in class_dino_pairs_by_class.items():
        record_by_path = records_by_class_and_path[normalized_class_name]
        bioclip_candidate_pairs = bioclip_candidates_by_class[normalized_class_name]

        unique_bioclip_paths = sorted({path for pair in bioclip_candidate_pairs for path in pair})
        if unique_bioclip_paths and bioclip_preprocess is not None and bioclip_model is not None:
            _progress(
                progress_fn,
                f"Encoding BioCLIP refinement embeddings for class '{normalized_class_name}' ({len(unique_bioclip_paths)} images).",
            )
            bioclip_embeddings = _encode_with_oom_retries(
                _encode_bioclip_with_components,
                paths=[Path(record_by_path[path].absolute_path) for path in unique_bioclip_paths],
                label=f"BioCLIP class '{normalized_class_name}'",
                preprocess_val=bioclip_preprocess,
                model=bioclip_model,
                batch_size=effective_batch_size,
                device=device,
                amp_dtype=amp_dtype,
                progress_fn=progress_fn,
                batch_progress_fn=lambda done, total: _progress(
                    progress_fn,
                    f"BioCLIP class '{normalized_class_name}' encoded {done}/{total} images.",
                ),
            )
            bioclip_scores.update(
                _compute_pair_similarity(
                    bioclip_embeddings,
                    path_keys=unique_bioclip_paths,
                    pairs=bioclip_candidate_pairs,
                )
            )

        for path_a, path_b in sorted(class_dino_pairs.keys()):
            record_a = record_by_path[path_a]
            record_b = record_by_path[path_b]
            phash_distance = _phash_distance(record_a.phash_hex, record_b.phash_hex)
            dino_cosine = class_dino_pairs.get((path_a, path_b), float("-inf"))
            bioclip_cosine = bioclip_scores.get((path_a, path_b), float("-inf"))
            exact_match = bool(record_a.exact_hash and record_a.exact_hash == record_b.exact_hash)

            should_auto_merge = (
                exact_match
                or phash_distance <= PHASH_AUTO_MAX_DISTANCE
                or (dino_cosine >= DINO_AUTO_MIN and bioclip_cosine >= BIOCLIP_AUTO_MIN)
            )
            should_review = (
                phash_distance <= PHASH_REVIEW_MAX_DISTANCE
                or dino_cosine >= DINO_REVIEW_MIN
                or bioclip_cosine >= BIOCLIP_REVIEW_MIN
            )
            if should_auto_merge:
                uf.union(path_a, path_b)
            elif should_review:
                review_pairs.append(
                    ReviewPair(
                        pair_type="same_class_review",
                        class_a=record_a.normalized_class_name,
                        class_b=record_b.normalized_class_name,
                        path_a=path_a,
                        path_b=path_b,
                        exact_match=exact_match,
                        phash_distance=phash_distance,
                        dino_cosine=float(dino_cosine if math.isfinite(dino_cosine) else -1.0),
                        bioclip_cosine=float(bioclip_cosine if math.isfinite(bioclip_cosine) else -1.0),
                        adjacency_distance=_adjacency_distance(record_a, record_b),
                        review_rank=_review_rank(record_a, record_b),
                        decision="review",
                        reason="borderline same-class similarity; adjacency only affects review ordering",
                    )
                )

    if bioclip_model is not None:
        bioclip_model.to("cpu")
        del bioclip_model
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    record_lookup = {record.relative_path: record for record in valid_records}
    review_pairs_total = len(review_pairs)
    review_pairs, review_clusters, auto_resolved_review_clusters = _triage_review_clusters(
        review_pairs=review_pairs,
        record_lookup=record_lookup,
        uf=uf,
    )
    high_risk_review_clusters = [row for row in review_clusters if row.resolution != "auto_resolve"]
    label_risk_by_path, label_review_candidates, label_risk_summary = _build_label_risk_audit(
        records=valid_records,
        review_pairs=review_pairs,
        auto_resolved_review_clusters=auto_resolved_review_clusters,
        blocking_conflicts=blocking_conflicts,
    )

    review_pairs = sorted(
        review_pairs,
        key=lambda item: (
            item.class_a,
            item.review_rank,
            -max(float(item.dino_cosine), float(item.bioclip_cosine)),
            item.phash_distance,
            item.path_a,
            item.path_b,
        ),
    )

    families: Dict[tuple[str, str], List[ImageRecord]] = defaultdict(list)
    for record in valid_records:
        root_id = uf.find(record.relative_path)
        families[(record.normalized_class_name, root_id)].append(record)

    review_excluded_paths = {
        str(path) for pair in review_pairs for path in (pair.path_a, pair.path_b) if str(path).strip()
    }
    family_details: Dict[tuple[str, str], Dict[str, Any]] = {}
    for family_key, items in families.items():
        ordered_items = sorted(items, key=lambda record: record.relative_path)
        canonical_record = _select_canonical_record(ordered_items)
        canonical_label_risk = label_risk_by_path.get(
            canonical_record.relative_path, _clear_label_risk(canonical_record)
        )
        review_excluded = any(item.relative_path in review_excluded_paths for item in ordered_items)
        family_has_synthetic = any(item.synthetic_hint for item in ordered_items)
        family_has_eval_risk = any(item.eval_quality_risk for item in ordered_items)
        family_has_source_style_risk = any(item.source_style_risk for item in ordered_items)
        family_label_risk_levels = sorted(
            {
                label_risk_by_path.get(item.relative_path, _clear_label_risk(item)).label_risk_level
                for item in ordered_items
            },
            key=_risk_rank,
            reverse=True,
        )
        bundle_key = _bundle_key_for_family(
            family_id=str(family_key[1]),
            family_records=ordered_items,
        )
        if review_excluded:
            family_role = "review_excluded"
        elif canonical_record.source_style_risk:
            family_role = "source_style_risk_family"
        elif canonical_record.synthetic_hint:
            family_role = "synthetic_family"
        elif canonical_record.eval_quality_risk:
            family_role = "eval_risk_family"
        elif family_has_synthetic:
            family_role = "canonical_with_synthetic_derivatives"
        elif family_has_eval_risk:
            family_role = "canonical_with_eval_risk_derivatives"
        elif canonical_label_risk.label_risk_level != LABEL_RISK_CLEAR:
            family_role = "label_risk_family"
        elif len(ordered_items) > 1:
            family_role = "canonical_with_derivatives"
        else:
            family_role = "independent_singleton"
        family_details[family_key] = {
            "canonical_record": canonical_record,
            "canonical_relative_path": canonical_record.relative_path,
            "eval_eligible": (
                (not review_excluded)
                and (not canonical_record.synthetic_hint)
                and (not canonical_record.eval_quality_risk)
                and (not canonical_record.source_style_risk)
                and canonical_label_risk.label_risk_level == LABEL_RISK_CLEAR
            ),
            "family_role": family_role,
            "family_size": len(ordered_items),
            "review_excluded": review_excluded,
            "family_has_synthetic": family_has_synthetic,
            "family_has_eval_risk": family_has_eval_risk,
            "family_has_source_style_risk": family_has_source_style_risk,
            "family_label_risk_levels": "|".join(family_label_risk_levels),
            "bundle_key": bundle_key,
        }

    family_assignments: Dict[str, str] = {}
    class_health: Dict[str, Any] = {}
    blocking_issues: List[str] = []
    skipped_classes: List[Dict[str, Any]] = []
    skipped_class_names: set[str] = set()
    included_class_names: set[str] = set()
    for class_name in sorted({record.normalized_class_name for record in valid_records}):
        class_families = sorted(
            [
                (family_root, items, family_details[(family_class, family_root)])
                for (family_class, family_root), items in families.items()
                if family_class == class_name
            ],
            key=lambda item: item[0],
        )
        eval_candidate_families = [
            (family_root, [detail["canonical_record"]])
            for family_root, _items, detail in class_families
            if detail["eval_eligible"]
        ]
        target = _estimate_grouped_split_counts(len(eval_candidate_families))
        health_entry = {
            "image_count": sum(len(items) for _, items, _detail in class_families),
            "family_count": len(class_families),
            "eval_eligible_family_count": len(eval_candidate_families),
            "continual_only_family_count": len(class_families) - len(eval_candidate_families),
            "targets": {"continual": target[0], "val": target[1], "test": target[2]},
            "synthetic_hint_count": sum(
                1 for _, items, _detail in class_families for item in items if item.synthetic_hint
            ),
            "eval_risk_count": sum(
                1 for _, items, _detail in class_families for item in items if item.eval_quality_risk
            ),
            "source_style_risk_count": sum(
                1 for _, items, _detail in class_families for item in items if item.source_style_risk
            ),
            "label_risk_count": sum(
                1
                for _, items, _detail in class_families
                for item in items
                if label_risk_by_path.get(item.relative_path, _clear_label_risk(item)).label_risk_level
                != LABEL_RISK_CLEAR
            ),
            "review_excluded_family_count": sum(
                1 for _family_root, _items, detail in class_families if detail["review_excluded"]
            ),
            "source_like_bundle_count": len(
                {
                    str(detail["bundle_key"])
                    for _family_root, _items, detail in class_families
                    if str(detail.get("bundle_key", "")).strip()
                }
            ),
            "source_style_bundle_count": len(
                {
                    str(detail["bundle_key"])
                    for _family_root, _items, detail in class_families
                    if str(detail.get("bundle_key", "")).startswith(("style:", "layout:"))
                }
            ),
        }
        class_health[class_name] = health_entry
        if len(eval_candidate_families) == 0:
            skip_reason = "no evaluation-eligible families after grouped prep"
            health_entry["runtime_action"] = "skipped"
            health_entry["skip_reason"] = skip_reason
            skipped_class_names.add(class_name)
            skipped_classes.append(_skipped_class_entry(class_name, skip_reason, health_entry))
            continue
        if len(eval_candidate_families) < 3:
            shortage_reason = (
                f"Class '{class_name}' has only {len(eval_candidate_families)} "
                "evaluation-eligible family/families after grouped prep."
            )
            if normalized_under_min_eval_policy == "skip":
                health_entry["runtime_action"] = "skipped"
                health_entry["skip_reason"] = shortage_reason
                skipped_class_names.add(class_name)
                skipped_classes.append(_skipped_class_entry(class_name, shortage_reason, health_entry))
            else:
                health_entry["runtime_action"] = "blocked"
                blocking_issues.append(shortage_reason)
            continue
        bundled_eval_families: Dict[str, List[ImageRecord]] = defaultdict(list)
        bundle_to_family_ids: Dict[str, List[str]] = defaultdict(list)
        for family_root, _items, detail in class_families:
            if not detail["eval_eligible"]:
                continue
            bundle_key = str(detail.get("bundle_key") or f"family:{family_root}")
            bundled_eval_families[bundle_key].append(detail["canonical_record"])
            bundle_to_family_ids[bundle_key].append(family_root)
        if len(bundled_eval_families) < 3:
            shortage_reason = (
                f"Class '{class_name}' has only {len(bundled_eval_families)} "
                "source-like eval bundle(s) after grouped prep."
            )
            if normalized_under_min_eval_policy == "skip":
                health_entry["runtime_action"] = "skipped"
                health_entry["skip_reason"] = shortage_reason
                skipped_class_names.add(class_name)
                skipped_classes.append(_skipped_class_entry(class_name, shortage_reason, health_entry))
            else:
                health_entry["runtime_action"] = "blocked"
                blocking_issues.append(shortage_reason)
            continue
        health_entry["runtime_action"] = "included"
        included_class_names.add(class_name)
        assignments = _assign_splits_for_units(
            units=[(bundle_key, records) for bundle_key, records in bundled_eval_families.items()],
            targets=target,
        )
        for bundle_key, split_name in assignments.items():
            for family_root in bundle_to_family_ids.get(bundle_key, []):
                family_assignments[family_root] = split_name

    if skipped_classes and not included_class_names and not blocking_issues:
        if normalized_under_min_eval_policy == "skip":
            blocking_issues.append("No classes remain after applying under-minimum evaluation skip policy.")
        else:
            blocking_issues.append("No classes remain after skipping zero evaluation-eligible classes.")

    manifest_rows: List[Dict[str, Any]] = []
    for (class_name, family_root), items in sorted(families.items(), key=lambda item: (item[0][0], item[0][1])):
        family_detail = family_details[(class_name, family_root)]
        family_id = f"{class_name}__{family_root[:12]}"
        runtime_skipped = class_name in skipped_class_names
        assigned_split = "skipped" if runtime_skipped else family_assignments.get(family_root, "continual")
        canonical_relative_path = str(family_detail["canonical_relative_path"])
        for item in sorted(items, key=lambda record: record.relative_path):
            label_risk = label_risk_by_path.get(item.relative_path, _clear_label_risk(item))
            is_family_canonical = bool(item.relative_path == canonical_relative_path)
            split_name = (
                "skipped"
                if runtime_skipped
                else assigned_split
                if bool(family_detail["eval_eligible"]) and is_family_canonical
                else "continual"
            )
            train_only_reasons: List[str] = []
            if item.synthetic_hint:
                train_only_reasons.append("synthetic_hint")
            if item.eval_quality_risk:
                train_only_reasons.append("eval_quality_risk")
            if item.source_style_risk:
                train_only_reasons.append("source_style_risk")
            if label_risk.label_risk_level in {LABEL_RISK_TRAIN_ONLY, LABEL_RISK_REVIEW}:
                train_only_reasons.append(f"label_{label_risk.label_risk_level}")
            train_only_routed = bool(split_name == "continual" and train_only_reasons)
            label_risk.train_only_routed = bool(train_only_routed)
            manifest_rows.append(
                {
                    "relative_path": item.relative_path,
                    "raw_class_name": item.raw_class_name,
                    "normalized_class_name": item.normalized_class_name,
                    "source_hint": item.source_hint,
                    "source_like_group": item.source_like_group,
                    "source_style_group": item.source_style_group,
                    "source_style_risk": item.source_style_risk,
                    "source_style_reason": item.source_style_reason,
                    "synthetic_hint": item.synthetic_hint,
                    "eval_quality_risk": item.eval_quality_risk,
                    "width": item.width,
                    "height": item.height,
                    "aspect_ratio_bucket": item.aspect_ratio_bucket,
                    "resolution_bucket": item.resolution_bucket,
                    "border_layout": item.border_layout,
                    "compression_style": item.compression_style,
                    "blur_score": round(item.blur_score, 4),
                    "brightness_mean": round(item.brightness_mean, 4),
                    "exact_hash": item.exact_hash,
                    "phash_hex": item.phash_hex,
                    "family_id": family_id,
                    "family_size": int(family_detail["family_size"]),
                    "family_role": str(family_detail["family_role"]),
                    "family_eval_eligible": bool(family_detail["eval_eligible"]),
                    "family_bundle_key": str(family_detail.get("bundle_key") or ""),
                    "family_canonical_relative_path": canonical_relative_path,
                    "is_family_canonical": is_family_canonical,
                    "label_risk_level": label_risk.label_risk_level,
                    "label_risk_reason": label_risk.label_risk_reason,
                    "label_risk_score": round(float(label_risk.label_risk_score), 4),
                    "train_only_routed": train_only_routed,
                    "train_only_route_reason": "|".join(train_only_reasons),
                    "canonical_eval_safe": bool(
                        is_family_canonical
                        and not item.synthetic_hint
                        and not item.eval_quality_risk
                        and not item.source_style_risk
                        and label_risk.label_risk_level == LABEL_RISK_CLEAR
                    ),
                    "family_assignment": assigned_split,
                    "runtime_skipped": bool(runtime_skipped),
                    "split": split_name,
                }
            )

    summary = {
        "schema_version": "v1_grouped_data_prep",
        "crop_name": str(crop_name),
        "part_name": str(part_name),
        "under_min_eval_policy": normalized_under_min_eval_policy,
        "requested_device": requested_device,
        "embedding_device": device,
        "source_root": str(class_root.resolve()),
        "summary": {
            "total_images": len(records),
            "readable_images": len(valid_records),
            "excluded_images": len([record for record in records if record.excluded_reason]),
            "same_class_review_pairs": len(review_pairs),
            "same_class_review_pairs_total": review_pairs_total,
            "same_class_review_clusters_total": len(review_clusters),
            "same_class_auto_resolved_clusters": len(auto_resolved_review_clusters),
            "same_class_high_risk_clusters": len(high_risk_review_clusters),
            "cross_class_conflicts": len(blocking_conflicts),
            "blocking_issues": len(blocking_issues),
            "skipped_classes": len(skipped_classes),
            "materialized_classes": len(included_class_names),
            "skipped_images": sum(
                len(items)
                for (class_name, _family_root), items in families.items()
                if class_name in skipped_class_names
            ),
            "eval_eligible_families": sum(1 for detail in family_details.values() if detail["eval_eligible"]),
            "continual_only_families": sum(1 for detail in family_details.values() if not detail["eval_eligible"]),
            "eval_risk_images": sum(1 for record in valid_records if record.eval_quality_risk),
            "source_style_risk_images": sum(1 for record in valid_records if record.source_style_risk),
            "source_style_risk_groups": sum(1 for row in source_style_group_rows if row.get("source_style_risk")),
            "label_review_candidates": len(label_review_candidates),
            "label_train_only_risk_images": label_risk_summary["level_counts"][LABEL_RISK_TRAIN_ONLY],
            "label_blocking_conflict_images": label_risk_summary["level_counts"][LABEL_RISK_BLOCKING],
            "train_only_routed_images": sum(1 for row in manifest_rows if row.get("train_only_routed")),
            "review_excluded_paths": len(review_excluded_paths),
            "excluded_reason_breakdown": {
                "unreadable": len([record for record in records if record.excluded_reason == "unreadable"]),
                "unhashable": len([record for record in records if record.excluded_reason == "unhashable"]),
                "missing_after_scan": availability_exclusions["missing_after_scan"],
                "unreadable_after_scan": availability_exclusions["unreadable_after_scan"],
            },
            "adjacency_used_for_review_ranking_only": True,
        },
        "normalization_report": normalization_report,
        "class_health": class_health,
        "blocking_issues": blocking_issues,
        "skipped_classes": skipped_classes,
        "runtime_ready": not blocking_issues and not blocking_conflicts,
        "prepared_runtime_root": str(
            (DEFAULT_RUNTIME_ROOT / build_prepared_dataset_key(crop_name, part_name)).resolve()
        ),
        "ood_handoff_checklist": {
            "status": "pending",
            "message": "Prepare a separate runtime_dataset/<dataset_key>/ood tree after ID-side prep completes.",
        },
    }
    summary["human_review_packet_path"] = HUMAN_REVIEW_PACKET_FILENAME
    summary["label_review_summary_path"] = LABEL_REVIEW_SUMMARY_FILENAME
    label_risk_summary["train_only_routed_count"] = sum(1 for row in manifest_rows if row.get("train_only_routed"))
    label_risk_summary["review_queue_path"] = "label_review_candidates.csv"

    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "prep_summary.json", summary, ensure_ascii=False)
    write_json(artifact_root / "label_normalization_report.json", normalization_report, ensure_ascii=False)
    write_json(artifact_root / "label_risk_summary.json", label_risk_summary, ensure_ascii=False)
    write_json(artifact_root / "class_health_report.json", class_health, ensure_ascii=False)
    write_json(
        artifact_root / "proposed_split_manifest.json",
        {
            "schema_version": "v1_grouped_split_manifest",
            "crop_name": str(crop_name),
            "part_name": str(part_name),
            "dataset_key": build_prepared_dataset_key(crop_name, part_name),
            "source_root": str(class_root.resolve()),
            "blocking_issues": list(blocking_issues),
            "skipped_classes": list(skipped_classes),
            "runtime_ready": bool(summary["runtime_ready"]),
            "rows": manifest_rows,
        },
        ensure_ascii=False,
    )
    write_json(
        artifact_root / "ood_handoff_checklist.json",
        summary["ood_handoff_checklist"],
        ensure_ascii=False,
    )
    write_csv_rows(artifact_root / "dataset_manifest.csv", [asdict(record) for record in records])
    write_csv_rows(artifact_root / "family_manifest.csv", manifest_rows)
    write_csv_rows(artifact_root / "source_style_groups.csv", source_style_group_rows)
    write_csv_rows(artifact_root / "label_review_candidates.csv", label_review_candidates)
    write_csv_rows(artifact_root / "same_class_review_candidates.csv", [asdict(pair) for pair in review_pairs])
    write_csv_rows(artifact_root / "same_class_review_clusters.csv", [asdict(cluster) for cluster in review_clusters])
    write_csv_rows(
        artifact_root / "same_class_auto_resolved_clusters.csv",
        [asdict(cluster) for cluster in auto_resolved_review_clusters],
    )
    write_csv_rows(
        artifact_root / "same_class_high_risk_clusters.csv",
        [asdict(cluster) for cluster in high_risk_review_clusters],
    )
    write_csv_rows(artifact_root / "cross_class_conflicts.csv", [asdict(pair) for pair in blocking_conflicts])
    write_csv_rows(
        artifact_root / "exact_duplicates.csv",
        [
            {
                "normalized_class_name": key[0],
                "exact_hash": key[1],
                "relative_paths": "|".join(sorted(record.relative_path for record in values)),
                "count": len(values),
            }
            for key, values in sorted(by_hash.items())
            if len(values) > 1
        ],
    )
    human_review_packet = build_human_review_packet(summary, artifact_root=artifact_root)
    write_json(artifact_root / HUMAN_REVIEW_PACKET_FILENAME, human_review_packet, ensure_ascii=False)
    label_review_summary = build_label_review_summary(
        summary,
        label_risk_summary=label_risk_summary,
        human_review_packet=human_review_packet,
        label_review_candidates=label_review_candidates,
    )
    write_json(artifact_root / LABEL_REVIEW_SUMMARY_FILENAME, label_review_summary, ensure_ascii=False)
    refresh_prep_guided_artifacts(
        artifact_root,
        overview_updates={
            "crop_name": str(crop_name),
            "part_name": str(part_name),
            "runtime_ready": bool(summary["runtime_ready"]),
            "prepared_runtime_root": str(
                (DEFAULT_RUNTIME_ROOT / build_prepared_dataset_key(crop_name, part_name)).resolve()
            ),
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a grouped runtime dataset with duplicate-aware audit.")
    parser.add_argument("--root", type=Path, required=True, help="Flat class-root dataset.")
    parser.add_argument("--crop", type=str, required=True, help="Crop name.")
    parser.add_argument(
        "--part", type=str, default="unspecified", help="Part name used for prepared runtime dataset naming."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help=f"Artifact output root (default: {DEFAULT_ARTIFACT_ROOT})",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=Path,
        default=Path("config") / "plant_taxonomy.json",
        help="Taxonomy path used for class normalization.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Embedding device (cpu or cuda).")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS, help="Neighbors per image.")
    parser.add_argument(
        "--under-min-eval-policy",
        type=str,
        default="block",
        choices=("block", "skip"),
        help="Whether classes with <3 eval families/source bundles should block or be skipped.",
    )
    parser.add_argument("--materialize", action="store_true", help="Materialize runtime dataset if no blockers.")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=DEFAULT_RUNTIME_ROOT,
        help=f"Prepared runtime dataset root (default: {DEFAULT_RUNTIME_ROOT})",
    )
    parser.add_argument(
        "--ood-root",
        type=Path,
        default=None,
        help="Optional repo-local or explicit OOD tree to materialize into runtime_dataset/<dataset_key>/ood.",
    )
    parser.add_argument(
        "--oe-root",
        type=Path,
        default=None,
        help="Optional OE tree to materialize into runtime_dataset/<dataset_key>/oe.",
    )
    args = parser.parse_args()

    summary = build_grouped_dataset_plan(
        class_root=args.root,
        crop_name=args.crop,
        part_name=args.part,
        artifact_root=args.artifact_root,
        taxonomy_path=args.taxonomy_path,
        device=args.device,
        batch_size=args.batch_size,
        neighbors=args.neighbors,
        under_min_eval_policy=args.under_min_eval_policy,
    )
    print(json.dumps(summary, indent=2))
    if args.materialize:
        materialize_grouped_runtime_dataset(
            class_root=args.root,
            crop_name=args.crop,
            part_name=args.part,
            artifact_root=args.artifact_root,
            runtime_root=args.runtime_root,
            ood_root=args.ood_root,
            oe_root=args.oe_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
