"""Visual prototype-bank builder for router handoff evidence."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from src.data.dataset_release_runtime import DEFAULT_MATERIALIZED_DATASET_ROOT
from src.router.taxonomy_registry import SUPPORTED_SPLITS, artifact_sha256, now_utc_timestamp, split_target_id
from src.shared.json_utils import ensure_parent, write_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_BACKEND = "image_stats_v1"
BIOCLIP_OPEN_CLIP_BACKEND = "bioclip_open_clip"
DEFAULT_BIOCLIP_MODEL_ID = "imageomics/bioclip-2.5-vith14"


@dataclass(frozen=True)
class ImageFeatureRecord:
    path: Path
    target_id: str
    split: str
    class_label: str
    vector: tuple[float, ...]
    sha256: str
    source_kind: str = "dataset"


@dataclass(frozen=True)
class HardNegativeFeatureRecord:
    path: Path
    negative_for_target_id: str
    source_expected_target: str
    class_label: str
    vector: tuple[float, ...]
    sha256: str
    source_kind: str = "curated_hard_negative"


def iter_dataset_images(
    dataset_root: Path,
    *,
    splits: Iterable[str] = SUPPORTED_SPLITS,
    include_ood: bool = False,
    max_images_per_class: int | None = None,
) -> Iterable[tuple[str, str, str, Path]]:
    if not dataset_root.exists():
        return []

    split_names = tuple(splits)
    for target_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir() and "__" in path.name):
        split_target_id(target_dir.name)
        for split_name in split_names:
            if not include_ood and split_name.lower() in {"ood", "oe"}:
                continue
            split_dir = target_dir / split_name
            if not split_dir.exists():
                continue
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                emitted = 0
                for image_path in sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
                    yield target_dir.name, split_name, class_dir.name, image_path
                    emitted += 1
                    if max_images_per_class is not None and emitted >= max_images_per_class:
                        break


def image_stats_vector(image_path: Path) -> tuple[float, ...]:
    try:
        from PIL import Image, ImageStat
    except ImportError as exc:  # pragma: no cover - exercised only in runtimes without Pillow
        raise RuntimeError("Pillow is required to build image_stats_v1 router prototypes") from exc

    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((64, 64))
        stat = ImageStat.Stat(rgb)
        width, height = image.size
        means = [float(value) / 255.0 for value in stat.mean]
        stddev = [float(value) / 255.0 for value in stat.stddev]
        hist = rgb.histogram()
        channel_features: list[float] = []
        for channel in range(3):
            values = hist[channel * 256 : (channel + 1) * 256]
            bins = [sum(values[index : index + 16]) / float(64 * 64) for index in range(0, 256, 16)]
            channel_features.extend(bins)
        aspect_ratio = float(width) / float(height) if height else 0.0
    return tuple(round(value, 8) for value in (*means, *stddev, aspect_ratio, *channel_features))


@lru_cache(maxsize=2)
def _load_open_clip_encoder(model_id: str, device: str) -> tuple[Any, Any, Any]:
    try:
        import open_clip
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional Colab/runtime packages
        raise RuntimeError("open_clip and torch are required for bioclip_open_clip prototypes") from exc

    hub_model_id = str(model_id)
    if "/" in hub_model_id and not hub_model_id.startswith("hf-hub:"):
        hub_model_id = f"hf-hub:{hub_model_id}"
    model, _, preprocess_val = open_clip.create_model_and_transforms(hub_model_id)
    model = model.to(device)
    model.eval()
    return torch, model, preprocess_val


def open_clip_image_vector(
    image_path: Path,
    *,
    model_id: str = DEFAULT_BIOCLIP_MODEL_ID,
    device: str = "cpu",
) -> tuple[float, ...]:
    torch, model, preprocess_val = _load_open_clip_encoder(model_id, device)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a runtime dependency
        raise RuntimeError("Pillow is required to build bioclip_open_clip router prototypes") from exc

    with Image.open(image_path) as image:
        image_tensor = preprocess_val(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        embedding = model.encode_image(image_tensor)
        embedding = torch.nn.functional.normalize(embedding, dim=-1).to(dtype=torch.float32)
    return tuple(round(float(value), 8) for value in embedding.detach().cpu().numpy()[0].tolist())


def _image_vector_cache_key(
    image_path: Path,
    *,
    embedding_backend: str,
    embedding_model_id: str,
    device: str,
) -> tuple[str, int, int, str, str, str] | None:
    try:
        resolved = image_path.resolve()
        stat = resolved.stat()
    except OSError:
        return None
    return (
        str(resolved),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        str(embedding_backend),
        str(embedding_model_id),
        str(device),
    )


@lru_cache(maxsize=2048)
def _cached_image_vector(
    image_path: str,
    _mtime_ns: int,
    _size: int,
    embedding_backend: str,
    embedding_model_id: str,
    device: str,
) -> tuple[float, ...]:
    path = Path(image_path)
    if embedding_backend == DEFAULT_BACKEND:
        return image_stats_vector(path)
    if embedding_backend == BIOCLIP_OPEN_CLIP_BACKEND:
        return open_clip_image_vector(path, model_id=embedding_model_id, device=device)
    raise ValueError(f"Unsupported prototype backend {embedding_backend!r}")


def image_vector(
    image_path: Path,
    *,
    embedding_backend: str = DEFAULT_BACKEND,
    embedding_model_id: str = DEFAULT_BIOCLIP_MODEL_ID,
    device: str = "cpu",
) -> tuple[float, ...]:
    cache_key = _image_vector_cache_key(
        image_path,
        embedding_backend=embedding_backend,
        embedding_model_id=embedding_model_id,
        device=device,
    )
    if cache_key is not None:
        return _cached_image_vector(*cache_key)
    if embedding_backend == DEFAULT_BACKEND:
        return image_stats_vector(image_path)
    if embedding_backend == BIOCLIP_OPEN_CLIP_BACKEND:
        return open_clip_image_vector(image_path, model_id=embedding_model_id, device=device)
    raise ValueError(f"Unsupported prototype backend {embedding_backend!r}")


def centroid(vectors: Iterable[tuple[float, ...]]) -> tuple[float, ...]:
    materialized = list(vectors)
    if not materialized:
        return ()
    width = len(materialized[0])
    totals = [0.0] * width
    for vector in materialized:
        if len(vector) != width:
            raise ValueError("All vectors must share the same width")
        for index, value in enumerate(vector):
            totals[index] += float(value)
    return tuple(round(total / len(materialized), 8) for total in totals)


def euclidean_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        raise ValueError("Vector widths differ")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def dispersion(vectors: Iterable[tuple[float, ...]], center: tuple[float, ...]) -> float:
    materialized = list(vectors)
    if not materialized or not center:
        return 0.0
    return round(sum(euclidean_distance(vector, center) for vector in materialized) / len(materialized), 8)


def _summarize_records(records: list[ImageFeatureRecord]) -> dict[str, Any]:
    by_target: dict[str, list[ImageFeatureRecord]] = defaultdict(list)
    by_class: dict[tuple[str, str], list[ImageFeatureRecord]] = defaultdict(list)
    for record in records:
        by_target[record.target_id].append(record)
        by_class[(record.target_id, record.class_label)].append(record)

    target_prototypes: dict[str, Any] = {}
    for target_id, target_records in sorted(by_target.items()):
        crop, part = split_target_id(target_id)
        target_vectors = [record.vector for record in target_records]
        target_center = centroid(target_vectors)
        split_counts: dict[str, int] = defaultdict(int)
        for record in target_records:
            split_counts[record.split] += 1
        class_labels = sorted({record.class_label for record in target_records})
        target_prototypes[target_id] = {
            "target_id": target_id,
            "crop": crop,
            "part": part,
            "sample_count": len(target_records),
            "split_counts": dict(sorted(split_counts.items())),
            "class_labels": class_labels,
            "centroid": list(target_center),
            "dispersion": dispersion(target_vectors, target_center),
        }

    class_prototypes: dict[str, Any] = {}
    for (target_id, class_label), class_records in sorted(by_class.items()):
        class_vectors = [record.vector for record in class_records]
        class_center = centroid(class_vectors)
        key = f"{target_id}::{class_label}"
        split_counts: dict[str, int] = defaultdict(int)
        for record in class_records:
            split_counts[record.split] += 1
        class_prototypes[key] = {
            "target_id": target_id,
            "class_label": class_label,
            "sample_count": len(class_records),
            "split_counts": dict(sorted(split_counts.items())),
            "centroid": list(class_center),
            "dispersion": dispersion(class_vectors, class_center),
        }

    return {
        "target_prototypes": target_prototypes,
        "class_prototypes": class_prototypes,
    }


def _summarize_hard_negatives(records: list[HardNegativeFeatureRecord]) -> dict[str, Any]:
    by_target: dict[str, list[HardNegativeFeatureRecord]] = defaultdict(list)
    for record in records:
        by_target[record.negative_for_target_id].append(record)

    hard_negative_prototypes: dict[str, Any] = {}
    for target_id, target_records in sorted(by_target.items()):
        vectors = [record.vector for record in target_records]
        center = centroid(vectors)
        source_expected_targets = sorted({record.source_expected_target for record in target_records if record.source_expected_target})
        hard_negative_prototypes[target_id] = {
            "negative_for_target_id": target_id,
            "sample_count": len(target_records),
            "source_expected_targets": source_expected_targets,
            "class_labels": sorted({record.class_label for record in target_records if record.class_label}),
            "centroid": list(center),
            "dispersion": dispersion(vectors, center),
        }
    return {"hard_negative_prototypes": hard_negative_prototypes}


def iter_curation_manifest_rows(curation_root: Path) -> Iterable[dict[str, str]]:
    if not curation_root:
        return []
    for filename, role in (
        ("prototype_positive_manifest.csv", "prototype_positive"),
        ("prototype_hard_negative_manifest.csv", "prototype_hard_negative"),
    ):
        manifest_path = curation_root / filename
        if not manifest_path.is_file():
            continue
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                materialized = dict(row)
                materialized["_curation_role"] = role
                materialized["_manifest_path"] = str(manifest_path)
                yield materialized


def _required_curation_manifest_paths(curation_root: Path) -> tuple[Path, Path]:
    return (
        curation_root / "prototype_positive_manifest.csv",
        curation_root / "prototype_hard_negative_manifest.csv",
    )


def resolve_curation_image_path(row: dict[str, Any], *, repo_root: Path) -> Path | None:
    for key in ("resolved_image", "source"):
        value = str(row.get(key) or "").strip()
        if not value:
            continue
        if value.startswith("staged_external:"):
            value = value.split(":", 1)[1]
        normalized = value.replace("\\", "/")
        candidate = Path(normalized)
        if candidate.is_absolute() and candidate.is_file():
            return candidate
        marker = "bitirmeprojesi/"
        if marker in normalized:
            value = normalized.split(marker, 1)[1]
        resolved = repo_root / value
        if resolved.is_file():
            return resolved
    return None


def _curation_class_label(row: dict[str, Any]) -> str:
    for key in ("corrected_class", "expected_class", "prototype_class_label"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _negative_for_target_id(row: dict[str, Any]) -> str:
    expected_target = str(row.get("expected_target") or "").strip()
    prototype_target = str(row.get("prototype_target") or "").strip()
    if prototype_target and prototype_target != expected_target:
        return prototype_target
    return ""


def load_curation_feature_records(
    *,
    curation_root: Path | None,
    repo_root: Path,
    embedding_backend: str,
    embedding_model_id: str,
    device: str,
) -> tuple[list[ImageFeatureRecord], list[HardNegativeFeatureRecord], list[dict[str, str]]]:
    if not curation_root:
        return [], [], []
    missing_manifests = [
        str(path) for path in _required_curation_manifest_paths(curation_root) if not path.is_file()
    ]
    if missing_manifests:
        raise FileNotFoundError(
            "Curation root was provided but required prototype curation manifest(s) are missing: "
            + ", ".join(missing_manifests)
        )

    manifest_rows = list(iter_curation_manifest_rows(curation_root))
    if not manifest_rows:
        raise ValueError(f"Curation root {curation_root} contains no prototype curation rows")

    positive_records: list[ImageFeatureRecord] = []
    hard_negative_records: list[HardNegativeFeatureRecord] = []
    skipped: list[dict[str, str]] = []
    for row in manifest_rows:
        role = str(row.get("_curation_role") or "")
        image_path = resolve_curation_image_path(row, repo_root=repo_root)
        if image_path is None:
            skipped.append({"image_id": str(row.get("image_id") or ""), "role": role, "reason": "image_missing"})
            continue
        try:
            vector = image_vector(
                image_path,
                embedding_backend=embedding_backend,
                embedding_model_id=embedding_model_id,
                device=device,
            )
            digest = artifact_sha256(image_path)
        except Exception as exc:
            skipped.append(
                {
                    "image_id": str(row.get("image_id") or ""),
                    "role": role,
                    "path": str(image_path),
                    "reason": str(exc),
                }
            )
            continue

        if role == "prototype_positive":
            target_id = str(row.get("expected_target") or "").strip()
            class_label = _curation_class_label(row)
            if not target_id or "__" not in target_id or not class_label:
                skipped.append(
                    {
                        "image_id": str(row.get("image_id") or ""),
                        "role": role,
                        "reason": "missing_target_or_class",
                    }
                )
                continue
            positive_records.append(
                ImageFeatureRecord(
                    path=image_path,
                    target_id=target_id,
                    split="curated",
                    class_label=class_label,
                    vector=vector,
                    sha256=digest,
                    source_kind="curated_positive",
                )
            )
        elif role == "prototype_hard_negative":
            negative_for_target_id = _negative_for_target_id(row)
            if not negative_for_target_id or "__" not in negative_for_target_id:
                reason = (
                    "same_target_hard_negative_not_used"
                    if row.get("prototype_target")
                    else "missing_negative_target"
                )
                skipped.append({"image_id": str(row.get("image_id") or ""), "role": role, "reason": reason})
                continue
            hard_negative_records.append(
                HardNegativeFeatureRecord(
                    path=image_path,
                    negative_for_target_id=negative_for_target_id,
                    source_expected_target=str(row.get("expected_target") or "").strip(),
                    class_label=_curation_class_label(row),
                    vector=vector,
                    sha256=digest,
                )
            )
    if not positive_records and not hard_negative_records:
        skipped_reasons = Counter(str(row.get("reason") or "unknown") for row in skipped)
        reason_summary = ", ".join(f"{reason}={count}" for reason, count in sorted(skipped_reasons.items()))
        raise ValueError(
            f"Curation root {curation_root} loaded zero usable prototype curation rows "
            f"from {len(manifest_rows)} manifest row(s); skipped reasons: {reason_summary or 'none'}"
        )
    return positive_records, hard_negative_records, skipped


def build_prototype_bank(
    *,
    dataset_root: Path = DEFAULT_MATERIALIZED_DATASET_ROOT,
    curation_root: Path | None = None,
    repo_root: Path = Path("."),
    embedding_backend: str = DEFAULT_BACKEND,
    embedding_model_id: str = DEFAULT_BIOCLIP_MODEL_ID,
    device: str = "cpu",
    splits: Iterable[str] = SUPPORTED_SPLITS,
    include_ood: bool = False,
    max_images_per_class: int | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if embedding_backend not in {DEFAULT_BACKEND, BIOCLIP_OPEN_CLIP_BACKEND}:
        raise ValueError(
            f"Unsupported prototype backend {embedding_backend!r}; "
            f"supported backends are {DEFAULT_BACKEND!r} and {BIOCLIP_OPEN_CLIP_BACKEND!r}"
        )

    records: list[ImageFeatureRecord] = []
    hard_negative_records: list[HardNegativeFeatureRecord] = []
    skipped: list[dict[str, str]] = []
    for target_id, split_name, class_label, image_path in iter_dataset_images(
        dataset_root,
        splits=splits,
        include_ood=include_ood,
        max_images_per_class=max_images_per_class,
    ):
        try:
            vector = image_vector(
                image_path,
                embedding_backend=embedding_backend,
                embedding_model_id=embedding_model_id,
                device=device,
            )
            digest = artifact_sha256(image_path)
        except Exception as exc:
            skipped.append({"path": str(image_path), "reason": str(exc)})
            continue
        records.append(
            ImageFeatureRecord(
                path=image_path,
                target_id=target_id,
                split=split_name,
                class_label=class_label,
                vector=vector,
                sha256=digest,
            )
        )

    curation_positive_records, curation_hard_negative_records, curation_skipped = load_curation_feature_records(
        curation_root=curation_root,
        repo_root=repo_root,
        embedding_backend=embedding_backend,
        embedding_model_id=embedding_model_id,
        device=device,
    )
    records.extend(curation_positive_records)
    hard_negative_records.extend(curation_hard_negative_records)
    skipped.extend(curation_skipped)

    summaries = _summarize_records(records)
    hard_negative_summaries = _summarize_hard_negatives(hard_negative_records)
    payload = {
        "schema_version": "router_prototype_bank.v1",
        "created_at": created_at or now_utc_timestamp(),
        "embedding_backend": embedding_backend,
        "source_roots": {
            "dataset_root": str(dataset_root),
            "curation_root": str(curation_root) if curation_root else None,
            "splits": list(splits),
            "include_ood": include_ood,
            "max_images_per_class": max_images_per_class,
            "embedding_model_id": embedding_model_id if embedding_backend == BIOCLIP_OPEN_CLIP_BACKEND else None,
            "embedding_device": device if embedding_backend == BIOCLIP_OPEN_CLIP_BACKEND else None,
        },
        "summary": {
            "target_count": len(summaries["target_prototypes"]),
            "class_prototype_count": len(summaries["class_prototypes"]),
            "sample_count": len(records),
            "dataset_sample_count": len(records) - len(curation_positive_records),
            "curation_positive_count": len(curation_positive_records),
            "hard_negative_count": len(hard_negative_records),
            "hard_negative_target_count": len(hard_negative_summaries["hard_negative_prototypes"]),
            "skipped_count": len(skipped),
            "targets": sorted(summaries["target_prototypes"]),
        },
        **summaries,
        **hard_negative_summaries,
        "source_hashes": {
            f"{record.source_kind}::{record.target_id}::{record.split}::{record.class_label}::{record.path.name}": record.sha256
            for record in records
        },
        "hard_negative_source_hashes": {
            f"{record.negative_for_target_id}::{record.source_expected_target}::{record.class_label}::{record.path.name}": record.sha256
            for record in hard_negative_records
        },
        "skipped": skipped,
    }
    return payload


def write_prototype_bank(payload: dict[str, Any], output_path: Path) -> Path:
    return write_json(output_path, payload, ensure_ascii=False, sort_keys=False)


def write_router_prototype_summary(
    *,
    output_path: Path,
    registry_payload: dict[str, Any] | None,
    prototype_payload: dict[str, Any],
) -> Path:
    registry_summary = registry_payload.get("summary", {}) if isinstance(registry_payload, dict) else {}
    prototype_summary = prototype_payload.get("summary", {})
    lines = [
        "# Router Prototype Artifact Summary",
        "",
        f"- Registry targets: {registry_summary.get('target_count', 0)}",
        f"- Registry unresolved crops: {registry_summary.get('unresolved_count', 0)}",
        f"- Prototype targets: {prototype_summary.get('target_count', 0)}",
        f"- Class prototypes: {prototype_summary.get('class_prototype_count', 0)}",
        f"- Prototype samples: {prototype_summary.get('sample_count', 0)}",
        f"- Curated positive samples: {prototype_summary.get('curation_positive_count', 0)}",
        f"- Hard negative samples: {prototype_summary.get('hard_negative_count', 0)}",
        f"- Skipped images: {prototype_summary.get('skipped_count', 0)}",
        f"- Embedding backend: {prototype_payload.get('embedding_backend')}",
        "",
        "Targets:",
    ]
    for target_id in prototype_summary.get("targets", []):
        target = prototype_payload.get("target_prototypes", {}).get(target_id, {})
        lines.append(f"- `{target_id}`: {target.get('sample_count', 0)} samples")
    resolved = ensure_parent(output_path)
    resolved.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return resolved
