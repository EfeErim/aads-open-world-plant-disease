#!/usr/bin/env python3
"""Direct adapter smoke-test helpers for Colab notebooks."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.core.config_manager import get_config
from src.data.transforms import build_image_transform, preprocess_image
from src.shared.json_utils import read_json_dict

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_ROBUST_VIEWS = ("full_resize", "resize_pad", "center_crop")
IMAGE_MEAN_PAD_RGB = (124, 116, 104)
DEFAULT_EXPLANATION_METHOD = "occlusion_sensitivity"
SUPPORTED_EXPLANATION_METHODS = {"occlusion_sensitivity", "attention_map"}

# Explanation method hyperparameters.
DEFAULT_EXPLANATION_GRID_SIZE = 7
DEFAULT_CONFIDENCE_SPREAD_WARNING_THRESHOLD = 0.20

DEFAULT_DISCOVERY_ROOTS = (
    Path("."),
    Path("outputs"),
    Path("models"),
    Path("models") / "adapters",
)
SKIP_DISCOVERY_DIR_NAMES = {
    ".git",
    ".hg",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".runtime_tmp",
    ".venv",
    "__pycache__",
    "node_modules",
}
ADAPTER_LAYOUT_RESERVED_NAMES = {
    "adapter",
    "adapters",
    "adapter_export",
    "artifacts",
    "checkpoint_state",
    "colab_notebook_training",
    "models",
    "outputs",
    "runs",
    "telemetry",
}
RUN_ARTIFACT_MARKERS = {"outputs", "telemetry", "checkpoint_state", "artifacts"}


@lru_cache(maxsize=128)
def _load_inference_settings(config_env: Optional[str]) -> tuple[Path, int, float]:
    """Load inference settings from config. Cache expires after each call in long-running processes."""
    inference_cfg = dict(dict(get_config(environment=config_env)).get("inference", {}))
    return (
        Path(str(inference_cfg.get("adapter_root", "models/adapters"))),
        int(inference_cfg.get("target_size", 224)),
        float(
            inference_cfg.get(
                "confidence_spread_warning_threshold",
                DEFAULT_CONFIDENCE_SPREAD_WARNING_THRESHOLD,
            )
        ),
    )


def _default_adapter_root(config_env: Optional[str]) -> Path:
    return _load_inference_settings(config_env)[0]


def _target_size(config_env: Optional[str]) -> int:
    return _load_inference_settings(config_env)[1]


def _confidence_spread_warning_threshold(config_env: Optional[str]) -> float:
    return _load_inference_settings(config_env)[2]


def _normalize_crop_name(crop_name: str) -> str:
    normalized = str(crop_name).strip().lower()
    if not normalized:
        raise ValueError("crop_name must be a non-empty string")
    return normalized


def _read_json_if_exists(path: Path, *, strict: bool = False) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json_dict(path)
    except Exception as exc:
        if strict:
            raise ValueError(f"Failed to parse JSON metadata at {path}: {exc}") from exc
        return {}


def _extract_crop_name_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None

    context = payload.get("context", {})
    if not isinstance(context, dict):
        context = {}

    for candidate in (
        payload.get("crop"),
        payload.get("crop_name"),
        context.get("crop"),
        context.get("crop_name"),
    ):
        if not candidate:
            continue
        try:
            return _normalize_crop_name(str(candidate))
        except ValueError:
            continue
    return None


def _infer_crop_name_from_meta(adapter_dir: Path, *, strict_metadata: bool = False) -> Optional[str]:
    meta = _read_json_if_exists(adapter_dir / "adapter_meta.json", strict=strict_metadata)
    class_to_idx = meta.get("class_to_idx", {})
    if not isinstance(class_to_idx, dict):
        return None

    candidate_counts: Dict[str, int] = {}
    for raw_name in class_to_idx.keys():
        class_name = str(raw_name).strip().lower()
        if not class_name or class_name == "healthy":
            continue
        prefix = class_name.split("_", 1)[0].strip()
        if not prefix or prefix in {"unknown", "ood"}:
            continue
        candidate_counts[prefix] = candidate_counts.get(prefix, 0) + 1

    if not candidate_counts:
        return None
    inferred = max(candidate_counts.items(), key=lambda item: (item[1], item[0]))[0]
    return _normalize_crop_name(inferred)


def _iter_crop_metadata_candidates(adapter_dir: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    bases = [adapter_dir, *list(adapter_dir.parents)[:5]]
    relative_candidates = (
        Path("crop_info.json"),
        Path("production_readiness.json"),
        Path("training") / "summary.json",
        Path("artifacts") / "crop_info.json",
        Path("artifacts") / "production_readiness.json",
        Path("artifacts") / "training" / "summary.json",
    )
    for base in bases:
        for relative in relative_candidates:
            candidate = base / relative
            if candidate in seen:
                continue
            seen.add(candidate)
            yield candidate


def _infer_crop_name_from_adapter_dir(adapter_dir: Path, *, strict_metadata: bool = False) -> Optional[str]:
    layout_crop, _layout_part = _infer_adapter_layout_crop_part(adapter_dir)
    if layout_crop is not None:
        return layout_crop

    for candidate in _iter_crop_metadata_candidates(adapter_dir):
        inferred = _extract_crop_name_from_payload(
            _read_json_if_exists(candidate, strict=strict_metadata)
        )
        if inferred is not None:
            return inferred
    if adapter_dir.name == "continual_sd_lora_adapter" and adapter_dir.parent.name:
        parent_name = adapter_dir.parent.name.strip().lower()
        grandparent_name = adapter_dir.parent.parent.name.strip().lower()
        if (
            grandparent_name
            and parent_name not in ADAPTER_LAYOUT_RESERVED_NAMES
            and grandparent_name not in ADAPTER_LAYOUT_RESERVED_NAMES
        ):
            return grandparent_name
        parent_crop = parent_name
        if parent_crop not in ADAPTER_LAYOUT_RESERVED_NAMES:
            return parent_crop
    inferred_from_meta = _infer_crop_name_from_meta(adapter_dir, strict_metadata=strict_metadata)
    if inferred_from_meta is not None:
        return inferred_from_meta
    return None


def _infer_adapter_layout_crop_part(adapter_dir: Path) -> tuple[Optional[str], Optional[str]]:
    if adapter_dir.name != "continual_sd_lora_adapter":
        return None, None
    parent_part = adapter_dir.parent.name.strip().lower()
    grandparent_crop = adapter_dir.parent.parent.name.strip().lower()
    if not parent_part or not grandparent_crop:
        return None, None
    if grandparent_crop in ADAPTER_LAYOUT_RESERVED_NAMES or parent_part in ADAPTER_LAYOUT_RESERVED_NAMES:
        return None, None
    return grandparent_crop, parent_part


def _infer_part_name_from_adapter_dir(adapter_dir: Path) -> Optional[str]:
    _layout_crop, layout_part = _infer_adapter_layout_crop_part(adapter_dir)
    return layout_part


def _infer_enclosing_run_crop_part(adapter_dir: Path) -> tuple[Optional[str], Optional[str]]:
    parts = list(adapter_dir.parts)
    lowered = [str(part).strip().lower() for part in parts]
    for index, part in enumerate(lowered):
        if part != "runs":
            continue
        if index + 4 >= len(lowered):
            continue
        crop_name = lowered[index + 1]
        part_name = lowered[index + 2]
        next_marker = lowered[index + 4]
        if next_marker not in RUN_ARTIFACT_MARKERS:
            continue
        if (
            not crop_name
            or not part_name
            or crop_name in ADAPTER_LAYOUT_RESERVED_NAMES
            or part_name in ADAPTER_LAYOUT_RESERVED_NAMES
        ):
            continue
        return crop_name, part_name
    return None, None


def _adapter_layout_matches_enclosing_run(adapter_dir: Path) -> bool:
    layout_crop, layout_part = _infer_adapter_layout_crop_part(adapter_dir)
    run_crop, run_part = _infer_enclosing_run_crop_part(adapter_dir)
    if layout_crop is not None and run_crop is not None and layout_crop != run_crop:
        return False
    if layout_part is not None and run_part is not None and layout_part != run_part:
        return False
    return True


def _resolve_crop_name(crop_name: Optional[str], *, adapter_dir: Path) -> str:
    if crop_name is not None:
        return _normalize_crop_name(crop_name)
    inferred = _infer_crop_name_from_adapter_dir(adapter_dir, strict_metadata=True)
    if inferred is not None:
        return inferred
    raise ValueError(
        "crop_name is required when it cannot be inferred from the adapter path or crop_info.json. "
        "Set CROP_NAME manually, or pass an adapter path that already implies the crop."
    )


def _infer_run_id(adapter_dir: Path) -> str:
    parts = list(adapter_dir.parts)
    reserved = {
        "artifacts",
        "adapter_export",
        "continual_sd_lora_adapter",
        "checkpoint_state",
        "outputs",
        "telemetry",
    }
    if "runs" in parts:
        idx = parts.index("runs")
        marker_offsets = [
            offset
            for offset, part in enumerate(parts[idx + 1 :], start=idx + 1)
            if part in {"outputs", "telemetry", "checkpoint_state"}
        ]
        if marker_offsets:
            candidate = str(parts[marker_offsets[0] - 1]).strip()
            if candidate and candidate not in reserved:
                return candidate
        for candidate_part in reversed(parts[idx + 1 :]):
            candidate = str(candidate_part).strip()
            if candidate and candidate not in reserved:
                return candidate
    if "telemetry" in parts:
        idx = parts.index("telemetry")
        if idx + 1 < len(parts):
            candidate = str(parts[idx + 1]).strip()
            if candidate and candidate not in reserved:
                return candidate
    return ""


def _adapter_source_rank(adapter_dir: Path) -> int:
    normalized = adapter_dir.as_posix().lower()
    if (
        "/runs/" in normalized
        and "/outputs/colab_notebook_training/" in normalized
        and normalized.endswith("/continual_sd_lora_adapter")
    ):
        return 0
    if (
        "/runs/" in normalized
        and "/checkpoint_state/artifacts/adapter_export/" in normalized
        and normalized.endswith("/continual_sd_lora_adapter")
    ):
        return 4
    if (
        "/runs/" in normalized
        and "/telemetry/artifacts/adapter_export/" in normalized
        and normalized.endswith("/continual_sd_lora_adapter")
    ):
        return 2
    if (
        "/runs/" in normalized
        and "/artifacts/adapter_export/" in normalized
        and normalized.endswith("/continual_sd_lora_adapter")
    ):
        return 1
    if (
        "/telemetry/" in normalized
        and "/artifacts/adapter_export/" in normalized
        and normalized.endswith("/continual_sd_lora_adapter")
    ):
        return 3
    if "/models/adapters/" in normalized and normalized.endswith("/continual_sd_lora_adapter"):
        return 5
    if (
        "/outputs/colab_notebook_training/" in normalized
        and normalized.endswith("/continual_sd_lora_adapter")
    ):
        return 6
    return 7


def _candidate_identity_key(
    *,
    adapter_dir: Path,
    crop_name: Optional[str],
    part_name: Optional[str],
    run_id: str,
) -> str:
    if run_id:
        return f"run:{run_id}:{crop_name or 'unknown'}:{part_name or 'unknown'}"
    return f"path:{adapter_dir.resolve()}"


def _is_adapter_dir(path: Path) -> bool:
    return path.is_dir() and path.name == "continual_sd_lora_adapter" and (path / "adapter_meta.json").exists()


def _iter_explicit_adapter_dir_candidates(path: Path) -> Iterable[Path]:
    if path.name == "adapter_meta.json":
        yield path.parent

    for relative in (
        Path("."),
        Path("continual_sd_lora_adapter"),
        Path("adapter_export"),
        Path("adapter_export") / "continual_sd_lora_adapter",
        Path("artifacts") / "adapter_export",
        Path("artifacts") / "adapter_export" / "continual_sd_lora_adapter",
        Path("outputs") / "colab_notebook_training",
        Path("outputs") / "colab_notebook_training" / "continual_sd_lora_adapter",
    ):
        yield path / relative


def _iter_adapter_root_candidates(path: Path, *, crop_key: str, part_key: Optional[str] = None) -> Iterable[Path]:
    yield path
    yield path / "continual_sd_lora_adapter"
    if part_key:
        yield path / crop_key / part_key / "continual_sd_lora_adapter"
        yield path / crop_key / part_key
    yield path / crop_key
    yield path / crop_key / "continual_sd_lora_adapter"


def _first_adapter_dir(candidates: Iterable[Path]) -> Optional[Path]:
    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if _is_adapter_dir(candidate):
            return candidate
    return None


def _resolve_adapter_dir(
    crop_name: Optional[str],
    *,
    adapter_dir: Optional[str | Path] = None,
    adapter_root: Optional[str | Path] = None,
    part_name: Optional[str] = None,
    config_env: Optional[str] = "colab",
) -> Path:
    if adapter_dir is not None:
        root = Path(adapter_dir)
        explicit_candidates = list(_iter_explicit_adapter_dir_candidates(root))
        if crop_name is not None:
            explicit_candidates.extend(
                _iter_adapter_root_candidates(
                    root,
                    crop_key=_normalize_crop_name(crop_name),
                    part_key=str(part_name or "").strip().lower() or None,
                )
            )
        resolved = _first_adapter_dir(explicit_candidates)
        if resolved is not None:
            return resolved
        raise FileNotFoundError(
            "Could not resolve an adapter bundle from "
            f"adapter_dir={root}. Pass the adapter asset directory, its parent export directory, "
            "or adapter_meta.json."
        )

    base_root = Path(adapter_root) if adapter_root is not None else _default_adapter_root(config_env)
    if crop_name is None:
        resolved = _first_adapter_dir((base_root, base_root / "continual_sd_lora_adapter"))
        if resolved is not None:
            return resolved
        raise ValueError(
            "crop_name is required when resolving from ADAPTER_ROOT or the configured adapter_root. "
            "Set CROP_NAME manually, or use ADAPTER_DIR to point at a specific adapter export."
        )

    crop_key = _normalize_crop_name(crop_name)
    resolved = _first_adapter_dir(
        _iter_adapter_root_candidates(
            base_root,
            crop_key=crop_key,
            part_key=str(part_name or "").strip().lower() or None,
        )
    )
    if resolved is not None:
        return resolved

    discovered = discover_adapter_candidates([base_root], crop_name=crop_key)
    if discovered:
        normalized_part = str(part_name or "").strip().lower()
        if normalized_part:
            for candidate in discovered:
                if str(candidate.get("part_name") or "").strip().lower() == normalized_part:
                    return Path(str(candidate["adapter_dir"]))
        for candidate in discovered:
            candidate_part = str(candidate.get("part_name") or "").strip().lower()
            if not candidate_part or candidate_part == "unspecified":
                return Path(str(candidate["adapter_dir"]))
        return Path(str(discovered[0]["adapter_dir"]))

    raise FileNotFoundError(
        f"Adapter not found for crop '{crop_key}' under adapter_root={base_root}. "
        "Expected either <adapter_root>/<crop>/<part>/continual_sd_lora_adapter/, "
        "<adapter_root>/<crop>/continual_sd_lora_adapter/, <adapter_root>/<crop>/, or a direct adapter asset directory."
    )


def _build_adapter(crop_name: str, *, device: str) -> IndependentCropAdapter:
    return IndependentCropAdapter(crop_name=_normalize_crop_name(crop_name), device=device)


def _adapter_context_cache_key(resolved_dir: Path) -> tuple[str, int, int]:
    normalized_dir = str(resolved_dir.resolve(strict=False))
    meta_path = resolved_dir / "adapter_meta.json"
    if not meta_path.exists():
        return normalized_dir, 0, 0
    stat = meta_path.stat()
    return normalized_dir, int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=16)
def _cached_load_adapter_context(
    resolved_dir_key: str,
    crop_key: str,
    device: str,
    meta_mtime_ns: int,
    meta_size: int,
) -> tuple[Path, str, IndependentCropAdapter]:
    resolved_dir = Path(resolved_dir_key)
    adapter = _build_adapter(crop_key, device=device)
    adapter.load_adapter(str(resolved_dir))
    return resolved_dir, crop_key, adapter


def _load_adapter_context(
    crop_name: Optional[str],
    *,
    adapter_dir: Optional[str | Path] = None,
    adapter_root: Optional[str | Path] = None,
    part_name: Optional[str] = None,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
) -> tuple[Path, str, IndependentCropAdapter]:
    resolved_dir = _resolve_adapter_dir(
        crop_name,
        adapter_dir=adapter_dir,
        adapter_root=adapter_root,
        part_name=part_name,
        config_env=config_env,
    )
    crop_key = _resolve_crop_name(crop_name, adapter_dir=resolved_dir)
    resolved_dir_key, meta_mtime_ns, meta_size = _adapter_context_cache_key(resolved_dir)
    return _cached_load_adapter_context(
        resolved_dir_key,
        crop_key,
        device,
        meta_mtime_ns,
        meta_size,
    )


def _read_adapter_meta(adapter_dir: Path) -> Dict[str, Any]:
    return read_json_dict(adapter_dir / "adapter_meta.json")


def _summary_from_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    class_to_idx = meta.get("class_to_idx", {})
    ordered_classes = [
        name
        for name, _ in sorted(
            ((str(name), int(index)) for name, index in dict(class_to_idx).items()),
            key=lambda item: item[1],
        )
    ]
    return {
        "crop_name": str(meta.get("crop_name", "") or ""),
        "part_name": str(meta.get("part_name", "") or ""),
        "backbone_model_name": str(dict(meta.get("backbone", {})).get("model_name", "")),
        "class_names": ordered_classes,
        "class_count": len(ordered_classes),
        "fusion": dict(meta.get("fusion", {})),
        "target_modules_resolved": [str(item) for item in list(meta.get("target_modules_resolved", []))],
        "ood_calibration_version": int(cast(int, dict(meta.get("ood_calibration", {})).get("version", 0))),
        "adapter_runtime": dict(meta.get("adapter_runtime", {})),
    }


def _candidate_label(
    *,
    adapter_dir: Path,
    crop_name: Optional[str],
    backbone_model_name: str,
    class_count: int,
    run_id: str,
    metadata_error: str = "",
) -> str:
    label_bits = [crop_name or "unknown-crop", f"{class_count} classes"]
    if backbone_model_name:
        label_bits.append(backbone_model_name)
    if run_id:
        label_bits.append(f"run={run_id}")
    if metadata_error:
        label_bits.append("metadata-warning")
    label_bits.append(str(adapter_dir))
    return " | ".join(label_bits)


def _iter_adapter_meta_paths(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DISCOVERY_DIR_NAMES]
        if "adapter_meta.json" in filenames:
            yield Path(dirpath) / "adapter_meta.json"


def _normalize_search_roots(search_roots: Sequence[str | Path]) -> List[Path]:
    resolved_pairs = [(Path(raw_root), Path(raw_root).resolve(strict=False)) for raw_root in search_roots]
    kept: List[tuple[Path, Path]] = []
    for index, (root, root_key) in enumerate(resolved_pairs):
        if any(
            index != other_index and (root_key == other_key or root_key.is_relative_to(other_key))
            for other_index, (_other_root, other_key) in enumerate(resolved_pairs)
        ):
            continue
        if any(root_key == existing_key for _existing_root, existing_key in kept):
            continue
        kept.append((root, root_key))
    return [root for root, _root_key in kept]


def discover_adapter_candidates(
    search_roots: Optional[Sequence[str | Path]] = None,
    *,
    crop_name: Optional[str] = None,
    collapse_run_mirrors: bool = True,
) -> List[Dict[str, Any]]:
    """Search one or more roots for adapter bundles and return selection-ready metadata."""
    requested_crop = _normalize_crop_name(crop_name) if crop_name else None
    roots = _normalize_search_roots(search_roots or DEFAULT_DISCOVERY_ROOTS)
    candidates_by_identity: Dict[str, Dict[str, Any]] = {}
    seen: set[str] = set()

    for root in roots:
        for meta_path in sorted(_iter_adapter_meta_paths(root)):
            adapter_dir = meta_path.parent
            if not _is_adapter_dir(adapter_dir):
                continue
            if not _adapter_layout_matches_enclosing_run(adapter_dir):
                continue
            adapter_key = str(adapter_dir.resolve())
            if adapter_key in seen:
                continue
            seen.add(adapter_key)

            metadata_error = ""
            try:
                inferred_crop = _infer_crop_name_from_adapter_dir(adapter_dir, strict_metadata=True)
            except ValueError as exc:
                inferred_crop = None
                metadata_error = str(exc)
            if requested_crop is not None and inferred_crop not in {None, requested_crop}:
                continue

            meta = _read_adapter_meta(adapter_dir)
            meta_summary = _summary_from_meta(meta)
            layout_crop, layout_part = _infer_adapter_layout_crop_part(adapter_dir)
            candidate_crop = layout_crop or inferred_crop or meta_summary.get("crop_name") or requested_crop
            candidate_part = (
                layout_part
                or str(meta_summary.get("part_name") or "").strip().lower()
                or _infer_part_name_from_adapter_dir(adapter_dir)
                or None
            )
            run_id = _infer_run_id(adapter_dir)
            if collapse_run_mirrors:
                identity_key = _candidate_identity_key(
                    adapter_dir=adapter_dir,
                    crop_name=candidate_crop,
                    part_name=candidate_part,
                    run_id=run_id,
                )
            else:
                identity_key = f"path:{adapter_dir.resolve()}"
            candidate = {
                "adapter_dir": str(adapter_dir),
                "crop_name": candidate_crop,
                "part_name": candidate_part,
                "backbone_model_name": meta_summary["backbone_model_name"],
                "class_count": meta_summary["class_count"],
                "class_names": list(meta_summary["class_names"]),
                "fusion": dict(meta_summary["fusion"]),
                "target_modules_resolved": list(meta_summary["target_modules_resolved"]),
                "ood_calibration_version": int(meta_summary["ood_calibration_version"]),
                "run_id": run_id,
                "metadata_error": metadata_error,
            }
            existing = candidates_by_identity.get(identity_key)
            if existing is not None:
                if not collapse_run_mirrors:
                    continue
                existing_rank = _adapter_source_rank(Path(str(existing.get("adapter_dir", ""))))
                candidate_rank = _adapter_source_rank(adapter_dir)
                if (candidate_rank, len(str(adapter_dir))) >= (
                    existing_rank,
                    len(str(existing.get("adapter_dir", ""))),
                ):
                    continue
            candidates_by_identity[identity_key] = candidate

    candidates = sorted(
        candidates_by_identity.values(),
        key=lambda item: (
            str(item.get("crop_name") or ""),
            str(item.get("run_id") or ""),
            _adapter_source_rank(Path(str(item.get("adapter_dir", "")))),
            str(item.get("adapter_dir", "")),
        ),
    )
    for candidate in candidates:
        adapter_dir = Path(str(candidate["adapter_dir"]))
        candidate["display_name"] = _candidate_label(
            adapter_dir=adapter_dir,
            crop_name=candidate.get("crop_name"),
            backbone_model_name=str(candidate.get("backbone_model_name", "")),
            class_count=int(candidate.get("class_count", 0)),
            run_id=str(candidate.get("run_id", "")),
            metadata_error=str(candidate.get("metadata_error", "")),
        )
    return candidates


def _flatten_prediction(
    image_name: str,
    payload: Dict[str, Any],
    *,
    resolved_adapter_dir: Path,
    view_name: Optional[str] = None,
) -> Dict[str, Any]:
    disease = dict(payload.get("disease", {}))
    ood = dict(payload.get("ood_analysis", {}))
    row = {
        "image_name": str(image_name),
        "adapter_dir": str(resolved_adapter_dir),
        "status": str(payload.get("status", "")),
        "predicted_class": disease.get("name"),
        "predicted_index": disease.get("class_index"),
        "confidence": float(disease.get("confidence", 0.0)),
        "is_ood": bool(ood.get("is_ood", False)),
        "score_method": str(ood.get("score_method", "ensemble")),
        "primary_score": float(ood.get("primary_score", 0.0)),
        "decision_threshold": float(ood.get("decision_threshold", 0.0)),
        "calibration_version": int(ood.get("calibration_version", 0)),
        "ood_analysis": ood,
        "raw_payload": payload,
    }
    if view_name is not None:
        row["view_name"] = str(view_name)
    return row


def _normalize_pil_image(image: Image.Image, *, target_size: int) -> Any:
    transform = build_image_transform(int(target_size), training=False)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image)


def _prepare_view_image(image: Image.Image, *, target_size: int, view_name: str) -> Image.Image:
    rgb_image = image.convert("RGB")
    if view_name == "full_resize":
        return rgb_image.resize((int(target_size), int(target_size)), Image.Resampling.BILINEAR)

    if view_name == "resize_pad":
        width, height = rgb_image.size
        longest = max(width, height, 1)
        scale = float(target_size) / float(longest)
        resized = rgb_image.resize(
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (int(target_size), int(target_size)), IMAGE_MEAN_PAD_RGB)
        paste_x = (int(target_size) - resized.width) // 2
        paste_y = (int(target_size) - resized.height) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas

    if view_name == "center_crop":
        width, height = rgb_image.size
        shortest = max(1, min(width, height))
        scale = float(target_size) / float(shortest)
        resized = rgb_image.resize(
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            Image.Resampling.BILINEAR,
        )
        left = max(0, (resized.width - int(target_size)) // 2)
        top = max(0, (resized.height - int(target_size)) // 2)
        cropped = ImageOps.crop(
            resized,
            border=cast(
                Any,
                (
                    left,
                    top,
                    max(0, resized.width - int(target_size) - left),
                    max(0, resized.height - int(target_size) - top),
                ),
            ),
        )
        if cropped.size != (int(target_size), int(target_size)):
            cropped = ImageOps.fit(
                resized,
                (int(target_size), int(target_size)),
                method=Image.Resampling.BILINEAR,
                centering=(0.5, 0.5),
            )
        return cropped.convert("RGB")

    raise ValueError(
        f"Unsupported robust smoke view '{view_name}'. Expected one of: {', '.join(DEFAULT_ROBUST_VIEWS)}."
    )


def _prepare_view_tensor(image: Image.Image, *, target_size: int, view_name: str) -> Any:
    if view_name == "full_resize":
        return preprocess_image(image.convert("RGB"), target_size=target_size)
    return _normalize_pil_image(
        _prepare_view_image(image, target_size=int(target_size), view_name=view_name),
        target_size=int(target_size),
    )


def _view_error_row(
    image_name: str,
    *,
    resolved_adapter_dir: Path,
    view_name: str,
    error: Exception,
) -> Dict[str, Any]:
    row = _error_row(Path(image_name), resolved_adapter_dir=resolved_adapter_dir, error=error)
    row["view_name"] = str(view_name)
    return row


def _predict_image_view(
    image_name: str,
    image: Image.Image,
    *,
    adapter: IndependentCropAdapter,
    target_size: int,
    resolved_adapter_dir: Path,
    view_name: str,
) -> Dict[str, Any]:
    image_tensor = _prepare_view_tensor(image, target_size=target_size, view_name=view_name)
    payload = adapter.predict_with_ood(image_tensor)
    return _flatten_prediction(
        image_name,
        payload,
        resolved_adapter_dir=resolved_adapter_dir,
        view_name=view_name,
    )


def _predict_payload_confidence_for_class(
    adapter: IndependentCropAdapter,
    image_tensor: Any,
    *,
    target_class_index: int,
) -> float:
    payload = adapter.predict_with_ood(image_tensor)
    disease = dict(payload.get("disease", {}))
    try:
        predicted_index = int(cast(int, disease.get("class_index")))
    except (TypeError, ValueError):
        return 0.0
    if predicted_index != int(target_class_index):
        return 0.0
    try:
        return float(disease.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _occlusion_sensitivity_grid(
    image_tensor: Any,
    *,
    adapter: IndependentCropAdapter,
    target_class_index: int,
    baseline_confidence: float,
    grid_size: int,
) -> List[List[float]]:
    import torch

    if not torch.is_tensor(image_tensor):
        raise ValueError("Occlusion visualization requires a torch tensor image.")
    if image_tensor.ndim != 3:
        raise ValueError("Occlusion visualization expects an image tensor shaped [C, H, W].")

    resolved_grid = int(max(2, min(16, grid_size)))
    _, height, width = image_tensor.shape
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Image tensor has invalid spatial dimensions: height={height}, width={width}. "
            "Occlusion visualization requires positive spatial extent."
        )
    heatmap = torch.zeros((resolved_grid, resolved_grid), dtype=torch.float32)
    baseline = float(max(0.0, baseline_confidence))

    masked = image_tensor.detach().clone()

    for row_index in range(resolved_grid):
        top = int(round(row_index * height / resolved_grid))
        bottom = int(round((row_index + 1) * height / resolved_grid))
        for col_index in range(resolved_grid):
            left = int(round(col_index * width / resolved_grid))
            right = int(round((col_index + 1) * width / resolved_grid))
            original_patch = masked[:, top:bottom, left:right].clone()
            masked[:, top:bottom, left:right] = 0.0
            occluded_confidence = _predict_payload_confidence_for_class(
                adapter,
                masked,
                target_class_index=int(target_class_index),
            )
            heatmap[row_index, col_index] = max(0.0, baseline - occluded_confidence)
            masked[:, top:bottom, left:right] = original_patch

    del masked
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    max_drop = float(heatmap.max().item()) if heatmap.numel() else 0.0
    if max_drop > 0.0:
        heatmap = heatmap / max_drop
    return [[round(float(value), 6) for value in row] for row in heatmap.tolist()]


def _add_occlusion_visualization(
    result: Dict[str, Any],
    image: Image.Image,
    *,
    adapter: IndependentCropAdapter,
    target_size: int,
    view_name: str,
    grid_size: int,
) -> None:
    if str(result.get("status", "")).strip().lower() == "error":
        return
    predicted_index = result.get("predicted_index")
    if predicted_index is None:
        return
    try:
        target_class_index = int(predicted_index)
        baseline_confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        return

    view_image = _prepare_view_image(image, target_size=int(target_size), view_name=view_name)
    view_tensor = _normalize_pil_image(view_image, target_size=int(target_size))
    grid = _occlusion_sensitivity_grid(
        view_tensor,
        adapter=adapter,
        target_class_index=target_class_index,
        baseline_confidence=baseline_confidence,
        grid_size=int(grid_size),
    )
    result["visualization"] = {
        "status": "success",
        "method": "occlusion_sensitivity",
        "explanation": (
            "Patches are hidden one at a time; brighter cells mark larger drops in the "
            "predicted-class confidence. This is a model-agnostic diagnostic, not proof of causality."
        ),
        "view_name": str(view_name),
        "target_class_index": target_class_index,
        "target_class_name": result.get("predicted_class"),
        "baseline_confidence": baseline_confidence,
        "grid_size": int(max(2, min(16, grid_size))),
        "heatmap": grid,
    }


def _normalization_error_visualization(method: str, error: Exception) -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "method": str(method),
        "error": str(error),
    }


def _normalize_explanation_method(explanation_method: str | None) -> str:
    normalized = str(explanation_method or DEFAULT_EXPLANATION_METHOD).strip().lower()
    if normalized in {"occlusion", "occlusion_map"}:
        return "occlusion_sensitivity"
    if normalized in {"attention", "attention_rollout", "attention_weights"}:
        return "attention_map"
    if normalized not in SUPPORTED_EXPLANATION_METHODS:
        raise ValueError(
            "Unsupported explanation_method: "
            f"{explanation_method}. Expected one of: {', '.join(sorted(SUPPORTED_EXPLANATION_METHODS))}."
        )
    return normalized


def _patch_grid_shape(patch_count: int, config: Any) -> tuple[int, int]:
    import math

    count = int(max(1, patch_count))
    square = int(math.sqrt(count))
    if square * square == count:
        return square, square

    image_size = getattr(config, "image_size", None)
    patch_size = getattr(config, "patch_size", None)
    try:
        if isinstance(image_size, (list, tuple)):
            image_height, image_width = int(image_size[0]), int(image_size[1])
        else:
            image_height = image_width = int(cast(int, image_size))
        patch = int(cast(int, patch_size))
        rows = max(1, image_height // max(1, patch))
        cols = max(1, image_width // max(1, patch))
        if rows * cols == count:
            return rows, cols
    except (TypeError, ValueError):
        pass

    return 1, count


def _normalize_heatmap_tensor(values: Any, *, rows: int, cols: int) -> List[List[float]]:

    heatmap = values.detach().float().reshape(int(rows), int(cols)).cpu()
    heatmap = heatmap - heatmap.min()
    max_value = float(heatmap.max().item()) if heatmap.numel() else 0.0
    if max_value > 0.0:
        heatmap = heatmap / max_value
    return [[round(float(value), 6) for value in row] for row in heatmap.tolist()]


def _extract_attention_heatmap(
    image_tensor: Any,
    *,
    adapter: IndependentCropAdapter,
) -> Dict[str, Any]:
    import torch

    if not torch.is_tensor(image_tensor):
        raise ValueError("Attention visualization requires a torch tensor image.")
    if image_tensor.ndim != 3:
        raise ValueError("Attention visualization expects an image tensor shaped [C, H, W].")

    trainer = adapter.trainer
    model = getattr(trainer, "adapter_model", None)
    if model is None:
        raise RuntimeError("Adapter model is not initialized; attention tensors cannot be captured.")

    config = _resolve_attention_config(model, trainer)
    if config is None:
        raise RuntimeError("Loaded adapter model does not expose a config for attention capture.")

    previous_attn_impl = getattr(config, "_attn_implementation", None)
    attn_impl_changed = False
    if hasattr(config, "_attn_implementation") and previous_attn_impl != "eager":
        setattr(config, "_attn_implementation", "eager")
        attn_impl_changed = True

    try:
        if hasattr(trainer, "set_eval_mode"):
            trainer.set_eval_mode()
        with torch.inference_mode():
            device = getattr(trainer, "device", None)
            if device is None:
                raise RuntimeError("Trainer does not have a 'device' attribute for attention extraction.")
            outputs = model(
                image_tensor.unsqueeze(0).to(device, non_blocking=True),
                output_attentions=True,
                output_hidden_states=True,
            )
    finally:
        if attn_impl_changed:
            try:
                setattr(config, "_attn_implementation", previous_attn_impl)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to restore attention implementation to '{previous_attn_impl}' after inference. "
                    f"Config state may be inconsistent. Error: {exc}"
                ) from exc

    attentions = getattr(outputs, "attentions", None)
    if not attentions or len(attentions) == 0:
        raise RuntimeError(
            "The loaded backbone did not return attentions. DINOv3 ViT requires an eager attention backend; "
            "DINOv3 ConvNeXT-style backbones do not expose attention maps."
        )

    layer_attention = attentions[-1]
    if layer_attention.ndim != 4:
        raise RuntimeError(f"Expected attention tensor [B, H, T, T], got shape {tuple(layer_attention.shape)}.")

    token_count = int(layer_attention.shape[-1])
    register_count = int(getattr(config, "num_register_tokens", 0) or 0)
    patch_start = 1 + max(0, register_count)
    if patch_start >= token_count:
        raise RuntimeError(
            f"Attention tensor has {token_count} tokens, leaving no patch tokens after CLS/register tokens."
        )

    patch_attention = layer_attention[0, :, 0, patch_start:].mean(dim=0)
    rows, cols = _patch_grid_shape(int(patch_attention.numel()), config)
    return {
        "heatmap": _normalize_heatmap_tensor(patch_attention, rows=rows, cols=cols),
        "attention_layer": -1,
        "attention_head_aggregation": "mean",
        "source_token": "cls",
        "register_tokens": register_count,
        "patch_tokens": int(patch_attention.numel()),
        "grid_size": rows if rows == cols else [rows, cols],
        "attention_backend": "eager",
    }


def _resolve_attention_config(model: Any, trainer: Any) -> Any:
    for candidate in (
        getattr(model, "config", None),
        getattr(getattr(model, "base_model", None), "config", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None),
        getattr(trainer, "config", None),
    ):
        if candidate is not None and (
            hasattr(candidate, "_attn_implementation")
            or hasattr(candidate, "num_register_tokens")
            or hasattr(candidate, "patch_size")
        ):
            return candidate
    return None


def _add_attention_visualization(
    result: Dict[str, Any],
    image: Image.Image,
    *,
    adapter: IndependentCropAdapter,
    target_size: int,
    view_name: str,
) -> None:
    if str(result.get("status", "")).strip().lower() == "error":
        return

    try:
        view_image = _prepare_view_image(image, target_size=int(target_size), view_name=view_name)
        view_tensor = _normalize_pil_image(view_image, target_size=int(target_size))
        attention_payload = _extract_attention_heatmap(view_tensor, adapter=adapter)
        result["visualization"] = {
            "status": "success",
            "method": "attention_map",
            "explanation": (
                "Last-layer CLS-to-patch attention averaged across heads. This shows attention routing, "
                "not a causal explanation of the prediction."
            ),
            "view_name": str(view_name),
            "target_class_index": result.get("predicted_index"),
            "target_class_name": result.get("predicted_class"),
            **attention_payload,
        }
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        result["visualization"] = _normalization_error_visualization("attention_map", exc)


def _add_prediction_visualization(
    result: Dict[str, Any],
    image: Image.Image,
    *,
    adapter: IndependentCropAdapter,
    target_size: int,
    view_name: str,
    grid_size: int,
    explanation_method: str,
) -> None:
    method = _normalize_explanation_method(explanation_method)
    if method == "attention_map":
        _add_attention_visualization(
            result,
            image,
            adapter=adapter,
            target_size=target_size,
            view_name=view_name,
        )
        return

    _add_occlusion_visualization(
        result,
        image,
        adapter=adapter,
        target_size=target_size,
        view_name=view_name,
        grid_size=grid_size,
    )


def _normalize_requested_views(robust_views: Optional[Sequence[str]]) -> List[str]:
    raw_views = list(robust_views or DEFAULT_ROBUST_VIEWS)
    if not raw_views:
        raise ValueError("robust_views must contain at least one view name when robust mode is enabled.")
    normalized: List[str] = []
    seen: set[str] = set()
    for view_name in raw_views:
        normalized_name = str(view_name).strip().lower()
        if not normalized_name:
            continue
        if normalized_name in seen:
            continue
        if normalized_name not in DEFAULT_ROBUST_VIEWS:
            raise ValueError(
                f"Unsupported robust smoke view '{normalized_name}'. "
                f"Expected one of: {', '.join(DEFAULT_ROBUST_VIEWS)}."
            )
        seen.add(normalized_name)
        normalized.append(normalized_name)
    if not normalized:
        raise ValueError("robust_views must contain at least one supported view name.")
    return normalized


def _view_consistency_summary(
    views: Sequence[Dict[str, Any]],
    *,
    primary_view: str,
    confidence_spread_warning_threshold: float = DEFAULT_CONFIDENCE_SPREAD_WARNING_THRESHOLD,
) -> Dict[str, Any]:
    successful = [row for row in views if row.get("status") != "error"]
    successful_names = [str(row.get("view_name", "")) for row in successful if row.get("view_name")]
    failed_names = [str(row.get("view_name", "")) for row in views if row.get("status") == "error" and row.get("view_name")]
    predicted_classes = {
        str(row.get("view_name", "")): row.get("predicted_class")
        for row in successful
        if row.get("view_name")
    }
    ood_votes = {
        str(row.get("view_name", "")): row.get("is_ood")
        for row in successful
        if row.get("view_name")
    }
    confidences = [
        float(row.get("confidence", 0.0))
        for row in successful
        if row.get("confidence") is not None
    ]
    warning_codes: List[str] = []
    distinct_classes = {value for value in predicted_classes.values() if value is not None}
    if len(distinct_classes) > 1:
        warning_codes.append("view_class_disagreement")
    distinct_ood = {value for value in ood_votes.values() if value is not None}
    if len(distinct_ood) > 1:
        warning_codes.append("view_ood_disagreement")
    if confidences:
        confidence_min = min(confidences)
        confidence_max = max(confidences)
        confidence_spread = confidence_max - confidence_min
        if confidence_spread >= confidence_spread_warning_threshold:
            warning_codes.append("view_confidence_spread_high")
    else:
        confidence_min = None
        confidence_max = None
        confidence_spread = None
    if failed_names:
        warning_codes.append("view_error_present")
    return {
        "primary_view": str(primary_view),
        "successful_views": successful_names,
        "failed_views": failed_names,
        "warning_codes": warning_codes,
        "predicted_classes": predicted_classes,
        "ood_votes": ood_votes,
        "confidence_min": confidence_min,
        "confidence_max": confidence_max,
        "confidence_spread": confidence_spread,
        "stable": len(successful_names) >= 2 and not warning_codes,
    }


def _uncertainty_diagnostics(
    row: Dict[str, Any],
    *,
    view_consistency: Dict[str, Any],
) -> Dict[str, Any]:
    raw_ood_analysis = row.get("ood_analysis")
    ood_analysis = dict(raw_ood_analysis) if isinstance(raw_ood_analysis, dict) else {}
    warning_codes: List[str] = []
    status = str(row.get("status", "")).strip().lower()
    if status == "error":
        warning_codes.append("prediction_error")
    else:
        warning_codes.append("confidence_not_calibrated")
        if bool(row.get("is_ood")):
            warning_codes.append("ood_flagged")
    sure_confidence_reject = ood_analysis.get("sure_confidence_reject")
    if sure_confidence_reject is True:
        warning_codes.append("sure_confidence_reject")
    conformal_set = ood_analysis.get("conformal_set")
    conformal_set_size = ood_analysis.get("conformal_set_size")
    if conformal_set_size is None and isinstance(conformal_set, list):
        conformal_set_size = len(conformal_set)
    if isinstance(conformal_set_size, int) and conformal_set_size > 1:
        warning_codes.append("conformal_set_wide")
    if not bool(view_consistency.get("stable")):
        warning_codes.append("view_instability")

    diagnostics: Dict[str, Any] = {
        "confidence_source": "top1_softmax",
        "confidence_is_calibrated": False,
        "status": status,
        "warning_codes": warning_codes,
    }
    if row.get("error"):
        diagnostics["error"] = str(row.get("error"))
    if "candidate_scores" in ood_analysis:
        diagnostics["candidate_scores"] = dict(ood_analysis.get("candidate_scores", {}))
    if "candidate_thresholds" in ood_analysis:
        diagnostics["candidate_thresholds"] = dict(ood_analysis.get("candidate_thresholds", {}))
    if "sure_confidence_reject" in ood_analysis:
        diagnostics["sure_confidence_reject"] = bool(ood_analysis.get("sure_confidence_reject"))
    if "sure_semantic_ood" in ood_analysis:
        diagnostics["sure_semantic_ood"] = bool(ood_analysis.get("sure_semantic_ood"))
    if conformal_set is not None:
        diagnostics["conformal_set"] = list(conformal_set)
    if conformal_set_size is not None:
        diagnostics["conformal_set_size"] = int(conformal_set_size)
    return diagnostics


def _predict_single_image_robust(
    image_path: Path,
    *,
    adapter: IndependentCropAdapter,
    target_size: int,
    resolved_adapter_dir: Path,
    config_env: Optional[str],
    robust_views: Sequence[str],
    explain_prediction: bool = False,
    explanation_grid_size: int = DEFAULT_EXPLANATION_GRID_SIZE,
    explanation_method: str = DEFAULT_EXPLANATION_METHOD,
) -> Dict[str, Any]:
    image_name = image_path.name
    rows: List[Dict[str, Any]] = []
    with Image.open(image_path) as opened_image:
        base_image = opened_image.convert("RGB")
        for view_name in robust_views:
            try:
                rows.append(
                    _predict_image_view(
                        image_name,
                        base_image,
                        adapter=adapter,
                        target_size=target_size,
                        resolved_adapter_dir=resolved_adapter_dir,
                        view_name=view_name,
                    )
                )
            except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as exc:
                rows.append(
                    _view_error_row(
                        image_name,
                        resolved_adapter_dir=resolved_adapter_dir,
                        view_name=view_name,
                        error=exc,
                    )
                )

    primary_view = str(robust_views[0])
    primary_row = next(
        (row for row in rows if row.get("view_name") == primary_view),
        _view_error_row(
            image_name,
            resolved_adapter_dir=resolved_adapter_dir,
            view_name=primary_view,
            error=RuntimeError(f"Primary view '{primary_view}' did not produce a result."),
        ),
    )
    result = dict(primary_row)
    result["image_path"] = str(image_path)
    result["views"] = rows
    result["view_consistency"] = _view_consistency_summary(
        rows,
        primary_view=primary_view,
        confidence_spread_warning_threshold=_confidence_spread_warning_threshold(config_env),
    )
    result["uncertainty_diagnostics"] = _uncertainty_diagnostics(
        result,
        view_consistency=result["view_consistency"],
    )
    if explain_prediction:
        _add_prediction_visualization(
            result,
            base_image,
            adapter=adapter,
            target_size=target_size,
            view_name=primary_view,
            grid_size=explanation_grid_size,
            explanation_method=explanation_method,
        )
    return result


def _predict_image_row(
    image_path: Path,
    *,
    adapter: IndependentCropAdapter,
    target_size: int,
    resolved_adapter_dir: Path,
    explain_prediction: bool = False,
    explanation_grid_size: int = DEFAULT_EXPLANATION_GRID_SIZE,
    explanation_method: str = DEFAULT_EXPLANATION_METHOD,
) -> Dict[str, Any]:
    with Image.open(image_path) as image:
        base_image = image.convert("RGB")
        row = _predict_image_view(
            image_path.name,
            base_image,
            adapter=adapter,
            target_size=target_size,
            resolved_adapter_dir=resolved_adapter_dir,
            view_name="full_resize",
        )
        if explain_prediction:
            _add_prediction_visualization(
                row,
                base_image,
                adapter=adapter,
                target_size=target_size,
                view_name="full_resize",
                grid_size=explanation_grid_size,
                explanation_method=explanation_method,
            )
    row.pop("view_name", None)
    row["image_path"] = str(image_path)
    return row


def _heat_color(value: float) -> tuple[int, int, int, int]:
    intensity = max(0.0, min(1.0, float(value)))
    red = int(255)
    green = int(round(224 * (1.0 - intensity) + 80 * intensity))
    blue = int(round(130 * (1.0 - intensity)))
    alpha = int(round(35 + 165 * intensity))
    return red, green, blue, alpha


def _heatmap_overlay_image(base_image: Image.Image, heatmap: Sequence[Sequence[float]]) -> Image.Image:
    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rows = len(heatmap)
    cols = max((len(row) for row in heatmap), default=0)
    if rows <= 0 or cols <= 0:
        return base_image.convert("RGBA")
    width, height = base_image.size
    for row_index, row in enumerate(heatmap):
        for col_index, value in enumerate(row):
            left = int(round(col_index * width / cols))
            right = int(round((col_index + 1) * width / cols))
            top = int(round(row_index * height / rows))
            bottom = int(round((row_index + 1) * height / rows))
            draw.rectangle((left, top, right, bottom), fill=_heat_color(float(value)))
    return Image.alpha_composite(base_image.convert("RGBA"), overlay).convert("RGB")


def build_prediction_visualization_images(
    image_path: str | Path,
    result: Dict[str, Any],
) -> Dict[str, Image.Image]:
    """Build notebook-display images from a prediction result visualization payload."""
    visualization = dict(result.get("visualization", {}))
    if visualization.get("method") not in {"occlusion_sensitivity", "attention_map"}:
        return {}
    heatmap = visualization.get("heatmap")
    if not isinstance(heatmap, list) or not heatmap:
        return {}
    view_name = str(visualization.get("view_name") or "full_resize")
    grid_size = int(visualization.get("grid_size") or len(heatmap))
    with Image.open(image_path) as opened:
        view_image = _prepare_view_image(
            opened.convert("RGB"),
            target_size=grid_size * 64,
            view_name=view_name,
        )
    overlay = _heatmap_overlay_image(view_image, heatmap)
    return {
        "model_view": view_image,
        "heatmap_overlay": overlay,
        "occlusion_overlay": overlay,
    }


def _error_row(image_path: Path, *, resolved_adapter_dir: Path, error: Exception) -> Dict[str, Any]:
    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "adapter_dir": str(resolved_adapter_dir),
        "status": "error",
        "predicted_class": None,
        "predicted_index": None,
        "confidence": 0.0,
        "is_ood": None,
        "score_method": None,
        "primary_score": None,
        "decision_threshold": None,
        "calibration_version": None,
        "ood_analysis": None,
        "raw_payload": None,
        "error": str(error),
    }


def load_adapter_summary(
    crop_name: Optional[str],
    adapter_dir: Optional[str | Path] = None,
    adapter_root: Optional[str | Path] = None,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    *,
    part_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve an adapter bundle and return a compact metadata summary."""
    resolved_dir = _resolve_adapter_dir(
        crop_name,
        adapter_dir=adapter_dir,
        adapter_root=adapter_root,
        part_name=part_name,
        config_env=config_env,
    )
    crop_key = _resolve_crop_name(crop_name, adapter_dir=resolved_dir)
    meta = _read_adapter_meta(resolved_dir)
    meta_summary = _summary_from_meta(meta)
    class_to_idx = dict(meta.get("class_to_idx", {}))
    summary = {
        "crop_name": crop_key,
        "part_name": str(meta.get("part_name", "") or ""),
        "model_name": str(dict(meta.get("backbone", {})).get("model_name", "")),
        "engine": str(meta.get("engine", "continual_sd_lora")),
        "schema_version": str(meta.get("schema_version", "v6")),
        "is_trained": True,
        "num_classes": len(class_to_idx),
        "class_to_idx": {str(k): int(v) for k, v in class_to_idx.items()},
        "ood_calibration_version": int(dict(meta.get("ood_calibration", {})).get("version", 0)),
    }
    summary.update(meta_summary)
    # Keep resolved crop/part values authoritative when metadata is missing.
    summary["crop_name"] = crop_key
    if not str(summary.get("part_name") or "").strip():
        summary["part_name"] = "unspecified"
    summary["resolved_adapter_dir"] = str(resolved_dir)
    return summary


def predict_single_image(
    image_path: str | Path,
    crop_name: Optional[str],
    adapter_dir: Optional[str | Path] = None,
    adapter_root: Optional[str | Path] = None,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    enable_robust_smoke: bool = False,
    robust_views: Optional[Sequence[str]] = None,
    explain_prediction: bool = False,
    explanation_grid_size: int = DEFAULT_EXPLANATION_GRID_SIZE,
    explanation_method: str = DEFAULT_EXPLANATION_METHOD,
    *,
    part_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single direct adapter prediction for a smoke test."""
    resolved_dir, _, adapter = _load_adapter_context(
        crop_name,
        adapter_dir=adapter_dir,
        adapter_root=adapter_root,
        part_name=part_name,
        config_env=config_env,
        device=device,
    )
    image_ref = Path(image_path)
    if enable_robust_smoke:
        return _predict_single_image_robust(
            image_ref,
            adapter=adapter,
            target_size=_target_size(config_env),
            resolved_adapter_dir=resolved_dir,
            config_env=config_env,
            robust_views=_normalize_requested_views(robust_views),
            explain_prediction=bool(explain_prediction),
            explanation_grid_size=int(explanation_grid_size),
            explanation_method=explanation_method,
        )
    return _predict_image_row(
        image_ref,
        adapter=adapter,
        target_size=_target_size(config_env),
        resolved_adapter_dir=resolved_dir,
        explain_prediction=bool(explain_prediction),
        explanation_grid_size=int(explanation_grid_size),
        explanation_method=explanation_method,
    )


def _iter_image_candidates(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def predict_image_folder(
    image_dir: str | Path,
    crop_name: Optional[str],
    adapter_dir: Optional[str | Path] = None,
    adapter_root: Optional[str | Path] = None,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    *,
    part_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run direct adapter predictions for all supported images in a folder."""
    folder = Path(image_dir)
    if not folder.is_dir():
        raise NotADirectoryError(
            f"image_dir must be a directory containing images, but got: {folder}"
        )

    resolved_dir, _, adapter = _load_adapter_context(
        crop_name,
        adapter_dir=adapter_dir,
        adapter_root=adapter_root,
        part_name=part_name,
        config_env=config_env,
        device=device,
    )
    target_size = _target_size(config_env)

    rows: List[Dict[str, Any]] = []
    for image_path in _iter_image_candidates(folder):
        try:
            row = _predict_image_row(
                image_path,
                adapter=adapter,
                target_size=target_size,
                resolved_adapter_dir=resolved_dir,
            )
            row["error"] = ""
        except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as exc:
            row = _error_row(image_path, resolved_adapter_dir=resolved_dir, error=exc)
        rows.append(row)
    return rows


__all__ = [
    "DEFAULT_EXPLANATION_METHOD",
    "SUPPORTED_EXPLANATION_METHODS",
    "build_prediction_visualization_images",
    "discover_adapter_candidates",
    "load_adapter_summary",
    "predict_single_image",
    "predict_image_folder",
]
