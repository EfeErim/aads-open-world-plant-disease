"""Embedding backends and neighbor search for grouped dataset preparation."""

from __future__ import annotations

import gc
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageOps
from sklearn.neighbors import NearestNeighbors

DEFAULT_CPU_EMBEDDING_BATCH_SIZE = 4


DEFAULT_T4_EMBEDDING_BATCH_SIZE = 4


DEFAULT_SMALL_GPU_EMBEDDING_BATCH_SIZE = 6


DEFAULT_MID_GPU_EMBEDDING_BATCH_SIZE = 8


DEFAULT_LARGE_GPU_EMBEDDING_BATCH_SIZE = 12


def _resolve_amp_dtype(device: str) -> Any:
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    major, _minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def _resolve_embedding_device(device: str) -> str:
    requested = str(device or "cpu").strip() or "cpu"
    if requested.startswith("cuda"):
        import torch

        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
        if not cuda_available:
            return "cpu"
    return requested


def _system_memory_gb() -> Optional[float]:
    try:
        sysconf = getattr(os, "sysconf", None)
        if not callable(sysconf):
            return None
        pages = sysconf("SC_PHYS_PAGES")
        page_size = sysconf("SC_PAGE_SIZE")
        if pages and page_size:
            return float(pages * page_size) / (1024**3)
    except (AttributeError, OSError, ValueError):
        return None
    return None


def resolve_safe_embedding_batch_size(device: str, requested_batch_size: Optional[int] = None) -> int:
    """Choose a conservative Notebook 0 embedding batch size for Colab RAM limits."""
    if requested_batch_size is not None:
        return max(1, int(requested_batch_size))

    resolved_device = _resolve_embedding_device(device)
    if not str(resolved_device).startswith("cuda"):
        return DEFAULT_CPU_EMBEDDING_BATCH_SIZE

    import torch

    try:
        props = torch.cuda.get_device_properties(0)
        vram_gb = float(getattr(props, "total_memory", 0.0)) / (1024**3)
        gpu_name = str(getattr(props, "name", "") or "").lower()
    except Exception:
        return DEFAULT_T4_EMBEDDING_BATCH_SIZE

    system_gb = _system_memory_gb()
    if "t4" in gpu_name or vram_gb <= 16.5 or (system_gb is not None and system_gb <= 14.0):
        return DEFAULT_T4_EMBEDDING_BATCH_SIZE
    if vram_gb <= 24.0:
        return DEFAULT_SMALL_GPU_EMBEDDING_BATCH_SIZE
    if vram_gb <= 35.0:
        return DEFAULT_MID_GPU_EMBEDDING_BATCH_SIZE
    return DEFAULT_LARGE_GPU_EMBEDDING_BATCH_SIZE


def _release_embedding_batch_memory(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    if "cuda" in message and ("out of memory" in message or "oom" in message):
        return True
    try:
        import torch

        return isinstance(exc, torch.cuda.OutOfMemoryError)
    except Exception:
        return False


def _load_dinov3_components(model_id: str, *, device: str = "cpu") -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModel

    load_kwargs: Dict[str, Any] = {}
    amp_dtype = _resolve_amp_dtype(device)
    if amp_dtype is not None:
        load_kwargs["dtype"] = amp_dtype
    processor = AutoImageProcessor.from_pretrained(model_id)
    try:
        model = AutoModel.from_pretrained(model_id, **load_kwargs)
    except TypeError:
        model = AutoModel.from_pretrained(model_id)
    model.eval()
    return processor, model


def _autocast_context(*, device: str, amp_dtype: Any) -> Any:
    import torch

    enabled = amp_dtype is not None and str(device).startswith("cuda") and torch.cuda.is_available()
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _load_bioclip_components(model_id: str, *, device: str = "cpu") -> tuple[Any, Any]:
    import open_clip
    import torch

    hub_model_id = f"hf-hub:{model_id}" if not str(model_id).startswith("hf-hub:") else str(model_id)
    create_kwargs: Dict[str, Any] = {}
    amp_dtype = _resolve_amp_dtype(device)
    if amp_dtype is not None:
        create_kwargs["precision"] = "bf16" if amp_dtype == torch.bfloat16 else "fp16"
    try:
        model, _, preprocess_val = open_clip.create_model_and_transforms(hub_model_id, **create_kwargs)
    except TypeError:
        model, _, preprocess_val = open_clip.create_model_and_transforms(hub_model_id)
    model.eval()
    return preprocess_val, model


def _encode_dinov3(paths: Sequence[Path], *, model_id: str, batch_size: int, device: str) -> np.ndarray:
    processor, model = _load_dinov3_components(model_id, device=device)
    return _encode_dinov3_with_components(
        paths,
        processor=processor,
        model=model,
        batch_size=batch_size,
        device=device,
        amp_dtype=_resolve_amp_dtype(device),
    )


def _encode_dinov3_with_components(
    paths: Sequence[Path],
    *,
    processor: Any,
    model: Any,
    batch_size: int,
    device: str,
    amp_dtype: Any = None,
    progress_fn: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    import torch

    embeddings: List[np.ndarray] = []
    total = len(paths)
    effective_batch_size = max(1, int(batch_size))
    for start in range(0, len(paths), effective_batch_size):
        batch_paths = paths[start : start + effective_batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as raw:
                images.append(ImageOps.exif_transpose(raw.convert("RGB")))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        with torch.inference_mode():
            with _autocast_context(device=device, amp_dtype=amp_dtype):
                outputs = model(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    batch = outputs.pooler_output
                else:
                    batch = outputs.last_hidden_state[:, 0]
        batch = torch.nn.functional.normalize(batch, dim=-1).to(dtype=torch.float32)
        embeddings.append(batch.detach().cpu().numpy())
        del images, inputs, outputs, batch
        _release_embedding_batch_memory(device)
        if callable(progress_fn):
            progress_fn(min(start + len(batch_paths), total), total)
    return np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)


def _encode_bioclip(paths: Sequence[Path], *, model_id: str, batch_size: int, device: str) -> np.ndarray:
    preprocess_val, model = _load_bioclip_components(model_id, device=device)
    return _encode_bioclip_with_components(
        paths,
        preprocess_val=preprocess_val,
        model=model,
        batch_size=batch_size,
        device=device,
        amp_dtype=_resolve_amp_dtype(device),
    )


def _encode_bioclip_with_components(
    paths: Sequence[Path],
    *,
    preprocess_val: Any,
    model: Any,
    batch_size: int,
    device: str,
    amp_dtype: Any = None,
    progress_fn: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    import torch

    embeddings: List[np.ndarray] = []
    total = len(paths)
    effective_batch_size = max(1, int(batch_size))
    for start in range(0, len(paths), effective_batch_size):
        batch_paths = paths[start : start + effective_batch_size]
        tensors = []
        for path in batch_paths:
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw.convert("RGB"))
            tensors.append(preprocess_val(image))
        image_tensor = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        with torch.inference_mode():
            with _autocast_context(device=device, amp_dtype=amp_dtype):
                batch = model.encode_image(image_tensor)
        batch = torch.nn.functional.normalize(batch, dim=-1).to(dtype=torch.float32)
        embeddings.append(batch.detach().cpu().numpy())
        del tensors, image_tensor, batch
        _release_embedding_batch_memory(device)
        if callable(progress_fn):
            progress_fn(min(start + len(batch_paths), total), total)
    return np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)


def _encode_with_oom_retries(
    encode_fn: Callable[..., np.ndarray],
    *,
    paths: Sequence[Path],
    batch_size: int,
    progress_fn: Optional[Callable[[str], None]],
    batch_progress_fn: Optional[Callable[[int, int], None]] = None,
    label: str,
    oom_predicate: Optional[Callable[[BaseException], bool]] = None,
    **kwargs: Any,
) -> np.ndarray:
    current_batch_size = max(1, int(batch_size))
    while True:
        try:
            return encode_fn(
                paths,
                batch_size=current_batch_size,
                progress_fn=batch_progress_fn,
                **kwargs,
            )
        except RuntimeError as exc:
            if not (oom_predicate or _is_cuda_oom)(exc) or current_batch_size <= 1:
                raise
            next_batch_size = max(1, current_batch_size // 2)
            _progress(
                progress_fn,
                f"{label} CUDA OOM at batch_size={current_batch_size}; retrying with batch_size={next_batch_size}.",
            )
            _release_embedding_batch_memory(str(kwargs.get("device", "")))
            current_batch_size = next_batch_size


def _compute_neighbor_pairs(
    embeddings: np.ndarray,
    *,
    paths: Sequence[str],
    neighbors: int,
) -> Dict[tuple[str, str], float]:
    if embeddings.size == 0 or len(paths) < 2:
        return {}
    # Guard against rare NaN/Inf rows from model outputs or normalization.
    finite_mask = np.isfinite(embeddings).all(axis=1)
    if not bool(np.all(finite_mask)):
        embeddings = embeddings[finite_mask]
        paths = [path for path, keep in zip(paths, finite_mask) if bool(keep)]
    if embeddings.size == 0 or len(paths) < 2:
        return {}
    neigh = NearestNeighbors(
        n_neighbors=min(len(paths), max(2, int(neighbors))),
        metric="cosine",
        algorithm="brute",
    )
    neigh.fit(embeddings)
    distances, indices = neigh.kneighbors(embeddings)
    pairs: Dict[tuple[str, str], float] = {}
    for row_index, path_a in enumerate(paths):
        for distance, col_index in zip(distances[row_index][1:], indices[row_index][1:]):
            path_b = paths[int(col_index)]
            if path_a == path_b:
                continue
            pair = (min(path_a, path_b), max(path_a, path_b))
            cosine = 1.0 - float(distance)
            pairs[pair] = max(cosine, pairs.get(pair, float("-inf")))
    return pairs


def _progress(progress_fn: Optional[Callable[[str], None]], message: str) -> None:
    if callable(progress_fn):
        progress_fn(str(message))
