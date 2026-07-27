"""Runtime scoring helpers for CLIP/BioCLIP-backed router inference."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.router.confidence_utils import build_open_set_rejection_reasons
from src.router.prompt_clip_utils import (
    aggregate_prompt_logits,
    build_prompt_ensemble,
    get_clip_logit_scale,
    get_prompt_templates_for_type,
    open_set_unknown_prompts,
)
from src.router.runtime_surface import coerce_bool

logger = logging.getLogger(__name__)


def _strict_scoring_errors(runtime: Any) -> bool:
    configured = getattr(runtime, "strict_scoring_errors", None)
    if configured is None:
        configured = getattr(runtime, "strict_model_loading", None)
    if configured is None:
        return True
    return coerce_bool(configured, default=True)


@dataclass
class ClipScoreRequest:
    """Request-local state so one scoring flow does not re-encode the same image."""

    image: Image.Image
    image_embedding: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class PromptMatrixPlan:
    labels: Tuple[str, ...]
    templates: Tuple[str, ...]
    flattened_prompts: Tuple[str, ...]

    @property
    def label_count(self) -> int:
        return len(self.labels)

    @property
    def template_count(self) -> int:
        return len(self.templates)


def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings / embeddings.norm(dim=-1, keepdim=True)


def _open_clip_components(runtime: Any) -> tuple[Dict[str, Any], Any]:
    processor = runtime.bioclip_processor
    model = runtime.bioclip
    if not isinstance(processor, dict) or model is None:
        raise RuntimeError("open_clip encoding requested before BioCLIP runtime was initialized.")
    return processor, model


def _processor_components(runtime: Any) -> tuple[Any, Any]:
    processor = runtime.bioclip_processor
    model = runtime.bioclip
    if processor is None or model is None:
        raise RuntimeError("BioCLIP scoring requested before BioCLIP runtime was initialized.")
    return processor, model


def _resolve_logits_per_image(model_outputs: Any, model: Any) -> torch.Tensor:
    if hasattr(model_outputs, "logits_per_image") and model_outputs.logits_per_image is not None:
        return model_outputs.logits_per_image
    if hasattr(model_outputs, "image_embeds") and hasattr(model_outputs, "text_embeds"):
        image_embeds = _normalize_embeddings(model_outputs.image_embeds)
        text_embeds = _normalize_embeddings(model_outputs.text_embeds)
        logit_scale = get_clip_logit_scale(model)
        return (image_embeds @ text_embeds.T) * logit_scale
    raise RuntimeError("BioCLIP model output does not provide logits_per_image or embeddable outputs")


def _build_open_clip_image_batch(
    preprocess: Any,
    images: List[Image.Image],
    device: Any,
) -> torch.Tensor:
    processed_images = [preprocess(image) for image in images]
    return torch.stack(processed_images, dim=0).to(device)


def _limit_prompt_templates(
    runtime: Any,
    *,
    label_type: str,
    num_prompts: Optional[int],
) -> List[str]:
    templates = get_prompt_templates_for_type(runtime.vlm_config, label_type)
    if num_prompts is None:
        return templates
    try:
        prompt_limit = int(num_prompts)
    except (TypeError, ValueError, OverflowError):
        prompt_limit = 0
    if prompt_limit > 0:
        return templates[:prompt_limit]
    return templates


def _average_label_scores(label_score_lists: Dict[str, List[float]]) -> Dict[str, float]:
    return {
        label: float(np.mean(scores)) if scores else 0.0
        for label, scores in label_score_lists.items()
    }


def _best_label_and_score(label_scores: Dict[str, float]) -> tuple[str, float]:
    best_label = max(label_scores, key=lambda label: label_scores[label]) if label_scores else "unknown"
    return best_label, label_scores.get(best_label, 0.0)


def _build_prompt_matrix_plan(
    runtime: Any,
    *,
    labels: List[str],
    label_type: str,
    num_prompts: Optional[int],
) -> PromptMatrixPlan:
    templates = tuple(_limit_prompt_templates(runtime, label_type=label_type, num_prompts=num_prompts))
    flattened_prompts = tuple(
        template.format(term=label)
        for template in templates
        for label in labels
    )
    return PromptMatrixPlan(
        labels=tuple(labels),
        templates=templates,
        flattened_prompts=flattened_prompts,
    )


def _build_limited_prompt_batch(
    labels: List[str],
    *,
    label_type: str,
    vlm_config: Dict[str, Any],
    num_prompts: Optional[int],
) -> tuple[List[str], List[int]]:
    try:
        prompt_limit = int(num_prompts) if num_prompts is not None else 0
    except (TypeError, ValueError, OverflowError):
        prompt_limit = 0

    prompt_texts: List[str] = []
    prompt_to_class: List[int] = []
    for class_index, label in enumerate(labels):
        class_prompts = build_prompt_ensemble(
            label=label,
            label_type=label_type,
            vlm_config=vlm_config,
        )
        if not class_prompts:
            class_prompts = [str(label)]
        if prompt_limit > 0:
            class_prompts = class_prompts[:prompt_limit]
        prompt_texts.extend(class_prompts)
        prompt_to_class.extend([class_index] * len(class_prompts))
    return prompt_texts, prompt_to_class


def _score_prompt_probabilities(
    runtime: Any,
    prompts: List[str],
    *,
    request: Optional[ClipScoreRequest] = None,
    images: Optional[List[Image.Image]] = None,
    image_batch_size: Optional[int] = None,
    image_embeddings: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if request is not None:
        try:
            if runtime.bioclip_backend == "open_clip":
                image_embedding = ensure_open_clip_image_embedding(runtime, request)
                text_embeds = get_open_clip_text_embeddings(runtime, prompts)
                if runtime.bioclip is None:
                    raise RuntimeError("open_clip scoring requested before BioCLIP runtime was initialized.")
                logit_scale = get_clip_logit_scale(runtime.bioclip)
                with torch.no_grad():
                    logits_per_image = (image_embedding @ text_embeds.T) * logit_scale
                return torch.softmax(logits_per_image, dim=-1).detach().cpu()
            return _score_processor_batch(runtime, request.image, prompts).detach().cpu()
        except Exception as exc:
            logger.error("Encoding failed: %s", exc)
            if _strict_scoring_errors(runtime):
                raise RuntimeError("Prompt scoring failed during router inference.") from exc
            return torch.zeros((1, len(prompts)), dtype=torch.float32)

    if images is None:
        raise ValueError("Either request or images must be provided for prompt scoring.")
    return encode_and_score_batch(
        runtime,
        images,
        prompts,
        image_batch_size=image_batch_size,
        image_embeddings=image_embeddings,
    )


def _prompt_matrix_to_label_scores(
    probabilities: torch.Tensor,
    *,
    labels: Tuple[str, ...],
    template_count: int,
) -> Dict[str, float]:
    if probabilities.numel() <= 0 or template_count <= 0 or not labels:
        return {}
    score_matrix = probabilities.reshape(template_count, len(labels))
    label_means = score_matrix.mean(dim=0)
    return {
        labels[index]: float(label_means[index].item())
        for index in range(len(labels))
    }


def _prompt_tensor_to_batch_label_scores(
    probabilities: torch.Tensor,
    *,
    labels: Tuple[str, ...],
    template_count: int,
) -> List[Dict[str, float]]:
    if probabilities.numel() <= 0 or template_count <= 0 or not labels:
        return [{} for _ in range(int(probabilities.shape[0]))]
    score_tensor = probabilities.reshape(int(probabilities.shape[0]), template_count, len(labels))
    label_means = score_tensor.mean(dim=1)
    return [
        {
            labels[label_index]: float(label_means[image_index, label_index].item())
            for label_index in range(len(labels))
        }
        for image_index in range(int(label_means.shape[0]))
    ]


def _score_processor_batch(
    runtime: Any,
    images: Image.Image | List[Image.Image],
    prompts: List[str],
) -> torch.Tensor:
    processor, model = _processor_components(runtime)
    model_inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    model_inputs = {key: value.to(runtime.device) for key, value in model_inputs.items()}

    with torch.no_grad():
        model_outputs = model(**model_inputs)
        logits_per_image = _resolve_logits_per_image(model_outputs, model)
    return torch.softmax(logits_per_image, dim=-1)


def _aggregate_prompt_probabilities(
    runtime: Any,
    *,
    request: ClipScoreRequest,
    text_prompts: List[str],
    prompt_to_class: List[int],
    class_count: int,
) -> torch.Tensor:
    probabilities = _score_prompt_probabilities(runtime, text_prompts, request=request)
    prompt_logits = torch.log(probabilities.clamp_min(1e-12))

    with torch.no_grad():
        logits = aggregate_prompt_logits(prompt_logits, prompt_to_class, class_count)
    return torch.softmax(logits, dim=-1)


def _aggregate_batch_prompt_probabilities(
    probabilities: torch.Tensor,
    *,
    prompt_to_class: List[int],
    class_count: int,
) -> torch.Tensor:
    if probabilities.numel() <= 0 or class_count <= 0:
        return torch.zeros((int(probabilities.shape[0]), max(0, class_count)), dtype=torch.float32)

    prompt_logits = torch.log(probabilities.clamp_min(1e-12))
    class_logits = torch.full(
        (int(prompt_logits.shape[0]), class_count),
        float("-inf"),
        dtype=prompt_logits.dtype,
        device=prompt_logits.device,
    )
    for prompt_index, class_index in enumerate(prompt_to_class):
        class_logits[:, class_index] = torch.maximum(class_logits[:, class_index], prompt_logits[:, prompt_index])
    return torch.softmax(class_logits, dim=-1).detach().cpu()


def _class_probabilities_to_label_scores(
    probabilities: torch.Tensor,
    *,
    labels: List[str],
) -> Dict[str, float]:
    if probabilities.numel() <= 0:
        return {}
    return {
        str(label): float(probabilities[0, index].item())
        for index, label in enumerate(labels)
    }


def _batch_class_probabilities_to_label_scores(
    probabilities: torch.Tensor,
    *,
    labels: List[str],
) -> List[Dict[str, float]]:
    if probabilities.numel() <= 0:
        return [{} for _ in range(int(probabilities.shape[0]))]
    return [
        {
            str(label): float(probabilities[image_index, label_index].item())
            for label_index, label in enumerate(labels)
        }
        for image_index in range(int(probabilities.shape[0]))
    ]


def get_open_clip_text_embeddings(runtime: Any, prompts: List[str]) -> torch.Tensor:
    """Get normalized text embeddings for prompts with lightweight in-memory caching."""
    cache_key = tuple(prompts)
    cached = runtime._open_clip_text_embedding_cache.get(cache_key)
    if cached is not None:
        return cached

    processor, model = _open_clip_components(runtime)

    tokenizer = processor["tokenizer"]
    text_tokens = tokenizer(prompts).to(runtime.device)
    with torch.no_grad():
        text_embeds = model.encode_text(text_tokens)
        text_embeds = _normalize_embeddings(text_embeds)

    max_cache_size = int(runtime.vlm_config.get("open_clip_text_cache_size", 64))
    if len(runtime._open_clip_text_embedding_cache) >= max_cache_size and max_cache_size > 0:
        oldest_key = next(iter(runtime._open_clip_text_embedding_cache))
        runtime._open_clip_text_embedding_cache.pop(oldest_key, None)
    if max_cache_size > 0:
        runtime._open_clip_text_embedding_cache[cache_key] = text_embeds
    return text_embeds


def ensure_open_clip_image_embedding(runtime: Any, request: ClipScoreRequest) -> torch.Tensor:
    """Encode a single image once per scoring request for open_clip scoring."""
    if request.image_embedding is not None:
        return request.image_embedding

    processor, model = _open_clip_components(runtime)

    preprocess = processor["preprocess"]
    image_tensor = preprocess(request.image).unsqueeze(0).to(runtime.device)
    with torch.no_grad():
        image_embeds = model.encode_image(image_tensor)
        request.image_embedding = _normalize_embeddings(image_embeds)
    return request.image_embedding


def ensure_open_clip_image_embeddings(
    runtime: Any,
    images: List[Image.Image],
    *,
    image_batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Encode many images with open_clip in image batches."""
    if not images:
        return torch.empty((0, 0), dtype=torch.float32, device=runtime.device)

    processor, model = _open_clip_components(runtime)

    preprocess = processor["preprocess"]
    batch_size = max(1, int(image_batch_size or len(images)))
    embeddings: List[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        image_tensor = _build_open_clip_image_batch(
            preprocess,
            images[start : start + batch_size],
            runtime.device,
        )
        with torch.no_grad():
            image_embeds = model.encode_image(image_tensor)
            embeddings.append(_normalize_embeddings(image_embeds))
    return torch.cat(embeddings, dim=0)


def score_open_clip_with_image_embedding(
    runtime: Any,
    image_embedding: torch.Tensor,
    prompts: List[str],
) -> List[float]:
    """Score prompts against a precomputed open_clip image embedding."""
    text_embeds = get_open_clip_text_embeddings(runtime, prompts)
    if runtime.bioclip is None:
        raise RuntimeError("open_clip scoring requested before BioCLIP runtime was initialized.")
    logit_scale = get_clip_logit_scale(runtime.bioclip)
    with torch.no_grad():
        logits_per_image = (image_embedding @ text_embeds.T) * logit_scale
        scores = torch.softmax(logits_per_image, dim=-1)[0]
    return scores.detach().cpu().numpy().tolist()


def score_open_clip_with_image_embeddings(
    runtime: Any,
    image_embeddings: torch.Tensor,
    prompts: List[str],
) -> torch.Tensor:
    """Score prompts against precomputed open_clip image embeddings."""
    text_embeds = get_open_clip_text_embeddings(runtime, prompts)
    if runtime.bioclip is None:
        raise RuntimeError("open_clip scoring requested before BioCLIP runtime was initialized.")
    logit_scale = get_clip_logit_scale(runtime.bioclip)
    with torch.no_grad():
        logits_per_image = (image_embeddings @ text_embeds.T) * logit_scale
    return torch.softmax(logits_per_image, dim=-1)


def encode_and_score(runtime: Any, request: ClipScoreRequest, prompts: List[str]) -> List[float]:
    """Encode image and score against text prompts."""
    try:
        if runtime.bioclip_backend == "open_clip":
            image_embedding = ensure_open_clip_image_embedding(runtime, request)
            return score_open_clip_with_image_embedding(runtime, image_embedding, prompts)

        scores = _score_processor_batch(runtime, request.image, prompts)[0]
        return scores.detach().cpu().numpy().tolist()
    except Exception as exc:
        logger.error("Encoding failed: %s", exc)
        if _strict_scoring_errors(runtime):
            raise RuntimeError("Prompt encoding failed during router inference.") from exc
        return [0.0] * len(prompts)


def encode_and_score_batch(
    runtime: Any,
    images: List[Image.Image],
    prompts: List[str],
    *,
    image_batch_size: Optional[int] = None,
    image_embeddings: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Encode many images and score against a shared prompt set."""
    if not images or not prompts:
        return torch.zeros((len(images), len(prompts)), dtype=torch.float32)

    try:
        if runtime.bioclip_backend == "open_clip":
            embeddings = image_embeddings
            if embeddings is None:
                embeddings = ensure_open_clip_image_embeddings(runtime, images, image_batch_size=image_batch_size)
            return score_open_clip_with_image_embeddings(runtime, embeddings, prompts).detach().cpu()

        batch_size = max(1, int(image_batch_size or len(images)))
        outputs: List[torch.Tensor] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            outputs.append(_score_processor_batch(runtime, chunk, prompts).detach().cpu())
        return torch.cat(outputs, dim=0) if outputs else torch.zeros((0, len(prompts)), dtype=torch.float32)
    except Exception as exc:
        logger.error("Batch encoding failed: %s", exc)
        if _strict_scoring_errors(runtime):
            raise RuntimeError("Batch prompt encoding failed during router inference.") from exc
        return torch.zeros((len(images), len(prompts)), dtype=torch.float32)


def clip_score_labels_ensemble(
    runtime: Any,
    image: Image.Image,
    labels: List[str],
    *,
    label_type: str = "generic",
    num_prompts: Optional[int] = None,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Score text labels against image with multiple prompts for robustness.

    Returns: (best_label, best_score, all_scores_dict)
    """
    if not labels:
        return "unknown", 0.0, {}

    text_prompts, prompt_to_class = _build_limited_prompt_batch(
        labels=labels,
        label_type=label_type,
        vlm_config=runtime.vlm_config,
        num_prompts=num_prompts,
    )
    if not text_prompts:
        return "unknown", 0.0, {}
    request = ClipScoreRequest(image=image)
    probabilities = _aggregate_prompt_probabilities(
        runtime,
        request=request,
        text_prompts=text_prompts,
        prompt_to_class=prompt_to_class,
        class_count=len(labels),
    )
    label_avg_scores = _class_probabilities_to_label_scores(
        probabilities,
        labels=labels,
    )
    best_label, best_score = _best_label_and_score(label_avg_scores)
    return best_label, best_score, label_avg_scores


def clip_score_labels_open_set(
    runtime: Any,
    image: Image.Image,
    labels: List[str],
    *,
    label_type: str = "generic",
    num_prompts: Optional[int] = None,
    open_set_enabled: bool = False,
    open_set_min_confidence: Optional[float] = None,
    open_set_margin: Optional[float] = None,
    unknown_label: str = "unknown",
    unknown_prompts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Score labels once and return open-set diagnostics for downstream gating."""
    if not labels:
        return {
            "label": str(unknown_label),
            "confidence": 0.0,
            "second_confidence": 0.0,
            "unknown_confidence": 1.0 if open_set_enabled else 0.0,
            "margin": 0.0,
            "label_scores": {},
            "rejection_reasons": [f"label resolved to {str(unknown_label).strip().lower() or 'unknown'}"],
        }

    text_prompts, prompt_to_class = _build_limited_prompt_batch(
        labels=labels,
        label_type=label_type,
        vlm_config=runtime.vlm_config,
        num_prompts=num_prompts,
    )
    if not text_prompts:
        return {
            "label": str(unknown_label),
            "confidence": 0.0,
            "second_confidence": 0.0,
            "unknown_confidence": 1.0 if open_set_enabled else 0.0,
            "margin": 0.0,
            "label_scores": {},
            "rejection_reasons": [f"label resolved to {str(unknown_label).strip().lower() or 'unknown'}"],
        }

    known_class_count = len(labels)
    class_count = known_class_count
    unknown_class_index: Optional[int] = None
    if open_set_enabled:
        unknown_class_index = class_count
        text_prompts.extend(
            list(unknown_prompts)
            if isinstance(unknown_prompts, list) and unknown_prompts
            else open_set_unknown_prompts(label_type=label_type, known_labels=labels)
        )
        prompt_to_class.extend([unknown_class_index] * max(0, len(text_prompts) - len(prompt_to_class)))
        class_count += 1

    request = ClipScoreRequest(image=image)
    probabilities = _aggregate_prompt_probabilities(
        runtime,
        request=request,
        text_prompts=text_prompts,
        prompt_to_class=prompt_to_class,
        class_count=class_count,
    )
    known_probs = probabilities[:, :known_class_count]
    label_scores = {
        labels[index]: float(known_probs[0, index].item())
        for index in range(known_class_count)
    }
    best_label, confidence = _best_label_and_score(label_scores)
    sorted_scores = sorted((float(score) for score in label_scores.values()), reverse=True)
    second_confidence = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    unknown_confidence = 0.0
    if open_set_enabled and unknown_class_index is not None:
        unknown_confidence = float(probabilities[:, unknown_class_index].item())

    margin = float(confidence) - float(second_confidence)
    rejection_reasons = (
        build_open_set_rejection_reasons(
            label=best_label,
            confidence=confidence,
            second_confidence=second_confidence,
            unknown_confidence=unknown_confidence,
            min_confidence=0.0 if open_set_min_confidence is None else float(open_set_min_confidence),
            margin_threshold=0.0 if open_set_margin is None else float(open_set_margin),
            unknown_label=unknown_label,
        )
        if open_set_enabled
        else []
    )
    return {
        "label": best_label,
        "confidence": float(confidence),
        "second_confidence": float(second_confidence),
        "unknown_confidence": float(unknown_confidence),
        "margin": float(margin),
        "label_scores": label_scores,
        "rejection_reasons": rejection_reasons,
    }


def clip_score_labels_ensemble_batch(
    runtime: Any,
    images: List[Image.Image],
    labels: List[str],
    *,
    label_type: str = "generic",
    num_prompts: Optional[int] = None,
    image_batch_size: Optional[int] = None,
) -> List[Tuple[str, float, Dict[str, float]]]:
    """Batch ensemble scoring for many images against a shared label set."""
    if not images:
        return []
    if not labels:
        return [("unknown", 0.0, {}) for _ in images]

    text_prompts, prompt_to_class = _build_limited_prompt_batch(
        labels=labels,
        label_type=label_type,
        vlm_config=runtime.vlm_config,
        num_prompts=num_prompts,
    )
    if not text_prompts:
        return [("unknown", 0.0, {}) for _ in images]
    shared_image_embeddings: Optional[torch.Tensor] = None
    if runtime.bioclip_backend == "open_clip":
        shared_image_embeddings = ensure_open_clip_image_embeddings(
            runtime,
            images,
            image_batch_size=image_batch_size,
        )

    batch_scores = _score_prompt_probabilities(
        runtime,
        text_prompts,
        images=images,
        image_batch_size=image_batch_size,
        image_embeddings=shared_image_embeddings,
    )
    class_probabilities = _aggregate_batch_prompt_probabilities(
        batch_scores,
        prompt_to_class=prompt_to_class,
        class_count=len(labels),
    )
    per_image_scores = _batch_class_probabilities_to_label_scores(
        class_probabilities,
        labels=labels,
    )
    results: List[Tuple[str, float, Dict[str, float]]] = []
    for label_avg_scores in per_image_scores:
        best_label, best_score = _best_label_and_score(label_avg_scores)
        results.append((best_label, best_score, label_avg_scores))
    return results


def _log_open_set_debug(
    *,
    label_type: str,
    labels: List[str],
    known_probs: torch.Tensor,
    known_class_count: int,
    unknown_confidence: float,
    class_index: int,
    confidence: float,
    second_confidence: torch.Tensor,
    margin: float,
    open_set_min_confidence: float,
    open_set_margin: float,
    rejection_reasons: List[str],
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    debug_info = {
        "label_type": label_type,
        "known_labels": labels,
        "known_class_probabilities": {
            labels[i]: float(known_probs[0, i].item()) if i < len(labels) else None
            for i in range(known_class_count)
        },
        "unknown_probability": unknown_confidence,
        "best_known_label": labels[class_index] if class_index < len(labels) else "unknown",
        "best_known_confidence": confidence,
        "second_known_confidence": float(second_confidence.item()),
        "margin_best_vs_second": margin,
        "threshold_min_confidence": open_set_min_confidence,
        "threshold_margin": open_set_margin,
        "rejection_reasons": list(rejection_reasons),
    }
    status = "REJECTED as unknown" if rejection_reasons else f"ACCEPTED {debug_info['best_known_label']}"
    logger.debug("[OPEN-SET] %s: %s", status, json.dumps(debug_info, indent=2, default=str))


def clip_score_labels(
    runtime: Any,
    image: Image.Image,
    labels: List[str],
    *,
    label_type: str = "generic",
) -> Tuple[str, float]:
    """Score text labels against image using CLIP/BioCLIP."""
    outcome = clip_score_labels_open_set(
        runtime,
        image,
        labels,
        label_type=label_type,
        open_set_enabled=bool(runtime.open_set_enabled and label_type == "crop"),
        open_set_min_confidence=getattr(runtime, "open_set_min_confidence", 0.0),
        open_set_margin=getattr(runtime, "open_set_margin", 0.0),
        unknown_label="unknown",
    )
    if bool(runtime.open_set_enabled and label_type == "crop"):
        known_probs = torch.tensor(
            [[float(outcome["label_scores"].get(label, 0.0)) for label in labels]],
            dtype=torch.float32,
        )
        try:
            class_index = labels.index(str(outcome["label"]))
        except ValueError:
            class_index = 0
        _log_open_set_debug(
            label_type=label_type,
            labels=labels,
            known_probs=known_probs,
            known_class_count=len(labels),
            unknown_confidence=float(outcome["unknown_confidence"]),
            class_index=class_index,
            confidence=float(outcome["confidence"]),
            second_confidence=torch.tensor([float(outcome["second_confidence"])], dtype=torch.float32),
            margin=float(outcome["margin"]),
            open_set_min_confidence=float(getattr(runtime, "open_set_min_confidence", 0.0)),
            open_set_margin=float(getattr(runtime, "open_set_margin", 0.0)),
            rejection_reasons=list(outcome["rejection_reasons"]),
        )
        if outcome["rejection_reasons"]:
            return "unknown", max(float(outcome["unknown_confidence"]), float(outcome["confidence"]))
    return str(outcome["label"]), float(outcome["confidence"])
