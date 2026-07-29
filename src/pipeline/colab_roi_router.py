"""Crop/part handoff and Grounding DINO fallback for the Colab ROI surface."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

from PIL import Image

StatusPrinter = Callable[[str], None]
GroundingDinoRunner = Callable[..., Dict[str, Any]]

ADAPTER_ALLOWED_ROUTER_STATUSES = {"ok", "trusted_hint_skipped", "skipped"}


DEFAULT_TARGET_ROI_BACKEND = "router_then_grounding_dino"


DEFAULT_GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


DEFAULT_GROUNDING_DINO_PROMPTS = (
    "tomato fruit.",
    "a tomato fruit.",
    "tomato fruits.",
    "fruit on tomato plant.",
    "tomato on plant.",
    "tomatoes.",
)


DEFAULT_GROUNDING_DINO_BOX_THRESHOLD = 0.15


DEFAULT_GROUNDING_DINO_TEXT_THRESHOLD = 0.10


DEFAULT_GROUNDING_DINO_MAX_CANDIDATES = 5


_GROUNDING_DINO_SESSION_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def _emit_status(status_printer: Optional[StatusPrinter], message: str) -> None:
    if status_printer is not None:
        status_printer(str(message))


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _router_payload(router_result: Dict[str, Any]) -> Dict[str, Any]:
    payload = router_result.get("router")
    return dict(payload) if isinstance(payload, dict) else {}


def _primary_detection(router_result: Dict[str, Any]) -> Dict[str, Any]:
    router_payload = _router_payload(router_result)
    primary = router_payload.get("primary_detection")
    if isinstance(primary, dict):
        return dict(primary)
    details = router_result.get("router_details")
    if isinstance(details, dict) and isinstance(details.get("primary_detection"), dict):
        return dict(details["primary_detection"])
    return {}


def _router_detections(router_result: Dict[str, Any]) -> list[Dict[str, Any]]:
    details = router_result.get("router_details")
    detections = details.get("detections") if isinstance(details, dict) else None
    if not isinstance(detections, list):
        router_payload = _router_payload(router_result)
        detections = router_payload.get("detections")
    normalized = [dict(item) for item in list(detections or []) if isinstance(item, dict)]
    primary = _primary_detection(router_result)
    if primary and not normalized:
        normalized.append(primary)
    return normalized


def _detection_confidence(detection: Dict[str, Any]) -> float:
    return float(detection.get("crop_confidence", detection.get("confidence", 0.0)) or 0.0)


def _detection_sort_key(detection: Dict[str, Any]) -> tuple[float, float, float]:
    quality = detection.get("quality_score")
    quality_score = float("-inf") if quality is None else float(quality)
    return (
        quality_score,
        _detection_confidence(detection),
        float(detection.get("part_confidence", 0.0) or 0.0),
    )


def _format_grounding_prompt(prompt: str) -> str:
    formatted = " ".join(str(prompt or "").strip().lower().split())
    if formatted and not formatted.endswith("."):
        formatted = f"{formatted}."
    return formatted


def _normalize_grounding_prompts(prompts: Optional[Sequence[str]]) -> tuple[str, ...]:
    normalized = tuple(_format_grounding_prompt(prompt) for prompt in (prompts or DEFAULT_GROUNDING_DINO_PROMPTS))
    return tuple(prompt for prompt in normalized if prompt)


def build_grounding_dino_prompts(crop: str, part: str) -> tuple[str, ...]:
    """Build lowercase, dot-terminated Grounding DINO prompts for one crop/part target."""
    crop_name = _normalize_text(crop)
    part_name = _normalize_text(part)
    prompts: list[str] = []
    if crop_name and part_name:
        prompts.extend(
            [
                f"{crop_name} {part_name}",
                f"a {crop_name} {part_name}",
                f"{crop_name} {part_name}s",
                f"{part_name} on {crop_name} plant",
                f"{crop_name} plant {part_name}",
            ]
        )
    if crop_name:
        prompts.append(f"{crop_name} plant")
    if part_name:
        prompts.append(part_name)
    return _normalize_grounding_prompts(prompts)


def _batch_input_ids(inputs: Any) -> Any:
    if hasattr(inputs, "input_ids"):
        return inputs.input_ids
    if isinstance(inputs, dict):
        return inputs.get("input_ids")
    return None


def _to_device_batch(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    return inputs


def _score_to_float(score: Any) -> float:
    if hasattr(score, "detach"):
        return float(score.detach().cpu().item())
    return float(score)


def _box_to_float_list(box: Any) -> list[float]:
    if hasattr(box, "detach"):
        box = box.detach().cpu().tolist()
    return [float(value) for value in box]


def _post_process_grounding_dino(
    processor: Any,
    outputs: Any,
    *,
    input_ids: Any,
    box_threshold: float,
    text_threshold: float,
    target_sizes: list[tuple[int, int]],
) -> Any:
    try:
        return processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            target_sizes=target_sizes,
        )
    except TypeError:
        return processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            target_sizes=target_sizes,
        )


def _grounding_dino_cache_key(*, model_id: str, device: str) -> tuple[str, str]:
    return (str(model_id).strip(), str(device or "cuda").strip().lower())


def clear_grounding_dino_cache() -> None:
    """Drop cached Grounding DINO components for the current Python session."""
    _GROUNDING_DINO_SESSION_CACHE.clear()


def _load_grounding_dino_components(
    *,
    model_id: str = DEFAULT_GROUNDING_DINO_MODEL_ID,
    device: str = "cuda",
    status_printer: Optional[StatusPrinter] = None,
) -> tuple[Any, Any]:
    cache_key = _grounding_dino_cache_key(model_id=model_id, device=device)
    cached = _GROUNDING_DINO_SESSION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _emit_status(status_printer, f"[GROUNDING_DINO] Loading model={model_id} device={device}...")
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()
    _GROUNDING_DINO_SESSION_CACHE[cache_key] = (processor, model)
    _emit_status(status_printer, "[GROUNDING_DINO] Ready.")
    return processor, model


def run_grounding_dino_target_detection(
    image: Image.Image,
    *,
    prompts: Optional[Sequence[str]] = None,
    model_id: str = DEFAULT_GROUNDING_DINO_MODEL_ID,
    device: str = "cuda",
    box_threshold: float = DEFAULT_GROUNDING_DINO_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_GROUNDING_DINO_TEXT_THRESHOLD,
    max_candidates: int = DEFAULT_GROUNDING_DINO_MAX_CANDIDATES,
    status_printer: Optional[StatusPrinter] = None,
) -> Dict[str, Any]:
    """Return the best Grounding DINO bbox for target ROI retrieval."""
    prompt_list = _normalize_grounding_prompts(prompts)
    if not prompt_list:
        return {"detections": [], "candidate_count": 0, "status": "no_prompts"}

    try:
        import torch

        processor, model = _load_grounding_dino_components(
            model_id=model_id,
            device=device,
            status_printer=status_printer,
        )
        detections: list[Dict[str, Any]] = []
        for prompt in prompt_list:
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = _to_device_batch(inputs, device)
            with torch.no_grad():
                outputs = model(**inputs)
            processed = _post_process_grounding_dino(
                processor,
                outputs,
                input_ids=_batch_input_ids(inputs),
                box_threshold=float(box_threshold),
                text_threshold=float(text_threshold),
                target_sizes=[image.size[::-1]],
            )
            for item in list(processed or []):
                boxes = item.get("boxes", [])
                scores = item.get("scores", [])
                labels = item.get("labels", [])
                for index, box in enumerate(list(boxes)):
                    score = _score_to_float(scores[index])
                    label = str(labels[index] if index < len(labels) else prompt)
                    bbox = _box_to_float_list(box)
                    detections.append(
                        {
                            "crop": "tomato",
                            "part": "fruit",
                            "crop_confidence": score,
                            "part_confidence": score,
                            "bbox": bbox,
                            "prompt": prompt,
                            "label": label,
                            "source": "grounding_dino",
                        }
                    )
        detections.sort(key=_detection_sort_key, reverse=True)
        kept = detections[: max(1, int(max_candidates))]
        status = "ok" if detections else "no_candidates"
        return {"detections": kept, "candidate_count": len(detections), "status": status}
    except Exception as exc:
        _emit_status(status_printer, f"[GROUNDING_DINO] skipped: {exc}")
        return {"detections": [], "candidate_count": 0, "status": "error", "error": str(exc)}


def resolve_router_handoff(router_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the adapter handoff fields from a router result."""
    primary = _primary_detection(router_result)
    crop = _normalize_text(router_result.get("crop") or primary.get("crop"))
    part = _normalize_text(router_result.get("part") or primary.get("part"))
    status = _normalize_text(router_result.get("status") or _router_payload(router_result).get("status"))
    return {
        "status": status,
        "crop": crop or None,
        "part": part or None,
        "router_confidence": float(router_result.get("router_confidence", primary.get("crop_confidence", 0.0)) or 0.0),
        "bbox": primary.get("bbox"),
        "primary_detection": primary,
        "adapter_allowed": (
            status in ADAPTER_ALLOWED_ROUTER_STATUSES
            and bool(crop)
            and crop != "unknown"
            and bool(part)
            and part != "unknown"
        ),
    }


def resolve_target_router_handoff(
    router_result: Dict[str, Any],
    *,
    target_crop: Optional[str],
    target_part: Optional[str],
    image: Optional[Image.Image] = None,
    target_roi_backend: str = DEFAULT_TARGET_ROI_BACKEND,
    grounding_dino_runner: Optional[GroundingDinoRunner] = None,
    grounding_dino_model_id: str = DEFAULT_GROUNDING_DINO_MODEL_ID,
    grounding_dino_prompts: Optional[Sequence[str]] = None,
    grounding_dino_box_threshold: float = DEFAULT_GROUNDING_DINO_BOX_THRESHOLD,
    grounding_dino_text_threshold: float = DEFAULT_GROUNDING_DINO_TEXT_THRESHOLD,
    grounding_dino_max_candidates: int = DEFAULT_GROUNDING_DINO_MAX_CANDIDATES,
    device: str = "cuda",
    status_printer: Optional[StatusPrinter] = None,
) -> Dict[str, Any]:
    """Prefer a router detection that matches the adapter target crop/part."""
    handoff = resolve_router_handoff(router_result)
    normalized_crop = _normalize_text(target_crop)
    normalized_part = _normalize_text(target_part)
    normalized_backend = _normalize_text(target_roi_backend) or "router_detections"
    primary = dict(handoff.get("primary_detection") or {})
    handoff.update(
        {
            "primary_crop": _normalize_text(primary.get("crop")) or handoff.get("crop"),
            "primary_part": _normalize_text(primary.get("part")) or handoff.get("part"),
            "primary_bbox": primary.get("bbox"),
            "target_crop": normalized_crop or None,
            "target_part": normalized_part or None,
            "target_roi_backend": normalized_backend,
            "target_prompt": None,
            "target_detection_confidence": None,
            "target_detection_source": "target_detection_missing",
            "grounding_dino_candidate_count": 0,
            "grounding_dino_status": "",
            "grounding_dino_error": "",
            "target_detection_found": False,
            "selected_detection_source": "primary_detection",
        }
    )
    if not normalized_crop:
        return handoff

    matches: list[Dict[str, Any]] = []
    for detection in _router_detections(router_result):
        detection_crop = _normalize_text(detection.get("crop"))
        detection_part = _normalize_text(detection.get("part"))
        if detection_crop != normalized_crop:
            continue
        if normalized_part and detection_part != normalized_part:
            continue
        matches.append(detection)
    if not matches:
        if normalized_backend in {"grounding_dino", "router_then_grounding_dino"} and image is not None:
            runner = grounding_dino_runner or run_grounding_dino_target_detection
            grounding_payload = runner(
                image,
                prompts=grounding_dino_prompts,
                model_id=grounding_dino_model_id,
                device=device,
                box_threshold=grounding_dino_box_threshold,
                text_threshold=grounding_dino_text_threshold,
                max_candidates=grounding_dino_max_candidates,
                status_printer=status_printer,
            )
            grounding_detections = [
                dict(item) for item in list((grounding_payload or {}).get("detections") or []) if isinstance(item, dict)
            ]
            handoff["grounding_dino_candidate_count"] = int((grounding_payload or {}).get("candidate_count", 0) or 0)
            handoff["grounding_dino_status"] = str((grounding_payload or {}).get("status") or "")
            handoff["grounding_dino_error"] = str((grounding_payload or {}).get("error") or "")
            if grounding_detections:
                selected = max(grounding_detections, key=_detection_sort_key)
                selected_crop = normalized_crop
                selected_part = normalized_part or _normalize_text(selected.get("part")) or None
                confidence = _detection_confidence(selected)
                handoff.update(
                    {
                        "crop": selected_crop,
                        "part": selected_part,
                        "router_confidence": confidence,
                        "bbox": selected.get("bbox"),
                        "primary_detection": selected,
                        "adapter_allowed": bool(selected_crop and selected_part and selected_part != "unknown"),
                        "target_prompt": selected.get("prompt"),
                        "target_detection_confidence": confidence,
                        "target_detection_source": "grounding_dino",
                        "target_detection_found": True,
                        "selected_detection_source": "grounding_dino",
                    }
                )
                return handoff
        handoff.update(
            {
                "crop": normalized_crop,
                "part": normalized_part or handoff.get("part"),
                "bbox": None,
                "adapter_allowed": bool(normalized_crop and normalized_part),
                "selected_detection_source": "target_detection_missing",
            }
        )
        return handoff

    selected = max(matches, key=_detection_sort_key)
    selected_crop = _normalize_text(selected.get("crop")) or normalized_crop
    selected_part = _normalize_text(selected.get("part")) or normalized_part or None
    handoff.update(
        {
            "crop": selected_crop,
            "part": selected_part,
            "router_confidence": _detection_confidence(selected),
            "bbox": selected.get("bbox"),
            "primary_detection": selected,
            "adapter_allowed": bool(selected_crop and selected_part and selected_part != "unknown"),
            "target_prompt": selected.get("prompt"),
            "target_detection_confidence": _detection_confidence(selected),
            "target_detection_source": "router_detection",
            "target_detection_found": True,
            "selected_detection_source": "router_detection",
        }
    )
    return handoff
