#!/usr/bin/env python3
"""Thin entrypoint for Colab router-only crop and part identification."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from PIL import Image

from src.core.config_manager import get_config
from src.pipeline.inference_payloads import build_router_skipped_analysis
from src.router.router_pipeline import RouterPipeline
from src.shared.contracts import RouterAnalysisResult

StatusPrinter = Callable[[str], None]
RouterCacheKey = tuple[str, str]

_ROUTER_SESSION_CACHE: dict[RouterCacheKey, RouterPipeline] = {}
_IMAGE_SESSION_CACHE: "OrderedDict[Tuple[str, int, int, Optional[int]], Image.Image]" = OrderedDict()
_MAX_IMAGE_SESSION_CACHE_ITEMS = 6


def _coerce_optional_max_side(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _prepare_router_image(
    image_path: str | Path,
    *,
    max_image_side: Optional[int],
    status_printer: Optional[StatusPrinter],
) -> Image.Image:
    clamped_side = _coerce_optional_max_side(max_image_side)
    cache_key: Optional[Tuple[str, int, int, Optional[int]]] = None
    image_path_obj = Path(image_path)
    try:
        stat_result = image_path_obj.stat()
        cache_key = (
            str(image_path_obj.resolve()),
            int(stat_result.st_mtime_ns),
            int(stat_result.st_size),
            clamped_side,
        )
    except Exception:
        cache_key = None

    if cache_key is not None:
        cached_image = _IMAGE_SESSION_CACHE.get(cache_key)
        if cached_image is not None:
            copied = cached_image.copy() if hasattr(cached_image, "copy") else cached_image
            return copied

    image = Image.open(image_path).convert("RGB")
    if clamped_side is None:
        final_image = image
    else:
        width, height = image.size
        longest_edge = max(width, height)
        if longest_edge <= clamped_side:
            final_image = image
        else:
            scale = float(clamped_side) / float(longest_edge)
            resized_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
            _emit_status(
                status_printer,
                f"[INFER] Downscaled image from {width}x{height} to {resized_size[0]}x{resized_size[1]} for faster router inference.",
            )
            final_image = image.resize(resized_size, resample=resampling)

    if cache_key is not None:
        _IMAGE_SESSION_CACHE[cache_key] = final_image.copy() if hasattr(final_image, "copy") else final_image
        while len(_IMAGE_SESSION_CACHE) > _MAX_IMAGE_SESSION_CACHE_ITEMS:
            _IMAGE_SESSION_CACHE.popitem(last=False)

    return final_image


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        import logging
        logging.debug(f'Coercing {value!r} to float failed; using default {default}')
        return float(default)


def _emit_status(status_printer: Optional[StatusPrinter], message: str) -> None:
    if status_printer is not None:
        status_printer(str(message))


def _build_router_payload(analysis: RouterAnalysisResult) -> Dict[str, Any]:
    detection = analysis.primary_detection
    return {
        "status": str(analysis.status or "ok"),
        "crop": None if detection is None else str(detection.crop or "unknown"),
        "part": None if detection is None else str(detection.part or "unknown"),
        "router_confidence": 0.0 if detection is None else float(detection.crop_confidence),
        "message": str(analysis.message or ""),
        "router": analysis.to_summary_dict(),
        "router_details": analysis.to_dict(),
    }


def _resolve_adapter_target(
    crop_name: Optional[str],
    part_name: Optional[str],
    *,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_crop = str(crop_name or "").strip().lower()
    normalized_part = str(part_name or "").strip().lower() or "unspecified"
    if not normalized_crop or normalized_crop == "unknown":
        return {
            "crop": None,
            "part": None,
            "adapter_dir": None,
            "exists": False,
        }
    inference_cfg = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}
    adapter_root = Path(str(inference_cfg.get("adapter_root", "models/adapters")))
    adapter_dir = adapter_root / normalized_crop / normalized_part / "continual_sd_lora_adapter"
    if not adapter_dir.exists():
        legacy_adapter_dir = adapter_root / normalized_crop / "continual_sd_lora_adapter"
        adapter_dir = legacy_adapter_dir if legacy_adapter_dir.exists() else adapter_dir
    return {
        "crop": normalized_crop,
        "part": normalized_part,
        "adapter_dir": str(adapter_dir),
        "exists": adapter_dir.exists(),
    }


def _extract_top_crop_candidates(analysis: RouterAnalysisResult, *, top_candidates: int) -> list[Dict[str, Any]]:
    best_by_crop: Dict[str, Dict[str, Any]] = {}
    for detection in analysis.detections:
        crop_name = str(detection.crop or "unknown").strip().lower() or "unknown"
        candidate = {
            "crop": crop_name,
            "part": str(detection.part or "unknown").strip().lower() or "unknown",
            "crop_confidence": float(detection.crop_confidence),
            "part_confidence": float(detection.part_confidence),
            "quality_score": None if detection.quality_score is None else float(detection.quality_score),
        }
        existing = best_by_crop.get(crop_name)
        if existing is None or candidate["crop_confidence"] > float(existing.get("crop_confidence", 0.0)):
            best_by_crop[crop_name] = candidate

    ordered = sorted(
        best_by_crop.values(),
        key=lambda item: float(item.get("crop_confidence", 0.0)),
        reverse=True,
    )
    return ordered[: max(1, int(top_candidates))]


def _build_router_diagnostics(
    analysis: RouterAnalysisResult,
    *,
    top_candidates: int,
) -> Dict[str, Any]:
    top_crop_candidates = _extract_top_crop_candidates(analysis, top_candidates=top_candidates)
    primary = analysis.primary_detection
    primary_payload = {} if primary is None else primary.to_dict()

    best_confidence = _coerce_float(primary_payload.get("crop_confidence"), 0.0)
    second_confidence = 0.0
    if len(top_crop_candidates) > 1:
        second_confidence = _coerce_float(top_crop_candidates[1].get("crop_confidence"), 0.0)

    return {
        "top_crop_candidates": top_crop_candidates,
        "crop_confidence_margin": round(best_confidence - second_confidence, 4),
        "raw_part_label": str(primary_payload.get("raw_part_label", "") or ""),
        "raw_part_confidence": _coerce_float(primary_payload.get("raw_part_confidence"), 0.0),
        "part_unknown_confidence": _coerce_float(primary_payload.get("part_unknown_confidence"), 0.0),
        "part_rejection_reason": str(primary_payload.get("part_rejection_reason", "") or ""),
    }


def _router_cache_key(*, config_env: Optional[str], device: str) -> RouterCacheKey:
    return (str(config_env or ""), str(device or "cuda").strip().lower())


def clear_router_cache() -> None:
    """Drop any router instance cached for this Python session."""
    _ROUTER_SESSION_CACHE.clear()
    _IMAGE_SESSION_CACHE.clear()


def ensure_router_ready(
    *,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    status_printer: Optional[StatusPrinter] = None,
    reuse_cached: bool = True,
    runtime_profile: Optional[str] = None,
) -> RouterPipeline:
    cache_key = _router_cache_key(config_env=config_env, device=device)
    cached_router = _ROUTER_SESSION_CACHE.get(cache_key) if reuse_cached else None
    if cached_router is not None:
        if runtime_profile and hasattr(cached_router, "set_runtime_profile"):
            cached_profile = str(getattr(cached_router, "active_profile", "") or "")
            if cached_profile != str(runtime_profile).strip():
                cached_router.set_runtime_profile(str(runtime_profile).strip())
                _emit_status(
                    status_printer,
                    f"[ROUTER] Applied runtime profile={str(runtime_profile).strip()} on cached router.",
                )
        if cached_router.is_ready():
            _emit_status(status_printer, f"[ROUTER] Reusing cached models on {device}.")
            _emit_status(status_printer, "[ROUTER] Ready.")
            return cached_router
        _ROUTER_SESSION_CACHE.pop(cache_key, None)

    _emit_status(status_printer, f"[ROUTER] Loading models on {device}...")
    config = get_config(environment=config_env)
    router = RouterPipeline(config=config, device=device)
    if runtime_profile and hasattr(router, "set_runtime_profile"):
        router.set_runtime_profile(str(runtime_profile).strip())
        _emit_status(status_printer, f"[ROUTER] Applied runtime profile={str(runtime_profile).strip()}.")
    router.load_models()
    if not router.is_ready():
        raise RuntimeError(
            "Router models failed to become ready for inference. "
            "Check router.vlm.enabled, model availability, and router dependency installation."
        )

    if reuse_cached:
        _ROUTER_SESSION_CACHE[cache_key] = router
    _emit_status(status_printer, "[ROUTER] Ready.")
    return router


def run_inference(
    image_path: str | Path,
    *,
    config_env: Optional[str] = "colab",
    crop_hint: Optional[str] = None,
    part_hint: Optional[str] = None,
    trust_crop_hint: bool = False,
    device: str = "cuda",
    status_printer: Optional[StatusPrinter] = None,
    reuse_router: bool = True,
    include_diagnostics: bool = False,
    top_candidates: int = 3,
    runtime_profile: Optional[str] = None,
    max_image_side: Optional[int] = None,
    include_adapter_target: bool = True,
) -> Dict[str, Any]:
    image_ref = Path(image_path)
    _emit_status(status_printer, f"[INFER] image={image_ref.name} device={device}")
    image = _prepare_router_image(
        image_path,
        max_image_side=max_image_side,
        status_printer=status_printer,
    )

    if crop_hint and trust_crop_hint:
        crop_name = str(crop_hint).strip().lower()
        part_name = str(part_hint).strip().lower() if part_hint else None
        _emit_status(
            status_printer,
            f"[ROUTER] Trusted hint; skipped router crop={crop_name} part={part_name or 'unknown'}",
        )
        analysis = build_router_skipped_analysis(
            crop_name=crop_name,
            part_name=part_name,
            router_confidence=1.0,
            status="trusted_hint_skipped",
            message="Router skipped because trust_crop_hint=True.",
        )
        payload = _build_router_payload(analysis)
        payload["runtime_profile"] = str(runtime_profile or "")
        if include_adapter_target:
            config = get_config(environment=config_env)
            payload["adapter_target"] = _resolve_adapter_target(payload.get("crop"), payload.get("part"), config=config)
        if include_diagnostics:
            payload["diagnostics"] = _build_router_diagnostics(
                analysis,
                top_candidates=top_candidates,
            )
        _emit_status(
            status_printer,
            f"[RESULT] status={payload['status']} crop={payload['crop'] or 'unknown'} "
            f"part={payload['part'] or 'unknown'} router_confidence={float(payload['router_confidence']):.3f}",
        )
        return payload

    router = ensure_router_ready(
        config_env=config_env,
        device=device,
        status_printer=status_printer,
        reuse_cached=reuse_router,
        runtime_profile=runtime_profile,
    )

    analysis = router.analyze_image_result(image)
    payload = _build_router_payload(analysis)
    if crop_hint:
        hinted_crop = str(crop_hint or "").strip().lower()
        detected_crop = str(payload.get("crop") or "").strip().lower()
        if not detected_crop or detected_crop == "unknown":
            payload["status"] = "router_uncertain"
            payload["message"] = (
                f"Router could not confirm untrusted crop_hint='{hinted_crop}'."
            )
        elif detected_crop != hinted_crop:
            payload["status"] = "router_uncertain"
            payload["message"] = (
                f"Router rejected untrusted crop_hint='{hinted_crop}' because it detected crop='{detected_crop}'."
            )
    payload["runtime_profile"] = str(getattr(router, "active_profile", runtime_profile or "") or "")
    if include_adapter_target:
        router_config = router.config if isinstance(getattr(router, "config", None), dict) else get_config(environment=config_env)
        payload["adapter_target"] = _resolve_adapter_target(payload.get("crop"), payload.get("part"), config=router_config)
    if include_diagnostics:
        payload["diagnostics"] = _build_router_diagnostics(
            analysis,
            top_candidates=top_candidates,
        )
    _emit_status(
        status_printer,
        f"[ROUTER] crop={payload['crop'] or 'unknown'} "
        f"part={payload['part'] or 'unknown'} "
        f"confidence={float(payload['router_confidence']):.3f}"
        + (f" message={payload['message']}" if payload["message"] else ""),
    )
    _emit_status(
        status_printer,
        f"[RESULT] status={payload['status']} crop={payload['crop'] or 'unknown'} "
        f"part={payload['part'] or 'unknown'} router_confidence={float(payload['router_confidence']):.3f}",
    )
    return payload


def run_inference_batch(
    image_paths: list[str | Path],
    *,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    status_printer: Optional[StatusPrinter] = None,
    reuse_router: bool = True,
    include_diagnostics: bool = False,
    top_candidates: int = 3,
    runtime_profile: Optional[str] = None,
    max_image_side: Optional[int] = None,
    include_adapter_target: bool = True,
) -> list[Dict[str, Any]]:
    """Run router-only crop/part identification for many images with one router batch."""
    if not image_paths:
        return []
    if include_diagnostics:
        return [
            run_inference(
                image_path,
                config_env=config_env,
                device=device,
                status_printer=status_printer,
                reuse_router=reuse_router,
                include_diagnostics=True,
                top_candidates=top_candidates,
                runtime_profile=runtime_profile,
                max_image_side=max_image_side,
                include_adapter_target=include_adapter_target,
            )
            for image_path in image_paths
        ]

    _emit_status(status_printer, f"[INFER] batch_size={len(image_paths)} device={device}")
    images = [
        _prepare_router_image(
            image_path,
            max_image_side=max_image_side,
            status_printer=status_printer,
        )
        for image_path in image_paths
    ]
    router = ensure_router_ready(
        config_env=config_env,
        device=device,
        status_printer=status_printer,
        reuse_cached=reuse_router,
        runtime_profile=runtime_profile,
    )
    analyses = router.analyze_images_result(images)
    payloads: list[Dict[str, Any]] = []
    router_config = router.config if isinstance(getattr(router, "config", None), dict) else get_config(environment=config_env)
    for image_path, analysis in zip(image_paths, analyses):
        payload = _build_router_payload(analysis)
        payload["runtime_profile"] = str(getattr(router, "active_profile", runtime_profile or "") or "")
        if include_adapter_target:
            payload["adapter_target"] = _resolve_adapter_target(payload.get("crop"), payload.get("part"), config=router_config)
        payloads.append(payload)
        _emit_status(
            status_printer,
            f"[ROUTER] image={Path(image_path).name} crop={payload['crop'] or 'unknown'} "
            f"part={payload['part'] or 'unknown'} confidence={float(payload['router_confidence']):.3f}",
        )
    if len(payloads) != len(image_paths):
        raise RuntimeError(f"Router batch returned {len(payloads)} results for {len(image_paths)} images.")
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description="Run router-only crop and part identification for a single image.")
    parser.add_argument("image", type=Path, help="Image path")
    parser.add_argument("--config-env", default="colab", help="Config environment override (default: colab)")
    parser.add_argument("--crop", dest="crop_hint", help="Optional untrusted crop hint")
    parser.add_argument("--part", dest="part_hint", help="Optional part hint")
    parser.add_argument("--trust-crop-hint", action="store_true", help="Trust --crop and skip router execution")
    parser.add_argument("--device", default="cuda", help="Torch device preference")
    parser.add_argument(
        "--runtime-profile",
        help="Optional router runtime profile override (for example: balanced or fast).",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Include router diagnostics (top candidates, confidence margin, rejection evidence).",
    )
    parser.add_argument(
        "--top-candidates",
        default=3,
        type=int,
        help="Number of crop candidates to include when --diagnostics is enabled.",
    )
    args = parser.parse_args()

    result = run_inference(
        args.image,
        config_env=args.config_env,
        crop_hint=args.crop_hint,
        part_hint=args.part_hint,
        trust_crop_hint=bool(args.trust_crop_hint),
        device=args.device,
        include_diagnostics=bool(args.diagnostics),
        top_candidates=int(args.top_candidates),
        runtime_profile=args.runtime_profile,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
