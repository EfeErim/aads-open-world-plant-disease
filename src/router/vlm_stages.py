from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from src.pipeline.inference_payloads import best_detection_from_analysis
from src.router import sam3_runtime
from src.router.runtime_surface import RouterRequestOptions


def analysis_to_batch_item(analysis: Any) -> Tuple[Dict[str, Any], float]:
    """Map analysis payload to route_batch item contract (crop dict + confidence)."""
    detection = best_detection_from_analysis(analysis)
    if detection:
        return (
            {
                'crop': detection.get('crop', 'unknown'),
                'part': detection.get('part', 'unknown'),
                'bbox': detection.get('bbox', [0, 0, 0, 0]),
            },
            detection.get('crop_confidence', 0.0),
        )

    return ({'crop': 'unknown', 'part': 'unknown', 'bbox': [0, 0, 0, 0]}, 0.0)

_SUPPORTED_TRANSFORMERS_MIN = (5, 1, 0)
_SUPPORTED_TRANSFORMERS_MAX_EXCLUSIVE = (5, 2, 0)
_SUPPORTED_TRANSFORMERS_RANGE = ">=5.1.0,<5.2.0"


def _parse_transformers_version(version: str) -> tuple[int, int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?", str(version))
    if match is None:
        raise ValueError(f"Unrecognized transformers version: {version!r}")
    major, minor, patch = match.groups(default="0")
    return int(major), int(minor), int(patch)


def check_vlm_dependencies(
    import_module: Callable[[str], Any] = __import__,
    transformers_version_provider: Optional[Callable[[], str]] = None,
) -> Dict[str, Any]:
    """Collect dependency diagnostics for VLM runtime without direct logging side effects."""
    result: Dict[str, Any] = {
        "transformers_warning": None,
        "missing_deps": [],
        "install_command": None,
    }

    try:
        version = (
            transformers_version_provider()
            if transformers_version_provider is not None
            else import_module("transformers").__version__
        )
        parsed_version = _parse_transformers_version(version)
        if not (_SUPPORTED_TRANSFORMERS_MIN <= parsed_version < _SUPPORTED_TRANSFORMERS_MAX_EXCLUSIVE):
            result["transformers_warning"] = (
                f"transformers {version} is outside the repo's validated range {_SUPPORTED_TRANSFORMERS_RANGE}. "
                "DINOv3 landed in Transformers 4.56.0+, but the maintained baseline is the latest validated 5.1.x "
                "line for the repo's DINOv3 + SAM3 surface. Install: !pip install 'transformers~=5.1.0'"
            )
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        result["transformers_warning"] = f"Could not check transformers version: {exc}"

    optional_packages = {
        "open_clip": "open-clip-torch",
    }
    missing_deps = []
    for package_name, pip_name in optional_packages.items():
        try:
            import_module(package_name)
        except ImportError:
            missing_deps.append(pip_name)

    result["missing_deps"] = missing_deps
    if missing_deps:
        result["install_command"] = f"!pip install {' '.join(missing_deps)}"

    return result

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineSurfaceConfig:
    router_config: Dict[str, Any]
    vlm_config: Dict[str, Any]
    base_vlm_config: Dict[str, Any]
    crop_mapping: Dict[str, Any]


def build_pipeline_surface_config(config: Dict[str, Any]) -> PipelineSurfaceConfig:
    router_config = config.get("router", {}) if isinstance(config.get("router"), dict) else {}
    raw_vlm_config = router_config.get("vlm", {}) if isinstance(router_config, dict) else {}
    vlm_config = copy.deepcopy(raw_vlm_config) if isinstance(raw_vlm_config, dict) else {}
    crop_mapping = router_config.get("crop_mapping", {}) if isinstance(router_config, dict) else {}
    crop_mapping = dict(crop_mapping) if isinstance(crop_mapping, dict) else {}
    return PipelineSurfaceConfig(
        router_config=dict(router_config),
        vlm_config=vlm_config,
        base_vlm_config=copy.deepcopy(vlm_config),
        crop_mapping=crop_mapping,
    )


def load_models(runtime: Any) -> None:
    logger.info("Loading VLM models...")

    if not runtime.enabled:
        logger.info("VLM pipeline is disabled; skipping model loading")
        runtime.models_loaded = False
        return

    diagnostics = check_vlm_dependencies()
    transformers_warning = diagnostics.get("transformers_warning")
    if transformers_warning:
        logger.warning(transformers_warning)

    missing_deps = diagnostics.get("missing_deps", [])
    if missing_deps:
        logger.warning("Missing optional dependencies: %s", ", ".join(missing_deps))
        install_cmd = diagnostics.get("install_command")
        if install_cmd:
            logger.warning("Install in Colab cell: %s", install_cmd)

    try:
        if runtime.model_source != "huggingface":
            raise ValueError(
                f"Unsupported VLM model_source '{runtime.model_source}'. Currently supported: 'huggingface'"
            )

        logger.info("Loading SAM3 + BioCLIP-2.5 pipeline...")
        logger.info("Note: First run downloads ~1-2 GB. This may take 2-5 minutes...")
        load_sam3_bioclip25(runtime)
        runtime.actual_pipeline = "sam3"
        runtime.models_loaded = True
        logger.info("SAM3 + BioCLIP-2.5 loaded successfully (pipeline=%s)", runtime.actual_pipeline)
    except Exception as exc:
        runtime.models_loaded = False
        if runtime.strict_model_loading:
            raise RuntimeError(f"Strict VLM model loading failed: {exc}") from exc
        logger.warning("SAM3 model loading failed. Models remain unloaded: %s", exc)


def load_sam3_bioclip25(runtime: Any) -> None:
    sam_id = str(runtime.model_ids.get("sam", ""))
    bioclip_id = str(runtime.model_ids.get("bioclip", ""))

    if sam_id.startswith("fake-") or bioclip_id.startswith("fake-"):
        logger.info("Using helper loaders for fake model ids")
        sam_loader = getattr(runtime, "_load_sam", None)
        clip_loader = getattr(runtime, "_load_clip_like_model", None)
        def _load_sam_via_helper(model_id: str):
            return load_sam(runtime, model_id)

        def _load_clip_via_helper(model_id: str):
            return load_clip_like_model(runtime, model_id)

        if not callable(sam_loader):
            sam_loader = _load_sam_via_helper
        if not callable(clip_loader):
            clip_loader = _load_clip_via_helper
        sam_processor, sam_model = sam_loader(sam_id)
        bioclip_processor, bioclip_model = clip_loader(bioclip_id)

        runtime._set_sam_runtime(sam_processor, sam_model, backend="sam3")
        runtime._set_bioclip_runtime(
            bioclip_processor,
            bioclip_model,
            backend=runtime.bioclip_backend or "transformers",
        )
        return

    import open_clip
    from transformers import Sam3Model, Sam3Processor

    logger.info("Loading SAM3...")
    sam3_processor = Sam3Processor.from_pretrained(sam_id)
    sam3_model = runtime._prepare_inference_model(Sam3Model.from_pretrained(sam_id))

    logger.info("Loading BioCLIP-2.5...")
    hub_model_id = f"hf-hub:{bioclip_id}"
    model, _, preprocess_val = open_clip.create_model_and_transforms(hub_model_id)
    tokenizer = open_clip.get_tokenizer(hub_model_id)
    model = runtime._prepare_inference_model(model)

    runtime._set_sam_runtime(sam3_processor, sam3_model, backend="sam3")
    runtime._set_bioclip_runtime(
        {
            "preprocess": preprocess_val,
            "tokenizer": tokenizer,
        },
        model,
        backend="open_clip",
    )


def load_sam(runtime: Any, model_id: str) -> Tuple[Any, Any]:
    sam2_requested = "sam2" in model_id.lower() or "hiera" in model_id.lower() or model_id.lower().endswith(".pt")
    if sam2_requested:
        try:
            import importlib

            ultralytics_module = importlib.import_module("ultralytics")
            sam_cls = getattr(ultralytics_module, "SAM")

            checkpoint = model_id
            if "/" in checkpoint and checkpoint.startswith("facebook/sam2"):
                checkpoint = runtime.vlm_config.get("sam2_checkpoint", "sam2_b.pt")

            model = sam_cls(checkpoint)
            runtime.sam_backend = "ultralytics"
            return {"backend": "ultralytics", "checkpoint": checkpoint}, model
        except Exception as exc:
            raise RuntimeError(
                "SAM-2 requires ultralytics in this pipeline configuration. "
                "Install ultralytics and ensure SAM-2 weights are accessible (e.g., checkpoint 'sam2_b.pt')."
            ) from exc

    from transformers import SamModel, SamProcessor

    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id)
    runtime.sam_backend = "transformers_sam"
    model = runtime._prepare_inference_model(model)
    return processor, model


def load_clip_like_model(runtime: Any, model_id: str) -> Tuple[Any, Any]:
    if "bioclip" in model_id.lower() or "imageomics" in model_id.lower():
        try:
            import open_clip

            hub_model_id = f"hf-hub:{model_id}"
            model, _, preprocess_val = open_clip.create_model_and_transforms(hub_model_id)
            tokenizer = open_clip.get_tokenizer(hub_model_id)
            model = runtime._prepare_inference_model(model)
            runtime.bioclip_backend = "open_clip"
            return {
                "preprocess": preprocess_val,
                "tokenizer": tokenizer,
            }, model
        except Exception as exc:
            logger.debug(
                f"Failed to load CLIP/BioCLIP model '{model_id}' via open_clip; will attempt transformers fallback",
                exc_info=exc,
            )

    try:
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id)
        model = runtime._prepare_inference_model(AutoModel.from_pretrained(model_id))
        runtime.bioclip_backend = "transformers"
        return processor, model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load CLIP/BioCLIP model '{model_id}' via both open_clip and transformers. "
            "For BioCLIP models, ensure open_clip_torch is installed."
        ) from exc


def analyze_batch_results(runtime: Any, batch: torch.Tensor, request: RouterRequestOptions) -> List[Any]:
    if runtime.enabled and runtime.models_loaded and runtime.actual_pipeline == "sam3":
        analyses = sam3_runtime.analyze_sam3_batch(
            runtime,
            batch,
            confidence_threshold=request.confidence_threshold,
            max_detections=request.max_detections,
        )
        return [runtime._normalize_router_analysis(analysis, request=request) for analysis in analyses]

    return [
        runtime.analyze_image_result(batch[index], options=request)
        for index in range(int(batch.shape[0]))
    ]


def route_batch_items(
    runtime: Any,
    batch: torch.Tensor,
    *,
    confidence_threshold: Optional[float] = None,
    max_detections: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    analyses = runtime.analyze_batch_result(
        batch,
        confidence_threshold=confidence_threshold,
        max_detections=max_detections,
    )

    crops_out: List[Dict[str, Any]] = []
    confs: List[float] = []
    for analysis in analyses:
        crop_item, confidence = analysis_to_batch_item(analysis)
        crops_out.append(crop_item)
        confs.append(confidence)
    return crops_out, confs
