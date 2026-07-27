#!/usr/bin/env python3
"""
VLM Pipeline for AADS-ULoRA
SAM3 + BioCLIP-2.5 only (fallback pipeline removed)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from PIL import Image

from src.router import clip_runtime, sam3_runtime
from src.router.label_normalization import normalize_part_label
from src.router.pipeline_flow_utils import build_process_image_response
from src.router.policy_taxonomy_utils import (
    apply_runtime_profile,
    build_policy_graph,
    load_crop_part_compatibility,
    load_taxonomy,
    policy_enabled,
    policy_value,
    resolve_requested_profile,
)
from src.router.roi_helpers import coerce_image_input
from src.router.runtime_surface import (
    RouterAnalyzer,
    coerce_bool,
    normalize_router_analysis_result,
    resolve_router_analyzer,
    resolve_runtime_controls,
)
from src.router.runtime_surface import (
    resolve_request_options as resolve_router_request_options,
)
from src.router.sam3_output_utils import (
    normalize_sam3_results,
    sam3_error_result,
)
from src.router.vlm_stages import (
    analyze_batch_results,
    build_pipeline_surface_config,
    route_batch_items,
)
from src.router.vlm_stages import (
    load_clip_like_model as load_clip_like_model_stage,
)
from src.router.vlm_stages import (
    load_models as load_models_stage,
)
from src.router.vlm_stages import (
    load_sam as load_sam_stage,
)
from src.router.vlm_stages import (
    load_sam3_bioclip25 as load_sam3_bioclip25_stage,
)
from src.shared.contracts import RouterAnalysisResult, RouterRequestOptions

logger = logging.getLogger(__name__)


def _normalize_crop_part_compatibility(crop_mapping: Dict[str, Any]) -> Dict[str, List[str]]:
    compatibility: Dict[str, List[str]] = {}
    for crop_name, crop_data in crop_mapping.items():
        if not isinstance(crop_data, dict):
            continue
        parts = crop_data.get('parts', [])
        if not isinstance(parts, list):
            continue
        normalized_crop = str(crop_name).strip().lower()
        normalized_parts: List[str] = []
        for part_name in parts:
            normalized_part = normalize_part_label(part_name)
            if normalized_part:
                normalized_parts.append(normalized_part)
        if normalized_crop and normalized_parts:
            compatibility[normalized_crop] = normalized_parts
    return compatibility


class RouterPipeline:
    """
     Unified VLM pipeline (SAM3 + BioCLIP-2.5 only).
    """

    def __init__(self, config: Dict, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        surface_config = build_pipeline_surface_config(config)
        self.vlm_config = surface_config.vlm_config
        self._base_vlm_config = surface_config.base_vlm_config
        self.active_profile: Optional[str] = None
        self.policy_graph: Dict[str, Dict[str, Any]] = {}
        self.set_runtime_profile(resolve_requested_profile(self.vlm_config), suppress_warning=True)

        # Pipeline mode is SAM3-only
        self.pipeline_mode = 'sam3'
        self.actual_pipeline: Optional[str] = None  # Will be set to 'sam3' after load_models

        self._refresh_runtime_controls()

        # Dynamic taxonomy support
        self.use_dynamic_taxonomy = coerce_bool(self.vlm_config.get('use_dynamic_taxonomy', False), default=False)
        self.taxonomy_path = self.vlm_config.get('taxonomy_path', 'config/plant_taxonomy.json')
        self.strict_taxonomy_loading = coerce_bool(
            self.vlm_config.get('strict_taxonomy_loading', self.use_dynamic_taxonomy),
            default=self.use_dynamic_taxonomy,
        )
        self.runtime_warnings: List[str] = []

        crop_mapping = surface_config.crop_mapping
        self.crop_labels: List[str] = []
        self.part_labels: List[str] = []
        self.crop_part_compatibility: Dict[str, List[str]] = {}
        self._initialize_taxonomy_surface(crop_mapping)

        # SAM model placeholders
        self.sam_model: Any = None
        self.sam_processor: Any = None

        # BioCLIP model placeholders (can be BioCLIP-2.5 or BioCLIP-2)
        self.bioclip: Any = None
        self.bioclip_processor: Any = None
        self._open_clip_text_embedding_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

        # Backend tracking
        self.sam_backend: Optional[str] = None
        self.bioclip_backend: Optional[str] = None
        self.models_loaded = False

        # Log GPU availability for debugging
        logger.info(f"RouterPipeline initialized on {self.device}")
        logger.info("Pipeline mode: sam3 (fallback disabled)")
        logger.info(
            "GPU available: %s, CUDA version: %s",
            torch.cuda.is_available(),
            torch.version.cuda if torch.cuda.is_available() else 'N/A',
        )
        if torch.cuda.is_available():
            logger.info(
                "GPU device: %s, Memory: %.1fGB",
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )

    def _initialize_taxonomy_surface(self, crop_mapping: Dict[str, Any]) -> None:
        taxonomy_available = self._try_load_taxonomy()
        if taxonomy_available:
            self._try_load_crop_part_compatibility()
        if not taxonomy_available:
            self._load_labels_from_config(crop_mapping)
        self._merge_configured_router_surface(crop_mapping)
        if not self.crop_part_compatibility:
            self.crop_part_compatibility = _normalize_crop_part_compatibility(crop_mapping)

    def _merge_configured_router_surface(self, crop_mapping: Dict[str, Any]) -> None:
        configured_compatibility = _normalize_crop_part_compatibility(crop_mapping)
        if not configured_compatibility:
            return

        part_labels_by_lower = {
            str(part).strip().lower(): part
            for part in self.part_labels
            if str(part).strip()
        }
        for crop_name, parts in configured_compatibility.items():
            for part_name in parts:
                if part_name not in part_labels_by_lower:
                    self.part_labels.append(part_name)
                    part_labels_by_lower[part_name] = part_name
            # Keep explicit router.crop_mapping authoritative for maintained crop-part surfaces.
            self.crop_part_compatibility[crop_name] = list(parts)

    def _try_load_taxonomy(self) -> bool:
        if not (self.use_dynamic_taxonomy or not self.vlm_config.get('crop_labels')):
            return False
        try:
            self.crop_labels, self.part_labels = load_taxonomy(self.taxonomy_path)
            logger.info(
                "Loaded taxonomy from %s: %d crops, %d parts",
                self.taxonomy_path,
                len(self.crop_labels),
                len(self.part_labels),
            )
            return True
        except Exception as e:
            warning_message = f"Failed to load taxonomy from {self.taxonomy_path}: {e}"
            if self.strict_taxonomy_loading:
                raise RuntimeError(warning_message) from e
            self._record_runtime_warning(warning_message)
            return False

    def _try_load_crop_part_compatibility(self) -> None:
        try:
            self.crop_part_compatibility = load_crop_part_compatibility(self.taxonomy_path)
            if self.crop_part_compatibility:
                logger.info(f"Loaded crop-part compatibility for {len(self.crop_part_compatibility)} crops")
        except Exception as e:
            warning_message = f"Failed to load crop-part compatibility from {self.taxonomy_path}: {e}"
            if self.strict_taxonomy_loading:
                raise RuntimeError(warning_message) from e
            self._record_runtime_warning(warning_message)

    def _record_runtime_warning(self, message: str) -> None:
        normalized = str(message or "").strip()
        if not normalized:
            return
        logger.warning(normalized)
        if normalized not in self.runtime_warnings:
            self.runtime_warnings.append(normalized)

    def _apply_runtime_warnings(self, result: RouterAnalysisResult) -> RouterAnalysisResult:
        if self.runtime_warnings:
            result.metadata.setdefault("runtime_warnings", list(self.runtime_warnings))
        return result

    def _load_labels_from_config(self, crop_mapping: Dict[str, Any]) -> None:
        self.crop_labels = list(self.vlm_config.get('crop_labels', list(crop_mapping.keys())))
        parts_from_mapping: List[str] = []
        for crop_data in crop_mapping.values():
            if isinstance(crop_data, dict):
                parts_from_mapping.extend(crop_data.get('parts', []))
        default_parts = sorted(set(parts_from_mapping))
        self.part_labels = list(self.vlm_config.get('part_labels', default_parts))
        logger.info(f"Using config labels: {len(self.crop_labels)} crops, {len(self.part_labels)} parts")

    def _refresh_policy_graph(self) -> None:
        """Refresh merged policy graph from defaults + configuration."""
        self.policy_graph = build_policy_graph(self.vlm_config)

    def _refresh_runtime_controls(self) -> None:
        """Refresh config-derived runtime controls after profile changes."""
        controls = resolve_runtime_controls(self.config, self.vlm_config)
        self.enabled = bool(controls["enabled"])
        self.confidence_threshold = float(controls["confidence_threshold"])
        self.max_detections = controls["max_detections"]
        self.open_set_enabled = bool(controls["open_set_enabled"])
        self.open_set_min_confidence = float(controls["open_set_min_confidence"])
        self.open_set_margin = float(controls["open_set_margin"])
        self.strict_model_loading = bool(controls["strict_model_loading"])
        self.strict_scoring_errors = bool(controls["strict_scoring_errors"])
        self.model_source = str(controls["model_source"])
        self.model_ids = dict(controls["model_ids"])

    def set_runtime_profile(self, profile_name: Optional[str], suppress_warning: bool = False) -> bool:
        """Apply named runtime profile to VLM config. Returns True if a profile was applied."""
        self.vlm_config, self.active_profile, applied = apply_runtime_profile(
            self._base_vlm_config,
            profile_name,
            suppress_warning=suppress_warning,
        )
        self._refresh_policy_graph()
        self._refresh_runtime_controls()
        if hasattr(self, '_open_clip_text_embedding_cache'):
            self._open_clip_text_embedding_cache.clear()
        return applied

    def _policy_value(self, stage: str, key: str, default: Any) -> Any:
        """Read value from policy stage first, then root vlm config, then fallback default."""
        return policy_value(self.policy_graph, self.vlm_config, stage, key, default)

    def _policy_enabled(self, stage: str, default: bool = True) -> bool:
        """Check if a policy stage is enabled."""
        return policy_enabled(self.policy_graph, stage, default)

    def _prepare_inference_model(self, model: Any) -> Any:
        if hasattr(model, 'to'):
            model = cast(Any, model).to(self.device)
        if hasattr(model, 'eval'):
            model.eval()
        return model

    @staticmethod
    def _coerce_sam3_prompt_sequence(prompt: Any) -> List[str]:
        raw_prompts = prompt if isinstance(prompt, (list, tuple)) else [prompt]
        cleaned: List[str] = []
        seen: set[str] = set()
        for item in raw_prompts:
            normalized = str(item).strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        return cleaned or ['plant']

    @staticmethod
    def _merge_sam3_field(chunks: List[Any]) -> Any:
        tensors: List[torch.Tensor] = []
        for chunk in chunks:
            if torch.is_tensor(chunk):
                tensor = chunk
            elif isinstance(chunk, (list, tuple)) and not chunk:
                continue
            else:
                try:
                    tensor = torch.as_tensor(chunk)
                except (TypeError, ValueError, RuntimeError):
                    continue
            if tensor.numel() <= 0:
                continue
            if tensor.ndim == 0:
                tensor = tensor.reshape(1)
            tensors.append(tensor)
        if not tensors:
            return torch.tensor([])
        return torch.cat(tensors, dim=0)

    @classmethod
    def _merge_sam3_results(cls, result_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not result_chunks:
            return normalize_sam3_results({}, empty_tensor_factory=lambda: torch.tensor([]))
        return {
            'masks': cls._merge_sam3_field([chunk.get('masks', []) for chunk in result_chunks]),
            'boxes': cls._merge_sam3_field([chunk.get('boxes', []) for chunk in result_chunks]),
            'scores': cls._merge_sam3_field([chunk.get('scores', []) for chunk in result_chunks]),
        }

    def _set_sam_runtime(self, processor: Any, model: Any, *, backend: str) -> None:
        self.sam_processor = processor
        self.sam_model = model
        self.sam_backend = backend

    def _set_bioclip_runtime(self, processor: Any, model: Any, *, backend: str) -> None:
        self.bioclip_processor = processor
        self.bioclip = model
        self.bioclip_backend = backend

    def _labels_for_type(self, label_type: str) -> List[str]:
        if label_type == 'crop':
            return self.crop_labels
        if label_type == 'part':
            return self.part_labels
        raise ValueError(f"label_type must be 'crop' or 'part', got {label_type}")

    def load_models(self):
        """Load SAM3 + BioCLIP-2.5 models (fallback disabled)."""
        return load_models_stage(self)
    def _load_sam3_bioclip25(self):
        """Load SAM3 and BioCLIP-2.5 models."""
        return load_sam3_bioclip25_stage(self)
    def is_ready(self) -> bool:
        """Return whether the pipeline is ready for real inference."""
        if not self.enabled:
            return False

        return bool(
            self.models_loaded
            and self.actual_pipeline == 'sam3'
            and self.sam_model is not None
            and self.sam_processor is not None
            and self.bioclip is not None
            and self.bioclip_processor is not None
            and self.sam_backend == 'sam3'
        )

    def _load_sam(self, model_id: str):
        """Load SAM model and processor."""
        return load_sam_stage(self, model_id)
    def _load_clip_like_model(self, model_id: str):
        """Load CLIP/BioCLIP-like model and processor."""
        return load_clip_like_model_stage(self, model_id)
    def _score_parts_conditioned_on_crop(
        self,
        roi_image: Image.Image,
        crop_label: str,
        candidate_parts: List[str],
        num_prompts: Optional[int] = None,
    ) -> Dict[str, float]:
        """Rescore compatible parts using crop-conditioned terms (e.g., 'strawberry leaf')."""
        if not crop_label or not candidate_parts:
            return {}

        conditioned_terms: List[str] = []
        term_to_part: Dict[str, str] = {}
        for part_name in candidate_parts:
            normalized_part = str(part_name).strip()
            if not normalized_part:
                continue
            term = f"{crop_label} {normalized_part}".strip()
            conditioned_terms.append(term)
            term_to_part[term] = normalized_part

        if not conditioned_terms:
            return {}

        _, _, conditioned_scores = clip_runtime.clip_score_labels_ensemble(
            self,
            roi_image,
            conditioned_terms,
            label_type='part',
            num_prompts=num_prompts,
        )

        merged: Dict[str, float] = {part: 0.0 for part in candidate_parts}
        for term, score in conditioned_scores.items():
            resolved_part: Optional[str] = term_to_part.get(term)
            if resolved_part:
                merged[resolved_part] = float(score)
        return merged

    def _classify_with_prompt_ensemble(
        self,
        image: Image.Image,
        label_type: str  # 'crop' or 'part'
    ) -> Tuple[str, float]:
        """Classify image with prompt-ensemble scoring."""
        labels = self._labels_for_type(label_type)

        if not labels:
            return 'unknown', 0.0

        return clip_runtime.clip_score_labels(self, image, labels, label_type=label_type)

    def _resolve_request_options(
        self,
        *,
        confidence_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        options: Optional[RouterRequestOptions] = None,
    ) -> RouterRequestOptions:
        return resolve_router_request_options(
            default_confidence_threshold=self.confidence_threshold,
            default_max_detections=self.max_detections,
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
            options=options,
        )

    def _normalize_router_analysis(
        self,
        analysis: RouterAnalysisResult | Dict[str, Any],
        *,
        request: RouterRequestOptions,
        status: str = "ok",
        message: str = "",
    ) -> RouterAnalysisResult:
        return normalize_router_analysis_result(
            analysis,
            request=request,
            status=status,
            message=message,
        )

    def process_image(
        self,
        image_tensor: torch.Tensor,
        confidence_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> Dict[str, Any]:
        """High-level processing entrypoint expected by tests.

        Returns a small summary dict including a 'status' and 'scenario'.
        If the pipeline is enabled, tests expect the scenario to be
        'diagnostic_scouting'.
        """
        analysis = self.analyze_image(
            image_tensor,
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
        )
        return build_process_image_response(analysis, enabled=bool(getattr(self, 'enabled', False)))

    def analyze_image_result(
        self,
        image_tensor: Any,
        *,
        options: Optional[RouterRequestOptions] = None,
        confidence_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> RouterAnalysisResult:
        """Analyze one image and return the typed router contract."""
        request = self._resolve_request_options(
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
            options=options,
        )
        pil_image, image_size = coerce_image_input(image_tensor)

        if not (self.enabled and self.models_loaded):
            return self._apply_runtime_warnings(RouterAnalysisResult(
                status="unavailable",
                message="vlm_pipeline_unavailable",
                detections=[],
                image_size=image_size,
                processing_time_ms=0.0,
                request=request,
            ))

        analyzer = self._resolve_analyzer_for_active_pipeline()
        if analyzer is None:
            return self._apply_runtime_warnings(RouterAnalysisResult(
                status="unavailable",
                message="vlm_pipeline_analyzer_unavailable",
                detections=[],
                image_size=image_size,
                processing_time_ms=0.0,
                request=request,
            ))

        return self._apply_runtime_warnings(self._normalize_router_analysis(
            analyzer(
                pil_image,
                image_size,
                request.confidence_threshold,
                request.max_detections,
            ),
            request=request,
        ))

    def analyze_image(
        self,
        image_tensor: Any,
        confidence_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze an image using SAM3 + BioCLIP-2.5.

        Args:
            image_tensor: Preprocessed image tensor
            confidence_threshold: Minimum confidence for detections
            max_detections: Maximum detections to return; None/<=0 means no cap

        Returns:
            Dictionary with analysis results
        """
        return self.analyze_image_result(
            image_tensor,
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
        ).to_dict()

    def analyze_batch_result(
        self,
        batch: torch.Tensor,
        *,
        options: Optional[RouterRequestOptions] = None,
        confidence_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> List[RouterAnalysisResult]:
        """Analyze a tensor batch and return typed router results."""
        request = self._resolve_request_options(
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
            options=options,
        )
        return analyze_batch_results(self, batch, request)

    def analyze_images_result(
        self,
        images: List[Any],
        *,
        options: Optional[RouterRequestOptions] = None,
        confidence_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> List[RouterAnalysisResult]:
        """Analyze many image objects and return typed router results."""
        request = self._resolve_request_options(
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
            options=options,
        )
        if self.enabled and self.models_loaded and self.actual_pipeline == "sam3":
            analyses = sam3_runtime.analyze_sam3_images(
                self,
                images,
                confidence_threshold=request.confidence_threshold,
                max_detections=request.max_detections,
            )
            return [self._normalize_router_analysis(analysis, request=request) for analysis in analyses]
        return [self.analyze_image_result(image, options=request) for image in images]

    def _resolve_analyzer_for_active_pipeline(self) -> Optional[RouterAnalyzer]:
        """Resolve analysis function for active pipeline."""
        analyzers: Dict[str, RouterAnalyzer] = {
            'sam3': self._analyze_image_sam3,
        }
        return resolve_router_analyzer(self.actual_pipeline, analyzers)

    def _analyze_image_sam3(
        self,
        pil_image: Image.Image,
        image_size: Tuple[int, int, int],
        confidence_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analyze using SAM3 + BioCLIP-2.5 pipeline."""
        context = sam3_runtime.build_request_context(
            self,
            pil_image=pil_image,
            image_size=image_size,
            confidence_threshold=float(
                self.confidence_threshold if confidence_threshold is None else confidence_threshold
            ),
            max_detections=max_detections,
        )
        return sam3_runtime.analyze_sam3_image(self, context)

    def _run_sam3_single_prompt(self, image: Image.Image, prompt: str, threshold: float = 0.7) -> Dict[str, Any]:
        """Run SAM3 instance segmentation for one text prompt."""
        try:
            processor = self.sam_processor
            model = self.sam_model
            if processor is None or model is None:
                raise RuntimeError("SAM3 runtime requested before SAM models were initialized.")

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            return normalize_sam3_results(results, empty_tensor_factory=lambda: torch.tensor([]))
        except Exception as e:
            logger.error(f"SAM3 inference failed: {e}")
            return sam3_error_result(e)

    def _run_sam3(self, image: Image.Image, prompt: Any, threshold: float = 0.7) -> Dict[str, Any]:
        """Run SAM3 instance segmentation with one or more text prompts."""
        prompt_sequence = self._coerce_sam3_prompt_sequence(prompt)
        results_by_prompt: List[Dict[str, Any]] = []
        errors: List[str] = []
        for text_prompt in prompt_sequence:
            result = self._run_sam3_single_prompt(image, text_prompt, threshold=threshold)
            if result.get('error'):
                errors.append(str(result.get('error')))
                continue
            results_by_prompt.append(result)
        if results_by_prompt:
            return self._merge_sam3_results(results_by_prompt)
        if errors:
            return sam3_error_result(RuntimeError("; ".join(errors)))
        return normalize_sam3_results({}, empty_tensor_factory=lambda: torch.tensor([]))

    def _run_sam3_batch_single_prompt(
        self,
        images: List[Image.Image],
        prompt: str,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Run SAM3 instance segmentation on many images for one text prompt."""
        processor = self.sam_processor
        model = self.sam_model
        if processor is None or model is None:
            raise RuntimeError("SAM3 runtime requested before SAM models were initialized.")
        if not images:
            return []

        text_prompts = [prompt] * len(images)
        inputs = processor(images=images, text=text_prompts, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)

        original_sizes = inputs.get("original_sizes")
        target_sizes = original_sizes.tolist() if hasattr(original_sizes, "tolist") else original_sizes
        raw_results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=threshold,
            target_sizes=target_sizes,
        )
        if not isinstance(raw_results, list):
            raw_results = [raw_results]
        if len(raw_results) != len(images):
            raise RuntimeError(
                f"SAM3 batch post-process returned {len(raw_results)} results for {len(images)} images."
            )
        return [
            normalize_sam3_results(result, empty_tensor_factory=lambda: torch.tensor([]))
            for result in raw_results
        ]

    def _run_sam3_batch(self, images: List[Image.Image], prompt: Any, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Run SAM3 instance segmentation on many images with one or more text prompts."""
        prompt_sequence = self._coerce_sam3_prompt_sequence(prompt)
        if not images:
            return []

        merged_by_image: List[List[Dict[str, Any]]] = [[] for _ in images]
        errors: List[str] = []
        for text_prompt in prompt_sequence:
            try:
                prompt_results = self._run_sam3_batch_single_prompt(images, text_prompt, threshold=threshold)
            except Exception as exc:
                errors.append(str(exc))
                continue
            for index, result in enumerate(prompt_results):
                merged_by_image[index].append(result)

        if any(result_chunks for result_chunks in merged_by_image):
            return [self._merge_sam3_results(result_chunks) for result_chunks in merged_by_image]
        if errors:
            raise RuntimeError("; ".join(errors))
        return [normalize_sam3_results({}, empty_tensor_factory=lambda: torch.tensor([])) for _ in images]

    def route_batch(
        self,
        batch: torch.Tensor,
        confidence_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> Tuple[List[Dict], List[float]]:
        """Process a batch of images through the router pipeline.

        Args:
            batch: Tensor of shape [batch_size, 3, height, width]

        Returns:
            Tuple of (crops_out, confs) where:
            - crops_out: List of crop prediction dicts for each image
            - confs: List of confidence scores for each image
        """
        return route_batch_items(
            self,
            batch,
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
        )
