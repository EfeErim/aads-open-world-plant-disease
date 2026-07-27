"""Diagnostic scouting helper extracted from the main VLM pipeline module."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from src.shared.contracts import RouterAnalysisResult

logger = logging.getLogger(__name__)


class DiagnosticScoutingAnalyzer:
    """Simplified analyzer for crop classification using VLM models."""

    def __init__(self, config: Dict, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = config
        self.vlm_pipeline: Optional[Any] = None
        self.vlm_pipeline_init_error = ""
        # Resolve lazily to avoid module import cycles.
        try:
            from src.router.router_pipeline import RouterPipeline

            self.vlm_pipeline = RouterPipeline(config, device=device)
        except Exception as exc:
            self.vlm_pipeline_init_error = f"{exc.__class__.__name__}: {exc}"
            logger.warning(
                "Diagnostic scouting could not initialize the VLM pipeline: %s",
                self.vlm_pipeline_init_error,
            )

        vlm_conf = config.get("router", {}).get("vlm", {}) if isinstance(config.get("router"), dict) else {}
        self.confidence_threshold = config.get("vlm_confidence_threshold", vlm_conf.get("confidence_threshold", 0.8))
        configured_max = config.get("vlm_max_detections", vlm_conf.get("max_detections", 0))
        try:
            configured_max_int = int(configured_max)
        except (TypeError, ValueError, OverflowError):
            configured_max_int = 0
        self.max_detections = None if configured_max_int <= 0 else configured_max_int

        logger.info("DiagnosticScoutingAnalyzer initialized on %s", self.device)

    def quick_assessment(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Quick wrapper that returns compact assessment status."""
        explanation: Dict[str, Any]
        if self.vlm_pipeline is not None and getattr(self.vlm_pipeline, "enabled", False):
            try:
                analysis = self.vlm_pipeline.process_image(image_tensor)
                explanation = {"analysis": analysis}
                status = "ok"
            except Exception as e:
                explanation = {"error": str(e)}
                status = "error"
        else:
            if self.vlm_pipeline is None and self.vlm_pipeline_init_error:
                explanation = {
                    "reason": "vlm_pipeline_init_failed",
                    "error": self.vlm_pipeline_init_error,
                }
                status = "error"
            else:
                explanation = {
                    "reason": "vlm_pipeline_disabled" if self.vlm_pipeline is not None else "no_vlm_pipeline"
                }
                status = "skipped"

        return {"status": status, "explanation": explanation}

    def analyze_image(self, image_tensor: torch.Tensor, crop_hint: Optional[str] = None) -> Dict[str, Any]:
        """Analyze image for crop identification."""
        del crop_hint

        if self.vlm_pipeline is None:
            return {
                "status": "error",
                "crop": "unknown",
                "part": "unknown",
                "confidence": 0.0,
                "detections": [],
                "message": (
                    f"vlm_pipeline_init_failed: {self.vlm_pipeline_init_error}"
                    if self.vlm_pipeline_init_error
                    else "vlm_pipeline_unavailable"
                ),
            }

        try:
            if hasattr(self.vlm_pipeline, "analyze_image_result"):
                analysis_result = self.vlm_pipeline.analyze_image_result(
                    image_tensor,
                    confidence_threshold=self.confidence_threshold,
                    max_detections=self.max_detections,
                )
            else:
                analysis_result = RouterAnalysisResult.from_dict(
                    self.vlm_pipeline.analyze_image(
                        image_tensor,
                        confidence_threshold=self.confidence_threshold,
                        max_detections=self.max_detections,
                    )
                )

            detections = analysis_result.detections
            normalized_detections = []
            best = analysis_result.primary_detection

            for det in detections:
                crop_conf = float(det.crop_confidence)
                normalized = {
                    "crop": det.crop,
                    "part": det.part,
                    "confidence": crop_conf,
                    "bbox": det.bbox,
                    "mask": det.mask,
                }
                normalized_detections.append(normalized)

            if best is None:
                best_crop = "unknown"
                best_part = "unknown"
                best_confidence = 0.0
            else:
                best_crop = best.crop
                best_part = best.part
                best_confidence = float(best.crop_confidence)

            return {
                "status": str(analysis_result.status or "ok"),
                "crop": best_crop,
                "part": best_part,
                "confidence": best_confidence,
                "detections": normalized_detections,
                "processing_time_ms": float(analysis_result.processing_time_ms),
            }
        except Exception as e:
            return {
                "status": "error",
                "crop": "unknown",
                "part": "unknown",
                "confidence": 0.0,
                "detections": [],
                "message": str(e),
            }
