"""Typed contracts shared across training, inference, and Colab helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _coerce_image_size_triplet(payload: Any) -> Tuple[int, int, int]:
    raw = tuple(int(value) for value in list(payload))
    if len(raw) != 3:
        raise ValueError(f"image_size must contain exactly 3 integers, got {raw!r}")
    return (raw[0], raw[1], raw[2])


@dataclass
class OODAnalysis:
    score_method: str = "ensemble"
    primary_score: float = 0.0
    decision_threshold: float = 0.0
    is_ood: bool = False
    calibration_version: int = 0
    candidate_scores: Optional[Dict[str, float]] = None
    candidate_thresholds: Optional[Dict[str, float]] = None
    mahalanobis_z: Optional[float] = None
    energy_z: Optional[float] = None
    knn_distance: Optional[float] = None
    # --- Radially Scaled L2 Normalization ---
    radial_beta: Optional[float] = None
    # --- SURE+ Double Scoring ---
    sure_semantic_score: Optional[float] = None
    sure_confidence_score: Optional[float] = None
    sure_semantic_ood: Optional[bool] = None
    sure_confidence_reject: Optional[bool] = None
    # --- Conformal Prediction ---
    conformal_set: Optional[List[str]] = None
    conformal_coverage: Optional[float] = None
    conformal_set_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "score_method": str(self.score_method or "ensemble"),
            "primary_score": float(self.primary_score),
            "decision_threshold": float(self.decision_threshold),
            "is_ood": bool(self.is_ood),
            "calibration_version": int(self.calibration_version),
        }
        if self.candidate_scores is not None:
            payload["candidate_scores"] = {str(key): float(value) for key, value in self.candidate_scores.items()}
        if self.candidate_thresholds is not None:
            payload["candidate_thresholds"] = {
                str(key): float(value)
                for key, value in self.candidate_thresholds.items()
            }
        if self.mahalanobis_z is not None:
            payload["mahalanobis_z"] = float(self.mahalanobis_z)
        if self.energy_z is not None:
            payload["energy_z"] = float(self.energy_z)
        if self.knn_distance is not None:
            payload["knn_distance"] = float(self.knn_distance)
        if self.radial_beta is not None:
            payload["radial_beta"] = float(self.radial_beta)
        if self.sure_semantic_score is not None:
            payload["sure_semantic_score"] = float(self.sure_semantic_score)
        if self.sure_confidence_score is not None:
            payload["sure_confidence_score"] = float(self.sure_confidence_score)
        if self.sure_semantic_ood is not None:
            payload["sure_semantic_ood"] = bool(self.sure_semantic_ood)
        if self.sure_confidence_reject is not None:
            payload["sure_confidence_reject"] = bool(self.sure_confidence_reject)
        if self.conformal_set is not None:
            payload["conformal_set"] = list(self.conformal_set)
        if self.conformal_coverage is not None:
            payload["conformal_coverage"] = float(self.conformal_coverage)
        if self.conformal_set_size is not None:
            payload["conformal_set_size"] = int(self.conformal_set_size)
        return payload

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "OODAnalysis":
        data = dict(payload or {})
        conformal_set_raw = data.get("conformal_set")
        candidate_scores_raw = data.get("candidate_scores")
        candidate_thresholds_raw = data.get("candidate_thresholds")
        return cls(
            score_method=str(data.get("score_method", "ensemble") or "ensemble"),
            primary_score=float(data.get("primary_score", 0.0)),
            decision_threshold=float(data.get("decision_threshold", 0.0)),
            is_ood=bool(data.get("is_ood", False)),
            calibration_version=int(data.get("calibration_version", 0)),
            candidate_scores=(
                None
                if not isinstance(candidate_scores_raw, dict)
                else {str(key): float(value) for key, value in candidate_scores_raw.items()}
            ),
            candidate_thresholds=(
                None
                if not isinstance(candidate_thresholds_raw, dict)
                else {str(key): float(value) for key, value in candidate_thresholds_raw.items()}
            ),
            mahalanobis_z=(
                None if data.get("mahalanobis_z") is None else float(data.get("mahalanobis_z", 0.0))
            ),
            energy_z=None if data.get("energy_z") is None else float(data.get("energy_z", 0.0)),
            knn_distance=(
                None if data.get("knn_distance") is None else float(data.get("knn_distance", 0.0))
            ),
            radial_beta=(
                None if data.get("radial_beta") is None else float(data.get("radial_beta", 0.0))
            ),
            sure_semantic_score=(
                None if data.get("sure_semantic_score") is None
                else float(data.get("sure_semantic_score", 0.0))
            ),
            sure_confidence_score=(
                None if data.get("sure_confidence_score") is None
                else float(data.get("sure_confidence_score", 0.0))
            ),
            sure_semantic_ood=(
                None if data.get("sure_semantic_ood") is None
                else bool(data.get("sure_semantic_ood", False))
            ),
            sure_confidence_reject=(
                None if data.get("sure_confidence_reject") is None
                else bool(data.get("sure_confidence_reject", False))
            ),
            conformal_set=(
                None if conformal_set_raw is None else [str(c) for c in conformal_set_raw]
            ),
            conformal_coverage=(
                None if data.get("conformal_coverage") is None
                else float(data.get("conformal_coverage", 0.0))
            ),
            conformal_set_size=(
                None if data.get("conformal_set_size") is None
                else int(data.get("conformal_set_size", 0))
            ),
        )


@dataclass
class RouterRequestOptions:
    confidence_threshold: float = 0.0
    max_detections: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "confidence_threshold": float(self.confidence_threshold),
        }
        if self.max_detections is not None:
            payload["max_detections"] = int(self.max_detections)
        return payload

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "RouterRequestOptions":
        data = dict(payload or {})
        max_detections = data.get("max_detections")
        return cls(
            confidence_threshold=float(data.get("confidence_threshold", 0.0)),
            max_detections=None if max_detections is None else int(max_detections),
        )


@dataclass
class RouterDetection:
    crop: str = "unknown"
    part: str = "unknown"
    crop_confidence: float = 0.0
    part_confidence: float = 0.0
    bbox: Optional[List[float]] = None
    mask: Any = None
    sam3_score: Optional[float] = None
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def sort_key(self) -> Tuple[float, float, float]:
        quality = float("-inf") if self.quality_score is None else float(self.quality_score)
        return (quality, float(self.crop_confidence), float(self.part_confidence))

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "crop": str(self.crop or "unknown"),
            "part": str(self.part or "unknown"),
            "crop_confidence": float(self.crop_confidence),
            "part_confidence": float(self.part_confidence),
        }
        if self.bbox is not None:
            payload["bbox"] = [float(value) for value in self.bbox]
        if self.mask is not None:
            payload["mask"] = self.mask
        if self.sam3_score is not None:
            payload["sam3_score"] = float(self.sam3_score)
        if self.quality_score is not None:
            payload["quality_score"] = float(self.quality_score)
        for key, value in self.metadata.items():
            if key not in payload:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "RouterDetection":
        data = dict(payload or {})
        bbox_raw = data.pop("bbox", None)
        mask = data.pop("mask", None)
        sam3_score = data.pop("sam3_score", None)
        quality_score = data.pop("quality_score", data.pop("_quality_score", None))
        crop = data.pop("crop", "unknown")
        part = data.pop("part", "unknown")
        crop_confidence = data.pop("crop_confidence", data.pop("confidence", 0.0))
        part_confidence = data.pop("part_confidence", 0.0)
        return cls(
            crop=str(crop or "unknown"),
            part=str(part or "unknown"),
            crop_confidence=float(crop_confidence),
            part_confidence=float(part_confidence),
            bbox=(
                None
                if bbox_raw is None
                else [float(value) for value in list(bbox_raw)]
            ),
            mask=mask,
            sam3_score=None if sam3_score is None else float(sam3_score),
            quality_score=None if quality_score is None else float(quality_score),
            metadata=data,
        )


@dataclass
class RouterAnalysisResult:
    status: str = "ok"
    message: str = ""
    detections: List[RouterDetection] = field(default_factory=list)
    primary_detection: Optional[RouterDetection] = None
    detections_count: Optional[int] = None
    image_size: Optional[Tuple[int, int, int]] = None
    processing_time_ms: float = 0.0
    request: Optional[RouterRequestOptions] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_detections: List[RouterDetection] = []
        for detection in self.detections:
            if isinstance(detection, RouterDetection):
                normalized_detections.append(detection)
            else:
                normalized_detections.append(RouterDetection.from_dict(detection))
        indexed_detections = list(enumerate(normalized_detections))
        indexed_detections.sort(
            key=lambda item: (
                0 if item[1].quality_score is not None else 1,
                0.0 if item[1].quality_score is None else -float(item[1].quality_score),
                item[0],
            ),
        )
        self.detections = [detection for _index, detection in indexed_detections]

        if self.primary_detection is not None and not isinstance(self.primary_detection, RouterDetection):
            self.primary_detection = RouterDetection.from_dict(self.primary_detection)
        if self.primary_detection is None and self.detections:
            self.primary_detection = self.detections[0]

        if self.detections_count is None:
            if self.detections:
                self.detections_count = len(self.detections)
            elif self.primary_detection is not None:
                self.detections_count = 1
            else:
                self.detections_count = 0

        if self.image_size is not None:
            self.image_size = _coerce_image_size_triplet(self.image_size)
        if self.request is not None and not isinstance(self.request, RouterRequestOptions):
            self.request = RouterRequestOptions.from_dict(self.request)

    def to_dict(self, *, include_request: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": str(self.status or "ok"),
            "detections": [detection.to_dict() for detection in self.detections],
            "detections_count": int(self.detections_count or 0),
            "processing_time_ms": float(self.processing_time_ms),
        }
        if self.message:
            payload["message"] = str(self.message)
        if self.primary_detection is not None:
            payload["primary_detection"] = self.primary_detection.to_dict()
        if self.image_size is not None:
            payload["image_size"] = (
                int(self.image_size[0]),
                int(self.image_size[1]),
                int(self.image_size[2]),
            )
        if include_request and self.request is not None:
            payload["request"] = self.request.to_dict()
        for key, value in self.metadata.items():
            if key not in payload:
                payload[key] = value
        return payload

    def to_summary_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": str(self.status or "ok"),
            "message": str(self.message or ""),
            "detections_count": int(self.detections_count or 0),
        }
        if self.primary_detection is not None:
            payload["primary_detection"] = self.primary_detection.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "RouterAnalysisResult":
        data = dict(payload or {})
        detections_raw = data.pop("detections", [])
        primary_detection_raw = data.pop("primary_detection", None)
        detections_count = data.pop("detections_count", None)
        image_size_raw = data.pop("image_size", None)
        processing_time_ms = data.pop("processing_time_ms", 0.0)
        request_raw = data.pop("request", None)
        status = data.pop("status", "ok")
        message = data.pop("message", "")
        return cls(
            status=str(status or "ok"),
            message=str(message or ""),
            detections=[
                detection if isinstance(detection, RouterDetection) else RouterDetection.from_dict(detection)
                for detection in list(detections_raw or [])
            ],
            primary_detection=(
                None
                if primary_detection_raw is None
                else (
                    primary_detection_raw
                    if isinstance(primary_detection_raw, RouterDetection)
                    else RouterDetection.from_dict(primary_detection_raw)
                )
            ),
            detections_count=None if detections_count is None else int(detections_count),
            image_size=(
                None
                if image_size_raw is None
                else _coerce_image_size_triplet(image_size_raw)
            ),
            processing_time_ms=float(processing_time_ms),
            request=(
                None
                if request_raw is None
                else (
                    request_raw
                    if isinstance(request_raw, RouterRequestOptions)
                    else RouterRequestOptions.from_dict(request_raw)
                )
            ),
            metadata=data,
        )


@dataclass
class InputGuardAnalysis:
    enabled: bool = False
    decision: str = "disabled"
    is_plant_like: bool = True
    method: str = "disabled"
    plant_score: float = 0.0
    non_plant_score: float = 0.0
    margin: float = 0.0
    reason: str = ""
    debug_scores: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "enabled": bool(self.enabled),
            "decision": str(self.decision or "disabled"),
            "is_plant_like": bool(self.is_plant_like),
            "method": str(self.method or "disabled"),
            "plant_score": float(self.plant_score),
            "non_plant_score": float(self.non_plant_score),
            "margin": float(self.margin),
            "reason": str(self.reason or ""),
        }
        if self.debug_scores is not None:
            payload["debug_scores"] = {
                str(key): float(value)
                for key, value in self.debug_scores.items()
            }
        return payload

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "InputGuardAnalysis":
        data = dict(payload or {})
        debug_scores_raw = data.get("debug_scores")
        return cls(
            enabled=bool(data.get("enabled", False)),
            decision=str(data.get("decision", "disabled") or "disabled"),
            is_plant_like=bool(data.get("is_plant_like", True)),
            method=str(data.get("method", "disabled") or "disabled"),
            plant_score=float(data.get("plant_score", 0.0)),
            non_plant_score=float(data.get("non_plant_score", 0.0)),
            margin=float(data.get("margin", 0.0)),
            reason=str(data.get("reason", "") or ""),
            debug_scores=(
                None
                if not isinstance(debug_scores_raw, dict)
                else {str(key): float(value) for key, value in debug_scores_raw.items()}
            ),
        )


@dataclass
class InferenceResult:
    status: str
    crop: Optional[str] = None
    part: Optional[str] = None
    router_confidence: float = 0.0
    diagnosis: Optional[str] = None
    diagnosis_index: Optional[int] = None
    confidence: float = 0.0
    message: str = ""
    ood_analysis: Optional[OODAnalysis] = None
    conformal_set: Optional[List[str]] = None
    router: Optional[RouterAnalysisResult] = None
    input_guard: Optional[InputGuardAnalysis] = None
    selected_adapter: Optional[str] = None
    final_decision: str = ""
    reason_code: str = ""

    def __post_init__(self) -> None:
        if not self.final_decision:
            if self.status == "success" and self.diagnosis:
                self.final_decision = "answered"
            elif self.status in {"unknown_crop", "non_plant_rejected", "ood_rejected"}:
                self.final_decision = "unknown"
            elif self.status == "invalid_input":
                self.final_decision = "error"
            else:
                self.final_decision = "review"
        if not self.reason_code:
            if self.final_decision == "answered":
                self.reason_code = "router_and_adapter_accepted"
            else:
                self.reason_code = str(self.status or "unspecified")

    def to_dict(self, *, include_ood: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": str(self.status),
            "crop": self.crop,
            "part": self.part,
            "router_confidence": float(self.router_confidence),
            "diagnosis": self.diagnosis,
            "confidence": float(self.confidence),
            "final_decision": str(self.final_decision or "review"),
            "reason_code": str(self.reason_code or "unspecified"),
        }
        if self.selected_adapter:
            payload["selected_adapter"] = str(self.selected_adapter)
        if self.diagnosis_index is not None:
            payload["diagnosis_index"] = int(self.diagnosis_index)
        if self.message:
            payload["message"] = str(self.message)
        if include_ood and self.ood_analysis is not None:
            payload["ood_analysis"] = self.ood_analysis.to_dict()
        if include_ood:
            resolved_conformal_set = self.conformal_set
            if resolved_conformal_set is None and self.ood_analysis is not None:
                resolved_conformal_set = self.ood_analysis.conformal_set
            if resolved_conformal_set is not None:
                payload["conformal_set"] = list(resolved_conformal_set)
        if self.router is not None:
            payload["router"] = self.router.to_summary_dict()
        if self.input_guard is not None:
            payload["input_guard"] = self.input_guard.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "InferenceResult":
        data = dict(payload or {})
        diagnosis_index = data.get("diagnosis_index")
        conformal_set_raw = data.get("conformal_set")
        return cls(
            status=str(data.get("status", "unknown")),
            crop=data.get("crop"),
            part=data.get("part"),
            router_confidence=float(data.get("router_confidence", 0.0)),
            diagnosis=data.get("diagnosis"),
            diagnosis_index=None if diagnosis_index is None else int(diagnosis_index),
            confidence=float(data.get("confidence", 0.0)),
            message=str(data.get("message", "")),
            ood_analysis=(
                OODAnalysis.from_dict(data.get("ood_analysis"))
                if data.get("ood_analysis") is not None
                else None
            ),
            conformal_set=(
                None if conformal_set_raw is None else [str(c) for c in conformal_set_raw]
            ),
            router=(
                RouterAnalysisResult.from_dict(data.get("router"))
                if data.get("router") is not None
                else None
            ),
            input_guard=(
                InputGuardAnalysis.from_dict(data.get("input_guard"))
                if data.get("input_guard") is not None
                else None
            ),
            selected_adapter=data.get("selected_adapter"),
            final_decision=str(data.get("final_decision", "")),
            reason_code=str(data.get("reason_code", "")),
        )


@dataclass
class AdapterMetadata:
    schema_version: str = "v6"
    engine: str = "continual_sd_lora"
    crop_name: str = ""
    part_name: str = ""
    trainer_config: Dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    backbone: Dict[str, Any] = field(default_factory=dict)
    fusion: Dict[str, Any] = field(default_factory=dict)
    class_to_idx: Dict[str, int] = field(default_factory=dict)
    ood_calibration: Dict[str, Any] = field(default_factory=dict)
    ood_state: Dict[str, Any] = field(default_factory=dict)
    target_modules_resolved: List[str] = field(default_factory=list)
    adapter_runtime: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "engine": str(self.engine),
            "crop_name": str(self.crop_name),
            "part_name": str(self.part_name),
            "trainer_config": dict(self.trainer_config),
            "config_hash": str(self.config_hash),
            "backbone": dict(self.backbone),
            "fusion": dict(self.fusion),
            "class_to_idx": {str(k): int(v) for k, v in self.class_to_idx.items()},
            "ood_calibration": dict(self.ood_calibration),
            "ood_state": dict(self.ood_state),
            "target_modules_resolved": [str(item) for item in self.target_modules_resolved],
            "adapter_runtime": dict(self.adapter_runtime),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "AdapterMetadata":
        data = dict(payload or {})
        return cls(
            schema_version=str(data.get("schema_version", "v6")),
            engine=str(data.get("engine", "continual_sd_lora")),
            crop_name=str(data.get("crop_name", "")),
            part_name=str(data.get("part_name", "")),
            trainer_config=dict(data.get("trainer_config", {})),
            config_hash=str(data.get("config_hash", "")),
            backbone=dict(data.get("backbone", {})),
            fusion=dict(data.get("fusion", {})),
            class_to_idx={str(k): int(v) for k, v in dict(data.get("class_to_idx", {})).items()},
            ood_calibration=dict(data.get("ood_calibration", {})),
            ood_state=dict(data.get("ood_state", {})),
            target_modules_resolved=[str(item) for item in list(data.get("target_modules_resolved", []))],
            adapter_runtime=dict(data.get("adapter_runtime", {})),
        )


@dataclass
class CheckpointRecord:
    name: str
    path: Path
    created_at: str
    global_step: int
    epoch: int
    reason: str
    is_best: bool = False
    val_loss: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "path": str(self.path),
            "created_at": self.created_at,
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "reason": str(self.reason),
            "is_best": bool(self.is_best),
        }
        if self.val_loss is not None:
            payload["val_loss"] = float(self.val_loss)
        return payload

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "CheckpointRecord":
        data = dict(payload or {})
        val_loss = data.get("val_loss")
        return cls(
            name=str(data.get("name", "")),
            path=Path(str(data.get("path", ""))),
            created_at=str(data.get("created_at", "")),
            global_step=int(data.get("global_step", 0)),
            epoch=int(data.get("epoch", 0)),
            reason=str(data.get("reason", "")),
            is_best=bool(data.get("is_best", False)),
            val_loss=None if val_loss is None else float(val_loss),
        )
