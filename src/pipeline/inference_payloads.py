"""Helpers for assembling stable inference payloads."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.shared.contracts import (
    InferenceResult,
    InputGuardAnalysis,
    OODAnalysis,
    RouterAnalysisResult,
    RouterDetection,
)


def best_detection_from_analysis(analysis: Any) -> Dict[str, Any]:
    """Return the runtime-selected primary detection from an analysis payload."""
    router_analysis = normalize_router_analysis(analysis)
    if router_analysis.primary_detection is None:
        return {}
    return router_analysis.primary_detection.to_dict()


def normalize_ood_analysis(payload: Any) -> Optional[OODAnalysis]:
    if isinstance(payload, OODAnalysis):
        return payload
    if not isinstance(payload, dict) or not payload:
        return None
    return OODAnalysis.from_dict(payload)


def normalize_router_analysis(
    payload: Any,
    *,
    status: str = "ok",
    message: str = "",
) -> RouterAnalysisResult:
    if isinstance(payload, RouterAnalysisResult):
        result = payload
    elif isinstance(payload, dict):
        result = RouterAnalysisResult.from_dict(payload)
    else:
        result = RouterAnalysisResult(status=status, message=message)
    if not result.status:
        result.status = status
    if message and not result.message:
        result.message = message
    return result


def build_router_skipped_analysis(
    *,
    crop_name: str,
    part_name: str | None,
    router_confidence: float,
    status: str = "skipped",
    message: str = "Router skipped because crop_hint was provided.",
) -> RouterAnalysisResult:
    return RouterAnalysisResult(
        status=str(status or "skipped"),
        message=str(message or "Router skipped because crop_hint was provided."),
        primary_detection=RouterDetection(
            crop=str(crop_name or "unknown"),
            part=str(part_name or "unknown"),
            crop_confidence=float(router_confidence),
            part_confidence=1.0 if part_name else 0.0,
        ),
        detections_count=1,
    )


def build_unknown_crop_result(
    *,
    part_name: str | None,
    router_confidence: float,
    include_ood: bool,
    router_analysis: RouterAnalysisResult | Dict[str, Any] | None = None,
) -> InferenceResult:
    normalized_router = normalize_router_analysis(router_analysis)
    return InferenceResult(
        status="unknown_crop",
        crop=None,
        part=part_name,
        router_confidence=float(router_confidence),
        diagnosis=None,
        confidence=0.0,
        message="Router could not resolve a supported crop.",
        ood_analysis=None,
        router=normalized_router,
        final_decision="unknown",
        reason_code="router_unknown_crop",
    )


def build_router_unavailable_result(*, message: str, include_ood: bool) -> InferenceResult:
    router_analysis = RouterAnalysisResult(
        status="unavailable",
        message=str(message),
        detections_count=0,
    )
    return InferenceResult(
        status="router_unavailable",
        crop=None,
        part=None,
        router_confidence=0.0,
        diagnosis=None,
        confidence=0.0,
        message=str(message),
        ood_analysis=None,
        router=router_analysis,
        final_decision="review",
        reason_code="router_unavailable",
    )


def build_invalid_input_result(*, message: str) -> InferenceResult:
    return InferenceResult(
        status="invalid_input",
        crop=None,
        part=None,
        router_confidence=0.0,
        diagnosis=None,
        confidence=0.0,
        message=str(message),
        ood_analysis=None,
        router=None,
        final_decision="error",
        reason_code="input_unreadable",
    )


def build_router_uncertain_result(
    *,
    crop_name: str | None,
    part_name: str | None,
    router_confidence: float,
    message: str,
    include_ood: bool,
    router_analysis: RouterAnalysisResult | Dict[str, Any] | None = None,
) -> InferenceResult:
    normalized_router = normalize_router_analysis(router_analysis)
    if not normalized_router.message:
        normalized_router.message = str(message)
    resolved_crop_name = _resolve_uncertain_crop_name(normalized_router, crop_name)
    return InferenceResult(
        status="router_uncertain",
        crop=resolved_crop_name,
        part=part_name,
        router_confidence=float(router_confidence),
        diagnosis=None,
        confidence=0.0,
        message=str(message),
        ood_analysis=None,
        router=normalized_router,
        final_decision="review",
        reason_code="router_uncertain",
    )


def _resolve_uncertain_crop_name(normalized_router: RouterAnalysisResult, crop_name: str | None) -> str | None:
    resolved_crop_name = str(crop_name or "").strip() or None
    if resolved_crop_name is None:
        detection = getattr(normalized_router, "primary_detection", None)
        if detection is not None:
            candidate_crop = str(getattr(detection, "crop", "") or "").strip().lower()
            if candidate_crop and candidate_crop != "unknown":
                resolved_crop_name = candidate_crop
    return resolved_crop_name


def build_adapter_unavailable_result(
    *,
    crop_name: str,
    part_name: str | None,
    router_confidence: float,
    message: str,
    include_ood: bool,
    router_analysis: RouterAnalysisResult | Dict[str, Any] | None = None,
) -> InferenceResult:
    normalized_router = normalize_router_analysis(router_analysis)
    return InferenceResult(
        status="adapter_unavailable",
        crop=crop_name,
        part=part_name,
        router_confidence=float(router_confidence),
        diagnosis=None,
        confidence=0.0,
        message=str(message),
        ood_analysis=None,
        router=normalized_router,
        selected_adapter=f"{crop_name}__{part_name or 'unknown'}",
        final_decision="review",
        reason_code="adapter_unavailable",
    )


def build_non_plant_rejected_result(
    *,
    crop_name: str | None,
    part_name: str | None,
    router_confidence: float,
    input_guard: InputGuardAnalysis,
    include_ood: bool,
    router_analysis: RouterAnalysisResult | Dict[str, Any] | None = None,
) -> InferenceResult:
    normalized_router = normalize_router_analysis(router_analysis)
    reason = str(input_guard.reason or "Input guard rejected the image as non-plant.")
    return InferenceResult(
        status="non_plant_rejected",
        crop=crop_name,
        part=part_name,
        router_confidence=float(router_confidence),
        diagnosis=None,
        confidence=0.0,
        message=reason,
        ood_analysis=None,
        router=normalized_router,
        input_guard=input_guard,
        final_decision="unknown",
        reason_code="router_non_plant_rejected",
    )


def build_success_result(
    *,
    crop_name: str,
    part_name: str | None,
    router_confidence: float,
    result: Dict[str, Any],
    include_ood: bool,
    router_analysis: RouterAnalysisResult | Dict[str, Any] | None = None,
    expected_ood_calibration_version: int | None = None,
) -> InferenceResult:
    disease = result.get("disease", {}) if isinstance(result, dict) else {}
    raw_ood = result.get("ood_analysis") if isinstance(result, dict) else None
    normalized_router = normalize_router_analysis(router_analysis)
    required_ood_fields = {
        "score_method",
        "primary_score",
        "decision_threshold",
        "is_ood",
        "calibration_version",
    }
    ood_metadata_valid = isinstance(raw_ood, dict) and required_ood_fields <= set(raw_ood)
    normalized_ood = normalize_ood_analysis(raw_ood) if ood_metadata_valid else None
    adapter_name = f"{crop_name}__{part_name or 'unknown'}"
    calibration_mismatch = (
        normalized_ood is not None
        and expected_ood_calibration_version is not None
        and int(expected_ood_calibration_version) != int(normalized_ood.calibration_version)
    )
    if normalized_ood is None:
        status = "adapter_ood_unavailable"
        diagnosis = None
        diagnosis_index = None
        confidence = 0.0
        final_decision = "review"
        reason_code = "adapter_ood_metadata_missing"
        message = "Adapter OOD metadata is missing or incomplete; inference failed closed."
    elif calibration_mismatch:
        status = "adapter_ood_unavailable"
        diagnosis = None
        diagnosis_index = None
        confidence = 0.0
        final_decision = "review"
        reason_code = "adapter_ood_calibration_mismatch"
        message = (
            "Adapter OOD calibration version does not match the loaded bundle; inference failed closed."
        )
    elif normalized_ood.is_ood:
        status = "ood_rejected"
        diagnosis = None
        diagnosis_index = None
        confidence = 0.0
        final_decision = "unknown"
        reason_code = "adapter_ood_rejected"
        message = "Adapter rejected the input as out of distribution."
    else:
        status = str(result.get("status", "success"))
        diagnosis = disease.get("name")
        diagnosis_index = None if disease.get("class_index") is None else int(disease.get("class_index"))
        confidence = float(disease.get("confidence", 0.0))
        final_decision = "answered" if diagnosis else "review"
        reason_code = "router_and_adapter_accepted" if diagnosis else "adapter_diagnosis_missing"
        message = str(result.get("message") or "")
    return InferenceResult(
        status=status,
        crop=crop_name,
        part=part_name,
        router_confidence=float(router_confidence),
        diagnosis=diagnosis,
        diagnosis_index=diagnosis_index,
        confidence=confidence,
        message=message,
        ood_analysis=normalized_ood,
        conformal_set=(
            None
            if normalized_ood is None or normalized_ood.conformal_set is None
            else list(normalized_ood.conformal_set)
        ),
        router=normalized_router,
        selected_adapter=adapter_name,
        final_decision=final_decision,
        reason_code=reason_code,
    )
