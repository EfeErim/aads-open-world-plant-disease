#!/usr/bin/env python3
"""Thin Notebook 8 helper for Notebook 1 router result -> adapter prediction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from scripts.colab_router_adapter_inference import ensure_router_ready
from src.router.prototype_reconciler import reconcile_router_handoff
from src.workflows.inference import InferenceWorkflow

StatusPrinter = Callable[[str], None]
WorkflowFactory = Callable[..., Any]
WorkflowCacheKey = Tuple[str, str, str]

_ADAPTER_ALLOWED_ROUTER_STATUSES = {"ok", "trusted_hint_skipped", "skipped"}
_FINAL_DEMO_SUPPORTED_CROPS = frozenset({"tomato", "strawberry", "grape", "apricot"})
_FINAL_DEMO_SUPPORTED_PARTS = frozenset({"leaf", "fruit"})
_WORKFLOW_SESSION_CACHE: Dict[WorkflowCacheKey, Any] = {}


def _emit_status(status_printer: Optional[StatusPrinter], message: str) -> None:
    if status_printer is not None:
        status_printer(str(message))


def _normalize_optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text or text == "none":
        return None
    return text


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm_label(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _diagnosis_crosses_part(part: Optional[str], diagnosis: Any) -> bool:
    part_key = _normalize_optional_text(part)
    diagnosis_key = _norm_label(diagnosis)
    if part_key == "fruit":
        return "leaf" in diagnosis_key or "yaprak" in diagnosis_key
    if part_key == "leaf":
        return "fruit" in diagnosis_key or "meyve" in diagnosis_key
    return False


def _router_payload(router_result: Dict[str, Any]) -> Dict[str, Any]:
    payload = router_result.get("router")
    return dict(payload) if isinstance(payload, dict) else {}


def _resolve_router_handoff(router_result: Dict[str, Any]) -> Dict[str, Any]:
    router_payload = _router_payload(router_result)
    primary_detection = router_payload.get("primary_detection")
    primary = dict(primary_detection) if isinstance(primary_detection, dict) else {}
    crop = _normalize_optional_text(router_result.get("crop") or primary.get("crop"))
    part = _normalize_optional_text(router_result.get("part") or primary.get("part"))
    status = str(router_result.get("status") or router_payload.get("status") or "").strip().lower()
    message = str(router_result.get("message") or router_payload.get("message") or "")
    return {
        "status": status,
        "crop": crop,
        "part": part,
        "message": message,
        "router": router_payload,
    }


def _final_demo_handoff_rejection(crop: Optional[str], part: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not crop or crop == "unknown":
        return "unknown_crop", "Router could not resolve a supported crop."
    if crop not in _FINAL_DEMO_SUPPORTED_CROPS:
        return "unknown_crop", f"Router crop '{crop}' is outside the final demo supported crop set."
    if not part or part == "unknown":
        return "router_uncertain", "Router could not resolve a supported plant part."
    if part not in _FINAL_DEMO_SUPPORTED_PARTS:
        return "router_uncertain", f"Router part '{part}' is outside the final demo supported part set."
    return None, None


def clear_auto_prediction_workflow_cache() -> None:
    """Drop Notebook 8 workflow and adapter instances cached for this session."""
    _WORKFLOW_SESSION_CACHE.clear()


def _workflow_cache_key(*, config_env: Optional[str], device: str, adapter_root: Optional[str | Path]) -> WorkflowCacheKey:
    return (
        str(config_env or ""),
        str(device or "cuda").strip().lower(),
        str(Path(adapter_root).resolve()) if adapter_root is not None else "",
    )


def _resolve_workflow(
    *,
    config_env: Optional[str],
    device: str,
    adapter_root: Optional[str | Path],
    status_printer: Optional[StatusPrinter],
    workflow_factory: WorkflowFactory,
) -> Any:
    if workflow_factory is not InferenceWorkflow:
        return workflow_factory(
            environment=config_env,
            device=device,
            adapter_root=adapter_root,
            status_callback=status_printer,
        )

    cache_key = _workflow_cache_key(config_env=config_env, device=device, adapter_root=adapter_root)
    workflow = _WORKFLOW_SESSION_CACHE.get(cache_key)
    if workflow is None:
        workflow = workflow_factory(
            environment=config_env,
            device=device,
            adapter_root=adapter_root,
            status_callback=status_printer,
        )
        _WORKFLOW_SESSION_CACHE[cache_key] = workflow
    workflow.runtime.status_callback = status_printer
    if workflow.runtime.router is None:
        workflow.runtime.router = ensure_router_ready(
            config_env=config_env,
            device=device,
            status_printer=status_printer,
        )
    return workflow


def resolve_router_adapter_handoff(
    image_path: str | Path,
    *,
    router_result: Dict[str, Any],
    enable_prototype_reconciler: Optional[bool] = None,
    prototype_bank_path: Optional[str | Path] = None,
    taxonomy_registry_path: Optional[str | Path] = None,
    prototype_min_similarity: Optional[float] = None,
    prototype_min_margin: Optional[float] = None,
    prototype_min_negative_gap: Optional[float] = None,
    prototype_target_policies: Optional[Dict[str, Any]] = None,
    expected_target_id: Optional[str] = None,
    expected_class_label: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(router_result, dict):
        raise TypeError("router_result must be a dictionary produced by Notebook 1 analysis.")

    handoff = _resolve_router_handoff(router_result)
    status = str(handoff["status"] or "").strip().lower()
    crop = handoff["crop"]
    part = handoff["part"]
    reconciliation_payload: Dict[str, Any] = {
        "enabled": False,
        "vlm_crop": crop,
        "vlm_part": part,
        "reconciled_crop": crop,
        "reconciled_part": part,
        "reconcile_decision": "disabled",
    }
    reconciler_enabled = _coerce_bool(
        enable_prototype_reconciler
        if enable_prototype_reconciler is not None
        else os.getenv("AADS_ENABLE_PROTOTYPE_RECONCILER"),
        default=False,
    )
    resolved_prototype_path = prototype_bank_path or os.getenv("AADS_ROUTER_PROTOTYPE_BANK")
    resolved_registry_path = taxonomy_registry_path or os.getenv("AADS_TAXONOMY_REGISTRY")
    if reconciler_enabled:
        if not resolved_prototype_path or not resolved_registry_path:
            reconciliation_payload.update(
                {
                    "enabled": True,
                    "reconcile_decision": "unavailable",
                    "reason": "prototype_bank_or_taxonomy_registry_missing",
                }
            )
        else:
            try:
                decision = reconcile_router_handoff(
                    image_path=image_path,
                    router_crop=crop,
                    router_part=part,
                    router_status=status,
                    prototype_payload=resolved_prototype_path,
                    registry_payload=resolved_registry_path,
                    min_similarity=_coerce_float(
                        prototype_min_similarity or os.getenv("AADS_PROTOTYPE_MIN_SIMILARITY"),
                        default=0.20,
                    ),
                    min_margin=_coerce_float(
                        prototype_min_margin or os.getenv("AADS_PROTOTYPE_MIN_MARGIN"),
                        default=0.03,
                    ),
                    min_negative_gap=_coerce_float(
                        prototype_min_negative_gap or os.getenv("AADS_PROTOTYPE_MIN_NEGATIVE_GAP"),
                        default=0.0,
                    ),
                    target_policies=prototype_target_policies,
                    expected_target_id=expected_target_id,
                    expected_class_label=expected_class_label,
                )
                reconciliation_payload = {"enabled": True, **decision.to_payload()}
                if decision.decision in {"accept_router", "use_prototype"}:
                    crop = decision.crop
                    part = decision.part
                    if status not in _ADAPTER_ALLOWED_ROUTER_STATUSES:
                        status = "ok"
                else:
                    status = "router_uncertain"
            except Exception as exc:
                reconciliation_payload.update(
                    {
                        "enabled": True,
                        "reconcile_decision": "error",
                        "reason": str(exc),
                    }
                )
    rejection_status, rejection_message = _final_demo_handoff_rejection(crop, part)
    adapter_allowed = (
        rejection_status is None
        and status in _ADAPTER_ALLOWED_ROUTER_STATUSES
        and bool(crop)
        and bool(part)
    )
    return {
        "adapter_allowed": adapter_allowed,
        "status": status,
        "crop": crop,
        "part": part,
        "message": handoff["message"],
        "router_confidence": float(router_result.get("router_confidence", 0.0) or 0.0),
        "router": handoff["router"],
        "rejection_status": rejection_status,
        "rejection_message": rejection_message,
        "prototype_reconciliation": reconciliation_payload,
    }


def router_handoff_skip_result(handoff: Dict[str, Any], *, status_printer: Optional[StatusPrinter] = None) -> Dict[str, Any]:
    status = str(handoff.get("status") or "").strip().lower()
    crop = _normalize_optional_text(handoff.get("crop"))
    part = _normalize_optional_text(handoff.get("part"))
    rejection_status = handoff.get("rejection_status")
    rejection_message = handoff.get("rejection_message")
    if rejection_status is not None:
        result_status = str(rejection_status)
        default_message = str(rejection_message or "Router result is outside the final demo supported surface.")
    elif not crop or crop == "unknown":
        result_status = "unknown_crop"
        default_message = "Router could not resolve a supported crop."
    elif not part or part == "unknown":
        result_status = "router_uncertain"
        default_message = "Router could not resolve a supported plant part."
    else:
        result_status = status or "router_unavailable"
        default_message = "Router result is not eligible for adapter prediction."
    message = str(handoff.get("message") or default_message)
    final_decision = "unknown" if result_status in {"unknown_crop", "unsupported_crop", "unsupported_part"} else "review"
    _emit_status(status_printer, f"[AUTO] Adapter skipped: status={status or 'unknown'} crop={crop or 'unknown'}")
    return {
        "status": result_status,
        "crop": crop,
        "part": part,
        "router_confidence": float(handoff.get("router_confidence", 0.0) or 0.0),
        "diagnosis": None,
        "confidence": 0.0,
        "message": message,
        "selected_adapter": None,
        "final_decision": final_decision,
        "reason_code": f"router_{result_status}",
        "router": dict(handoff.get("router") or {}),
        "router_handoff": {
            "adapter_ran": False,
            "source_status": status,
            "reason": message,
            "prototype_reconciliation": dict(handoff.get("prototype_reconciliation") or {}),
        },
    }


def run_auto_router_adapter_prediction(
    image_path: str | Path,
    *,
    router_result: Dict[str, Any],
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    adapter_root: Optional[str | Path] = None,
    return_ood: bool = True,
    status_printer: Optional[StatusPrinter] = None,
    workflow_factory: WorkflowFactory = InferenceWorkflow,
    enable_prototype_reconciler: Optional[bool] = None,
    prototype_bank_path: Optional[str | Path] = None,
    taxonomy_registry_path: Optional[str | Path] = None,
    prototype_min_similarity: Optional[float] = None,
    prototype_min_margin: Optional[float] = None,
    prototype_min_negative_gap: Optional[float] = None,
    prototype_target_policies: Optional[Dict[str, Any]] = None,
    expected_target_id: Optional[str] = None,
    expected_class_label: Optional[str] = None,
    handoff_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run adapter prediction from an already-computed Notebook 1 router result.

    Notebook 8 deliberately gets crop/part routing from Notebook 1's maintained
    cell scripts, then calls the canonical inference workflow with a trusted
    handoff so router behavior is not duplicated in this wrapper.
    """
    handoff_result = handoff_result or resolve_router_adapter_handoff(
        image_path,
        router_result=router_result,
        enable_prototype_reconciler=enable_prototype_reconciler,
        prototype_bank_path=prototype_bank_path,
        taxonomy_registry_path=taxonomy_registry_path,
        prototype_min_similarity=prototype_min_similarity,
        prototype_min_margin=prototype_min_margin,
        prototype_min_negative_gap=prototype_min_negative_gap,
        prototype_target_policies=prototype_target_policies,
        expected_target_id=expected_target_id,
        expected_class_label=expected_class_label,
    )
    status = str(handoff_result.get("status") or "").strip().lower()
    crop = _normalize_optional_text(handoff_result.get("crop"))
    part = _normalize_optional_text(handoff_result.get("part"))
    reconciliation_payload = dict(handoff_result.get("prototype_reconciliation") or {})
    adapter_allowed = bool(handoff_result.get("adapter_allowed"))

    if not adapter_allowed:
        return router_handoff_skip_result(handoff_result, status_printer=status_printer)

    _emit_status(
        status_printer,
        f"[AUTO] Router handoff accepted crop={crop} part={part or 'unknown'}; loading adapter.",
    )
    workflow = _resolve_workflow(
        config_env=config_env,
        device=device,
        adapter_root=adapter_root,
        status_printer=status_printer,
        workflow_factory=workflow_factory,
    )
    payload = workflow.predict(
        image_path,
        crop_hint=crop,
        part_hint=part,
        return_ood=return_ood,
        trust_crop_hint=True,
    )
    combined = dict(payload)
    if _diagnosis_crosses_part(part, combined.get("diagnosis")):
        unsafe_diagnosis = combined.get("diagnosis")
        combined.update(
            {
                "status": "router_uncertain",
                "diagnosis": None,
                "confidence": 0.0,
                "message": (
                    f"Adapter diagnosis '{unsafe_diagnosis}' conflicts with routed part '{part}'; "
                    "returning review instead of a final answer."
                ),
                "unsafe_diagnosis": unsafe_diagnosis,
                "final_decision": "review",
                "reason_code": "adapter_part_conflict",
            }
        )
    combined["router_source"] = dict(handoff_result.get("router") or {})
    combined["router_handoff"] = {
        "adapter_ran": True,
        "source_status": status,
        "crop": crop,
        "part": part,
        "prototype_reconciliation": reconciliation_payload,
    }
    return combined


__all__ = [
    "clear_auto_prediction_workflow_cache",
    "resolve_router_adapter_handoff",
    "router_handoff_skip_result",
    "run_auto_router_adapter_prediction",
]
