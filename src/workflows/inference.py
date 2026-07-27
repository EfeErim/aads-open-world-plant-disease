"""Canonical inference workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.pipeline.router_adapter_runtime import RouterAdapterRuntime
from src.shared.contracts import InferenceResult


class InferenceWorkflow:
    """Stable app-facing entrypoint for router-driven inference."""

    def __init__(
        self,
        *,
        config: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None,
        device: str = "cuda",
        adapter_root: Optional[str | Path] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.runtime = RouterAdapterRuntime(
            config=config,
            environment=environment,
            device=device,
            adapter_root=adapter_root,
            status_callback=status_callback,
        )

    def predict_result(
        self,
        image: Any,
        *,
        crop_hint: Optional[str] = None,
        part_hint: Optional[str] = None,
        return_ood: bool = True,
        trust_crop_hint: bool = False,
    ) -> InferenceResult:
        return self.runtime.predict_result(
            image,
            crop_hint=crop_hint,
            part_hint=part_hint,
            return_ood=return_ood,
            trust_crop_hint=trust_crop_hint,
        )

    def predict(
        self,
        image: Any,
        *,
        crop_hint: Optional[str] = None,
        part_hint: Optional[str] = None,
        return_ood: bool = True,
        trust_crop_hint: bool = False,
    ) -> Dict[str, Any]:
        return self.predict_result(
            image,
            crop_hint=crop_hint,
            part_hint=part_hint,
            return_ood=return_ood,
            trust_crop_hint=trust_crop_hint,
        ).to_dict(include_ood=return_ood)
