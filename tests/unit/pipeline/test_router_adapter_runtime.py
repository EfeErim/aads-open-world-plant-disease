import os
from pathlib import Path

from PIL import Image

from src.pipeline.router_adapter_runtime import RouterAdapterRuntime
from src.shared.contracts import InferenceResult, InputGuardAnalysis, RouterAnalysisResult


class FakeRouter:
    def __init__(self):
        self.loaded = False

    def load_models(self):
        self.loaded = True

    def analyze_image(self, image):
        return {
            "detections": [
                {
                    "crop": "tomato",
                    "part": "leaf",
                    "crop_confidence": 0.94,
                }
            ]
        }


class FakeAdapter:
    def __init__(self, crop_name, device="cpu"):
        self.crop_name = crop_name
        self.device = device
        self.loaded_path = None

    def load_adapter(self, adapter_dir):
        self.loaded_path = adapter_dir

    def predict_with_ood(self, image):
        return {
            "status": "success",
            "disease": {"class_index": 0, "name": "healthy", "confidence": 0.91},
            "ood_analysis": {
                "score_method": "ensemble",
                "primary_score": 0.12,
                "decision_threshold": 0.8,
                "candidate_scores": {"ensemble": 0.12},
                "candidate_thresholds": {"ensemble": 0.8},
                "is_ood": False,
                "calibration_version": 3,
                "conformal_set": ["healthy"],
            },
        }


def _write_adapter_meta(root: Path, crop: str, part: str = "leaf") -> Path:
    adapter_dir = root / crop / part / "continual_sd_lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_meta.json").write_text(
        "{\"crop_name\": \"%s\", \"part_name\": \"%s\"}" % (crop, part),
        encoding="utf-8",
    )
    return adapter_dir



def test_predict_routes_and_loads_adapter(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato", "leaf")

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {
                "adapter_root": str(adapter_root),
                "target_size": 224,
                "allow_cross_part_adapter_fallback": True,
            },
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    monkeypatch.setattr(runtime, "_build_router", lambda: FakeRouter())
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "success"
    assert result["crop"] == "tomato"
    assert result["part"] == "leaf"
    assert result["router"]["status"] == "ok"
    assert result["router"]["detections_count"] == 1
    assert result["router"]["primary_detection"]["crop"] == "tomato"
    assert result["diagnosis"] == "healthy"
    assert result["conformal_set"] == ["healthy"]
    assert result["ood_analysis"]["score_method"] == "ensemble"
    assert result["ood_analysis"]["primary_score"] == 0.12
    assert {"score_method", "primary_score", "decision_threshold", "is_ood", "calibration_version"} <= set(
        result["ood_analysis"].keys()
    )



def test_predict_returns_adapter_unavailable_when_assets_missing(monkeypatch, tmp_path):
    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(tmp_path / "models"), "target_size": 224},
        },
        device="cpu",
        adapter_root=tmp_path / "models",
    )
    monkeypatch.setattr(runtime, "_build_router", lambda: FakeRouter())

    typed_result = runtime.predict_result(Image.new("RGB", (32, 32), color="green"))
    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "adapter_unavailable"
    assert result["crop"] == "tomato"
    assert result["router"]["primary_detection"]["crop"] == "tomato"
    assert typed_result.ood_analysis is None
    assert "ood_analysis" not in result



def test_trusted_crop_hint_bypasses_router(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato", "leaf")

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {
                "adapter_root": str(adapter_root),
                "target_size": 224,
                "allow_cross_part_adapter_fallback": True,
            },
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    def _fail_router():
        raise AssertionError("router should not be built when crop_hint is provided")

    monkeypatch.setattr(runtime, "_build_router", _fail_router)
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))

    result = runtime.predict(
        Image.new("RGB", (32, 32), color="green"),
        crop_hint="tomato",
        part_hint="leaf",
        trust_crop_hint=True,
    )

    assert result["status"] == "success"
    assert result["router_confidence"] == 1.0
    assert result["router"] == {
        "status": "trusted_hint_skipped",
        "message": "Router skipped because trust_crop_hint=True.",
        "detections_count": 1,
        "primary_detection": {
            "crop": "tomato",
            "part": "leaf",
            "crop_confidence": 1.0,
            "part_confidence": 1.0,
        },
    }


def test_untrusted_crop_hint_must_agree_with_router(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato", "leaf")

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(adapter_root), "target_size": 224},
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    class PotatoRouter(FakeRouter):
        def analyze_image(self, image):
            del image
            return {
                "detections": [
                    {
                        "crop": "potato",
                        "part": "leaf",
                        "crop_confidence": 0.94,
                    }
                ]
            }

    monkeypatch.setattr(runtime, "_build_router", lambda: PotatoRouter())
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"), crop_hint="tomato", part_hint="leaf")

    assert result["status"] == "router_uncertain"
    assert result["crop"] == "potato"
    assert "rejected untrusted crop_hint='tomato'" in result["message"]
    assert result["router"]["primary_detection"]["crop"] == "potato"



def test_part_hint_does_not_override_router_part_when_router_runs(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato", "leaf")

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(adapter_root), "target_size": 224},
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    class FruitRouter(FakeRouter):
        def analyze_image(self, image):
            del image
            return {
                "detections": [
                    {
                        "crop": "tomato",
                        "part": "fruit",
                        "crop_confidence": 0.94,
                    }
                ]
            }

    monkeypatch.setattr(runtime, "_build_router", lambda: FruitRouter())
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"), part_hint="leaf")

    assert result["status"] == "adapter_unavailable"
    assert result["part"] == "fruit"
    assert result["router"]["primary_detection"]["part"] == "fruit"


def test_predict_returns_adapter_unavailable_on_part_mismatch_without_cross_part_fallback(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato", "leaf")

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {
                "adapter_root": str(adapter_root),
                "target_size": 224,
                "allow_cross_part_adapter_fallback": False,
            },
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    class FruitRouter(FakeRouter):
        def analyze_image(self, image):
            del image
            return {
                "detections": [
                    {
                        "crop": "tomato",
                        "part": "fruit",
                        "crop_confidence": 0.94,
                    }
                ]
            }

    monkeypatch.setattr(runtime, "_build_router", lambda: FruitRouter())

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "adapter_unavailable"
    assert "part 'fruit'" in result["message"]


def test_predict_can_opt_in_to_cross_part_adapter_fallback(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato", "leaf")

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {
                "adapter_root": str(adapter_root),
                "target_size": 224,
                "allow_cross_part_adapter_fallback": True,
            },
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    class FruitRouter(FakeRouter):
        def analyze_image(self, image):
            del image
            return {
                "detections": [
                    {
                        "crop": "tomato",
                        "part": "fruit",
                        "crop_confidence": 0.94,
                    }
                ]
            }

    monkeypatch.setattr(runtime, "_build_router", lambda: FruitRouter())
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "success"
    assert result["part"] == "fruit"



def test_unknown_crop_payload_when_router_returns_nothing(monkeypatch, tmp_path):
    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(tmp_path / "models"), "target_size": 224},
        },
        device="cpu",
        adapter_root=tmp_path / "models",
    )

    class EmptyRouter(FakeRouter):
        def analyze_image(self, image):
            return {"detections": []}

    monkeypatch.setattr(runtime, "_build_router", lambda: EmptyRouter())

    typed_result = runtime.predict_result(Image.new("RGB", (32, 32), color="green"))
    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "unknown_crop"
    assert result["crop"] is None
    assert result["router"] == {
        "status": "ok",
        "message": "",
        "detections_count": 0,
    }
    assert typed_result.ood_analysis is None
    assert "ood_analysis" not in result



def test_predict_returns_router_unavailable_when_router_returns_no_payload(monkeypatch, tmp_path):
    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(tmp_path / "models"), "target_size": 224},
        },
        device="cpu",
        adapter_root=tmp_path / "models",
    )

    class NullRouter(FakeRouter):
        def analyze_image(self, image):
            del image
            return None

    monkeypatch.setattr(runtime, "_build_router", lambda: NullRouter())

    typed_result = runtime.predict_result(Image.new("RGB", (32, 32), color="green"))
    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "router_unavailable"
    assert "Router returned no analysis payload" in result["message"]
    assert result["router"]["status"] == "unavailable"
    assert typed_result.ood_analysis is None
    assert "ood_analysis" not in result



def test_predict_returns_router_unavailable_when_router_reports_unavailable_status(monkeypatch, tmp_path):
    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(tmp_path / "models"), "target_size": 224},
        },
        device="cpu",
        adapter_root=tmp_path / "models",
    )

    class UnavailableAnalysisRouter(FakeRouter):
        def analyze_image_result(self, image):
            del image
            return RouterAnalysisResult(status="unavailable", message="router backend missing", detections=[])

    monkeypatch.setattr(runtime, "_build_router", lambda: UnavailableAnalysisRouter())

    typed_result = runtime.predict_result(Image.new("RGB", (32, 32), color="green"))
    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "router_unavailable"
    assert "router backend missing" in result["message"]
    assert result["router"]["status"] == "unavailable"
    assert typed_result.ood_analysis is None
    assert "ood_analysis" not in result



def test_unknown_crop_status_updates_include_router_message(monkeypatch, tmp_path):
    status_lines = []

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(tmp_path / "models"), "target_size": 224},
        },
        device="cpu",
        adapter_root=tmp_path / "models",
        status_callback=status_lines.append,
    )

    class DiagnosticRouter(FakeRouter):
        def analyze_image_result(self, image):
            del image
            return RouterAnalysisResult(
                status="ok",
                message="No SAM3 instances for prompts=plant,leaf threshold=0.60.",
                detections=[],
            )

    monkeypatch.setattr(runtime, "_build_router", lambda: DiagnosticRouter())

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "unknown_crop"
    assert result["router"] == {
        "status": "ok",
        "message": "No SAM3 instances for prompts=plant,leaf threshold=0.60.",
        "detections_count": 0,
    }
    assert status_lines == [
        "[ROUTER] Loading models on cpu...",
        "[ROUTER] Ready.",
        (
            "[ROUTER] crop=unknown part=unknown confidence=0.000 "
            "message=No SAM3 instances for prompts=plant,leaf threshold=0.60."
        ),
        "[RESULT] status=unknown_crop router_confidence=0.000",
    ]



def test_predict_result_returns_typed_contract(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato", "fruit")

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(adapter_root), "target_size": 224},
        },
        device="cpu",
        adapter_root=adapter_root,
    )
    class FruitRouter(FakeRouter):
        def analyze_image(self, image):
            del image
            return {
                "detections": [
                    {
                        "crop": "tomato",
                        "part": "fruit",
                        "crop_confidence": 0.94,
                    }
                ]
            }

    monkeypatch.setattr(runtime, "_build_router", lambda: FruitRouter())
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))

    result = runtime.predict_result(Image.new("RGB", (32, 32), color="green"))

    assert isinstance(result, InferenceResult)
    assert result.status == "success"
    assert result.ood_analysis is not None
    assert result.router is not None
    assert result.router.primary_detection is not None



def test_predict_fails_closed_when_adapter_response_omits_ood_metadata(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato")

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(adapter_root), "target_size": 224},
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    class MissingOODAdapter(FakeAdapter):
        def predict_with_ood(self, image):
            del image
            return {
                "status": "success",
                "disease": {"class_index": 0, "name": "healthy", "confidence": 0.91},
            }

    monkeypatch.setattr(runtime, "_build_router", lambda: FakeRouter())
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: MissingOODAdapter(crop_name, device="cpu"))

    typed_result = runtime.predict_result(Image.new("RGB", (32, 32), color="green"))
    payload = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert typed_result.status == "adapter_ood_unavailable"
    assert typed_result.diagnosis is None
    assert typed_result.final_decision == "review"
    assert typed_result.reason_code == "adapter_ood_metadata_missing"
    assert typed_result.ood_analysis is None
    assert "ood_analysis" not in payload
    assert "conformal_set" not in payload
    assert payload["diagnosis"] is None
    assert payload["final_decision"] == "review"
    assert payload["reason_code"] == "adapter_ood_metadata_missing"


def test_predict_ood_rejection_overrides_high_class_confidence(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato")
    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(adapter_root), "target_size": 224},
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    class OODAdapter(FakeAdapter):
        def predict_with_ood(self, image):
            payload = super().predict_with_ood(image)
            payload["disease"]["confidence"] = 0.999
            payload["ood_analysis"]["primary_score"] = 0.95
            payload["ood_analysis"]["decision_threshold"] = 0.8
            payload["ood_analysis"]["is_ood"] = True
            return payload

    monkeypatch.setattr(runtime, "_build_router", lambda: FakeRouter())
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: OODAdapter(crop_name, device="cpu"))

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "ood_rejected"
    assert result["diagnosis"] is None
    assert result["confidence"] == 0.0
    assert result["final_decision"] == "unknown"
    assert result["reason_code"] == "adapter_ood_rejected"
    assert result["selected_adapter"] == "tomato__leaf"


def test_predict_fails_closed_on_ood_calibration_version_mismatch(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato")
    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(adapter_root), "target_size": 224},
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    class MismatchedAdapter(FakeAdapter):
        ood_calibration_version = 4

    monkeypatch.setattr(runtime, "_build_router", lambda: FakeRouter())
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: MismatchedAdapter(crop_name, device="cpu"))

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "adapter_ood_unavailable"
    assert result["diagnosis"] is None
    assert result["final_decision"] == "review"
    assert result["reason_code"] == "adapter_ood_calibration_mismatch"


def test_predict_returns_explicit_error_for_corrupt_image(tmp_path):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not-an-image")
    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}},
            "inference": {"adapter_root": str(tmp_path / "models"), "target_size": 224},
        },
        device="cpu",
        adapter_root=tmp_path / "models",
    )

    result = runtime.predict(corrupt)

    assert result["status"] == "invalid_input"
    assert result["final_decision"] == "error"
    assert result["reason_code"] == "input_unreadable"
    assert result["diagnosis"] is None



def test_predict_rejects_uncertain_router_handoff(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato")

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {
                "adapter_root": str(adapter_root),
                "target_size": 224,
                "router_min_confidence": 0.65,
                "router_min_margin": 0.10,
            },
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    class OrderedRouter(FakeRouter):
        def analyze_image(self, image):
            return {
                "detections": [
                    {
                        "crop": "tomato",
                        "part": "leaf",
                        "crop_confidence": 0.61,
                    },
                    {
                        "crop": "potato",
                        "part": "leaf",
                        "crop_confidence": 0.95,
                    },
                ]
            }

    monkeypatch.setattr(runtime, "_build_router", lambda: OrderedRouter())
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "router_uncertain"
    assert result["crop"] == "tomato"
    assert result["router_confidence"] == 0.61
    assert result["router"]["primary_detection"]["crop"] == "tomato"
    assert "min_confidence=0.650" in result["message"]
    assert "alternate_crop=potato" in result["message"]



def test_predict_emits_status_updates_for_notebook_surfaces(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato")
    status_lines = []

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(adapter_root), "target_size": 224},
        },
        device="cpu",
        adapter_root=adapter_root,
        status_callback=status_lines.append,
    )

    monkeypatch.setattr(runtime, "_build_router", lambda: FakeRouter())
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "success"
    assert status_lines == [
        "[ROUTER] Loading models on cpu...",
        "[ROUTER] Ready.",
        "[ROUTER] crop=tomato part=leaf confidence=0.940",
        "[ADAPTER] Loading adapter for crop=tomato part=leaf...",
        "[ADAPTER] Ready crop=tomato part=leaf",
        "[RESULT] status=success crop=tomato diagnosis=healthy confidence=0.910 ood=False",
    ]



def test_input_guard_rejects_non_plant_before_adapter_loading(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato")

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {
                "adapter_root": str(adapter_root),
                "target_size": 224,
                "input_guard": {"enabled": True},
            },
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    def _reject_guard(image, *, requested_part):
        del image, requested_part
        return InputGuardAnalysis(
            enabled=True,
            decision="non_plant_rejected",
            is_plant_like=False,
            method="bioclip_prompt_plantness",
            plant_score=0.21,
            non_plant_score=0.66,
            margin=-0.45,
            reason="non_plant_score exceeded plant_score by configured margin",
        )

    monkeypatch.setattr(runtime, "_build_router", lambda: FakeRouter())
    monkeypatch.setattr(runtime, "_evaluate_input_guard", _reject_guard)
    monkeypatch.setattr(
        runtime,
        "_build_adapter",
        lambda crop_name: (_ for _ in ()).throw(AssertionError("adapter should not load")),
    )

    result = runtime.predict(Image.new("RGB", (32, 32), color="white"))

    assert result["status"] == "non_plant_rejected"
    assert result["crop"] == "tomato"
    assert result["diagnosis"] is None
    assert "ood_analysis" not in result
    assert result["input_guard"]["decision"] == "non_plant_rejected"
    assert result["input_guard"]["plant_score"] == 0.21
    assert result["router"]["primary_detection"]["crop"] == "tomato"


def test_input_guard_pass_payload_is_included_on_success(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato")

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {
                "adapter_root": str(adapter_root),
                "target_size": 224,
                "input_guard": {"enabled": True},
            },
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    monkeypatch.setattr(runtime, "_build_router", lambda: FakeRouter())
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))
    monkeypatch.setattr(
        runtime,
        "_evaluate_input_guard",
        lambda image, *, requested_part: InputGuardAnalysis(
            enabled=True,
            decision="pass",
            is_plant_like=True,
            method="bioclip_prompt_plantness",
            plant_score=0.73,
            non_plant_score=0.18,
            margin=0.55,
        ),
    )

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "success"
    assert result["input_guard"]["decision"] == "pass"
    assert result["input_guard"]["is_plant_like"] is True
    assert result["input_guard"]["margin"] == 0.55


def test_input_guard_runs_for_trusted_crop_hint(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato", "leaf")
    guard_calls = []

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {
                "adapter_root": str(adapter_root),
                "target_size": 224,
                "allow_cross_part_adapter_fallback": True,
                "input_guard": {"enabled": True},
            },
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    def _pass_guard(image, *, requested_part):
        del image
        guard_calls.append(requested_part)
        return InputGuardAnalysis(
            enabled=True,
            decision="pass",
            is_plant_like=True,
            method="bioclip_prompt_plantness",
            plant_score=0.73,
            non_plant_score=0.18,
            margin=0.55,
        )

    monkeypatch.setattr(
        runtime,
        "_build_router",
        lambda: (_ for _ in ()).throw(AssertionError("router should not route")),
    )
    monkeypatch.setattr(runtime, "_evaluate_input_guard", _pass_guard)
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))

    result = runtime.predict(
        Image.new("RGB", (32, 32), color="green"),
        crop_hint="tomato",
        part_hint="leaf",
        trust_crop_hint=True,
    )

    assert result["status"] == "success"
    assert result["input_guard"]["decision"] == "pass"
    assert result["router"]["status"] == "trusted_hint_skipped"
    assert guard_calls == ["leaf"]


def test_load_adapter_reloads_when_bundle_changes_in_place(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    adapter_dir = _write_adapter_meta(adapter_root, "tomato")
    built_adapters = []

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(adapter_root), "target_size": 224},
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    def _build_adapter(crop_name):
        adapter = FakeAdapter(crop_name, device="cpu")
        built_adapters.append(adapter)
        return adapter

    monkeypatch.setattr(runtime, "_build_adapter", _build_adapter)

    first = runtime.load_adapter("tomato")
    meta_path = adapter_dir / "adapter_meta.json"
    stat_before = meta_path.stat()
    meta_path.write_text('{"schema_version":"v6"}', encoding="utf-8")
    os.utime(meta_path, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns + 1_000_000_000))

    second = runtime.load_adapter("tomato")

    assert first is not second
    assert len(built_adapters) == 2



def test_load_adapter_keeps_cached_bundle_when_only_non_metadata_files_change(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    adapter_dir = _write_adapter_meta(adapter_root, "tomato")
    classifier_path = adapter_dir / "classifier.pth"
    classifier_path.write_text("v1", encoding="utf-8")
    built_adapters = []

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(adapter_root), "target_size": 224},
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    def _build_adapter(crop_name):
        adapter = FakeAdapter(crop_name, device="cpu")
        built_adapters.append(adapter)
        return adapter

    monkeypatch.setattr(runtime, "_build_adapter", _build_adapter)

    first = runtime.load_adapter("tomato")
    stat_before = classifier_path.stat()
    classifier_path.write_text("v2", encoding="utf-8")
    os.utime(classifier_path, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns + 1_000_000_000))

    second = runtime.load_adapter("tomato")

    assert first is second
    assert len(built_adapters) == 1



def test_runtime_rejects_unavailable_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr("src.training.services.runtime.torch.cuda.is_available", lambda: False)

    try:
        RouterAdapterRuntime(
            config={"inference": {"adapter_root": str(tmp_path / "models"), "target_size": 224}},
            device="cuda",
            adapter_root=tmp_path / "models",
        )
    except RuntimeError as exc:
        assert "CUDA is not available" in str(exc)
    else:
        raise AssertionError("RouterAdapterRuntime was expected to reject unavailable CUDA.")



def test_predict_returns_router_unavailable_when_router_never_becomes_ready(monkeypatch, tmp_path):
    status_lines = []

    class UnreadyRouter(FakeRouter):
        def is_ready(self):
            return False

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(tmp_path / "models"), "target_size": 224},
        },
        device="cpu",
        adapter_root=tmp_path / "models",
        status_callback=status_lines.append,
    )
    monkeypatch.setattr(runtime, "_build_router", lambda: UnreadyRouter())

    first = runtime.predict(Image.new("RGB", (32, 32), color="green"))
    second = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert first["status"] == "router_unavailable"
    assert second["status"] == "router_unavailable"
    assert first["router"] == {
        "status": "unavailable",
        "message": first["message"],
        "detections_count": 0,
    }
    assert second["router"] == {
        "status": "unavailable",
        "message": second["message"],
        "detections_count": 0,
    }
    assert "Router runtime unavailable" in first["message"]
    assert status_lines == [
        "[ROUTER] Loading models on cpu...",
        "[ROUTER] Unavailable.",
        f"[RESULT] status=router_unavailable message={first['message']}",
        "[ROUTER] Loading models on cpu...",
        "[ROUTER] Unavailable.",
        f"[RESULT] status=router_unavailable message={second['message']}",
    ]


def test_predict_reload_adapter_when_adapter_meta_changes(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    adapter_dir = _write_adapter_meta(adapter_root, "tomato")
    build_calls = []

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(adapter_root), "target_size": 224},
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    def _build_adapter(crop_name):
        build_calls.append(crop_name)
        return FakeAdapter(crop_name, device="cpu")

    monkeypatch.setattr(runtime, "_build_adapter", _build_adapter)

    first = runtime.load_adapter("tomato")
    second = runtime.load_adapter("tomato")
    assert first is second
    assert build_calls == ["tomato"]

    meta_path = adapter_dir / "adapter_meta.json"
    meta_path.write_text('{"version": 2}', encoding="utf-8")

    third = runtime.load_adapter("tomato")
    assert third is not second
    assert build_calls == ["tomato", "tomato"]


def test_cross_part_fallback_does_not_cross_crop_when_metadata_omits_crop(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    tomato_dir = adapter_root / "tomato" / "leaf" / "continual_sd_lora_adapter"
    tomato_dir.mkdir(parents=True, exist_ok=True)
    (tomato_dir / "adapter_meta.json").write_text('{"part_name": "leaf"}', encoding="utf-8")

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"pepper": {"parts": ["fruit"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {
                "adapter_root": str(adapter_root),
                "target_size": 224,
                "allow_cross_part_adapter_fallback": True,
            },
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    class PepperRouter(FakeRouter):
        def analyze_image(self, image):
            del image
            return {
                "detections": [
                    {
                        "crop": "pepper",
                        "part": "fruit",
                        "crop_confidence": 0.94,
                    }
                ]
            }

    monkeypatch.setattr(runtime, "_build_router", lambda: PepperRouter())
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "adapter_unavailable"
    assert result["crop"] == "pepper"
    assert "pepper" in result["message"]
