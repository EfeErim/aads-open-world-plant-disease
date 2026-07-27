from PIL import Image

from src.core.config_manager import ConfigurationManager
from src.pipeline.router_adapter_runtime import RouterAdapterRuntime


class FakeRouter:
    def load_models(self):
        return None

    def analyze_image(self, image):
        return {
            "detections": [
                {"crop": "tomato", "part": "leaf", "crop_confidence": 0.97}
            ]
        }


class FakeAdapter:
    def __init__(self, crop_name, device="cpu"):
        self.crop_name = crop_name

    def load_adapter(self, adapter_dir):
        return None

    def predict_with_ood(self, image):
        return {
            "status": "success",
            "disease": {"class_index": 0, "name": "healthy", "confidence": 0.93},
            "ood_analysis": {
                "score_method": "ensemble",
                "primary_score": 0.1,
                "decision_threshold": 0.7,
                "candidate_scores": {"ensemble": 0.1},
                "candidate_thresholds": {"ensemble": 0.7},
                "is_ood": False,
                "calibration_version": 1,
            },
        }


def test_runtime_integration_with_config_manager_and_fake_router(monkeypatch, tmp_path):
    cfg = ConfigurationManager(config_dir="config", environment="colab").load_all_configs()

    adapter_root = tmp_path / "models" / "adapters"
    adapter_dir = adapter_root / "tomato" / "leaf" / "continual_sd_lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_meta.json").write_text("{}", encoding="utf-8")

    runtime = RouterAdapterRuntime(config=cfg, device="cpu", adapter_root=adapter_root)
    monkeypatch.setattr(runtime, "_build_router", lambda: FakeRouter())
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "success"
    assert result["crop"] == "tomato"
    assert result["diagnosis"] == "healthy"
    assert result["router"]["status"] == "ok"
    assert result["router"]["primary_detection"]["crop"] == "tomato"


def test_runtime_retries_with_fresh_router_after_startup_failure(monkeypatch, tmp_path):
    cfg = ConfigurationManager(config_dir="config", environment="colab").load_all_configs()

    adapter_root = tmp_path / "models" / "adapters"
    adapter_dir = adapter_root / "tomato" / "leaf" / "continual_sd_lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_meta.json").write_text("{}", encoding="utf-8")

    runtime = RouterAdapterRuntime(config=cfg, device="cpu", adapter_root=adapter_root)
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))

    build_calls = {"count": 0}

    class FailingRouter:
        def load_models(self):
            return None

        def is_ready(self):
            return False

        def analyze_image(self, image):
            del image
            return {"detections": []}

    class HealthyRouter:
        def load_models(self):
            return None

        def is_ready(self):
            return True

        def analyze_image(self, image):
            del image
            return {
                "detections": [
                    {"crop": "tomato", "part": "leaf", "crop_confidence": 0.96}
                ]
            }

    def _build_router():
        build_calls["count"] += 1
        if build_calls["count"] == 1:
            return FailingRouter()
        return HealthyRouter()

    monkeypatch.setattr(runtime, "_build_router", _build_router)

    first = runtime.predict(Image.new("RGB", (32, 32), color="green"))
    second = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert first["status"] == "router_unavailable"
    assert first["router"] == {
        "status": "unavailable",
        "message": first["message"],
        "detections_count": 0,
    }
    assert second["status"] == "success"
    assert second["crop"] == "tomato"
    assert second["router"]["primary_detection"]["crop"] == "tomato"
    assert build_calls["count"] == 2
