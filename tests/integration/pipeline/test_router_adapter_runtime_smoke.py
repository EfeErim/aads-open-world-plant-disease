from types import SimpleNamespace
from PIL import Image

from src.pipeline.router_adapter_runtime import RouterAdapterRuntime


class FakeAdapter:
    def __init__(self):
        self.part_name = "unspecified"

    def load_adapter(self, path):
        pass

    def predict_with_ood(self, image_tensor):
        # Return a dict expected by build_success_result
        return {
            "status": "success",
            "disease": {"name": "healthy", "class_index": 0, "confidence": 0.99},
            "ood_analysis": None,
        }


def test_router_adapter_runtime_trusted_hint_smoke(monkeypatch):
    runtime = RouterAdapterRuntime(device="cpu")
    # Skip input guard
    runtime.input_guard_enabled = False

    # Monkeypatch load_adapter to return our fake adapter
    monkeypatch.setattr(runtime, "load_adapter", lambda crop, part_name=None, adapter_dir=None: FakeAdapter())

    # Monkeypatch preprocess to return a dummy tensor
    monkeypatch.setattr("src.pipeline.router_adapter_runtime.preprocess_image", lambda img, target_size=224: b"tensor")

    img = Image.new("RGB", (32, 32), color=(255, 255, 255))
    result = runtime.predict(img, crop_hint="Apple", trust_crop_hint=True, return_ood=False)
    assert isinstance(result, dict)
    assert result.get("crop") == "apple"
    assert result.get("status") in {"success", "adapter_unavailable", "router_uncertain"} or isinstance(result.get("status"), str)
