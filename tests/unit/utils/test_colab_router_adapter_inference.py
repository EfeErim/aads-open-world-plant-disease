from pathlib import Path

import pytest

from src.colab import router_adapter_inference as router_script
from src.shared.contracts import RouterAnalysisResult, RouterDetection


class _FakeImage:
    def __init__(self) -> None:
        self.convert_mode = ""

    def convert(self, mode: str):
        self.convert_mode = str(mode)
        return self


@pytest.fixture(autouse=True)
def _clear_router_session_cache():
    router_script.clear_router_cache()
    yield
    router_script.clear_router_cache()


def test_run_inference_runs_router_only_pipeline(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}
    fake_image = _FakeImage()
    status_messages: list[str] = []

    class FakeRouter:
        def __init__(self, *, config=None, device="cuda"):
            calls["init"] = {
                "config": config,
                "device": device,
            }

        def load_models(self):
            calls["load_models"] = True

        def is_ready(self):
            return True

        def analyze_image_result(self, image):
            calls["analyze_image_result"] = image
            return RouterAnalysisResult(
                status="ok",
                message="",
                detections=[
                    RouterDetection(
                        crop="tomato",
                        part="fruit",
                        crop_confidence=0.95,
                        part_confidence=0.72,
                    )
                ],
            )

    def _open_image(path: Path):
        calls["opened_image"] = Path(path)
        return fake_image

    monkeypatch.setattr(router_script, "get_config", lambda environment=None: {"environment": environment})
    monkeypatch.setattr(router_script, "RouterPipeline", FakeRouter)
    monkeypatch.setattr(router_script.Image, "open", _open_image)

    result = router_script.run_inference(
        tmp_path / "leaf.png",
        config_env="colab",
        device="cpu",
        status_printer=status_messages.append,
    )

    assert result == {
        "status": "ok",
        "crop": "tomato",
        "part": "fruit",
        "router_confidence": 0.95,
        "message": "",
        "router": {
            "status": "ok",
            "message": "",
            "detections_count": 1,
            "primary_detection": {
                "crop": "tomato",
                "part": "fruit",
                "crop_confidence": 0.95,
                "part_confidence": 0.72,
            },
        },
        "router_details": {
            "status": "ok",
            "detections": [
                {
                    "crop": "tomato",
                    "part": "fruit",
                    "crop_confidence": 0.95,
                    "part_confidence": 0.72,
                }
            ],
            "detections_count": 1,
            "processing_time_ms": 0.0,
            "primary_detection": {
                "crop": "tomato",
                "part": "fruit",
                "crop_confidence": 0.95,
                "part_confidence": 0.72,
            },
        },
        "adapter_target": {
            "crop": "tomato",
            "part": "fruit",
            "adapter_dir": str(Path("models/adapters/tomato/fruit/continual_sd_lora_adapter")),
            "exists": False,
        },
        "runtime_profile": "",
    }
    assert status_messages == [
        "[INFER] image=leaf.png device=cpu",
        "[ROUTER] Loading models on cpu...",
        "[ROUTER] Ready.",
        "[ROUTER] crop=tomato part=fruit confidence=0.950",
        "[RESULT] status=ok crop=tomato part=fruit router_confidence=0.950",
    ]
    assert calls["opened_image"] == tmp_path / "leaf.png"
    assert fake_image.convert_mode == "RGB"
    assert calls["init"] == {
        "config": {"environment": "colab"},
        "device": "cpu",
    }
    assert calls["load_models"] is True
    assert calls["analyze_image_result"] is fake_image


def test_run_inference_trusted_crop_hint_skips_router(monkeypatch, tmp_path: Path):
    fake_image = _FakeImage()
    status_messages: list[str] = []

    def _open_image(_path: Path):
        return fake_image

    monkeypatch.setattr(router_script.Image, "open", _open_image)
    monkeypatch.setattr(router_script, "get_config", lambda environment=None: {"environment": environment})
    monkeypatch.setattr(
        router_script,
        "RouterPipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("router should not be built")),
    )

    result = router_script.run_inference(
        tmp_path / "leaf.png",
        config_env="colab",
        crop_hint="tomato",
        part_hint="leaf",
        trust_crop_hint=True,
        device="cpu",
        status_printer=status_messages.append,
    )

    assert result == {
        "status": "trusted_hint_skipped",
        "crop": "tomato",
        "part": "leaf",
        "router_confidence": 1.0,
        "message": "Router skipped because trust_crop_hint=True.",
        "router": {
            "status": "trusted_hint_skipped",
            "message": "Router skipped because trust_crop_hint=True.",
            "detections_count": 1,
            "primary_detection": {
                "crop": "tomato",
                "part": "leaf",
                "crop_confidence": 1.0,
                "part_confidence": 1.0,
            },
        },
        "router_details": {
            "status": "trusted_hint_skipped",
            "message": "Router skipped because trust_crop_hint=True.",
            "detections": [],
            "detections_count": 1,
            "processing_time_ms": 0.0,
            "primary_detection": {
                "crop": "tomato",
                "part": "leaf",
                "crop_confidence": 1.0,
                "part_confidence": 1.0,
            },
        },
        "adapter_target": {
            "crop": "tomato",
            "part": "leaf",
            "adapter_dir": str(Path("models/adapters/tomato/leaf/continual_sd_lora_adapter")),
            "exists": False,
        },
        "runtime_profile": "",
    }
    assert status_messages == [
        "[INFER] image=leaf.png device=cpu",
        "[ROUTER] Trusted hint; skipped router crop=tomato part=leaf",
        "[RESULT] status=trusted_hint_skipped crop=tomato part=leaf router_confidence=1.000",
    ]


def test_run_inference_surfaces_part_abstention_message(monkeypatch, tmp_path: Path):
    fake_image = _FakeImage()
    status_messages: list[str] = []

    class FakeRouter:
        def __init__(self, *, config=None, device="cuda"):
            del config, device

        def load_models(self):
            return None

        def is_ready(self):
            return True

        def analyze_image_result(self, image):
            assert image is fake_image
            return RouterAnalysisResult(
                status="ok",
                message="Part abstained for crop=tomato: unknown_confidence (0.4600) >= confidence (0.4100)",
                detections=[
                    RouterDetection(
                        crop="tomato",
                        part="unknown",
                        crop_confidence=0.95,
                        part_confidence=0.0,
                        metadata={
                            "raw_part_label": "fruit",
                            "raw_part_confidence": 0.41,
                            "part_rejection_reason": "unknown_confidence (0.4600) >= confidence (0.4100)",
                        },
                    )
                ],
            )

    monkeypatch.setattr(router_script, "get_config", lambda environment=None: {"environment": environment})
    monkeypatch.setattr(router_script, "RouterPipeline", FakeRouter)
    monkeypatch.setattr(router_script.Image, "open", lambda _path: fake_image)

    result = router_script.run_inference(
        tmp_path / "fruit.png",
        config_env="colab",
        device="cpu",
        status_printer=status_messages.append,
    )

    assert result["part"] == "unknown"
    assert result["message"] == "Part abstained for crop=tomato: unknown_confidence (0.4600) >= confidence (0.4100)"
    assert result["router"]["primary_detection"]["raw_part_label"] == "fruit"
    assert status_messages == [
        "[INFER] image=fruit.png device=cpu",
        "[ROUTER] Loading models on cpu...",
        "[ROUTER] Ready.",
        (
            "[ROUTER] crop=tomato part=unknown confidence=0.950 "
            "message=Part abstained for crop=tomato: unknown_confidence (0.4600) >= confidence (0.4100)"
        ),
        "[RESULT] status=ok crop=tomato part=unknown router_confidence=0.950",
    ]
