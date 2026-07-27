import torch

from src.router.diagnostic_scouting import DiagnosticScoutingAnalyzer
from src.shared.contracts import RouterAnalysisResult, RouterDetection


class _FakePipeline:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def process_image(self, _image):
        return {"status": "ok", "detections": []}

    def analyze_image(self, _image, confidence_threshold=None, max_detections=None):
        del confidence_threshold, max_detections
        return {
            "detections": [
                {"crop": "tomato", "part": "leaf", "crop_confidence": 0.91, "bbox": [0, 0, 10, 10]},
                {"crop": "potato", "part": "leaf", "crop_confidence": 0.75, "bbox": [1, 1, 9, 9]},
            ],
            "processing_time_ms": 12.3,
        }


def test_quick_assessment_skips_when_pipeline_unavailable():
    analyzer = DiagnosticScoutingAnalyzer(config={}, device="cpu")
    analyzer.vlm_pipeline = None
    out = analyzer.quick_assessment(torch.randn(1, 3, 16, 16))
    assert out["status"] == "skipped"
    assert out["explanation"]["reason"] == "no_vlm_pipeline"


def test_quick_assessment_uses_pipeline_process_image():
    analyzer = DiagnosticScoutingAnalyzer(config={}, device="cpu")
    analyzer.vlm_pipeline = _FakePipeline(enabled=True)
    out = analyzer.quick_assessment(torch.randn(1, 3, 16, 16))
    assert out["status"] == "ok"
    assert "analysis" in out["explanation"]


def test_analyze_image_returns_error_when_pipeline_missing():
    analyzer = DiagnosticScoutingAnalyzer(config={}, device="cpu")
    analyzer.vlm_pipeline = None
    out = analyzer.analyze_image(torch.randn(1, 3, 16, 16))
    assert out["status"] == "error"
    assert out["message"] == "vlm_pipeline_unavailable"


def test_analyze_image_normalizes_detections_and_selects_best():
    analyzer = DiagnosticScoutingAnalyzer(config={}, device="cpu")
    analyzer.vlm_pipeline = _FakePipeline(enabled=True)
    out = analyzer.analyze_image(torch.randn(1, 3, 16, 16))
    assert out["status"] == "ok"
    assert out["crop"] == "tomato"
    assert out["part"] == "leaf"
    assert out["confidence"] == 0.91
    assert len(out["detections"]) == 2


def test_analyze_image_uses_shared_primary_detection_when_available():
    analyzer = DiagnosticScoutingAnalyzer(config={}, device="cpu")

    class _TypedPipeline(_FakePipeline):
        def analyze_image_result(self, _image, confidence_threshold=None, max_detections=None):
            del confidence_threshold, max_detections
            return RouterAnalysisResult(
                detections=[
                    RouterDetection(crop="tomato", part="leaf", crop_confidence=0.61),
                    RouterDetection(crop="potato", part="leaf", crop_confidence=0.95),
                ],
                primary_detection=RouterDetection(crop="tomato", part="leaf", crop_confidence=0.61),
            )

    analyzer.vlm_pipeline = _TypedPipeline(enabled=True)

    out = analyzer.analyze_image(torch.randn(1, 3, 16, 16))

    assert out["status"] == "ok"
    assert out["crop"] == "tomato"
    assert out["confidence"] == 0.61


def test_quick_assessment_reports_pipeline_init_failure():
    analyzer = DiagnosticScoutingAnalyzer(config={}, device="cpu")
    analyzer.vlm_pipeline = None
    analyzer.vlm_pipeline_init_error = "RuntimeError: missing weights"

    out = analyzer.quick_assessment(torch.randn(1, 3, 16, 16))

    assert out["status"] == "error"
    assert out["explanation"]["reason"] == "vlm_pipeline_init_failed"
    assert "missing weights" in out["explanation"]["error"]


def test_analyze_image_reports_pipeline_init_failure_when_constructor_failed():
    analyzer = DiagnosticScoutingAnalyzer(config={}, device="cpu")
    analyzer.vlm_pipeline = None
    analyzer.vlm_pipeline_init_error = "RuntimeError: missing weights"

    out = analyzer.analyze_image(torch.randn(1, 3, 16, 16))

    assert out["status"] == "error"
    assert "vlm_pipeline_init_failed" in out["message"]
    assert "missing weights" in out["message"]
