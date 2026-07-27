from PIL import Image

from src.shared.contracts import InferenceResult, OODAnalysis, RouterAnalysisResult, RouterDetection
from src.workflows.inference import InferenceWorkflow


class FakeRuntime:
    def predict_result(self, image, *, crop_hint=None, part_hint=None, return_ood=True, trust_crop_hint=False):
        assert isinstance(image, Image.Image)
        assert crop_hint == "tomato"
        assert part_hint == "leaf"
        assert return_ood is True
        assert trust_crop_hint is True
        return InferenceResult(
            status="success",
            crop="tomato",
            part="leaf",
            router_confidence=1.0,
            diagnosis="healthy",
            diagnosis_index=0,
            confidence=0.9,
            ood_analysis=OODAnalysis(
                score_method="ensemble",
                primary_score=0.1,
                decision_threshold=0.8,
                is_ood=False,
                calibration_version=2,
                conformal_set=["healthy"],
            ),
            router=RouterAnalysisResult(
                status="trusted_hint_skipped",
                message="Router skipped because trust_crop_hint=True.",
                primary_detection=RouterDetection(
                    crop="tomato",
                    part="leaf",
                    crop_confidence=1.0,
                    part_confidence=1.0,
                ),
                detections_count=1,
            ),
        )


def test_inference_workflow_wraps_runtime_result():
    workflow = InferenceWorkflow(config={}, device="cpu")
    workflow.runtime = FakeRuntime()  # type: ignore[assignment]

    result = workflow.predict_result(
        Image.new("RGB", (8, 8)),
        crop_hint="tomato",
        part_hint="leaf",
        trust_crop_hint=True,
    )

    assert isinstance(result, InferenceResult)
    assert result.crop == "tomato"

    payload = workflow.predict(
        Image.new("RGB", (8, 8)),
        crop_hint="tomato",
        part_hint="leaf",
        trust_crop_hint=True,
    )
    assert payload["diagnosis"] == "healthy"
    assert payload["ood_analysis"]["calibration_version"] == 2
    assert payload["conformal_set"] == ["healthy"]
    assert payload["router"]["status"] == "trusted_hint_skipped"
    assert payload["router"]["primary_detection"]["crop"] == "tomato"
