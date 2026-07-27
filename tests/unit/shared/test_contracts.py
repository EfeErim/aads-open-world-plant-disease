from src.shared.contracts import (
    InferenceResult,
    InputGuardAnalysis,
    OODAnalysis,
    RouterAnalysisResult,
    RouterDetection,
)


def test_router_analysis_result_preserves_router_order_without_quality_scores():
    result = RouterAnalysisResult(
        detections=[
            RouterDetection(crop="tomato", part="leaf", crop_confidence=0.61),
            RouterDetection(crop="potato", part="leaf", crop_confidence=0.95),
        ]
    )

    assert result.primary_detection is not None
    assert result.primary_detection.crop == "tomato"
    assert [detection.crop for detection in result.detections] == ["tomato", "potato"]


def test_router_analysis_result_prefers_quality_score_when_present():
    result = RouterAnalysisResult(
        detections=[
            RouterDetection(crop="tomato", part="leaf", crop_confidence=0.95, quality_score=0.40),
            RouterDetection(crop="potato", part="leaf", crop_confidence=0.61, quality_score=0.75),
        ]
    )

    assert result.primary_detection is not None
    assert result.primary_detection.crop == "potato"
    assert [detection.crop for detection in result.detections] == ["potato", "tomato"]


def test_router_analysis_result_preserves_explicit_primary_detection():
    explicit_primary = RouterDetection(crop="tomato", part="leaf", crop_confidence=0.61, quality_score=0.40)
    result = RouterAnalysisResult(
        detections=[
            explicit_primary,
            RouterDetection(crop="potato", part="leaf", crop_confidence=0.95, quality_score=0.75),
        ],
        primary_detection=explicit_primary,
    )

    assert result.primary_detection is not None
    assert result.primary_detection.crop == "tomato"
    assert [detection.crop for detection in result.detections] == ["potato", "tomato"]


def test_inference_result_serializes_router_summary_block():
    payload = InferenceResult(
        status="success",
        crop="tomato",
        part="leaf",
        router_confidence=0.91,
        diagnosis="healthy",
        confidence=0.88,
        router=RouterAnalysisResult(
            status="ok",
            message="",
            detections=[
                RouterDetection(crop="tomato", part="leaf", crop_confidence=0.91, quality_score=0.80),
                RouterDetection(crop="potato", part="leaf", crop_confidence=0.92, quality_score=0.20),
            ],
        ),
    ).to_dict(include_ood=False)

    assert payload["router"] == {
        "status": "ok",
        "message": "",
        "detections_count": 2,
        "primary_detection": {
            "crop": "tomato",
            "part": "leaf",
            "crop_confidence": 0.91,
            "part_confidence": 0.0,
            "quality_score": 0.8,
        },
    }


def test_inference_result_omits_conformal_set_when_ood_is_disabled():
    payload = InferenceResult(
        status="success",
        crop="tomato",
        diagnosis="healthy",
        confidence=0.88,
        ood_analysis=OODAnalysis(
            score_method="ensemble",
            primary_score=0.1,
            decision_threshold=0.8,
            is_ood=False,
            calibration_version=2,
            conformal_set=["healthy"],
        ),
    ).to_dict(include_ood=False)

    assert "ood_analysis" not in payload
    assert "conformal_set" not in payload


def test_inference_result_serializes_input_guard_payload():
    payload = InferenceResult(
        status="non_plant_rejected",
        confidence=0.0,
        input_guard=InputGuardAnalysis(
            enabled=True,
            decision="non_plant_rejected",
            is_plant_like=False,
            method="bioclip_prompt_plantness",
            plant_score=0.21,
            non_plant_score=0.66,
            margin=-0.45,
            reason="non_plant_score exceeded plant_score by configured margin",
            debug_scores={"generic_plant": 0.21, "animals_people": 0.66},
        ),
    ).to_dict(include_ood=True)

    assert payload["input_guard"] == {
        "enabled": True,
        "decision": "non_plant_rejected",
        "is_plant_like": False,
        "method": "bioclip_prompt_plantness",
        "plant_score": 0.21,
        "non_plant_score": 0.66,
        "margin": -0.45,
        "reason": "non_plant_score exceeded plant_score by configured margin",
        "debug_scores": {"generic_plant": 0.21, "animals_people": 0.66},
    }
