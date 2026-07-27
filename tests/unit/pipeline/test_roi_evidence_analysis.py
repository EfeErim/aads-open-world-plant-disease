from src.pipeline.roi_evidence_analysis import analyze_dual_view_evidence_rows, classify_dual_view_evidence_row


def _base_row(**overrides):
    row = {
        "image_path": "sample.jpg",
        "target_id": "tomato__fruit",
        "expected_label": "healthy",
        "diagnosis": "healthy",
        "crop": "tomato",
        "part": "fruit",
        "router_status": "ok",
        "router_crop": "tomato",
        "router_part": "fruit",
        "roi_quality_status": "roi_ok",
        "roi_evidence_status": "supports_full",
        "selected_detection_source": "router_detection",
        "semantic_roi_match": True,
        "requires_review": False,
        "full_confidence": 0.86,
        "full_confidence_review_threshold": 0.70,
        "full_ood_is_ood": False,
        "roi_ood_is_ood": False,
    }
    row.update(overrides)
    return row


def test_classify_dual_view_evidence_row_flags_router_bbox_adapter_and_review_miss():
    analysis = classify_dual_view_evidence_row(
        _base_row(
            expected_label="healthy",
            diagnosis="blight",
            router_crop="eggplant",
            router_part="unknown",
            roi_quality_status="roi_too_large",
            roi_evidence_status="target_detection_missing",
            selected_detection_source="target_detection_missing",
            semantic_roi_match=False,
            requires_review=False,
        )
    )

    assert analysis["failure_buckets"] == ["router", "bbox", "adapter", "review_gate"]
    assert "router_crop_mismatch" in analysis["reasons"]
    assert "router_part_unknown" in analysis["reasons"]
    assert "target_detection_missing" in analysis["reasons"]
    assert "roi_too_large" in analysis["reasons"]
    assert "full_image_prediction_error" in analysis["reasons"]
    assert "missed_prediction_error" in analysis["reasons"]


def test_analyze_dual_view_evidence_rows_summarizes_review_capture_and_targets():
    payload = analyze_dual_view_evidence_rows(
        [
            _base_row(expected_label="healthy", diagnosis="blight", requires_review=True),
            _base_row(image_path="correct.jpg", expected_label="healthy", diagnosis="healthy", requires_review=True),
            _base_row(
                image_path="ood.jpg",
                target_id="grape__leaf",
                crop="grape",
                part="leaf",
                router_crop="grape",
                router_part="leaf",
                expected_label="rust",
                diagnosis="rust",
                full_confidence=0.41,
                full_ood_is_ood=True,
            ),
        ]
    )

    assert payload["schema_version"] == "v1_dual_view_evidence_failure_analysis"
    assert payload["sample_count"] == 3
    assert payload["incorrect_count"] == 1
    assert payload["bucket_counts"]["adapter"] == 1
    assert payload["bucket_counts"]["confidence_ood"] == 1
    assert payload["bucket_counts"]["review_gate"] == 1
    assert payload["review_capture_rate_on_errors"] == 1.0
    assert payload["review_false_positive_rate_on_correct"] == 0.5
    assert payload["per_target"]["tomato__fruit"]["sample_count"] == 2
    assert payload["per_target"]["grape__leaf"]["bucket_counts"]["confidence_ood"] == 1
