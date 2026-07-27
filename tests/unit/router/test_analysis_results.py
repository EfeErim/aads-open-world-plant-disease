from src.router.analysis_results import (
    build_sam3_analysis_result,
    init_sam3_stage_timings,
    summarize_sam3_stage_timings,
)


def test_init_sam3_stage_timings_defaults():
    timings = init_sam3_stage_timings()
    assert timings == {
        'preprocess': 0.0,
        'sam3_inference': 0.0,
        'roi_total': 0.0,
        'roi_classification': 0.0,
        'postprocess': 0.0,
    }


def test_summarize_sam3_stage_timings_with_averages():
    summary = summarize_sam3_stage_timings(
        {
            'preprocess': 1.234,
            'sam3_inference': 10.567,
            'roi_total': 6.0,
            'roi_classification': 4.5,
            'postprocess': 0.499,
        },
        roi_seen=3,
        roi_classification_calls=2,
    )

    assert summary == {
        'preprocess': 1.23,
        'sam3_inference': 10.57,
        'roi_total': 6.0,
        'roi_classification': 4.5,
        'postprocess': 0.5,
        'avg_roi': 2.0,
        'avg_classification_call': 2.25,
    }


def test_build_sam3_analysis_result_contract():
    detections = [{'crop': 'tomato', 'crop_confidence': 0.9}]
    stage_summary = {'sam3_inference': 11.2}

    result = build_sam3_analysis_result(
        detections=detections,
        image_size=(3, 224, 224),
        elapsed_ms=33.0,
        stage_summary=stage_summary,
        roi_seen=2,
        roi_kept=1,
        roi_classification_calls=2,
        mask_count=4,
        sam3_threshold=0.6,
    )

    assert result['detections'] == detections
    assert result['processing_time_ms'] == 33.0
    assert result['stage_timings_ms'] == stage_summary
    assert result['roi_stats'] == {'seen': 2, 'retained': 1, 'classification_calls': 2}
    assert result['pipeline_type'] == 'sam3_bioclip25'
    assert result['sam3_instances'] == 4
    assert result['sam3_instances_retained'] == 1
