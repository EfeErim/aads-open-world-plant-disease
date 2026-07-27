from src.router.pipeline_flow_utils import (
    build_process_image_response,
    empty_analysis_result,
    resolve_active_analyzer,
    resolve_effective_max_detections,
)


def test_build_process_image_response_uses_diagnostic_scouting_when_enabled():
    analysis = {'detections': []}
    payload = build_process_image_response(analysis, enabled=True)

    assert payload == {
        'status': 'ok',
        'scenario': 'diagnostic_scouting',
        'analysis': analysis,
    }


def test_build_process_image_response_uses_single_detection_when_disabled():
    analysis = {'detections': [{'crop': 'tomato'}]}
    payload = build_process_image_response(analysis, enabled=False)

    assert payload['scenario'] == 'single_detection'


def test_empty_analysis_result_contract():
    payload = empty_analysis_result((3, 224, 224))
    assert payload == {
        'detections': [],
        'image_size': (3, 224, 224),
        'processing_time_ms': 0.0,
    }


def test_resolve_active_analyzer_normalizes_pipeline_name():
    sentinel = object()
    analyzers = {'sam3': sentinel}
    resolved = resolve_active_analyzer(' SAM3 ', analyzers)
    assert resolved is sentinel


def test_resolve_effective_max_detections_normalization():
    assert resolve_effective_max_detections(None) is None
    assert resolve_effective_max_detections(0) is None
    assert resolve_effective_max_detections(-1) is None
    assert resolve_effective_max_detections('3') == 3
    assert resolve_effective_max_detections('bad') is None
