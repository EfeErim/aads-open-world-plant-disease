from src.router.pipeline_flow_utils import (
    resolve_active_analyzer,
    resolve_effective_max_detections,
)


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
