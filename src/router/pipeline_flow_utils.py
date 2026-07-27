from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    TypeVar,
)

TAnalyzer = TypeVar("TAnalyzer", bound=Callable[..., Any])


def build_process_image_response(analysis: Dict[str, Any], enabled: bool) -> Dict[str, Any]:
    """Build stable process_image response payload and scenario."""
    if enabled:
        scenario = 'diagnostic_scouting'
    else:
        detection_count = len(analysis.get('detections', [])) if isinstance(analysis, dict) else 0
        scenario = 'single_detection' if detection_count == 1 else 'multiple'
    return {
        'status': 'ok',
        'scenario': scenario,
        'analysis': analysis,
    }


def empty_analysis_result(image_size: Tuple[int, int, int]) -> Dict[str, Any]:
    """Compatibility-safe empty analysis response when runtime pipeline is unavailable."""
    return {
        'detections': [],
        'image_size': image_size,
        'processing_time_ms': 0.0,
    }


def resolve_active_analyzer(
    actual_pipeline: Optional[str],
    analyzers: Dict[str, TAnalyzer],
) -> Optional[TAnalyzer]:
    """Resolve analyzer callable by normalized active pipeline key."""
    active_pipeline = str(actual_pipeline or '').strip().lower()
    return analyzers.get(active_pipeline)


def resolve_effective_max_detections(max_detections: Optional[int]) -> Optional[int]:
    """Normalize max detections contract: None/<=0 means no cap."""
    if max_detections is None:
        return None

    try:
        max_det_int = int(max_detections)
    except (TypeError, ValueError, OverflowError):
        max_det_int = 0
    return None if max_det_int <= 0 else max_det_int
