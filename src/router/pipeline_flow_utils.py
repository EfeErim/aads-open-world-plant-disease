from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    TypeVar,
)

TAnalyzer = TypeVar("TAnalyzer", bound=Callable[..., Any])


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
