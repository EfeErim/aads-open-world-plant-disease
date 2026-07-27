from typing import Any, Dict, List


def init_sam3_stage_timings() -> Dict[str, float]:
    """Initialize default SAM3 timing buckets in milliseconds."""
    return {
        'preprocess': 0.0,
        'sam3_inference': 0.0,
        'roi_total': 0.0,
        'roi_classification': 0.0,
        'postprocess': 0.0,
    }


def summarize_sam3_stage_timings(
    stage_timings_ms: Dict[str, float],
    roi_seen: int,
    roi_classification_calls: int,
) -> Dict[str, float]:
    """Build rounded stage timing summary including per-ROI averages."""
    avg_roi_ms = (float(stage_timings_ms.get('roi_total', 0.0)) / roi_seen) if roi_seen > 0 else 0.0
    avg_classification_ms = (
        float(stage_timings_ms.get('roi_classification', 0.0)) / roi_classification_calls
    ) if roi_classification_calls > 0 else 0.0
    return {
        'preprocess': round(float(stage_timings_ms.get('preprocess', 0.0)), 2),
        'sam3_inference': round(float(stage_timings_ms.get('sam3_inference', 0.0)), 2),
        'roi_total': round(float(stage_timings_ms.get('roi_total', 0.0)), 2),
        'roi_classification': round(float(stage_timings_ms.get('roi_classification', 0.0)), 2),
        'postprocess': round(float(stage_timings_ms.get('postprocess', 0.0)), 2),
        'avg_roi': round(avg_roi_ms, 2),
        'avg_classification_call': round(avg_classification_ms, 2),
    }


def build_sam3_analysis_result(
    detections: List[Dict[str, Any]],
    image_size: Any,
    elapsed_ms: float,
    stage_summary: Dict[str, float],
    roi_seen: int,
    roi_kept: int,
    roi_classification_calls: int,
    mask_count: int,
    sam3_threshold: float,
) -> Dict[str, Any]:
    """Assemble SAM3 analysis payload with stable response contract."""
    return {
        'detections': detections,
        'image_size': image_size,
        'processing_time_ms': elapsed_ms,
        'stage_timings_ms': stage_summary,
        'roi_stats': {
            'seen': roi_seen,
            'retained': roi_kept,
            'classification_calls': roi_classification_calls,
        },
        'pipeline_type': 'sam3_bioclip25',
        'sam3_instances': mask_count,
        'sam3_threshold': sam3_threshold,
        'sam3_instances_raw': mask_count,
        'sam3_instances_retained': len(detections),
    }
