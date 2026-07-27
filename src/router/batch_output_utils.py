from typing import Any, Dict, Tuple

from src.pipeline.inference_payloads import best_detection_from_analysis


def analysis_to_batch_item(analysis: Any) -> Tuple[Dict[str, Any], float]:
    """Map analysis payload to route_batch item contract (crop dict + confidence)."""
    detection = best_detection_from_analysis(analysis)
    if detection:
        return (
            {
                'crop': detection.get('crop', 'unknown'),
                'part': detection.get('part', 'unknown'),
                'bbox': detection.get('bbox', [0, 0, 0, 0]),
            },
            detection.get('crop_confidence', 0.0),
        )

    return ({'crop': 'unknown', 'part': 'unknown', 'bbox': [0, 0, 0, 0]}, 0.0)
