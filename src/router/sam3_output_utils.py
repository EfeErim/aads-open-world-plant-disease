from typing import Any, Dict


def normalize_sam3_results(results: Dict[str, Any], empty_tensor_factory: Any) -> Dict[str, Any]:
    """Normalize SAM3 post-process outputs to stable masks/boxes/scores contract."""
    masks = results.get('masks', empty_tensor_factory())
    boxes = results.get('boxes', empty_tensor_factory())
    scores = results.get('scores', empty_tensor_factory())

    return {
        'masks': masks if len(masks.shape) > 0 else [],
        'boxes': boxes if len(boxes.shape) > 0 else [],
        'scores': scores if len(scores.shape) > 0 else [],
    }


def sam3_error_result(error: Exception) -> Dict[str, Any]:
    """Build stable error payload for SAM3 inference failures."""
    return {'masks': [], 'boxes': [], 'scores': [], 'error': str(error)}
