import numpy as np
import pytest
import torch

from src.router.roi_helpers import (
    bbox_iou,
    coerce_image_input,
    select_best_detection,
    suppress_overlapping_detections,
)


def test_bbox_iou_returns_expected_overlap():
    box_a = [0, 0, 10, 10]
    box_b = [5, 5, 15, 15]
    iou = bbox_iou(box_a, box_b)
    assert iou == pytest.approx(25.0 / 175.0)


def test_bbox_iou_returns_zero_for_invalid_boxes():
    assert bbox_iou(None, [0, 0, 1, 1]) == 0.0
    assert bbox_iou([0, 0, 1], [0, 0, 1, 1]) == 0.0


def test_suppress_overlapping_detections_same_crop_only():
    detections = [
        {"crop": "tomato", "bbox": [0, 0, 10, 10], "score": 0.9},
        {"crop": "tomato", "bbox": [1, 1, 11, 11], "score": 0.8},
        {"crop": "potato", "bbox": [1, 1, 11, 11], "score": 0.7},
    ]
    kept = suppress_overlapping_detections(detections, iou_threshold=0.5, same_crop_only=True)
    assert len(kept) == 2
    assert kept[0]["crop"] == "tomato"
    assert kept[1]["crop"] == "potato"


def test_suppress_overlapping_detections_cross_crop():
    detections = [
        {"crop": "tomato", "bbox": [0, 0, 10, 10], "score": 0.9},
        {"crop": "potato", "bbox": [1, 1, 11, 11], "score": 0.8},
    ]
    kept = suppress_overlapping_detections(detections, iou_threshold=0.5, same_crop_only=False)
    assert len(kept) == 1
    assert kept[0]["crop"] == "tomato"


def test_select_best_detection_by_score():
    detections = [
        {"crop": "tomato", "score": 0.2},
        {"crop": "potato", "score": 0.9},
        {"crop": "corn", "score": 0.5},
    ]
    best = select_best_detection(detections)
    assert best is not None
    assert best["crop"] == "potato"


def test_select_best_detection_returns_none_for_empty_input():
    assert select_best_detection([]) is None


def test_coerce_image_input_scales_normalized_numpy_float_images():
    pil_image, image_size = coerce_image_input(np.full((4, 4, 3), 0.5, dtype=np.float32))

    assert image_size == (3, 4, 4)
    assert pil_image.getpixel((0, 0)) in {(127, 127, 127), (128, 128, 128)}


def test_coerce_image_input_normalizes_batched_tensor_shape():
    pil_image, image_size = coerce_image_input(torch.zeros(2, 3, 4, 5))

    assert image_size == (3, 4, 5)
    assert pil_image.size == (5, 4)
