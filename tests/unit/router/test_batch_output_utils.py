from src.router.batch_output_utils import analysis_to_batch_item


def test_analysis_to_batch_item_uses_first_detection():
    analysis = {
        'detections': [
            {'crop': 'tomato', 'part': 'leaf', 'bbox': [1, 2, 3, 4], 'crop_confidence': 0.91},
            {'crop': 'potato', 'part': 'stem', 'bbox': [0, 0, 1, 1], 'crop_confidence': 0.12},
        ]
    }

    item, conf = analysis_to_batch_item(analysis)

    assert item == {'crop': 'tomato', 'part': 'leaf', 'bbox': [1, 2, 3, 4]}
    assert conf == 0.91


def test_analysis_to_batch_item_defaults_when_missing_detection_keys():
    analysis = {'detections': [{}]}

    item, conf = analysis_to_batch_item(analysis)

    assert item == {'crop': 'unknown', 'part': 'unknown', 'bbox': [0, 0, 0, 0]}
    assert conf == 0.0


def test_analysis_to_batch_item_defaults_for_empty_or_invalid_payload():
    item_empty, conf_empty = analysis_to_batch_item({'detections': []})
    item_invalid, conf_invalid = analysis_to_batch_item(None)

    expected = {'crop': 'unknown', 'part': 'unknown', 'bbox': [0, 0, 0, 0]}
    assert item_empty == expected
    assert conf_empty == 0.0
    assert item_invalid == expected
    assert conf_invalid == 0.0
