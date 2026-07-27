from src.router.compatibility_utils import compatible_parts_for_crop


def test_compatible_parts_for_crop_filters_to_known_part_labels():
    result = compatible_parts_for_crop(
        crop_label='Tomato',
        crop_part_compatibility={'tomato': ['leaf', 'stem', 'unknown_part']},
        part_labels=['leaf', 'stem', 'fruit'],
    )

    assert result == ['leaf', 'stem']


def test_compatible_parts_for_crop_returns_empty_for_missing_or_unknown_crop():
    assert compatible_parts_for_crop('', {'tomato': ['leaf']}, ['leaf']) == []
    assert compatible_parts_for_crop('potato', {'tomato': ['leaf']}, ['leaf']) == []
