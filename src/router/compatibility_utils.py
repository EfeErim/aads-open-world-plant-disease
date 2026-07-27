from typing import Dict, List


def compatible_parts_for_crop(
    crop_label: str,
    crop_part_compatibility: Dict[str, List[str]],
    part_labels: List[str],
) -> List[str]:
    """Return configured compatible parts for a crop filtered to active part labels."""
    if not crop_label:
        return []

    crop_key = str(crop_label).strip().lower()
    allowed_parts = crop_part_compatibility.get(crop_key, [])
    if not allowed_parts:
        return []

    part_labels_by_lower = {str(label).strip().lower(): label for label in part_labels}
    return [part_labels_by_lower[part] for part in allowed_parts if part in part_labels_by_lower]
