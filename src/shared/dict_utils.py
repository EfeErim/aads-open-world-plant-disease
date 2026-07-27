from typing import Any, Dict, Mapping, Optional

JsonDict = Dict[str, Any]


def nested_dict(source: Optional[Mapping[str, Any]], *keys: str) -> JsonDict:
    """Safely traverse nested mapping keys and return a dict at the leaf.

    Returns an empty dict if any intermediate step is not a mapping or the
    final value is not a mapping.
    """
    current: Any = source
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key, {})
    return dict(current) if isinstance(current, Mapping) else {}
