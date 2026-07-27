"""Caching primitives for dataset loading."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Dict, Optional


class LRUCache:
    """Tiny TTL-capable cache for preprocessed validation images."""

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = int(max(1, capacity))
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.ttl_seconds: Optional[float] = None

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        if self._is_expired(key):
            self.__delitem__(key)
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def _is_expired(self, key: str) -> bool:
        """Return True if the given key has expired under the current TTL."""
        if self.ttl_seconds is None:
            return False
        age = time.time() - self.timestamps.get(key, 0.0)
        return age > self.ttl_seconds

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        self.timestamps[key] = time.time()
        while len(self.cache) > self.capacity:
            oldest, _ = self.cache.popitem(last=False)
            self.timestamps.pop(oldest, None)

    def clear(self) -> None:
        self.cache.clear()
        self.timestamps.clear()

    def set_ttl(self, seconds: Optional[float]) -> None:
        self.ttl_seconds = seconds

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, key: str) -> Optional[Any]:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.put(key, value)

    def __delitem__(self, key: str) -> None:
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
