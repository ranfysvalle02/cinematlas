"""A small, thread-safe LRU cache for query embeddings.

Search UIs repeat queries constantly (typeahead, pagination, back buttons), and each query costs a
Voyage round trip. Embeddings are deterministic for a given (model, text), so caching them can't
change results; keys are exact (no normalization) for the same reason.
"""

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any


class LRUCache:
    def __init__(self, maxsize: int):
        if maxsize < 0:
            raise ValueError(f"cache size must be >= 0, got {maxsize}")
        self.maxsize = maxsize
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        """Return the cached value, or compute and store it. Exceptions propagate and are never cached."""
        if self.maxsize == 0:
            return compute()
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
        value = compute()  # outside the lock: never block other threads on a network call
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    def info(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "size": len(self._data), "maxsize": self.maxsize}
