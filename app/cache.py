"""
Simple in-memory LRU cache for Q&A responses.

Caches identical questions to skip redundant retrieval + LLM calls.
Thread-safe for FastAPI's async workers.
"""

import threading
import time
from collections import OrderedDict
from typing import Any


class QACache:
    """LRU cache for /ask responses.  Keyed by normalised question text."""

    def __init__(self, max_size: int = 200, ttl: int = 3600):
        self._max = max_size
        self._ttl = ttl  # seconds
        self._store: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _normalize(question: str) -> str:
        """Normalise question for cache key — trim + lowercase."""
        return question.strip().lower()

    def get(self, question: str) -> dict[str, Any] | None:
        key = self._normalize(question)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, value = entry
            if time.time() - ts > self._ttl:
                del self._store[key]
                return None
            # Move to end (LRU)
            self._store.move_to_end(key)
            return value

    def set(self, question: str, response: dict):
        key = self._normalize(question)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            else:
                self._store[key] = (time.time(), response)
                while len(self._store) > self._max:
                    self._store.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            return {"size": len(self._store), "max": self._max, "ttl": self._ttl}


# Module-level singleton
qa_cache = QACache(max_size=200, ttl=3600)
