"""
In-memory LRU cache + persistent query cache with chunk-signature keys.

The in-memory cache handles exact-question dedup (fast, 1h TTL).
The persistent cache uses chunk-ID signature so cached answers are
invalidated when the index is rebuilt (chunks change → key changes).
"""
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
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


class PersistentQueryCache:
    """Persistent cache keyed by (normalized_query, chunk_signature).

    A cache hit means: same question + same retrieved chunks → same answer.
    When the index is rebuilt, chunk IDs change → old keys auto-invalidate.
    """

    def __init__(self, ttl_days: int = 7, path: str = "data/cache/query_cache.jsonl"):
        self._ttl_sec = ttl_days * 24 * 3600
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    def _normalize(self, q: str) -> str:
        return re.sub(r"[\s,。?!、,.?!]+", "", q.strip().lower())

    def _signature(self, chunk_ids: list[str]) -> str:
        return hashlib.md5(",".join(sorted(chunk_ids)).encode()).hexdigest()[:12]

    def _key(self, query: str, chunk_ids: list[str]) -> str:
        return f"{self._normalize(query)}::{self._signature(chunk_ids)}"

    def get(self, query: str, chunk_ids: list[str]) -> dict | None:
        k = self._key(query, chunk_ids)
        with self._lock:
            v = self._mem.get(k)
            if not v:
                return None
            if time.time() - v["ts"] > self._ttl_sec:
                del self._mem[k]
                return None
            return v

    def set(self, query: str, chunk_ids: list[str], answer: str, tier: str):
        k = self._key(query, chunk_ids)
        entry = {"ts": time.time(), "answer": answer, "tier": tier}
        with self._lock:
            self._mem[k] = entry
        self._append(k, entry)

    def _load(self):
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    self._mem[rec["k"]] = rec["v"]
                except Exception:
                    pass

    def _append(self, k: str, v: dict):
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")

    def stats(self) -> dict:
        now = time.time()
        with self._lock:
            valid = sum(1 for v in self._mem.values() if now - v["ts"] <= self._ttl_sec)
            return {"total": len(self._mem), "valid": valid, "ttl_days": self._ttl_sec // 86400}


# Module-level singletons
qa_cache = QACache(max_size=200, ttl=3600)
query_cache = PersistentQueryCache(ttl_days=7)
