"""
Cross-encoder reranker for two-stage retrieval.

A reranker re-scores a larger candidate set with a more expensive but more
accurate model, then returns a smaller set of top-ranked chunks for the LLM.
"""

import math
import threading
from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """Protocol for pluggable rerankers. Implementations receive a question
    and a list of candidate chunks, and return re-scored chunks in descending
    score order, limited to *top_k*."""

    def rerank(self, question: str, chunks: list[dict], top_k: int = 12) -> list[dict]:
        ...


class CrossEncoderReranker:
    """Reranker backed by a sentence-transformers cross-encoder (e.g. BGE-Reranker).

    A cross-encoder takes (query, document) pairs and produces a single
    relevance score per pair — more accurate than bi-encoder vector similarity,
    but too slow to run over the full corpus.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "auto"):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._load_lock = threading.Lock()
        self._gpu_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self):
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from sentence_transformers import CrossEncoder
            import torch

            device = self._device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"[Reranker] Loading {self._model_name} on {device} ...")
            self._model = CrossEncoder(
                self._model_name, device=device,
                local_files_only=True,
            )
            if device == "cpu":
                self._gpu_lock = threading.Lock()  # no-op lock for CPU

    def rerank(self, question: str, chunks: list[dict], top_k: int = 12) -> list[dict]:
        if not chunks:
            return []
        self._load()
        pairs = [(question, c["content"]) for c in chunks]
        with self._gpu_lock:
            logits = self._model.predict(pairs, show_progress_bar=False)
        for c, logit in zip(chunks, logits):
            c["rerank_score"] = float(logit)
            # Fuse original hybrid score with sigmoid(logit) to keep both signals
            sigmoid_score = 1.0 / (1.0 + math.exp(-float(logit)))
            original = c.get("score", 0.0)
            c["score"] = 0.4 * original + 0.6 * sigmoid_score
        chunks.sort(key=lambda c: c["score"], reverse=True)
        return chunks[:top_k]
