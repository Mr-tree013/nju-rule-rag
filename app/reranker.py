"""
Cross-encoder reranker for two-stage retrieval.

A reranker re-scores a larger candidate set with a more expensive but more
accurate model, then returns a smaller set of top-ranked chunks for the LLM.
"""

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

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self._model_name = model_name
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        print(f"[Reranker] Loading {self._model_name} ...")
        self._model = CrossEncoder(self._model_name)

    def rerank(self, question: str, chunks: list[dict], top_k: int = 12) -> list[dict]:
        if not chunks:
            return []
        self._load()
        # Build (query, doc) pairs
        pairs = [(question, c["content"]) for c in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)
        # Attach/override score and re-sort
        for c, s in zip(chunks, scores):
            c["rerank_score"] = float(s)
            c["score"] = float(s)  # downstream filter expects "score"
        chunks.sort(key=lambda c: c["score"], reverse=True)
        return chunks[:top_k]
