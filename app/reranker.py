"""
Two-stage retrieval rerankers: Cross-Encoder and LLM-based.

A reranker re-scores a larger candidate set with a more expensive but more
accurate model, then returns a smaller set of top-ranked chunks for the LLM.
"""

import json
import math
import re
import threading
from typing import Protocol, runtime_checkable

LLM_RERANKER_PROMPT = """你是 NJU 校规问答系统的检索排序员。给定一个学生问题和若干候选资料片段，请把这些片段按"对回答该问题的有用程度"从高到低排序。

学生问题: {query}

候选片段:
{candidates}

只输出排序结果，格式为 JSON 数组，如: [3,7,1,5,...]
最相关的在前。不要输出任何解释。"""


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

        # Min-max normalize cross-encoder scores to [0, 1] so fusion works
        # consistently regardless of raw score distribution (original, fine-tuned, etc.)
        raw_scores = [float(logit) for logit in logits]
        min_s, max_s = min(raw_scores), max(raw_scores)
        score_range = max_s - min_s
        if score_range > 1e-8:
            norm_scores = [(s - min_s) / score_range for s in raw_scores]
        else:
            norm_scores = [0.5] * len(raw_scores)

        for c, norm_score in zip(chunks, norm_scores):
            c["rerank_score"] = norm_score
            original = c.get("score", 0.0)
            c["orig_score"] = original
            c["score"] = 0.4 * original + 0.6 * norm_score
        chunks.sort(key=lambda c: c["score"], reverse=True)
        return chunks[:top_k]


class LLMReranker:
    """Listwise LLM-based reranker using Qwen3-8B for zero-training re-ranking.

    Splits candidates into batches, ranks each batch with an LLM, then runs a
    championship round on the merged top-N from each batch. Falls back to the
    cross-encoder on parse failures (if configured).
    """

    def __init__(
        self,
        llm_client,
        *,
        batch_size: int = 15,
        candidate_preview_chars: int = 200,
        temperature: float = 0.0,
        fallback_reranker=None,
    ):
        self._llm = llm_client
        self._batch_size = batch_size
        self._preview_chars = candidate_preview_chars
        self._temperature = temperature
        self._fallback = fallback_reranker

    def rerank(self, question: str, chunks: list[dict], top_k: int = 12) -> list[dict]:
        if not chunks:
            return []
        n = len(chunks)

        # Single batch — one LLM call
        if n <= self._batch_size:
            ranked = self._rank_batch(question, chunks)
            if ranked is not None:
                return self._fuse_and_sort(ranked, top_k)

        # Multi-batch: rank each batch, merge top-5 from each, championship round
        batches = [chunks[i:i + self._batch_size] for i in range(0, n, self._batch_size)]
        top_per_batch: list[dict] = []
        for batch in batches:
            ranked = self._rank_batch(question, batch)
            if ranked is None:
                return self._fallback_rerank(question, chunks, top_k)
            top_per_batch.extend(ranked[:5])

        # Dedup by source_id then content prefix (same chunk may appear in both)
        seen = set()
        deduped: list[dict] = []
        for c in top_per_batch:
            key = (c.get("source_id", ""), c.get("content", "")[:100])
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        # Championship round
        final = self._rank_batch(question, deduped)
        if final is None:
            return self._fallback_rerank(question, chunks, top_k)
        return self._fuse_and_sort(final, top_k)

    def _rank_batch(self, question: str, chunks: list[dict]) -> list[dict] | None:
        """Ask LLM to rank a batch. Returns reordered chunks or None on failure."""
        candidates_text = ""
        for i, c in enumerate(chunks, start=1):
            preview = c.get("content", "")[:self._preview_chars].replace("\n", " ")
            candidates_text += f"[{i}] {preview}\n"

        prompt = LLM_RERANKER_PROMPT.format(query=question, candidates=candidates_text)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self._llm.chat(messages, temperature=self._temperature)
            indices = self._parse_ranking(response, len(chunks))
            if indices is None:
                return None
            return [chunks[i - 1] for i in indices if 1 <= i <= len(chunks)]
        except Exception:
            return None

    @staticmethod
    def _parse_ranking(text: str, expected_n: int) -> list[int] | None:
        """Extract ranking from LLM response. Tolerant to surrounding text."""
        # Try to extract a JSON array from the response
        text = text.strip()
        # Remove markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        # Find the first [...] in the text
        match = re.search(r"\[([^\]]*)\]", text)
        if not match:
            return None
        try:
            indices = json.loads(f"[{match.group(1)}]")
            if isinstance(indices, list) and all(isinstance(i, int) for i in indices):
                # Validate: must have the right count and no out-of-range values
                if len(indices) == expected_n and all(1 <= i <= expected_n for i in indices):
                    return indices
                # Partial match: if LLM returned fewer, pad with missing
                if 0 < len(indices) < expected_n:
                    seen = set(indices)
                    for i in range(1, expected_n + 1):
                        if i not in seen:
                            indices.append(i)
                    return indices
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def _fallback_rerank(self, question: str, chunks: list[dict], top_k: int) -> list[dict]:
        if self._fallback is not None:
            return self._fallback.rerank(question, chunks, top_k)
        # No fallback — keep original order with neutral scores
        for c in chunks:
            c["rerank_score"] = 0.5
        chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
        return chunks[:top_k]

    @staticmethod
    def _fuse_and_sort(chunks: list[dict], top_k: int) -> list[dict]:
        """Assign rank-based scores and sort. First-ranked gets 1.0, last gets 0.0."""
        n = len(chunks)
        for rank, c in enumerate(chunks):
            c["rerank_score"] = 1.0 - (rank / max(n - 1, 1)) if n > 1 else 1.0
            original = c.get("score", 0.0)
            c["orig_score"] = original
            c["score"] = 0.4 * original + 0.6 * c["rerank_score"]
        chunks.sort(key=lambda c: c["score"], reverse=True)
        return chunks[:top_k]
