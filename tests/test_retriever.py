"""Tests for retrieval components."""
import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from rank_bm25 import BM25Okapi

from app.config import RetrievalWeights
from app.retriever import (
    BM25Retriever,
    HybridRetriever,
    VectorRetriever,
    default_tokenizer,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "test-0001",
            "source_id": "test",
            "title": "缓考管理规定",
            "content": "学生因故不能参加期末考试的，应当在考试前申请缓考。",
            "url": "",
            "priority": 1,
            "section": "第三条",
        },
        {
            "chunk_id": "test-0002",
            "source_id": "test",
            "title": "补考管理规定",
            "content": "补考一般在下一学期开学初进行，具体时间由本科生院通知。",
            "url": "",
            "priority": 2,
            "section": "第五条",
        },
        {
            "chunk_id": "test-0003",
            "source_id": "test",
            "title": "宿舍指南",
            "content": "仙林校区宿舍为四人间，上床下桌，配备独立卫浴。",
            "url": "",
            "priority": 4,
            "section": "宿舍",
        },
    ]


@pytest.fixture
def chunks_dir(sample_chunks):
    with TemporaryDirectory() as tmp:
        chunks_path = Path(tmp) / "chunks.jsonl"
        with open(chunks_path, "w", encoding="utf-8") as f:
            for c in sample_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        lookup_path = Path(tmp) / "chunk_lookup.json"
        lookup = {c["chunk_id"]: c for c in sample_chunks}
        with open(lookup_path, "w", encoding="utf-8") as f:
            json.dump(lookup, f, ensure_ascii=False)

        # Build a real BM25 index
        from app.retriever import default_tokenizer
        corpus = [c["content"] for c in sample_chunks]
        tokenized = [default_tokenizer(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        bm25_path = Path(tmp) / "bm25.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump({"model": bm25, "chunks": sample_chunks}, f)

        yield {
            "chunks_path": chunks_path,
            "lookup_path": lookup_path,
            "bm25_path": bm25_path,
            "tmp": Path(tmp),
        }


@pytest.fixture
def empty_dir():
    with TemporaryDirectory() as tmp:
        yield Path(tmp)


# ── Tokenizer ─────────────────────────────────────────────────────────


class TestDefaultTokenizer:
    def test_splits_chinese(self):
        tokens = default_tokenizer("缓考怎么申请")
        assert "缓考" in tokens
        assert "申请" in tokens
        assert len(tokens) > 1

    def test_handles_empty(self):
        tokens = default_tokenizer("")
        assert tokens == []


# ── BM25Retriever ─────────────────────────────────────────────────────


class TestBM25Retriever:
    def test_loads_from_prebuilt_index(self, chunks_dir):
        r = BM25Retriever(
            chunks_path=chunks_dir["chunks_path"],
            index_path=chunks_dir["bm25_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
        )
        assert r.is_loaded
        assert r.chunk_count == 3

    def test_fallback_when_no_index(self, chunks_dir, empty_dir):
        r = BM25Retriever(
            chunks_path=chunks_dir["chunks_path"],
            index_path=empty_dir / "nonexistent.pkl",
            chunk_lookup_path=chunks_dir["lookup_path"],
        )
        assert r.is_loaded  # fallback build should succeed
        assert r.chunk_count == 3

    def test_search_returns_relevant_chunks(self, chunks_dir):
        r = BM25Retriever(
            chunks_path=chunks_dir["chunks_path"],
            index_path=chunks_dir["bm25_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
        )
        results = r.search("缓考怎么申请", top_k=5)
        assert len(results) > 0
        # Should match the 缓考 chunk first
        assert results[0]["chunk_id"] == "test-0001"

    def test_search_empty_question(self, chunks_dir):
        r = BM25Retriever(
            chunks_path=chunks_dir["chunks_path"],
            index_path=chunks_dir["bm25_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
        )
        assert r.search("") == []
        assert r.search("   ") == []

    def test_search_no_match(self, chunks_dir):
        r = BM25Retriever(
            chunks_path=chunks_dir["chunks_path"],
            index_path=chunks_dir["bm25_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
        )
        results = r.search("火星")
        assert results == [] or all(c["score"] == 0 for c in results)

    def test_not_loaded_when_no_chunks(self, empty_dir):
        r = BM25Retriever(
            chunks_path=empty_dir / "nonexistent.jsonl",
            index_path=empty_dir / "nonexistent.pkl",
            chunk_lookup_path=empty_dir / "nonexistent.json",
        )
        assert not r.is_loaded
        assert r.chunk_count == 0

    def test_custom_tokenizer(self, chunks_dir):
        def fake_tokenizer(text):
            return ["fake"]

        r = BM25Retriever(
            chunks_path=chunks_dir["chunks_path"],
            index_path=chunks_dir["bm25_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
            tokenizer=fake_tokenizer,
        )
        # The tokenizer is used for fallback builds, not when loading prebuilt
        assert r._tokenizer is fake_tokenizer


# ── VectorRetriever ───────────────────────────────────────────────────


class TestVectorRetriever:
    def test_not_loaded_when_no_chroma_dir(self, chunks_dir, empty_dir):
        r = VectorRetriever(
            chroma_path=empty_dir / "chroma",
            chunks_path=chunks_dir["chunks_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
            enable=True,
        )
        assert not r.is_loaded

    def test_disabled_when_enable_false(self, chunks_dir, empty_dir):
        r = VectorRetriever(
            chroma_path=empty_dir / "chroma",
            chunks_path=chunks_dir["chunks_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
            enable=False,
        )
        assert not r.is_loaded
        assert r.search("test") == []

    def test_search_empty_returns_empty(self, chunks_dir, empty_dir):
        r = VectorRetriever(
            chroma_path=empty_dir / "chroma",
            chunks_path=chunks_dir["chunks_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
            enable=False,
        )
        assert r.search("") == []

    def test_chunk_count_from_lookup(self, chunks_dir, empty_dir):
        r = VectorRetriever(
            chroma_path=empty_dir / "chroma",
            chunks_path=chunks_dir["chunks_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
            enable=False,
        )
        assert r.chunk_count == 3


# ── HybridRetriever ───────────────────────────────────────────────────


class TestHybridRetriever:
    def test_bm25_only_when_vector_disabled(self, chunks_dir, empty_dir):
        bm25 = BM25Retriever(
            chunks_path=chunks_dir["chunks_path"],
            index_path=chunks_dir["bm25_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
        )
        vector = VectorRetriever(
            chroma_path=empty_dir / "chroma",
            chunks_path=chunks_dir["chunks_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
            enable=False,
        )
        hybrid = HybridRetriever(
            bm25=bm25,
            vector=vector,
            manifest_path=empty_dir / "manifest.json",
        )
        results = hybrid.search("缓考")
        assert len(results) > 0
        # With BM25-only fallback, weights shift to 0.80/0.00/0.20
        detail = results[0].get("score_detail", {})
        assert detail.get("vector_weight") == 0.0

    def test_empty_question_returns_empty(self, chunks_dir, empty_dir):
        bm25 = BM25Retriever(
            chunks_path=chunks_dir["chunks_path"],
            index_path=chunks_dir["bm25_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
        )
        vector = VectorRetriever(
            chroma_path=empty_dir / "chroma",
            chunks_path=chunks_dir["chunks_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
            enable=False,
        )
        hybrid = HybridRetriever(
            bm25=bm25,
            vector=vector,
            manifest_path=empty_dir / "manifest.json",
        )
        assert hybrid.search("") == []

    def test_status_reports_retriever_state(self, chunks_dir, empty_dir):
        bm25 = BM25Retriever(
            chunks_path=chunks_dir["chunks_path"],
            index_path=chunks_dir["bm25_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
        )
        vector = VectorRetriever(
            chroma_path=empty_dir / "chroma",
            chunks_path=chunks_dir["chunks_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
            enable=False,
        )
        hybrid = HybridRetriever(
            bm25=bm25,
            vector=vector,
            manifest_path=empty_dir / "manifest.json",
        )
        status = hybrid.status()
        assert status["bm25_loaded"] is True
        assert status["vector_loaded"] is False
        assert status["bm25_chunks"] == 3

    def test_custom_weights(self, chunks_dir, empty_dir):
        bm25 = BM25Retriever(
            chunks_path=chunks_dir["chunks_path"],
            index_path=chunks_dir["bm25_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
        )
        vector = VectorRetriever(
            chroma_path=empty_dir / "chroma",
            chunks_path=chunks_dir["chunks_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
            enable=False,
        )
        weights = RetrievalWeights(bm25=0.6, vector=0.2, priority=0.2)
        hybrid = HybridRetriever(
            bm25=bm25,
            vector=vector,
            manifest_path=empty_dir / "manifest.json",
            weights=weights,
        )
        results = hybrid.search("缓考")
        assert len(results) > 0

    def test_minmax_norm(self):
        items = [
            {"chunk_id": "a", "score": 0.0},
            {"chunk_id": "b", "score": 5.0},
            {"chunk_id": "c", "score": 10.0},
        ]
        normed = HybridRetriever._minmax_norm(items)
        assert normed[0]["score"] == 0.0
        assert normed[2]["score"] == 1.0
        assert normed[1]["score"] == 0.5

    def test_minmax_norm_all_same(self):
        items = [{"chunk_id": "a", "score": 5.0}, {"chunk_id": "b", "score": 5.0}]
        normed = HybridRetriever._minmax_norm(items)
        assert all(x["score"] == 1.0 for x in normed)

    def test_minmax_norm_empty(self):
        assert HybridRetriever._minmax_norm([]) == []

    def test_score_detail_present(self, chunks_dir, empty_dir):
        bm25 = BM25Retriever(
            chunks_path=chunks_dir["chunks_path"],
            index_path=chunks_dir["bm25_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
        )
        vector = VectorRetriever(
            chroma_path=empty_dir / "chroma",
            chunks_path=chunks_dir["chunks_path"],
            chunk_lookup_path=chunks_dir["lookup_path"],
            enable=False,
        )
        hybrid = HybridRetriever(
            bm25=bm25,
            vector=vector,
            manifest_path=empty_dir / "manifest.json",
        )
        results = hybrid.search("缓考")
        assert len(results) > 0
        detail = results[0]["score_detail"]
        assert "bm25_raw" in detail
        assert "vector_raw" in detail
        assert "priority_bonus" in detail
        assert "bm25_weight" in detail
        assert "vector_weight" in detail
