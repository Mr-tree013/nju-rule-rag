"""
Online retrieval module.

Provides a ``Retriever`` protocol and three implementations:
- ``BM25Retriever`` — keyword-based search via jieba + rank-bm25
- ``VectorRetriever`` — semantic search via ChromaDB + sentence-transformers
- ``HybridRetriever`` — weighted fusion of BM25 + vector + priority bonus

All retrievers gracefully degrade when their backing index is missing,
logging a warning and returning empty results.
"""

import json
import pickle
import threading
from pathlib import Path
from typing import Callable, Protocol

import chromadb
import jieba
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from app.config import RetrievalWeights
from app.errors import RetrievalError

# ── Tokenizer ────────────────────────────────────────────────────────


def default_tokenizer(text: str) -> list[str]:
    """Chinese word segmentation via jieba (default for BM25)."""
    return list(jieba.cut(text))


# ── Protocol ─────────────────────────────────────────────────────────


class Retriever(Protocol):
    """Structural interface for any chunk retriever."""

    def search(self, question: str, top_k: int = 5) -> list[dict]: ...

    @property
    def is_loaded(self) -> bool: ...

    @property
    def chunk_count(self) -> int: ...


# ── BM25 Retriever ───────────────────────────────────────────────────


class BM25Retriever:
    """
    BM25 keyword retriever.

    Loads a pre-built ``bm25.pkl`` index when available; falls back to
    building a temporary in-memory index from chunks otherwise (development
    convenience — prints a warning).
    """

    COLLECTION = "nju_rules"

    def __init__(
        self,
        chunks_path: Path,
        index_path: Path,
        chunk_lookup_path: Path,
        tokenizer: Callable[[str], list[str]] | None = None,
    ):
        self._chunks: list[dict] = []
        self._chunks_by_id: dict[str, dict] = {}
        self._bm25: BM25Okapi | None = None
        self._loaded = False
        self._tokenizer = tokenizer or default_tokenizer

        self._load_chunks(chunks_path, chunk_lookup_path)
        if index_path.exists():
            self._load_index(index_path)
        else:
            self._build_fallback()
        self._loaded = self._bm25 is not None

    # ── Loading ──────────────────────────────────────────────────

    def _load_chunks(self, chunks_path: Path, lookup_path: Path):
        if lookup_path.exists():
            with open(lookup_path, encoding="utf-8") as f:
                lookup = json.load(f)
            self._chunks = list(lookup.values())
            self._chunks_by_id = lookup
        elif chunks_path.exists():
            with open(chunks_path, encoding="utf-8") as f:
                for line in f:
                    if not (line := line.strip()):
                        continue
                    c = json.loads(line)
                    self._chunks.append(c)
                    self._chunks_by_id[c["chunk_id"]] = c

    def _load_index(self, path: Path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["model"]
            if "chunks" in data and data["chunks"]:
                self._chunks = data["chunks"]
                self._chunks_by_id = {c["chunk_id"]: c for c in data["chunks"]}
        except Exception as exc:
            print(f"[BM25Retriever] 加载索引失败 ({exc})，回退到动态构建。")
            self._build_fallback()

    def _build_fallback(self):
        if not self._chunks:
            return
        print(
            f"[BM25Retriever] 警告：未找到离线索引，"
            f"从 {len(self._chunks)} 个 chunk 动态构建 BM25。"
            " 生产环境请运行 scripts/build_index.py。"
        )
        try:
            corpus = [c["content"] for c in self._chunks]
            tokenized = [self._tokenizer(doc) for doc in corpus]
            self._bm25 = BM25Okapi(tokenized)
        except Exception as exc:
            print(f"[BM25Retriever] 动态构建失败: {exc}")

    # ── Properties ───────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    # ── Search ───────────────────────────────────────────────────

    def search(self, question: str, top_k: int = 10) -> list[dict]:
        if not self._bm25 or not self._chunks or not (question and question.strip()):
            return []

        try:
            tokens = self._tokenizer(question)
            scores = self._bm25.get_scores(tokens)
            scored = [(i, s) for i, s in enumerate(scores) if s > 0]
            scored.sort(key=lambda x: x[1], reverse=True)
            scored = scored[:top_k]

            results: list[dict] = []
            for idx, score in scored:
                if idx >= len(self._chunks):
                    continue
                c = self._chunks[idx]
                results.append(
                    {
                        "chunk_id": c["chunk_id"],
                        "source_id": c.get("source_id", ""),
                        "title": c["title"],
                        "content": c["content"],
                        "url": c.get("url", ""),
                        "priority": c.get("priority", 5),
                        "score": round(float(score), 4),
                    }
                )
            return results
        except Exception as exc:
            print(f"[BM25Retriever] 检索异常: {exc}")
            return []


# ── Vector Retriever ─────────────────────────────────────────────────


class VectorRetriever:
    """
    ChromaDB vector retriever.

    Uses a local sentence-transformers model for embedding.
    Gracefully degrades (returns empty results) when the Chroma directory
    is missing or fails to load.
    """

    COLLECTION = "nju_rules"

    def __init__(
        self,
        chroma_path: Path,
        chunks_path: Path,
        chunk_lookup_path: Path,
        embedding_model: str = "shibing624/text2vec-base-chinese",
        enable: bool = True,
    ):
        self._collection = None
        self._chunks_by_id: dict[str, dict] = {}
        self._loaded = False
        self._gpu_lock = threading.RLock()  # serializes GPU access to embedding model

        self._load_chunks(chunks_path, chunk_lookup_path)
        if enable:
            self._load_index(chroma_path, embedding_model)

    def _load_chunks(self, chunks_path: Path, lookup_path: Path):
        if lookup_path.exists():
            with open(lookup_path, encoding="utf-8") as f:
                self._chunks_by_id = json.load(f)
            return
        if not chunks_path.exists():
            return
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                if not (line := line.strip()):
                    continue
                c = json.loads(line)
                self._chunks_by_id[c["chunk_id"]] = c

    def _load_index(self, chroma_path: Path, embedding_model: str):
        if not chroma_path.exists():
            print("[VectorRetriever] Chroma 目录不存在，跳过向量检索。")
            return
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._embedding_model = self._load_embedding_model(embedding_model)

            self._collection = client.get_collection(self.COLLECTION)
            self._loaded = True
        except Exception as exc:
            print(f"[VectorRetriever] 加载 Chroma 失败 ({exc})，向量检索不可用。")

    @staticmethod
    def _load_embedding_model(model_name: str):
        """Try local path first, then HuggingFace model name, then give up."""
        from pathlib import Path as _Path
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check for model cached locally in the project.
        project_root = _Path(__file__).resolve().parent.parent
        local_model = project_root / "data" / "models" / model_name.replace("/", "--")
        # Also try without the org prefix (e.g. "text2vec-base-chinese")
        local_short = project_root / "data" / "models" / model_name.split("/")[-1]

        for candidate in (local_model, local_short):
            if candidate.exists():
                try:
                    return SentenceTransformer(str(candidate), device=device)
                except Exception:
                    pass

        # Fall back to HuggingFace (needs network).
        try:
            return SentenceTransformer(model_name, device=device)
        except Exception:
            return None

    @property
    def embedding_model(self):
        """Expose the loaded SentenceTransformer model for downstream use."""
        return self._embedding_model

    @property
    def gpu_lock(self) -> threading.RLock:
        """Lock that serializes GPU access to the embedding model."""
        return self._gpu_lock

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def chunk_count(self) -> int:
        return len(self._chunks_by_id)

    def search(self, question: str, top_k: int = 10) -> list[dict]:
        if not self._collection or not (question and question.strip()):
            return []

        try:
            if self._embedding_model is not None:
                with self._gpu_lock:
                    vec = self._embedding_model.encode(question).tolist()
                raw = self._collection.query(query_embeddings=[vec], n_results=top_k)
            else:
                raw = self._collection.query(query_texts=[question], n_results=top_k)
            ids_list = raw.get("ids", [[]])[0]
            distances_list = raw.get("distances", [[]])[0]

            results: list[dict] = []
            for cid, dist in zip(ids_list, distances_list):
                chunk = self._chunks_by_id.get(cid)
                if not chunk:
                    continue
                sim = max(0.0, 1.0 - dist / 2.0)
                results.append(
                    {
                        "chunk_id": cid,
                        "source_id": chunk.get("source_id", ""),
                        "title": chunk["title"],
                        "content": chunk["content"],
                        "url": chunk.get("url", ""),
                        "priority": chunk.get("priority", 5),
                        "score": round(sim, 4),
                    }
                )
            return results
        except Exception as exc:
            print(f"[VectorRetriever] 检索异常: {exc}")
            return []


# ── Hybrid Retriever ─────────────────────────────────────────────────


class HybridRetriever:
    """
    Weighted fusion of BM25 keyword search and Chroma vector search.

    Score formula::

        final = bm25_norm × w_bm25 + vector_norm × w_vector + priority_bonus × w_priority

    where ``priority_bonus = (6 - priority) / 5`` (priority 1 → 1.0, priority 5 → 0.2).

    If one sub-retriever is unavailable, weights are adjusted automatically.
    """

    # Sources whose chunks tend to match many queries via common question words
    # (Q&A format with lots of 怎么/什么/申请/流程).  Applied as a multiplier on
    # the final hybrid score to reduce false-top-1 hits.
    _QA_SOURCE_PENALTY: float = 0.85  # reduced from 0.65 — BGE-M3 semantic matching is strong enough

    _source_boost: dict[str, float] = {}

    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        manifest_path: Path,
        weights: RetrievalWeights | None = None,
        top_k: int = 5,
        source_boost: dict[str, float] | None = None,
    ):
        self._bm25 = bm25
        self._vector = vector
        self._weights = weights or RetrievalWeights()
        self._top_k = top_k
        self._manifest = self._read_manifest(manifest_path)
        # Per-source score multipliers: key=source_id prefix or full id, value=multiplier
        self._source_boost = source_boost or {}

    @staticmethod
    def _read_manifest(path: Path) -> dict:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @classmethod
    def _resolve_boost(cls, source_id: str) -> float:
        """Return the score multiplier for *source_id*.

        Q&A-style daily-life docs (nju-life-*) get a slight penalty because
        their common question words (怎么/什么/申请) cause BM25 over-matching.
        Official 本科生院 docs (nju-jw-*) get a slight boost.
        """
        # Explicit per-source overrides
        if source_id in cls._source_boost:
            return cls._source_boost[source_id]
        if source_id.startswith("nju-life-"):
            return cls._QA_SOURCE_PENALTY
        return 1.0

    # ── Status ───────────────────────────────────────────────────

    def status(self) -> dict:
        m = self._manifest
        return {
            "bm25_loaded": self._bm25.is_loaded,
            "bm25_chunks": self._bm25.chunk_count,
            "vector_loaded": self._vector.is_loaded,
            "manifest_built_at": m.get("built_at", ""),
            "manifest_chunk_count": m.get("chunk_count", 0),
            "manifest_status": m.get("status", "missing"),
        }

    # ── Search ───────────────────────────────────────────────────

    def search(self, question: str, top_k: int | None = None,
               source_filter: set[str] | None = None) -> list[dict]:
        k = top_k if top_k is not None else self._top_k
        if not question or not question.strip():
            return []

        bm25_raw = self._bm25.search(question, top_k=k)
        vector_raw = self._vector.search(question, top_k=k)

        use_bm25 = len(bm25_raw) > 0
        use_vector = len(vector_raw) > 0

        if not use_bm25 and not use_vector:
            return []

        # Select weight profile based on available retrievers
        if use_bm25 and use_vector:
            w = self._weights
        elif use_bm25:
            w = self._weights.fallback_bm25_only()
        else:
            w = self._weights.fallback_vector_only()

        bm25_norm = self._minmax_norm(bm25_raw)
        vector_norm = self._minmax_norm(vector_raw)

        merged: dict[str, dict] = {}
        for item in bm25_norm:
            cid = item["chunk_id"]
            merged[cid] = {
                "bm25_score": item["score"],
                "vector_score": 0.0,
                "priority": item["priority"],
                "chunk": item,
            }
        for item in vector_norm:
            cid = item["chunk_id"]
            if cid in merged:
                merged[cid]["vector_score"] = item["score"]
            else:
                merged[cid] = {
                    "bm25_score": 0.0,
                    "vector_score": item["score"],
                    "priority": item["priority"],
                    "chunk": item,
                }

        scored: list[tuple[str, float, dict]] = []
        for cid, data in merged.items():
            # Topic routing: skip chunks from excluded sources
            src_id = data["chunk"].get("source_id", "")
            if source_filter is not None and src_id not in source_filter:
                continue
            priority_bonus = (6 - data["priority"]) / 5.0
            final = (
                data["bm25_score"] * w.bm25
                + data["vector_score"] * w.vector
                + priority_bonus * w.priority
            )
            # Apply per-source score boost/penalty
            boost = self._resolve_boost(src_id)
            if boost != 1.0:
                final *= boost
            scored.append((cid, final, data))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:k]

        results: list[dict] = []
        for cid, final_score, data in scored:
            chunk = data["chunk"]
            results.append(
                {
                    "chunk_id": cid,
                    "source_id": chunk.get("source_id", ""),
                    "title": chunk["title"],
                    "content": chunk["content"],
                    "url": chunk.get("url", ""),
                    "priority": chunk["priority"],
                    "score": round(final_score, 4),
                    "score_detail": {
                        "bm25_raw": round(data["bm25_score"], 4),
                        "vector_raw": round(data["vector_score"], 4),
                        "priority_bonus": round(
                            (6 - data["priority"]) / 5.0 * w.priority, 4
                        ),
                        "bm25_weight": w.bm25,
                        "vector_weight": w.vector,
                        "priority_weight": w.priority,
                    },
                }
            )
        return results

    @staticmethod
    def _minmax_norm(items: list[dict]) -> list[dict]:
        if not items:
            return []
        scores = [x["score"] for x in items]
        mn, mx = min(scores), max(scores)
        if mx == mn:
            return [{**x, "score": 1.0} for x in items]
        return [{**x, "score": (x["score"] - mn) / (mx - mn)} for x in items]


# ── Convenience factory ──────────────────────────────────────────────


def create_retriever(
    chunks_path: Path | str = "data/chunks/chunks.jsonl",
    index_dir: Path | str = "data/index",
    embedding_model: str = "shibing624/text2vec-base-chinese",
    enable_vector: bool = True,
    weights: RetrievalWeights | None = None,
) -> HybridRetriever:
    """Build a ``HybridRetriever`` from paths on disk.

    This is the quick-start factory.  For production, prefer constructing
    each retriever explicitly via ``deps.create_retriever()``.
    """
    root = Path(__file__).resolve().parent.parent
    cp = _resolve(root, chunks_path)
    idx = _resolve(root, index_dir)

    bm25 = BM25Retriever(
        chunks_path=cp,
        index_path=idx / "bm25.pkl",
        chunk_lookup_path=idx / "chunk_lookup.json",
    )
    vector = VectorRetriever(
        chroma_path=idx / "chroma",
        chunks_path=cp,
        chunk_lookup_path=idx / "chunk_lookup.json",
        embedding_model=embedding_model,
        enable=enable_vector,
    )
    retriever = HybridRetriever(
        bm25=bm25,
        vector=vector,
        manifest_path=idx / "manifest.json",
        weights=weights,
    )
    status = retriever.status()
    print(
        f"[Retriever] BM25={status['bm25_loaded']}({status['bm25_chunks']} chunks), "
        f"Vector={status['vector_loaded']}"
    )
    return retriever


def _resolve(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p
