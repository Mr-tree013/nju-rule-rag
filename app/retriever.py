"""
在线检索模块。

从离线构建的索引加载 BM25 和 Chroma，对用户问题执行混合检索。

不负责：
- 构建索引（由 scripts/build_index.py 负责）
- 调用 LLM（由 app/llm_client.py 负责）

依赖：
- data/chunks/chunks.jsonl           → chunk 原始数据
- data/index/bm25_index.pkl          → BM25 索引（可选，缺失时从 chunks 临时构建）
- data/index/chroma/                 → 向量索引（可选，缺失时仅运行 BM25）
"""

import json
import os
import pickle
from pathlib import Path

import chromadb
import jieba
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from app.config import (
    BM25_TOP_K,
    CHUNKS_FILE,
    HYBRID_TOP_K,
    INDEX_DIR,
    VECTOR_TOP_K,
)

# --------------------------------------------------------------------
# 路径解析
# --------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


def _resolve(path_str):
    p = Path(path_str)
    if p.is_absolute():
        return p
    return ROOT / p


CHUNKS_PATH = _resolve(CHUNKS_FILE)
INDEX_PATH = _resolve(INDEX_DIR)
BM25_PATH = INDEX_PATH / "bm25.pkl"
CHROMA_PATH = INDEX_PATH / "chroma"
MANIFEST_PATH = INDEX_PATH / "manifest.json"
CHUNK_LOOKUP_PATH = INDEX_PATH / "chunk_lookup.json"
COLLECTION_NAME = "nju_rules"

# --------------------------------------------------------------------
# 分词工具
# --------------------------------------------------------------------


def tokenize(text):
    """中文分词，BM25 构建和搜索公用。"""
    return list(jieba.cut(text))


# --------------------------------------------------------------------
# BM25Retriever
# --------------------------------------------------------------------


class BM25Retriever:
    """
    BM25 关键词检索器。

    优先从离线构建的 bm25_index.pkl 加载索引；
    如果文件不存在，从 chunks 动态构建（仅开发 / 兜底使用，会打印 warning）。
    """

    def __init__(self):
        self._chunks = []
        self._chunks_by_id = {}
        self._bm25 = None
        self._loaded = False

        self._load_chunks()
        if BM25_PATH.exists():
            self._load_index()
        else:
            self._build_fallback()
        self._loaded = self._bm25 is not None

    # ---- 加载 ----

    def _load_chunks(self):
        """从 chunk_lookup.json 或 chunks.jsonl 加载。"""
        if CHUNK_LOOKUP_PATH.exists():
            self._load_chunk_lookup()
        elif CHUNKS_PATH.exists():
            self._load_chunks_jsonl()
        else:
            print(f"[BM25Retriever] chunks 数据不存在: {CHUNKS_PATH}")

    def _load_chunk_lookup(self):
        with open(CHUNK_LOOKUP_PATH, encoding="utf-8") as f:
            lookup = json.load(f)
        self._chunks = list(lookup.values())
        self._chunks_by_id = lookup

    def _load_chunks_jsonl(self):
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                self._chunks.append(c)
                self._chunks_by_id[c["chunk_id"]] = c

    def _load_index(self):
        try:
            with open(BM25_PATH, "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["model"]
            # 优先使用索引中自带的 chunks（与索引构建时一致）
            if "chunks" in data and data["chunks"]:
                self._chunks = data["chunks"]
                self._chunks_by_id = {c["chunk_id"]: c for c in data["chunks"]}
        except Exception as e:
            print(f"[BM25Retriever] 加载索引失败 ({e})，回退到动态构建。")
            self._build_fallback()

    def _build_fallback(self):
        if not self._chunks:
            return
        print(
            f"[BM25Retriever] 警告：未找到离线索引，从 {len(self._chunks)} 个 chunk 动态构建 BM25。"
            " 生产环境请运行 scripts/build_index.py。"
        )
        try:
            corpus = [c["content"] for c in self._chunks]
            tokenized = [tokenize(doc) for doc in corpus]
            self._bm25 = BM25Okapi(tokenized)
        except Exception as e:
            print(f"[BM25Retriever] 动态构建失败: {e}")

    # ---- 状态 ----

    @property
    def is_loaded(self):
        return self._loaded

    @property
    def chunk_count(self):
        return len(self._chunks)

    # ---- 检索 ----

    def search(self, question, top_k=None):
        """
        返回列表，每项包含：
          chunk_id, title, content, url, priority, score
        """
        if top_k is None:
            top_k = BM25_TOP_K

        if not self._bm25 or not self._chunks:
            return []

        if not question or not question.strip():
            return []

        try:
            tokens = tokenize(question)
            scores = self._bm25.get_scores(tokens)

            # 收集 (index, score) 对，按分数降序取 top_k
            scored = [(i, s) for i, s in enumerate(scores) if s > 0]
            scored.sort(key=lambda x: x[1], reverse=True)
            scored = scored[:top_k]

            results = []
            for idx, score in scored:
                if idx >= len(self._chunks):
                    continue
                c = self._chunks[idx]
                results.append(
                    {
                        "chunk_id": c["chunk_id"],
                        "title": c["title"],
                        "content": c["content"],
                        "url": c.get("url", ""),
                        "priority": c.get("priority", 5),
                        "score": round(float(score), 4),
                    }
                )
            return results
        except Exception as e:
            print(f"[BM25Retriever] 检索异常: {e}")
            return []


# --------------------------------------------------------------------
# VectorRetriever
# --------------------------------------------------------------------


class VectorRetriever:
    """
    Chroma 向量检索器。

    默认使用 Chroma 内置的 ONNX embedding（all-MiniLM-L6-v2）。
    如果 Chroma 目录不存在或加载失败，优雅降级，不中断服务。
    """

    def __init__(self):
        self._collection = None
        self._chunks_by_id = {}
        self._loaded = False

        self._load_chunks()
        self._load_index()

    def _load_chunks(self):
        """从 chunk_lookup.json 或 chunks.jsonl 加载。"""
        if CHUNK_LOOKUP_PATH.exists():
            with open(CHUNK_LOOKUP_PATH, encoding="utf-8") as f:
                self._chunks_by_id = json.load(f)
            return
        if not CHUNKS_PATH.exists():
            return
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                self._chunks_by_id[c["chunk_id"]] = c

    def _load_index(self):
        if not CHROMA_PATH.exists():
            print("[VectorRetriever] Chroma 目录不存在，跳过向量检索。")
            return
        try:
            client = chromadb.PersistentClient(
                path=str(CHROMA_PATH),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = client.get_collection(COLLECTION_NAME)
            self._loaded = True
        except Exception as e:
            print(f"[VectorRetriever] 加载 Chroma 失败 ({e})，向量检索不可用。")

    # ---- 状态 ----

    @property
    def is_loaded(self):
        return self._loaded

    # ---- 检索 ----

    def search(self, question, top_k=None):
        """
        返回列表，每项包含：
          chunk_id, title, content, url, priority, score

        score 由余弦距离转换而来（0=不相关 ~ 1=完全匹配）。
        """
        if top_k is None:
            top_k = VECTOR_TOP_K

        if not self._collection:
            return []

        if not question or not question.strip():
            return []

        try:
            raw = self._collection.query(
                query_texts=[question],
                n_results=top_k,
            )
            ids_list = raw.get("ids", [[]])[0]
            distances_list = raw.get("distances", [[]])[0]

            results = []
            for cid, dist in zip(ids_list, distances_list):
                chunk = self._chunks_by_id.get(cid)
                if not chunk:
                    continue
                # 余弦距离 ∈ [0, 2]，转为分数：距离 0=完美匹配→1.0，距离 2=完全相反→0.0
                sim = max(0.0, 1.0 - dist / 2.0)
                results.append(
                    {
                        "chunk_id": cid,
                        "title": chunk["title"],
                        "content": chunk["content"],
                        "url": chunk.get("url", ""),
                        "priority": chunk.get("priority", 5),
                        "score": round(sim, 4),
                    }
                )
            return results
        except Exception as e:
            print(f"[VectorRetriever] 检索异常: {e}")
            return []


# --------------------------------------------------------------------
# HybridRetriever
# --------------------------------------------------------------------


class HybridRetriever:
    """
    混合检索器，融合 BM25 关键词匹配与 Chroma 语义匹配。

    最终公式：
      final_score = bm25_norm * 0.5 + vector_norm * 0.4 + priority_score * 0.1

    其中 priority_score = (6 - priority) / 5，使 priority=1 → 1.0，priority=5 → 0.2。

    当一路不可用时，权重自动调整，保证仍能返回结果。
    """

    def __init__(self):
        self._bm25 = BM25Retriever()
        self._vector = VectorRetriever()

    # ---- 状态 ----

    def status(self):
        return {
            "bm25_loaded": self._bm25.is_loaded,
            "bm25_chunks": self._bm25.chunk_count,
            "vector_loaded": self._vector.is_loaded,
        }

    # ---- 检索 ----

    def search(self, question, top_k=None):
        """
        执行混合检索，返回 top_k 条去重结果。

        每条结果包含：
          chunk_id, title, content, url, priority, score, score_detail
        """
        if top_k is None:
            top_k = HYBRID_TOP_K

        if not question or not question.strip():
            return []

        # 两路并行召回
        bm25_raw = self._bm25.search(question, top_k=BM25_TOP_K)
        vector_raw = self._vector.search(question, top_k=VECTOR_TOP_K)

        use_bm25 = len(bm25_raw) > 0
        use_vector = len(vector_raw) > 0

        # 两路都不可用
        if not use_bm25 and not use_vector:
            return []

        # 归一化各路的分数
        bm25_norm = self._normalize(bm25_raw)
        vector_norm = self._normalize(vector_raw)

        # 融合权重：缺一路时另一路权重提高
        if use_bm25 and use_vector:
            w_bm25, w_vec = 0.5, 0.4
        elif use_bm25:
            w_bm25, w_vec = 0.9, 0.0
        else:
            w_bm25, w_vec = 0.0, 0.9

        # 按 chunk_id 合并去重，同时保留各路分数用于 score_detail
        merged = {}  # chunk_id → { "bm25_score", "vector_score", "priority", "chunk" }
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
                if merged[cid]["chunk"]["score"] < item["score"]:
                    # 如果向量分数更高，用向量的 chunk（可能 title/content 更精确）
                    pass
            else:
                merged[cid] = {
                    "bm25_score": 0.0,
                    "vector_score": item["score"],
                    "priority": item["priority"],
                    "chunk": item,
                }

        # 计算最终分数
        scored = []
        for cid, data in merged.items():
            priority_score = (6 - data["priority"]) / 5.0  # 1→1.0, 5→0.2
            final = (
                data["bm25_score"] * w_bm25
                + data["vector_score"] * w_vec
                + priority_score * 0.1
            )
            scored.append((cid, final, data))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        # 组装输出
        results = []
        for cid, final_score, data in scored:
            chunk = data["chunk"]
            results.append(
                {
                    "chunk_id": cid,
                    "title": chunk["title"],
                    "content": chunk["content"],
                    "url": chunk.get("url", ""),
                    "priority": chunk["priority"],
                    "score": round(final_score, 4),
                    "score_detail": {
                        "bm25_raw": round(data["bm25_score"], 4),
                        "vector_raw": round(data["vector_score"], 4),
                        "priority_bonus": round(
                            (6 - data["priority"]) / 5.0 * 0.1, 4
                        ),
                        "bm25_weight": w_bm25,
                        "vector_weight": w_vec,
                    },
                }
            )
        return results

    # ---- 工具 ----

    @staticmethod
    def _normalize(items):
        """Min-max 归一化到 [0, 1]，列表为空或全零时返回原列表。"""
        if not items:
            return []
        scores = [x["score"] for x in items]
        mn = min(scores)
        mx = max(scores)
        if mx == mn:
            return [{**x, "score": 1.0} for x in items]
        return [{**x, "score": (x["score"] - mn) / (mx - mn)} for x in items]


# --------------------------------------------------------------------
# 便捷入口
# --------------------------------------------------------------------

def create_retriever():
    """创建 HybridRetriever 实例并打印加载状态。"""
    retriever = HybridRetriever()
    status = retriever.status()
    print(f"[Retriever] BM25={status['bm25_loaded']}({status['bm25_chunks']} chunks), "
          f"Vector={status['vector_loaded']}")
    return retriever
