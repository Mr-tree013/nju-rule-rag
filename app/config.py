"""
Application configuration loaded from environment variables.

Usage:
    from app.config import create_settings
    settings = create_settings()  # reads os.environ
    warnings = settings.validate()
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── System prompt (long enough to warrant its own constant) ──────────

DEFAULT_SYSTEM_PROMPT = """你是一个南京大学本科新生校规与教务流程问答助手。

你必须严格遵守以下规则：

1. 只能依据下面提供的【参考资料片段】进行回答，不得使用任何外部知识。
2. 不得编造任何文件名、文件编号、具体日期、条款编号或链接。如果资料中没有，就说没有。
3. 如果认真阅读了所有资料片段后，确实没有任何一条包含与问题相关的内容，才可以说"抱歉，目前没有找到与您问题相关的足够可靠的校规依据"。请先仔细检查每条资料，不要因为前几条不相关就直接放弃。
4. 对于涉及退学、开除、处分、作弊、学位等高风险问题，只提供校规中已有的客观规定描述，不得对用户个人情况做出判断或结论。
5. 回答要简洁直接，控制在 500 字以内，适合在 QQ 群里快速阅读。
6. 不要在回答中提及"根据参考资料"、"资料显示"等引用词，直接给出答案即可。
7. 如果用户问题与校规、教务流程完全无关，礼貌说明你只能回答校规相关问题。
8. 回答格式：先给出简短的直接结论（1-2句），再列出关键要点（如有必要）。不要添加客套话和无关内容。
9. 资料片段可能包含多条不相关内容，请跳过无关内容，只基于相关的片段回答。如果后面的片段包含答案，用它回答。"""


# ── Settings ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievalWeights:
    """Weight configuration for hybrid retrieval score merging."""

    bm25: float = 0.45
    vector: float = 0.35
    priority: float = 0.20

    def validate(self) -> list[str]:
        total = self.bm25 + self.vector + self.priority
        if abs(total - 1.0) > 0.01:
            return [f"检索权重之和应为 1.0，当前为 {total}"]
        return []

    def fallback_bm25_only(self) -> "RetrievalWeights":
        return RetrievalWeights(bm25=0.80, vector=0.00, priority=0.20)

    def fallback_vector_only(self) -> "RetrievalWeights":
        return RetrievalWeights(bm25=0.00, vector=0.80, priority=0.20)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment variables."""

    # ── App ──────────────────────────────────────────────────────

    app_title: str = "NJU Rule RAG"

    # ── LLM ──────────────────────────────────────────────────────

    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""

    # ── Embedding (API mode) ─────────────────────────────────────

    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""

    # ── Data paths ───────────────────────────────────────────────

    data_dir: str = "data"
    chunks_file: str = "data/chunks/chunks.jsonl"
    index_dir: str = "data/index"

    # ── Retrieval ────────────────────────────────────────────────

    bm25_top_k: int = 10
    vector_top_k: int = 10
    hybrid_top_k: int = 5
    retrieval_weights: RetrievalWeights = field(default_factory=RetrievalWeights)
    min_reliable_score: float = 0.2
    high_risk_min_score: float = 0.25

    # ── Embedding (local) ────────────────────────────────────────

    enable_vector: bool = True
    local_embedding_model: str = "shibing624/text2vec-base-chinese"

    # ── LLM fallback ─────────────────────────────────────────────

    enable_llm_fallback: bool = False
    fallback_llm_api_key: str = ""
    fallback_llm_base_url: str = ""
    fallback_llm_model: str = ""

    # ── LLM retry ────────────────────────────────────────────────

    retry_count: int = 3
    retry_delays: tuple = (1, 2, 4)
    request_timeout: int = 60

    # ── Pipeline ─────────────────────────────────────────────────

    max_answer_length: int = 600
    max_context_chunks: int = 12
    max_chunks_per_source: int = 3
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # ── QQ Bot ───────────────────────────────────────────────────

    qq_bot_self_id: str = ""
    qq_bot_api_base_url: str = "http://127.0.0.1:8000"
    qq_bot_max_reply_length: int = 800
    qq_bot_request_timeout: int = 30

    # ── Validation ───────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Check for configuration problems.  Returns a list of warnings."""
        warnings = []

        if not self.llm_api_key:
            warnings.append("LLM_API_KEY 未设置，LLM 调用将失败")
        if not self.llm_model:
            warnings.append("LLM_MODEL 未设置，LLM 调用将失败")

        warnings.extend(self.retrieval_weights.validate())

        chunks = Path(self.chunks_file)
        if not chunks.exists():
            warnings.append(f"chunks 文件不存在: {self.chunks_file}")

        return warnings

    @property
    def project_root(self) -> Path:
        """Absolute path to the project root (parent of data_dir)."""
        data = Path(self.data_dir)
        if data.is_absolute():
            return data.parent
        return Path(__file__).resolve().parent.parent


# ── Factory ──────────────────────────────────────────────────────────


def create_settings() -> Settings:
    """Build a Settings instance from the current environment.

    Call after ``load_dotenv()`` so ``.env`` values are present in ``os.environ``.
    """
    return Settings(
        app_title=os.getenv("APP_TITLE", "NJU Rule RAG"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_model=os.getenv("LLM_MODEL", ""),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", ""),
        data_dir=os.getenv("DATA_DIR", "data"),
        chunks_file=os.getenv("CHUNKS_FILE", "data/chunks/chunks.jsonl"),
        index_dir=os.getenv("INDEX_DIR", "data/index"),
        bm25_top_k=_int("BM25_TOP_K", 10),
        vector_top_k=_int("VECTOR_TOP_K", 10),
        hybrid_top_k=_int("HYBRID_TOP_K", 5),
        retrieval_weights=RetrievalWeights(),
        min_reliable_score=_float("MIN_RELIABLE_SCORE", 0.2),
        high_risk_min_score=_float("HIGH_RISK_MIN_SCORE", 0.25),
        enable_vector=os.getenv("ENABLE_VECTOR", "true").lower() not in ("false", "0", "no"),
        local_embedding_model=os.getenv("LOCAL_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"),
        enable_llm_fallback=os.getenv("ENABLE_LLM_FALLBACK", "false").lower() in ("true", "1", "yes"),
        fallback_llm_api_key=os.getenv("FALLBACK_LLM_API_KEY", ""),
        fallback_llm_base_url=os.getenv("FALLBACK_LLM_BASE_URL", ""),
        fallback_llm_model=os.getenv("FALLBACK_LLM_MODEL", ""),
        retry_count=3,
        retry_delays=(1, 2, 4),
        request_timeout=60,
        max_answer_length=600,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        qq_bot_self_id=os.getenv("QQ_BOT_SELF_ID", ""),
        qq_bot_api_base_url=os.getenv("QQ_BOT_API_BASE_URL", "http://127.0.0.1:8000"),
        qq_bot_max_reply_length=800,
        qq_bot_request_timeout=30,
    )


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


# ── Deprecated: module-level accessors for backward compat ──────────

_settings: Settings | None = None


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = create_settings()
    return _settings


def get_settings() -> Settings:
    """Return the cached Settings singleton (lazy-loaded)."""
    return _get_settings()


# Module-level accessors matching the old API.
# New code should use ``settings = create_settings()`` directly.
def __getattr__(name: str):
    _map = {
        "APP_TITLE": lambda s: s.app_title,
        "LLM_API_KEY": lambda s: s.llm_api_key,
        "LLM_BASE_URL": lambda s: s.llm_base_url,
        "LLM_MODEL": lambda s: s.llm_model,
        "EMBEDDING_API_KEY": lambda s: s.embedding_api_key,
        "EMBEDDING_BASE_URL": lambda s: s.embedding_base_url,
        "EMBEDDING_MODEL": lambda s: s.embedding_model,
        "DATA_DIR": lambda s: s.data_dir,
        "CHUNKS_FILE": lambda s: s.chunks_file,
        "INDEX_DIR": lambda s: s.index_dir,
        "BM25_TOP_K": lambda s: s.bm25_top_k,
        "VECTOR_TOP_K": lambda s: s.vector_top_k,
        "HYBRID_TOP_K": lambda s: s.hybrid_top_k,
        "MIN_RELIABLE_SCORE": lambda s: s.min_reliable_score,
        "HIGH_RISK_MIN_SCORE": lambda s: s.high_risk_min_score,
        "LOCAL_EMBEDDING_MODEL": lambda s: s.local_embedding_model,
        "QQ_BOT_SELF_ID": lambda s: s.qq_bot_self_id,
    }
    if name in _map:
        return _map[name](_get_settings())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
