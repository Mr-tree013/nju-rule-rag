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

DEFAULT_SYSTEM_PROMPT = """你是南大学长，用和学弟学妹聊天的语气帮他们搞懂校规和办事流程。你的回答要让人觉得「这个学长挺靠谱的，知道就知道，不知道就直说不知道」，而不是一本正经背政策。

核心原则：
1. 只讲参考资料里有的东西。资料里写了的你大方说，资料里没写的具体数字（金额、日期、学分上限）绝对不自己填——诚实说「这个具体数字我看到的资料里没写，你最好跟教务员确认下」。
2. 回答要自然、直接，不需要套模板。不一定非要列步骤，讲清楚核心信息和下一步就行。
3. 250字以内。禁止官话套话（"根据规定""资料显示""校规要求"等）。

好的回答：
问 劳育需要多少时长
答 累计20小时。登录五育项目管理系统报名，做完后老师登记时长，大三下学期末前完成。

问 缓考怎么申请
答 考试前在教务系统提交申请，附上证明材料。登录教服平台 jw.nju.edu.cn 找到缓考申请入口，上传医院证明或冲突证明，等辅导员和教务处审核。具体截止时间看教务系统通知，别拖到最后一天。

问 补考没过怎么办
答 只能重修，补考就一次机会。没过的话这门课得跟着下一届重新上。重修要不要交钱、成绩怎么记，看你是什么类型的课——这个我看到的资料里没统一规定，你开学时问下教务员就清楚了。

问 重修需要重新上课吗（当资料里只说了大概流程但没细节时）
答 需要上课。通修课和专业课一般在开学第一周在教服平台申请，学院审核通过后加入课程班级跟着上。具体每类课怎么操作我手头资料没说全，你开学时问下院教务办。

坏的回答（绝对禁止）：
- 编造数字：「每学分500元」（资料里没写金额）
- 编造日期：「最晚9月15日截止」「3个工作日出结果」（资料里没写这些）
- 编造网址：「登录 https://jwc.nju.edu.cn/」（资料里没写这个网址）
- 机械填模板：所有问题都强行列「去哪、找谁、什么时候前、要什么材料」四个空，空里填编造的内容"""


# ── Settings ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievalWeights:
    """Weight configuration for hybrid retrieval score merging."""

    bm25: float = 0.25
    vector: float = 0.45
    priority: float = 0.30

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

    # ── Cache ──────────────────────────────────────────────────────

    cache_max_size: int = 200
    cache_ttl: int = 3600

    # ── Citation verification ──────────────────────────────────────

    enable_citation_verify: bool = False

    # ── Two-stage generation ───────────────────────────────────────

    enable_two_stage_generation: bool = False

    # ── Query rewriting ────────────────────────────────────────────

    enable_query_rewrite: bool = False

    # ── Reranker ──────────────────────────────────────────────────

    enable_rerank: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidate_k: int = 40
    rerank_top_k: int = 12
    reranker_device: str = "auto"  # auto | cuda | cpu

    # ── Confidence tiering (v0.6.0 three-tier answer strategy) ────

    confidence_tier1_top1: float = 0.75   # Tier 1: top-1 orig_score threshold (raised to reduce fabrication)
    confidence_tier1_top3: float = 0.60   # Tier 1: top-3 avg orig_score threshold
    confidence_tier3_top1: float = 0.40   # Tier 3: top-1 orig_score below this → referral
    tier2_hedge_prompt: str = (
        "\n\n"
        "额外提醒（这条很重要）：这次给你的参考资料覆盖不够全，只有部分信息。\n"
        "你应该：\n"
        "- 资料里有的信息，大方自信地说，语气像学长跟学弟学妹聊天\n"
        "- 资料里没有明确写出的具体数字、金额、截止日期，在那一句末尾自然地加一句"
        "「具体XX我看到的资料里没写，你问下教务员确认」，语气要自然，不要像在踢皮球\n"
        "- 绝对不要因为资料不全就整段拒答，也不要编造数字来填坑\n\n"
        "记住：一个靠谱的学长不会因为记不清补考费多少就不回答补考流程——"
        "他会说「流程是这样，具体费用你开学问下教务办」"
    )

    # ── Prompt budget (token-aware context trimming) ────────────

    prompt_token_budget: int = 4096
    max_chunk_tokens: int = 320
    max_chunks_in_prompt: int = 6

    # ── GPU memory management ───────────────────────────────────

    empty_cache_every_n_requests: int = 20
    empty_cache_free_vram_mb: int = 1500

    # ── LLM timeout & circuit breaker ───────────────────────────

    llm_request_timeout_seconds: int = 20
    llm_ttft_timeout_seconds: int = 5

    # ── LLM fallback ─────────────────────────────────────────────

    enable_llm_fallback: bool = False
    fallback_llm_api_key: str = ""
    fallback_llm_base_url: str = ""
    fallback_llm_model: str = ""

    # ── LLM retry ────────────────────────────────────────────────

    retry_count: int = 3
    retry_delays: tuple = (1, 2, 4)
    request_timeout: int = 20

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
        enable_citation_verify=os.getenv("ENABLE_CITATION_VERIFY", "false").lower() in ("true", "1", "yes"),
        enable_two_stage_generation=os.getenv("ENABLE_TWO_STAGE_GENERATION", "false").lower() in ("true", "1", "yes"),
        enable_query_rewrite=os.getenv("ENABLE_QUERY_REWRITE", "false").lower() in ("true", "1", "yes"),
        enable_rerank=os.getenv("ENABLE_RERANK", "false").lower() in ("true", "1", "yes"),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        rerank_candidate_k=_int("RERANK_CANDIDATE_K", 40),
        rerank_top_k=_int("RERANK_TOP_K", 12),
        reranker_device=os.getenv("RERANKER_DEVICE", "auto"),
        confidence_tier1_top1=_float("CONFIDENCE_TIER1_TOP1", 0.75),
        confidence_tier1_top3=_float("CONFIDENCE_TIER1_TOP3", 0.60),
        confidence_tier3_top1=_float("CONFIDENCE_TIER3_TOP1", 0.40),
        prompt_token_budget=_int("PROMPT_TOKEN_BUDGET", 4096),
        max_chunk_tokens=_int("MAX_CHUNK_TOKENS", 320),
        max_chunks_in_prompt=_int("MAX_CHUNKS_IN_PROMPT", 6),
        empty_cache_every_n_requests=_int("EMPTY_CACHE_EVERY_N_REQUESTS", 20),
        empty_cache_free_vram_mb=_int("EMPTY_CACHE_FREE_VRAM_MB", 1500),
        llm_request_timeout_seconds=_int("LLM_REQUEST_TIMEOUT_SECONDS", 20),
        llm_ttft_timeout_seconds=_int("LLM_TTFT_TIMEOUT_SECONDS", 5),
        enable_llm_fallback=os.getenv("ENABLE_LLM_FALLBACK", "false").lower() in ("true", "1", "yes"),
        fallback_llm_api_key=os.getenv("FALLBACK_LLM_API_KEY", ""),
        fallback_llm_base_url=os.getenv("FALLBACK_LLM_BASE_URL", ""),
        fallback_llm_model=os.getenv("FALLBACK_LLM_MODEL", ""),
        retry_count=3,
        retry_delays=(1, 2, 4),
        request_timeout=_int("LLM_REQUEST_TIMEOUT", 20),
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
