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

DEFAULT_SYSTEM_PROMPT = """你是南大学长，用和学弟学妹聊天的语气帮他们搞懂校规和办事流程。

你的第一准则：宁可说「这个我不确定，你问下教务员」，也绝不编造任何信息。一个靠谱的学长不会不懂装懂。

以下是五条硬规则，每条都不能违反：

【规则1：不编造数字】金额、日期、学分、次数、比例——资料里没写的数字一个都不准自己填。不确定就说「具体数字我看到的资料里没写，你问教务员确认下」。

【规则2：不编造流程步骤】只说资料里明确有的步骤。不要因为"看起来合理"就自己补步骤。比如资料只说"提交申请"，你就不要加"先找辅导员签字"——除非资料里写了要找辅导员。

【规则3：不编造网址和系统名】只提资料里出现的网址和系统名（如 jw.nju.edu.cn）。不要自己编。

【规则4：不跨问题混淆】不要把不同问题的规则混在一起。比如问的是补考，就不要套缓考的流程；问的是本科生，就不要套研究生的规则。

【规则5：不确定就标出来】资料不全时，在不确定的地方自然地说「具体XX我看到的资料里没写，建议问教务员」。不要因为信息不全就直接拒答——把知道的部分说出来，不知道的部分诚实标注。

风格要求：
- 自然聊天语气，200-300字，不要官话套话（"根据规定""资料显示""校规要求"）
- 不一定列步骤，讲清楚核心信息和下一步就行

好的回答示例：

问 劳育需要多少时长
答 累计20小时。登录五育项目管理系统报名，做完后老师登记时长，大三下学期末前完成就行。

问 缓考怎么申请
答 考试前在教务系统提交申请，附上证明材料。登录教服平台 jw.nju.edu.cn 找到缓考申请入口，上传医院证明或冲突证明，等辅导员和教务处审核。具体截止时间看教务系统通知，别拖到最后一天。

问 补考没过怎么办
答 只能重修，补考就一次机会。没过的话这门课得跟着下一届重新上。重修要不要交钱、成绩怎么记，看你是什么类型的课——这个我看到的资料里没统一规定，你开学时问下教务员就清楚了。

问 体育课项目有哪些（当资料里只提到部分信息时）
答 我看到的资料里提到的有篮球、足球、排球、羽毛球、网球、武术、健美操这些。具体每学期开哪些课、每个项目多少名额，你选课的时候在教务系统上看最准确。

问 退学后还能回来吗
答 可以的，但要重新参加高考或者通过成人高考录取。具体的复学条件和流程我手头资料没写全——这种特殊情况最好直接打教务处电话问清楚，比在网上查靠谱。

坏的回答（绝对禁止）：
- 编造数字：「每学分500元」「最晚9月15日截止」（资料没写）
- 编造流程：「先找辅导员签字→再去教务处盖章→最后交到…」（资料没写的步骤）
- 编造网址和系统名：「登录学生资助管理中心官网 https://jwc.nju.edu.cn/」
- 跨问题套用：问补考却回答缓考流程，问本科生却套研究生规则
- 资料不全时硬编完整答案，不标注哪些是推测"""


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
    reranker_type: str = "cross_encoder"  # cross_encoder | llm
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidate_k: int = 40
    rerank_top_k: int = 12
    reranker_device: str = "auto"  # auto | cuda | cpu

    # LLM-as-Reranker settings (when reranker_type=llm)
    llm_reranker_batch_size: int = 15
    llm_reranker_candidate_preview_chars: int = 200
    llm_reranker_temperature: float = 0.0
    llm_reranker_fallback_to_ce: bool = True

    # ── Topic routing (C track: pre-filter sources by question topic) ──

    # topic → source_ids mapping (derived from eval gold-source annotations)
    topic_route_map: dict[str, list[str]] = field(default_factory=lambda: {
        "选课": ["nju-jw-006", "nju-guide-036", "nju-guide-001", "nju-guide-002", "nju-jw-050", "nju-jw-056"],
        "补考": ["nju-jw-005", "nju-jw-029", "nju-jw-032", "nju-jw-033", "nju-guide-037", "nju-jw-050", "nju-jw-056"],
        "缓考": ["nju-jw-004", "nju-jw-028", "nju-jw-033", "nju-jw-045", "nju-jw-050"],
        "重修": ["nju-jw-005", "nju-guide-037", "nju-jw-050", "nju-jw-056"],
        "成绩/绩点": ["nju-jw-003", "nju-jw-043", "nju-guide-036", "nju-jw-051"],
        "处分/退学/学位": ["nju-jw-001", "nju-jw-002", "nju-jw-007", "nju-jw-010", "nju-jw-039", "nju-jw-057", "nju-jw-058"],
        "学业预警": ["nju-jw-001", "nju-jw-008"],
        "转专业": ["nju-guide-004", "nju-jw-011", "nju-jw-052", "nju-jw-053", "nju-jw-054", "nju-jw-055"],
        "辅修": ["nju-jw-015", "nju-jw-046", "nju-jw-055"],
        "学籍异动": ["nju-guide-004", "nju-jw-001", "nju-jw-011", "nju-jw-012", "nju-jw-053", "nju-jw-054"],
        "保研推免": ["nju-guide-005", "nju-jw-050", "nju-guide-047"],
        "考研": ["nju-guide-017"],
        "录取入学": ["nju-guide-001", "nju-guide-002", "nju-guide-003", "nju-guide-004", "nju-guide-032", "nju-life-007", "nju-jw-053"],
        "出国交流": ["nju-guide-008", "nju-guide-009", "nju-jw-014", "nju-jw-018", "nju-jw-041", "nju-jw-042"],
        "交换/课程认定": ["nju-guide-008", "nju-jw-014", "nju-jw-018", "nju-jw-019", "nju-jw-034", "nju-jw-041", "nju-jw-042"],
        "资助政策": ["nju-guide-022", "nju-guide-023", "nju-guide-024", "nju-guide-025", "nju-guide-026", "nju-guide-027", "nju-guide-028", "nju-guide-029", "nju-guide-040", "nju-guide-047", "nju-guide-048", "nju-guide-049", "nju-guide-050", "nju-guide-051"],
        "体育": ["nju-guide-016", "nju-guide-036"],
        "校历": ["nju-guide-019", "nju-guide-035", "nju-guide-039"],
        "校园生活": ["nju-life-001", "nju-life-002", "nju-life-003", "nju-life-004", "nju-life-005", "nju-life-007", "nju-life-008", "nju-life-009", "nju-guide-038", "nju-guide-051", "nju-guide-052"],
        "校园安全": ["nju-guide-020", "nju-guide-033"],
        "信息化工具": ["nju-guide-010", "nju-guide-011", "nju-guide-012", "nju-guide-013", "nju-guide-014", "nju-guide-015", "nju-guide-038"],
        "学生社团": ["nju-guide-006", "nju-guide-007", "nju-guide-034"],
        "浦口校区": ["nju-guide-021"],
        "周边生活": ["nju-guide-018", "nju-guide-038"],
    })

    # ── Confidence tiering (v0.6.0 three-tier answer strategy) ────

    confidence_tier1_top1: float = 0.80   # Tier 1: top-1 orig_score threshold (raised to reduce confident-but-wrong)
    confidence_tier1_top3: float = 0.65   # Tier 1: top-3 avg orig_score threshold
    confidence_tier3_top1: float = 0.40   # Tier 3: top-1 orig_score below this → direct referral
    tier2_hedge_prompt: str = (
        "\n\n"
        "重要: 这次给你的参考资料覆盖不全, 只有部分相关信息。下面的规则比平时更严格:\n\n"
        "你必须做到:\n"
        "- 只说你确定资料里写了的内容, 哪怕信息很少\n"
        "- 任何具体数字(金额/日期/学分/次数/比例) -- 资料里没写的, 一个都不准自己填\n"
        "- 任何流程步骤 -- 资料里没写的, 不要凭常识补\n"
        "- 在不确定的句子末尾, 自然地加一句 具体XX我看到的资料里没写, 建议问教务员\n\n"
        "禁止做的事:\n"
        "- 禁止因为资料不全就编造看似合理的信息来补全答案\n"
        "- 禁止把不同 topic 的规则混在一起(如把缓考规则套到补考上)\n"
        "- 禁止编造网址/系统名/部门名称\n\n"
        "核心原则: 宁可回答短但真实, 也不要长但有假。资料不全时你的价值不是"
        "编出完整答案, 而是诚实告诉学弟学妹哪些是确定的、哪些需要他们自己去确认。"
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
        reranker_type=os.getenv("RERANKER_TYPE", "cross_encoder"),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        rerank_candidate_k=_int("RERANK_CANDIDATE_K", 40),
        rerank_top_k=_int("RERANK_TOP_K", 12),
        reranker_device=os.getenv("RERANKER_DEVICE", "auto"),
        llm_reranker_batch_size=_int("LLM_RERANKER_BATCH_SIZE", 15),
        llm_reranker_candidate_preview_chars=_int("LLM_RERANKER_CANDIDATE_PREVIEW_CHARS", 200),
        llm_reranker_temperature=_float("LLM_RERANKER_TEMPERATURE", 0.0),
        llm_reranker_fallback_to_ce=os.getenv("LLM_RERANKER_FALLBACK_TO_CE", "true").lower() in ("true", "1", "yes"),
        confidence_tier1_top1=_float("CONFIDENCE_TIER1_TOP1", 0.80),
        confidence_tier1_top3=_float("CONFIDENCE_TIER1_TOP3", 0.65),
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
