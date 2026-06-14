"""
RAG question-answering pipeline.

Orchestrates: classify → retrieve → [rerank] → filter → prompt → LLM → format.
Each step is a named method so subclasses can override individual behaviours
without rewriting the whole flow.
"""

import time
import random
import threading
from typing import Any

from app.config import Settings
from app.errors import LLMError, EmptyQuestionError
from app.llm_client import LLMClient
from app.policy import (
    ClassificationResult,
    ResponseTemplates,
    RiskClassifier,
    RiskLevel,
)
from app.query_rewriter import QueryRewriter
from app.reranker import Reranker
from app.retriever import HybridRetriever, Retriever


class RAGPipeline:
    """
    Full RAG pipeline from question to answer dict.

    Dependencies are injected via the constructor — no global state.
    Override step methods in a subclass to customise behaviour.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        classifier: RiskClassifier | None = None,
        templates: ResponseTemplates | None = None,
        settings: Settings | None = None,
        fallback_llm: LLMClient | None = None,
        reranker: Reranker | None = None,
        query_rewriter: QueryRewriter | None = None,
    ):
        self._retriever = retriever
        self._llm = llm
        self._fallback_llm = fallback_llm
        self._reranker = reranker
        self._query_rewriter = query_rewriter
        self._classifier = classifier or RiskClassifier()
        self._templates = templates or ResponseTemplates()
        self._settings = settings or Settings()
        self._request_count = 0
        self._counter_lock = threading.Lock()

    # ── Main entry point ────────────────────────────────────────

    def answer(self, question: str,
               conversation_history: list[dict] | None = None) -> dict[str, Any]:
        """Run the full pipeline and return the ``/ask`` response dict.

        *conversation_history* is an optional list of {question, answer}
        from previous turns, injected into the prompt so the LLM can
        resolve anaphora ("那", "这个") across turns.
        """
        t_start = time.time()
        self._llm_used: str | None = None
        timing: dict[str, float] = {}

        # 1. Validate input
        if not question or not question.strip():
            return self._empty_question_response()

        # 1.5 Meta-questions — respond without retrieval
        meta = self._handle_meta_question(question)
        if meta:
            return meta

        # 2. Classify risk
        t0 = time.time()
        classification = self._classify(question)
        timing["classify_ms"] = round((time.time() - t0) * 1000)

        # 2.5 Rewrite query (optional — normalise colloquial → formal)
        search_query = question
        t0 = time.time()
        if self._query_rewriter and self._settings.enable_query_rewrite:
            search_query = self._rewrite_query(question)
        timing["rewrite_ms"] = round((time.time() - t0) * 1000)

        # 3.4. Topic routing — boost relevant sources (soft), don't hard-filter
        route_topic = self._classify_topic(question)
        route_boost_sids = self._get_source_filter(route_topic) if route_topic else None

        # 4. Retrieve (larger candidate pool if reranker enabled)
        t0 = time.time()
        try:
            if self._reranker and self._settings.enable_rerank:
                chunks = self._retrieve(search_query, top_k=self._settings.rerank_candidate_k)
            else:
                chunks = self._retrieve(search_query)
        except Exception:
            return self._fallback_response(question, classification, t_start, retrieval_count=0, timing=timing)

        # Soft boost: multiply scores of chunks from routed sources by 1.2
        if route_boost_sids:
            for c in chunks:
                if c.get("source_id", "") in route_boost_sids:
                    c["score"] = c.get("score", 0) * 1.2

        retrieval_count = len(chunks)
        timing["retrieve_ms"] = round((time.time() - t0) * 1000)

        # 3.5 Rerank (two-stage: coarse retrieval → fine cross-encoder)
        t0 = time.time()
        if self._reranker and self._settings.enable_rerank and chunks:
            chunks = self._rerank(search_query, chunks)
        timing["rerank_ms"] = round((time.time() - t0) * 1000)

        # 4. Filter by reliability
        reliable = self._filter_chunks(chunks, classification.level)

        # 5. No evidence → refusal
        if not reliable:
            result = self._no_evidence_response(question, classification, t_start, retrieval_count)
            result["debug"]["timing"] = timing
            return result

        # 6. Dedup & decide confidence tier
        t0 = time.time()
        top_chunks = self._dedup_chunks(reliable)
        confidence_tier, tier_top1, tier_top3 = self._decide_confidence_tier(top_chunks)
        timing["build_prompt_ms"] = round((time.time() - t0) * 1000)

        # 7. Check persistent query cache (chunk-signature key)
        chunk_ids = [c.get("chunk_id", "") for c in top_chunks]
        from app.cache import query_cache
        cached = query_cache.get(question, chunk_ids)
        if cached:
            result = self._format_response(
                question, cached["answer"], classification, top_chunks,
                t_start, retrieval_count, [],
                timing=timing, prompt_tokens=0, prompt_chunks=len(top_chunks),
                confidence_tier=cached.get("tier", confidence_tier),
                tier_top1=tier_top1, tier_top3=tier_top3,
                cached=True,
            )
            timing["format_ms"] = round((time.time() - t_start) * 1000)
            return result

        # 8. Tier 3: direct referral, skip LLM
        if confidence_tier == "3":
            result = self._tier3_response(
                question, classification, t_start, retrieval_count, timing,
                {"top1": tier_top1, "top3_avg": tier_top3},
            )
            return result

        # 9. Build prompt (inject hedge instructions for Tier 2)
        messages, prompt_tokens, prompt_chunks = self._build_prompt(
            question, top_chunks, classification.level, classification.is_process,
            confidence_tier=confidence_tier,
            conversation_history=conversation_history,
        )

        # 9. Call LLM (route high-risk to DeepSeek if enabled)
        t0 = time.time()
        try:
            if (self._settings.enable_high_risk_deepseek
                    and classification.level == RiskLevel.HIGH
                    and self._fallback_llm):
                answer_text = self._generate_with_fallback(messages)
            elif self._settings.enable_two_stage_generation:
                answer_text = self._generate_two_stage(question, top_chunks)
            else:
                answer_text = self._generate(messages)
        except LLMError:
            return self._fallback_response(question, classification, t_start, retrieval_count, timing=timing)
        timing["generate_ms"] = round((time.time() - t0) * 1000)

        # 10. NER fact check — verify entities against source chunks (Phase 1.1)
        citation_warnings: list[str] = []
        fact_check_debug: dict = {}
        if self._settings.enable_fact_check:
            from app.fact_check import apply_fact_check
            fc_result = apply_fact_check(answer_text, top_chunks, confidence_tier)
            answer_text = fc_result["answer"]
            new_tier = fc_result["tier"]
            fact_check_debug = fc_result["debug"]
            if new_tier != confidence_tier:
                if new_tier == "3":
                    # Full downgrade to referral — skip further processing
                    result = self._format_response(
                        question, answer_text, classification, top_chunks,
                        t_start, retrieval_count, citation_warnings,
                        timing=timing, prompt_tokens=prompt_tokens, prompt_chunks=prompt_chunks,
                        confidence_tier="3",
                        tier_top1=tier_top1, tier_top3=tier_top3,
                        fact_check_debug=fact_check_debug,
                    )
                    timing["format_ms"] = round((time.time() - t0) * 1000)
                    self._maybe_free_gpu_cache()
                    return result
                confidence_tier = new_tier
                if fact_check_debug.get("hedged", 0) > 0:
                    citation_warnings.append(
                        f"fact_check: {fact_check_debug['hedged']} entities hedged, "
                        f"{fact_check_debug.get('removed', 0)} removed"
                    )

        if self._settings.enable_citation_verify:
            citation_warnings.extend(self._verify_citations(answer_text, top_chunks))

        # 11. Format final response
        t0 = time.time()
        result = self._format_response(
            question, answer_text, classification, top_chunks,
            t_start, retrieval_count, citation_warnings,
            timing=timing, prompt_tokens=prompt_tokens, prompt_chunks=prompt_chunks,
            confidence_tier=confidence_tier,
            tier_top1=tier_top1, tier_top3=tier_top3,
            fact_check_debug=fact_check_debug,
        )
        timing["format_ms"] = round((time.time() - t0) * 1000)

        # 12. Write to persistent cache (skip Tier 3 — may become answerable later)
        if confidence_tier in ("1", "2") and not result.get("cached"):
            from app.cache import query_cache
            query_cache.set(question, chunk_ids, answer_text, confidence_tier)

        # 13. Periodic GPU cache cleanup
        self._maybe_free_gpu_cache()

        return result

    # ── Step methods (override in subclasses) ───────────────────

    def _handle_meta_question(self, question: str) -> dict | None:
        """Return a natural canned response for meta-questions and casual chat, or None."""
        q = question.strip().lower()

        # ── Bot introduction ──────────────────────────────────────

        intro_keywords = ("你是谁", "你是什么", "你是干啥", "介绍自己", "自我介绍",
                          "你的名字", "叫什么",
                          "你能干什么", "你能做什么", "你有什么功能", "你能干嘛",
                          "你可以做什么", "你会什么", "你会干啥", "你的能力",
                          "你怎么用", "如何使用你", "使用说明", "help", "帮助")
        for kw in intro_keywords:
            if kw in q:
                return self._meta_response(question,
                    "我是南鉴Bot，一个专注南京大学本科校规与教务流程的问答助手。\n\n"
                    "你可以直接问我：\n"
                    "  - 选课、缓考、补考、重修的流程和条件\n"
                    "  - 转专业、辅修、休学、交换的要求\n"
                    "  - 绩点计算、学业预警、毕业学分\n"
                    "  - 宿舍、校园卡、校医院、军训等校园生活问题\n\n"
                    "直接在群里 @我 或发 /问 加上你的问题就行。"
                )

        # ── Compliments (before greetings — avoid "你好厉害" → matched as "你好") ──

        compliment_keywords = ("你好厉害", "牛啊", "牛哇", "好聪明", "太棒了",
                               "厉害厉害", "你真棒", "好用", "不错啊", "可以啊",
                               "你好强", "nb", "niubi", "牛鼻")
        for kw in compliment_keywords:
            if kw in q:
                replies = [
                    "谢谢～我就是把校规资料整理了一下，能帮到你就好！",
                    "哈哈过奖了，主要是南大的校规资料比较齐全。有什么想问的？",
                    "嘿嘿谢谢！有问题随时发，我尽量答。",
                ]
                return self._meta_response(question, random.choice(replies))

        # ── Laughter / emoji ──────────────────────────────────────

        laugh_keywords = ("哈哈", "嘿嘿", "呵呵", "嘻嘻", "hhh", "233", "笑死",
                          "笑死我了", "好笑", "乐")
        for kw in laugh_keywords:
            if kw in q and len(q) <= 15:
                replies = [
                    "哈哈，有什么好玩的也给我讲讲？当然问校规问题更好～",
                    "😄 笑完了有问题随时问我，选课补考宿舍都行。",
                    "嘿嘿，开心就好！有啥校规方面的问题尽管问。",
                ]
                return self._meta_response(question, random.choice(replies))

        # ── Casual greetings (exact or short match) ───────────────

        greet_keywords = ("在吗", "在不在", "有人吗", "早啊", "早呀", "晚上好",
                          "下午好", "hello", "hi", "嗨", "你好呀", "哈喽", "你好啊")
        for kw in greet_keywords:
            if kw == q or q.startswith(kw):
                replies = [
                    "在呢～有什么校规或者教务流程想了解的？直接问我就行。",
                    "来了！你是想问什么校规相关的问题吗？",
                    "哈哈在的，有啥想问的直接发就行，不用客气。",
                ]
                return self._meta_response(question, random.choice(replies))

        # "你好" alone — only exact match (not "你好厉害")

        if q in ("你好", "你好呀", "你好啊", "你好哈"):
            replies = [
                "你好呀～有什么校规或者教务流程想了解的？",
                "嗨！想问什么直接发就行。",
            ]
            return self._meta_response(question, random.choice(replies))

        # "晚安" — don't redirect to school rules

        if q in ("晚安", "晚安啦", "拜拜", "再见", "bye", "bye bye"):
            replies = [
                "晚安～明天有问题再来找我。",
                "晚安！有需要再 @我。",
                "拜拜，有问题随时来问～",
            ]
            return self._meta_response(question, random.choice(replies))

        # ── Location questions ────────────────────────────────────

        location_keywords = ("这是哪里", "这是什么群", "这个群是干嘛", "这群干嘛",
                             "你是什么群", "什么群", "哪个群")
        for kw in location_keywords:
            if kw in q:
                replies = [
                    "这里是南大校规答疑群，我是群里的校规查询 Bot～有什么教务流程想了解的可以 @我 提问。",
                    "这个群是南大本科校规问答群，我是负责回答校规问题的机器人，补考、选课、缓考这些都能问我。",
                    "南大校规问答群！我是群里的校规查询助手，你想问什么直接 @我 就行。",
                ]
                return self._meta_response(question, random.choice(replies))

        # ── Thanks ────────────────────────────────────────────────

        thanks_keywords = ("谢谢", "谢谢啦", "谢啦", "感谢", "多谢", "谢谢bot",
                           "谢谢机器人", "谢谢学长")
        for kw in thanks_keywords:
            if kw in q and len(q) <= 10:
                replies = [
                    "不客气～还有其他问题随时问我。",
                    "没事！有需要再 @我。",
                    "应该的，有问题再来～",
                ]
                return self._meta_response(question, random.choice(replies))

        # ── Polite redirect for insults / irrelevant / troll ──────

        redirect_patterns = (
            "你有智力", "你傻", "sb", "傻逼", "傻b", "nm", "你懂吗",
            "你会思考吗", "你有意识", "你聪明", "笨蛋", "废物",
            "垃圾", "没用", "不好用", "人工智障",
            "聊天", "无聊", "讲个笑话", "开玩笑",
            "你是真人", "你是假的", "你是ai", "你是机器人",
            "介绍你自己", "介绍下自己",
            "我爱你", "我喜欢你", "爱你", "喜欢你", "么么哒",
            "牛逼不牛逼", "牛不牛", "六六六", "666",
            "你妈", "草泥马", "操你", "cnm", "fuck",
            # Emotional / casual chat — don't run RAG
            "好伤心", "好难过", "好累", "好无聊", "好烦", "好焦虑",
            "哭了", "想哭", "心累", "emo", "破防",
            "像人了一点", "像个人", "像人了",
            "吃了吗", "吃了没", "吃饭了", "在吃饭",
        )
        for p in redirect_patterns:
            if p in q:
                replies = [
                    "我是南大校规查询助手，还是聊点正经的吧～你想了解什么校规？",
                    "这个超出我的业务范围了哈哈，问点校规相关的？补考选课宿舍都行。",
                    "抱歉哈，我只能回答校规和教务流程问题。你试试问「补考没过怎么办」这种？",
                ]
                return self._meta_response(question, random.choice(replies))

        return None

    def _meta_response(self, question: str, answer: str) -> dict:
        return {
            "question": question,
            "answer": answer,
            "risk_level": "low",
            "need_human_confirm": False,
            "sources": [],
            "debug": {
                "retrieval_count": 0, "latency": 0,
                "audit_warnings": [], "citation_warnings": [],
                "llm_used": None, "cached": False,
                "timing": {},
            },
        }

    def _classify(self, question: str) -> ClassificationResult:
        return self._classifier.classify(question)

    # ── Topic routing (C track) ──────────────────────────────────

    # Keyword → topic mapping for routing. Order matters: first match wins.
    _topic_keywords: list[tuple[str, list[str]]] = [
        ("浦口校区", ["浦口", "浦口校区"]),
        ("学生社团", ["社团", "学生会", "百团大战"]),
        ("校园安全", ["报警", "安全", "诈骗", "保卫处", "紧急"]),
        ("信息化工具", ["校园网", "PJSD", "VPN", "正版软件", "邮箱", "统一身份认证", "南大APP", "企业微信"]),
        ("资助政策", ["奖学金", "助学金", "助学贷款", "困难补助", "学费减免", "勤工助学", "资助"]),
        ("体育", ["体测", "体育课", "军训"]),
        ("校历", ["校历", "放假", "寒假", "暑假", "开学时间", "学期"]),
        ("选课", ["选课", "学分上限", "通识课", "课程冲突"]),
        ("缓考", ["缓考"]),
        ("补考", ["补考"]),
        ("重修", ["重修"]),
        ("成绩/绩点", ["绩点", "GPA", "成绩", "平均分"]),
        ("转专业", ["转专业", "专业准入", "跨大类"]),
        ("辅修", ["辅修", "双学位"]),
        ("保研推免", ["保研", "推免"]),
        ("考研", ["考研", "研究生"]),
        ("学业预警", ["学业预警", "学业警示"]),
        ("处分/退学/学位", ["退学", "开除", "处分", "作弊", "学位", "毕业"]),
        ("交换/课程认定", ["交换", "课程认定", "交流学习"]),
        ("出国交流", ["出国", "留学", "公派"]),
        ("录取入学", ["报到", "入学", "二次选拔"]),
        ("学籍异动", ["休学", "复学", "学籍", "请假"]),
        ("校园生活", ["宿舍", "食堂", "校医院", "医保", "校园卡", "交通", "校车", "快递"]),
    ]

    def _classify_topic(self, question: str) -> str | None:
        """Classify question into a topic for source routing. Returns topic name or None."""
        q = question.lower()
        for topic, keywords in self._topic_keywords:
            for kw in keywords:
                if kw.lower() in q:
                    return topic
        return None

    def _get_source_filter(self, topic: str) -> set[str] | None:
        """Return source_ids relevant to a topic, or None for full corpus."""
        route_map = self._settings.topic_route_map
        sids = route_map.get(topic, [])
        if not sids:
            return None
        return set(sids)

    def _rewrite_query(self, question: str) -> str:
        """Normalise colloquial student questions for retrieval."""
        assert self._query_rewriter is not None
        rewritten = self._query_rewriter.rewrite(question)
        if rewritten and rewritten != question:
            self._rewritten_query = rewritten
            return rewritten
        return question

    def _retrieve(self, question: str, top_k: int | None = None,
                  source_filter: set[str] | None = None) -> list[dict]:
        return self._retriever.search(question, top_k=top_k, source_filter=source_filter)

    def _rerank(self, question: str, chunks: list[dict]) -> list[dict]:
        """Cross-encoder re-score and limit to rerank_top_k."""
        assert self._reranker is not None
        return self._reranker.rerank(question, chunks, self._settings.rerank_top_k)

    def _filter_chunks(self, chunks: list[dict], level: RiskLevel) -> list[dict]:
        min_score = self._settings.min_reliable_score
        if level == RiskLevel.HIGH:
            min_score = max(min_score, self._settings.high_risk_min_score)
        return [c for c in chunks if c["score"] >= min_score]

    def _dedup_chunks(self, chunks: list[dict]) -> list[dict]:
        """Keep top chunks, but limit per source to avoid single-doc bias.

        At most *max_chunks_per_source* from each source_id, then at most
        *max_context_chunks* total.  Default: 2 per source, 8 total.
        """
        limit = self._settings.max_context_chunks
        per_source = self._settings.max_chunks_per_source
        counts: dict[str, int] = {}
        result: list[dict] = []
        for c in chunks:
            sid = c.get("source_id", "")
            if counts.get(sid, 0) >= per_source:
                continue
            counts[sid] = counts.get(sid, 0) + 1
            result.append(c)
            if len(result) >= limit:
                break
        return result

    def _decide_confidence_tier(self, chunks: list[dict]) -> tuple[str, float, float]:
        """Three-tier confidence based on top chunks' original hybrid scores.

        Uses *orig_score* (pre-rerank hybrid score) for tiering, not the fused
        score, because hybrid scores have better dynamic range (0.2-0.75).

        Returns (tier, top1_score, top3_avg_score).
        """
        if not chunks:
            return ("3", 0.0, 0.0)
        orig_scores = sorted(
            [c.get("orig_score", c.get("score", 0)) for c in chunks],
            reverse=True,
        )
        top1 = orig_scores[0]
        top3_avg = sum(orig_scores[:3]) / min(3, len(orig_scores))
        s = self._settings
        if top1 >= s.confidence_tier1_top1 and top3_avg >= s.confidence_tier1_top3:
            return ("1", top1, top3_avg)
        elif top1 >= s.confidence_tier3_top1:
            return ("2", top1, top3_avg)
        else:
            return ("3", top1, top3_avg)

    def _tier3_response(
        self, question: str, classification, t_start: float, retrieval_count: int,
        timing: dict, tier_info: dict,
    ) -> dict:
        """Short referral response for Tier 3 — skip LLM entirely."""
        answer = (
            "这个问题我手头的校规资料里没有覆盖到，建议直接联系相关部门：\n"
            "  - 教务处 (025) 8968-1234\n"
            "  - 或通过教务系统 jw.nju.edu.cn 在线咨询"
        )
        return {
            "question": question,
            "answer": answer,
            "risk_level": classification.level.value,
            "need_human_confirm": classification.level == RiskLevel.HIGH,
            "sources": [],
            "debug": {
                "retrieval_count": retrieval_count,
                "latency": round(time.time() - t_start, 3),
                "audit_warnings": [],
                "citation_warnings": [],
                "llm_used": None,
                "cached": False,
                "timing": timing,
                "confidence_tier": "3",
                "tier_top1_score": tier_info.get("top1", 0),
                "tier_top3_avg": tier_info.get("top3_avg", 0),
            },
        }

    def _build_prompt(
        self, question: str, chunks: list[dict], level: RiskLevel,
        is_process: bool = False,
        confidence_tier: str = "1",
        conversation_history: list[dict] | None = None,
    ) -> tuple[list[dict[str, str]], int, int]:
        """Build the LLM prompt, applying token budget to trim chunks.

        *conversation_history* is injected between context and question
        so the LLM can resolve cross-turn anaphora.

        Returns (messages, estimated_prompt_tokens, chunk_count_used).
        """
        # Inject tier-specific instructions into system prompt
        system = self._settings.system_prompt
        if confidence_tier == "2":
            system += self._settings.tier2_hedge_prompt

        budget = self._settings.prompt_token_budget
        max_chunk_tok = self._settings.max_chunk_tokens
        max_chunks = self._settings.max_chunks_in_prompt

        # System prompt token estimate (Chinese chars ≈ tokens)
        system_tokens = len(system)
        question_tokens = len(question) + 50  # 50 for framing text
        overhead = system_tokens + question_tokens + 200  # 200 for formatting/metadata
        remaining = budget - overhead

        # Trim chunks to fit budget (each already in reranker order)
        context_parts = []
        used = 0
        for c in chunks[:max_chunks]:
            content = self._trim_chunk(c["content"], max_chunk_tok)
            section = c.get("article", c.get("section", "无"))
            part = f"[来源: {c['title']} | 条款: {section}]\n{content}"
            part_tokens = len(part)
            if remaining - part_tokens < 0 and context_parts:
                break  # don't exceed budget
            context_parts.append(part)
            remaining -= min(part_tokens, remaining)
            used += 1
            if remaining <= 0:
                break

        context = "\n\n---\n\n".join(context_parts) if context_parts else "（无参考资料）"

        # Append high-risk patch (after any tier-specific instructions already in `system`)
        if level == RiskLevel.HIGH:
            system += "\n\n本题为高风险，只给一般规定与办事入口，不给个人结论。"

        # Build user message — inject conversation history when available
        user_content = (
            f"以下是你可以使用的唯一信息来源，不要使用你自己的知识：\n\n"
            f"【参考资料】\n{context}"
        )
        if conversation_history:
            history_lines = []
            for h in conversation_history[-2:]:  # at most 2 recent turns
                history_lines.append(
                    f"用户: {h['question']}\n"
                    f"Bot: {h['answer'][:200]}"
                )
            user_content += "\n\n【对话上下文】\n" + "\n---\n".join(history_lines)
        user_content += f"\n\n【问题】\n{question}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        if is_process:
            messages.append(
                {
                    "role": "user",
                    "content": "（流程类问题。请分步骤列出操作流程，每步注明所需材料和办理入口。）",
                }
            )

        prompt_tokens = sum(len(m.get("content", "")) for m in messages)
        return messages, prompt_tokens, used

    def _build_context(self, chunks: list[dict]) -> str:
        """Build a concatenated context string (used by two-stage generation)."""
        if not chunks:
            return "（无参考资料）"
        parts = []
        for c in chunks:
            section = c.get("article", c.get("section", "无"))
            parts.append(f"[来源: {c['title']} | 条款: {section}]\n{c['content']}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count for Chinese text. ~1 token per char for CJK."""
        # Simple heuristic: count CJK chars as 1 token each, rest as length.
        cjk = sum(1 for ch in text if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿')
        non_cjk = len(text) - cjk
        return cjk + max(1, non_cjk // 3)  # ~3 chars/token for non-CJK text

    @staticmethod
    def _trim_chunk(content: str, max_tokens: int) -> str:
        """Trim a chunk to *max_tokens* preserving head and tail.

        Long chunks keep the first 60% and last 40% with an ellipsis marker
        in between — tail sections often carry effective dates and exceptions.
        """
        if len(content) <= max_tokens:
            return content
        head_size = int(max_tokens * 0.6)
        tail_size = max_tokens - head_size - 8  # 8 for ellipsis
        if tail_size < 20:
            return content[:max_tokens] + "…"
        return content[:head_size] + "\n……(原文略)……\n" + content[-tail_size:]

    def _generate(self, messages: list[dict]) -> str:
        """Call primary LLM; fall back to secondary on failure."""
        try:
            result = self._llm.chat(messages, temperature=0.15)
            if not result or len(result) == 0:
                q = ""
                if messages:
                    last = messages[-1]["content"]
                    import re
                    m = re.search(r'【用户问题】\s*(.+?)$', last, re.MULTILINE)
                    if m:
                        q = m.group(1)[:80]
                print(f"[retry] empty_response query={q}")
                result = self._llm.chat(messages, temperature=0.3)
                if not result or len(result) == 0:
                    print(f"[retry] STILL_EMPTY after retry query={q}")
            self._llm_used = self._llm.model
            return result
        except LLMError:
            if self._fallback_llm:
                self._llm_used = self._fallback_llm.model
                return self._fallback_llm.chat(messages, temperature=0.15)
            raise

    def _generate_stream(self, messages: list[dict]):
        """Stream LLM tokens via SSE. Yields content fragments."""
        try:
            yield from self._llm.chat_stream(messages, temperature=0.15)
            self._llm_used = self._llm.model
        except Exception:
            if self._fallback_llm:
                print("[LLM] 主模型失败，切换到回退模型")
                self._llm_used = self._fallback_llm.model
                yield from self._fallback_llm.chat_stream(messages, temperature=0.15)
            else:
                raise

    # ── GPU memory management ────────────────────────────────────

    def _maybe_free_gpu_cache(self):
        """Periodically release fragmented CUDA cache to prevent degradation."""
        with self._counter_lock:
            self._request_count += 1
            n = self._request_count

        interval = self._settings.empty_cache_every_n_requests
        if interval > 0 and n % interval == 0:
            try:
                import torch
                free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
                threshold_mb = self._settings.empty_cache_free_vram_mb
                if free_mb < threshold_mb:
                    print(f"[GPU] 空闲显存 {free_mb:.0f}MB < {threshold_mb}MB，释放缓存")
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ── Two-stage generation ────────────────────────────────────

    EXTRACT_PROMPT = """从以下校规资料中，提取与问题直接相关的关键事实。
规则：
- 只提取资料中明确写出的信息。数字（学分、金额、日期、时长）、网址、系统名称、操作步骤、条件要求。
- 一条事实一行。只写资料里有的，资料没有的绝对不写。
- 如果资料完全不包含相关信息，回复"无相关事实"。

资料：
{context}

问题：{question}
关键事实："""

    REWRITE_PROMPT = """你是南鉴Bot，帮同学整理校规信息。用直接好懂的话回答，**只用下面列出的事实信息**。

铁律：
1. 只能使用下面"事实信息"里明确写出的内容。不在事实列表里的数字、日期、金额、网址绝对不写。
2. 如果事实信息不完整（比如只有流程没有具体时间），诚实说"具体时间看教务系统通知"——不要自己填一个。
3. 语气自然直接，不用官话。200字以内。
4. 禁止"根据规定""资料显示""学弟学妹"等套话。

事实信息：
{facts}

问题：{question}
回答："""

    def _generate_two_stage(self, question: str, chunks: list[dict]) -> str:
        """Two-pass generation: extract facts → rewrite in plain language."""
        # Use only top 6 chunks for extraction (large context → Ollama timeout)
        extract_chunks = chunks[:6]
        context = self._build_context(extract_chunks)
        llm = self._llm
        self._llm_used = llm.model

        # Stage 1: extract key facts from chunks
        extract_msg = [{"role": "user", "content": self.EXTRACT_PROMPT.format(
            context=context, question=question,
        )}]
        facts = llm.chat(extract_msg, temperature=0.0)

        if not facts.strip() or "无相关事实" in facts:
            from app.policy import RiskLevel
            return self._generate(self._build_prompt(question, chunks, RiskLevel.LOW, False))

        # Stage 2: rewrite facts for students
        rewrite_msg = [{"role": "user", "content": self.REWRITE_PROMPT.format(
            facts=facts, question=question,
        )}]
        return llm.chat(rewrite_msg, temperature=0.3)


    def _generate_with_fallback(self, messages: list[dict]) -> str:
        """Route to DeepSeek fallback for high-risk questions."""
        assert self._fallback_llm is not None
        self._llm_used = self._fallback_llm.model
        print(f"[LLM] 高风险题分流到 {self._fallback_llm.model}")
        try:
            return self._fallback_llm.chat(messages, temperature=0.15)
        except LLMError:
            print("[LLM] 回退模型也失败，使用主模型")
            self._llm_used = self._llm.model
            return self._llm.chat(messages, temperature=0.15)

    def _format_response(
        self,
        question: str,
        answer_text: str,
        classification: ClassificationResult,
        chunks: list[dict],
        t_start: float,
        retrieval_count: int,
        citation_warnings: list[str] | None = None,
        timing: dict[str, float] | None = None,
        prompt_tokens: int = 0,
        prompt_chunks: int = 0,
        confidence_tier: str = "1",
        tier_top1: float = 0.0,
        tier_top3: float = 0.0,
        fact_check_debug: dict | None = None,
        cached: bool = False,
    ) -> dict[str, Any]:
        # Length cap
        limit = self._settings.max_answer_length
        truncated = False
        if len(answer_text) > limit:
            answer_text = answer_text[:limit] + "..."
            truncated = True

        # High-risk notice — template-driven, appended post-generation
        if classification.level == RiskLevel.HIGH:
            depts = [c.get("department", "") for c in chunks if c.get("department")]
            answer_text += "\n\n" + self._templates.high_risk_notice(question, depts)

        # Citation warnings — prepend to answer if significant issues found
        if citation_warnings and len(citation_warnings) > 3:
            answer_text = "⚠️ 以下回答可能缺乏足够的来源支撑，请谨慎参考：\n\n" + answer_text

        sources = self._extract_sources(chunks[:5])
        audit = self._audit_sources(chunks[:5])
        latency = round(time.time() - t_start, 2)

        debug: dict[str, Any] = {
            "retrieval_count": retrieval_count,
            "latency": latency,
            "audit_warnings": audit,
            "citation_warnings": citation_warnings or [],
            "llm_used": getattr(self, "_llm_used", None),
            "timing": timing or {},
            "prompt_tokens": prompt_tokens,
            "prompt_chunks": prompt_chunks,
            "truncated": truncated,
            "confidence_tier": confidence_tier,
            "tier_top1_score": round(tier_top1, 4),
            "tier_top3_avg": round(tier_top3, 4),
            "fact_check": fact_check_debug or {},
            "cached": cached,
        }

        return {
            "question": question,
            "answer": answer_text,
            "risk_level": classification.level,
            "need_human_confirm": self._classifier.needs_human_confirm(
                question, classification.level
            ),
            "sources": sources,
            "debug": debug,
        }

    # ── Helpers ─────────────────────────────────────────────────

    def _strip_unsupported_sentences(
        self, answer: str, chunks: list[dict], threshold: float = 0.05
    ) -> tuple[str, list[str]]:
        """Remove sentences with near-zero bigram overlap against source chunks.

        For Tier 2 answers where the LLM may still fabricate despite hedge prompts,
        this acts as a safety net: sentences that share < threshold bigrams with
        any source chunk are simply dropped.
        """
        import re
        if not chunks:
            return answer, []

        # Build source bigram set
        source_bigrams: set[str] = set()
        for c in chunks:
            content = c.get("content", "")
            for i in range(len(content) - 1):
                source_bigrams.add(content[i:i + 2])

        if not source_bigrams:
            return answer, []

        sentences = re.split(r"(?<=[。！？\n])", answer)
        kept = []
        removed = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 8:  # keep short fragments (connectors, transitions)
                if sent:
                    kept.append(sent)
                continue
            sent_bigrams: set[str] = set()
            for i in range(len(sent) - 1):
                sent_bigrams.add(sent[i:i + 2])
            if not sent_bigrams:
                kept.append(sent)
                continue
            overlap = len(sent_bigrams & source_bigrams) / len(sent_bigrams)
            if overlap >= threshold:
                kept.append(sent)
            else:
                removed.append(sent[:60] + "...")

        warnings = []
        if removed:
            warnings.append(f"[A.3] 移除{len(removed)}句无来源支撑的内容")

        return "".join(kept).strip(), warnings

    def _fact_check_entities(
        self, answer: str, chunks: list[dict]
    ) -> tuple[str, list[str]]:
        """Post-process: detect concrete entities in answer and verify against sources.

        Extracts numbers, dates, URLs, and phone numbers from the answer. If an
        entity doesn't appear verbatim in any source chunk, wraps its sentence
        with a soft hedge.

        Returns (modified_answer, hedge_warnings).
        """
        import re

        if not chunks:
            return answer, []

        # Build source text corpus
        source_text = "\n".join(c.get("content", "") for c in chunks)

        # Patterns for fabricatable entities
        patterns = [
            (r'\d{3,}\s*元', "金额"),
            (r'每\s*学分\s*\d+', "学分费用"),
            (r'\d+\s*元\s*/\s*学分', "学分费用"),
            (r'https?://[^\s。，）\)]+', "网址"),
            (r'\d{1,2}\s*月\s*\d{1,2}\s*日', "日期"),
            (r'\d{1,2}\s*个\s*工作日', "工作日"),
            (r'\d{3,}-\d{4,}', "电话"),
            (r'[1-9]\d?\s*周', "周数"),
            (r'第\s*\d+\s*周', "周数"),
        ]

        hedge_warnings = []
        modified = answer

        for pattern, label in patterns:
            for m in re.finditer(pattern, answer):
                entity = m.group()
                # Check if this entity appears verbatim in any source
                if entity not in source_text:
                    hedge_warnings.append(f"[{label}] {entity} 未在来源中找到")
                    # Add inline hedge after the sentence containing this entity
                    # Find the sentence boundary
                    start = m.start()
                    # Look back to find sentence start
                    sent_start = start
                    for sep in "。！？\n":
                        idx = answer.rfind(sep, 0, start)
                        if idx > sent_start:
                            sent_start = idx + 1
                    # Find sentence end
                    sent_end = len(answer)
                    for sep in "。！？\n":
                        idx = answer.find(sep, start)
                        if idx != -1 and idx < sent_end:
                            sent_end = idx
                    sentence = answer[sent_start:sent_end].strip()
                    if sentence and len(sentence) > 10:
                        # Wrap with hedge: append soft qualifier to this sentence
                        hedged = sentence + "（具体数字建议跟教务员确认）"
                        modified = modified.replace(sentence, hedged, 1)

        return modified, hedge_warnings

    def _verify_citations(self, answer: str, chunks: list[dict]) -> list[str]:
        """Lightweight check: do answer claims have source support?

        Splits the answer into sentence-level claims and checks token overlap
        with cited chunk content.  Returns a list of warning strings — empty
        means all claims are reasonably grounded.
        """
        import re
        sentences = re.split(r"[。！？\n]", answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 8]

        if not sentences or not chunks:
            return []

        # Build a token set for all cited chunks
        source_tokens: set[str] = set()
        for c in chunks:
            content = c.get("content", "")
            # Simple character bigram tokenization for Chinese
            for i in range(len(content) - 1):
                source_tokens.add(content[i:i + 2])

        warnings = []
        unsupported = 0
        for sent in sentences:
            sent_tokens: set[str] = set()
            for i in range(len(sent) - 1):
                sent_tokens.add(sent[i:i + 2])
            if not sent_tokens:
                continue
            overlap = len(sent_tokens & source_tokens) / len(sent_tokens)
            if overlap < 0.1:  # less than 10% token overlap
                unsupported += 1
                warnings.append(f"可能缺乏来源支撑: {sent[:50]}...")

        if unsupported > len(sentences) * 0.5:
            warnings.append(
                "注意：多条回答内容与检索到的来源匹配度较低，建议核实"
            )

        return warnings

    def _extract_sources(self, chunks: list[dict]) -> list[dict]:
        seen_titles: set[str] = set()
        unique = []
        for c in chunks:
            title = c["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            unique.append({
                "chunk_id": c["chunk_id"],
                "source_id": c.get("source_id", c["chunk_id"].rsplit("-", 1)[0]),
                "title": title,
                "url": c.get("url", ""),
                "priority": c.get("priority", 5),
                "fetched_at": c.get("fetched_at", ""),
            })
        return unique[:5]

    def _audit_sources(self, chunks: list[dict]) -> list[str]:
        """Check reliability of retrieved chunks.  Returns warnings.

        Implements rules from docs/risk_policy.md:
        - priority=5 (student handbook / unofficial) as sole source
        - chunks older than 3 years
        """
        warnings: list[str] = []
        if not chunks:
            return warnings

        priorities = {c.get("priority", 5) for c in chunks}
        if priorities == {5}:
            warnings.append("唯一来源为学生手册等非正式文件(priority=5)，建议核实")

        now = time.time()
        three_years = 3 * 365 * 24 * 3600
        for c in chunks:
            fetched = c.get("fetched_at", "")
            if fetched:
                try:
                    t = time.mktime(time.strptime(fetched, "%Y-%m-%d %H:%M:%S"))
                    if now - t > three_years:
                        warnings.append(
                            f"chunk {c['chunk_id']} 超过3年({fetched[:10]})，信息可能过时"
                        )
                except (ValueError, OverflowError):
                    pass
        return warnings

    def _empty_question_response(self) -> dict[str, Any]:
        return {
            "question": "",
            "answer": "请输入您的问题。",
            "risk_level": "low",
            "need_human_confirm": False,
            "sources": [],
            "debug": {
                "retrieval_count": 0, "latency": 0,
                "citation_warnings": [],
                "timing": {},
            },
        }

    def _no_evidence_response(
        self,
        question: str,
        classification: ClassificationResult,
        t_start: float,
        retrieval_count: int,
    ) -> dict[str, Any]:
        latency = round(time.time() - t_start, 2)
        if classification.level == RiskLevel.HIGH:
            result = self._templates.high_risk_no_evidence(question)
        else:
            result = self._templates.no_evidence(question)
            result["risk_level"] = classification.level
            result["need_human_confirm"] = self._classifier.needs_human_confirm(
                question, classification.level
            )
        result["debug"] = {
            "retrieval_count": retrieval_count,
            "latency": latency,
            "audit_warnings": [],
            "citation_warnings": [],
            "timing": {},
        }
        return result

    def _fallback_response(
        self,
        question: str,
        classification: ClassificationResult,
        t_start: float,
        retrieval_count: int,
        timing: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        latency = round(time.time() - t_start, 2)
        return {
            "question": question,
            "answer": "系统暂时不可用，请稍后再试。",
            "risk_level": classification.level,
            "need_human_confirm": True,
            "sources": [],
            "debug": {
                "retrieval_count": retrieval_count,
                "latency": latency,
                "audit_warnings": [],
                "citation_warnings": [],
                "timing": timing or {},
            },
        }


# ── Backward-compatible singleton ────────────────────────────────────

import threading

_pipeline: RAGPipeline | None = None
_lock = threading.Lock()


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                import time
                t0 = time.time()
                from app.deps import create_pipeline
                _pipeline = create_pipeline()
                print(f"[Pipeline] 初始化完成，耗时 {time.time() - t0:.1f}s")
    return _pipeline


def preload_pipeline() -> None:
    """Eagerly initialize the pipeline at startup (call once).

    Sends a warmup query to force all lazy-loaded models (embedding,
    reranker, LLM) to load into GPU memory before the first real request.
    """
    p = _get_pipeline()
    try:
        p.answer("预热")  # triggers _retrieve → _rerank → _generate
        print("[Pipeline] 预热完成，所有模型已加载。")
    except Exception:
        pass


def answer_question(question: str,
                    conversation_history: list[dict] | None = None) -> dict[str, Any]:
    """Backward-compatible entry point.  Prefer ``RAGPipeline.answer()``."""
    return _get_pipeline().answer(question, conversation_history=conversation_history)
