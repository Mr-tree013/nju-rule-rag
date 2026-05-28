"""
RAG question-answering pipeline.

Orchestrates: classify → retrieve → [rerank] → filter → prompt → LLM → format.
Each step is a named method so subclasses can override individual behaviours
without rewriting the whole flow.
"""

import time
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

    # ── Main entry point ────────────────────────────────────────

    def answer(self, question: str) -> dict[str, Any]:
        """Run the full pipeline and return the ``/ask`` response dict."""
        t_start = time.time()
        self._llm_used: str | None = None

        # 1. Validate input
        if not question or not question.strip():
            return self._empty_question_response()

        # 2. Classify risk
        classification = self._classify(question)

        # 2.5 Rewrite query (optional — normalise colloquial → formal)
        search_query = question
        if self._query_rewriter and self._settings.enable_query_rewrite:
            search_query = self._rewrite_query(question)

        # 3. Retrieve (larger candidate pool if reranker enabled)
        try:
            if self._reranker and self._settings.enable_rerank:
                chunks = self._retrieve(search_query, top_k=self._settings.rerank_candidate_k)
            else:
                chunks = self._retrieve(search_query)
        except Exception:
            return self._fallback_response(question, classification, t_start, retrieval_count=0)

        retrieval_count = len(chunks)

        # 3.5 Rerank (two-stage: coarse retrieval → fine cross-encoder)
        if self._reranker and self._settings.enable_rerank:
            chunks = self._rerank(search_query, chunks)

        # 4. Filter by reliability
        reliable = self._filter_chunks(chunks, classification.level)

        # 5. No evidence → refusal
        if not reliable:
            return self._no_evidence_response(question, classification, t_start, retrieval_count)

        # 6. Build prompt & call LLM — dedup by source, cap total
        top_chunks = self._dedup_chunks(reliable)
        messages = self._build_prompt(question, top_chunks, classification.level, classification.is_process)
        try:
            answer_text = self._generate(messages)
        except LLMError:
            return self._fallback_response(question, classification, t_start, retrieval_count)

        # 7. Verify citations (optional guardrail)
        citation_warnings: list[str] = []
        if self._settings.enable_citation_verify:
            citation_warnings = self._verify_citations(answer_text, top_chunks)

        # 8. Format final response
        return self._format_response(
            question, answer_text, classification, top_chunks,
            t_start, retrieval_count, citation_warnings,
        )

    # ── Step methods (override in subclasses) ───────────────────

    def _classify(self, question: str) -> ClassificationResult:
        return self._classifier.classify(question)

    def _rewrite_query(self, question: str) -> str:
        """Normalise colloquial student questions for retrieval."""
        assert self._query_rewriter is not None
        rewritten = self._query_rewriter.rewrite(question)
        if rewritten and rewritten != question:
            self._rewritten_query = rewritten
            return rewritten
        return question

    def _retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        return self._retriever.search(question, top_k=top_k)

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

    def _build_prompt(
        self, question: str, chunks: list[dict], level: RiskLevel,
        is_process: bool = False,
    ) -> list[dict[str, str]]:
        context = self._build_context(chunks)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._settings.system_prompt},
            {"role": "user", "content": f"【参考资料片段】\n\n{context}\n\n【用户问题】\n{question}"},
        ]
        if is_process:
            messages.append(
                {
                    "role": "user",
                    "content": "（这是一个流程类问题。请分步骤列出操作流程，每步注明所需材料和办理入口。）",
                }
            )
        if level == RiskLevel.HIGH:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "（注意：这是一个高风险问题。请只描述校规中已有的客观规定，"
                        "不要对用户个人情况做任何判断或结论。）"
                    ),
                }
            )
        return messages

    def _build_context(self, chunks: list[dict]) -> str:
        if not chunks:
            return "（无参考资料）"
        parts = []
        for c in chunks:
            section = c.get("article", c.get("section", "无"))
            parts.append(f"[来源: {c['title']} | 条款: {section}]\n{c['content']}")
        return "\n\n---\n\n".join(parts)

    def _generate(self, messages: list[dict]) -> str:
        """Call primary LLM; fall back to secondary on failure."""
        try:
            result = self._llm.chat(messages, temperature=0.2)
            self._llm_used = self._llm.model
            return result
        except LLMError:
            if self._fallback_llm:
                print("[LLM] 主模型失败，切换到回退模型")
                self._llm_used = self._fallback_llm.model
                return self._fallback_llm.chat(messages, temperature=0.2)
            raise

    def _generate_stream(self, messages: list[dict]):
        """Stream LLM tokens via SSE. Yields content fragments."""
        try:
            yield from self._llm.chat_stream(messages, temperature=0.2)
            self._llm_used = self._llm.model
        except Exception:
            if self._fallback_llm:
                print("[LLM] 主模型失败，切换到回退模型")
                self._llm_used = self._fallback_llm.model
                yield from self._fallback_llm.chat_stream(messages, temperature=0.2)
            else:
                raise

    def _format_response(
        self,
        question: str,
        answer_text: str,
        classification: ClassificationResult,
        chunks: list[dict],
        t_start: float,
        retrieval_count: int,
        citation_warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        # Length cap
        limit = self._settings.max_answer_length
        if len(answer_text) > limit:
            answer_text = answer_text[:limit] + "..."

        # High-risk notice (with department contacts from source metadata)
        if classification.level == RiskLevel.HIGH:
            depts = [c.get("department", "") for c in chunks if c.get("department")]
            answer_text += "\n\n" + self._templates.high_risk_notice(question, depts)

        # Citation warnings — prepend to answer if significant issues found
        if citation_warnings and len(citation_warnings) > 3:
            answer_text = "⚠️ 以下回答可能缺乏足够的来源支撑，请谨慎参考：\n\n" + answer_text

        sources = self._extract_sources(chunks[:5])
        audit = self._audit_sources(chunks[:5])
        latency = round(time.time() - t_start, 2)

        return {
            "question": question,
            "answer": answer_text,
            "risk_level": classification.level,
            "need_human_confirm": self._classifier.needs_human_confirm(
                question, classification.level
            ),
            "sources": sources,
            "debug": {
                "retrieval_count": retrieval_count,
                "latency": latency,
                "audit_warnings": audit,
                "citation_warnings": citation_warnings or [],
                "llm_used": getattr(self, "_llm_used", None),
            },
        }

    # ── Helpers ─────────────────────────────────────────────────

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
        return [
            {
                "chunk_id": c["chunk_id"],
                "source_id": c.get("source_id", c["chunk_id"].rsplit("-", 1)[0]),
                "title": c["title"],
                "url": c.get("url", ""),
                "priority": c.get("priority", 5),
                "fetched_at": c.get("fetched_at", ""),
            }
            for c in chunks
        ]

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
            "debug": {"retrieval_count": 0, "latency": 0, "citation_warnings": []},
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
        }
        return result

    def _fallback_response(
        self,
        question: str,
        classification: ClassificationResult,
        t_start: float,
        retrieval_count: int,
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
    """Eagerly initialize the pipeline at startup (call once)."""
    _get_pipeline()


def answer_question(question: str) -> dict[str, Any]:
    """Backward-compatible entry point.  Prefer ``RAGPipeline.answer()``."""
    return _get_pipeline().answer(question)
