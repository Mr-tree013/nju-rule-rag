"""
RAG question-answering pipeline.

Orchestrates: classify → retrieve → filter → prompt → LLM → format.
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
    ):
        self._retriever = retriever
        self._llm = llm
        self._classifier = classifier or RiskClassifier()
        self._templates = templates or ResponseTemplates()
        self._settings = settings or Settings()

    # ── Main entry point ────────────────────────────────────────

    def answer(self, question: str) -> dict[str, Any]:
        """Run the full pipeline and return the ``/ask`` response dict."""
        t_start = time.time()

        # 1. Validate input
        if not question or not question.strip():
            return self._empty_question_response()

        # 2. Classify risk
        classification = self._classify(question)

        # 3. Retrieve
        try:
            chunks = self._retrieve(question)
        except Exception:
            return self._fallback_response(question, classification, t_start, retrieval_count=0)

        retrieval_count = len(chunks)

        # 4. Filter by reliability
        reliable = self._filter_chunks(chunks, classification.level)

        # 5. No evidence → refusal
        if not reliable:
            return self._no_evidence_response(question, classification, t_start, retrieval_count)

        # 6. Build prompt & call LLM — dedup by source, cap total
        top_chunks = self._dedup_chunks(reliable)
        messages = self._build_prompt(question, top_chunks, classification.level)
        try:
            answer_text = self._generate(messages)
        except LLMError:
            return self._fallback_response(question, classification, t_start, retrieval_count)

        # 7. Format final response
        return self._format_response(question, answer_text, classification, top_chunks, t_start, retrieval_count)

    # ── Step methods (override in subclasses) ───────────────────

    def _classify(self, question: str) -> ClassificationResult:
        return self._classifier.classify(question)

    def _retrieve(self, question: str) -> list[dict]:
        return self._retriever.search(question)

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
        self, question: str, chunks: list[dict], level: RiskLevel
    ) -> list[dict[str, str]]:
        context = self._build_context(chunks)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._settings.system_prompt},
            {"role": "user", "content": f"【参考资料片段】\n\n{context}\n\n【用户问题】\n{question}"},
        ]
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
        return self._llm.chat(messages, temperature=0.2)

    def _format_response(
        self,
        question: str,
        answer_text: str,
        classification: ClassificationResult,
        chunks: list[dict],
        t_start: float,
        retrieval_count: int,
    ) -> dict[str, Any]:
        # Length cap
        limit = self._settings.max_answer_length
        if len(answer_text) > limit:
            answer_text = answer_text[:limit] + "..."

        # High-risk notice
        if classification.level == RiskLevel.HIGH:
            answer_text += "\n\n" + self._templates.high_risk_notice(question)

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
            },
        }

    # ── Helpers ─────────────────────────────────────────────────

    def _extract_sources(self, chunks: list[dict]) -> list[dict]:
        return [
            {
                "chunk_id": c["chunk_id"],
                "source_id": c.get("source_id", c["chunk_id"].rsplit("-", 1)[0]),
                "title": c["title"],
                "url": c.get("url", ""),
                "priority": c.get("priority", 5),
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
            "debug": {"retrieval_count": 0, "latency": 0},
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
