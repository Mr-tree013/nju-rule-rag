"""Tests for RAGPipeline with mocked dependencies."""
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.pipeline import RAGPipeline
from app.policy import ClassificationResult, RiskLevel
from app.errors import LLMError


@pytest.fixture
def mock_retriever():
    r = MagicMock()
    r.search.return_value = [
        {
            "chunk_id": "test-0001",
            "source_id": "test",
            "title": "缓考管理规定",
            "content": "学生因故不能参加期末考试的，应当在考试前申请缓考。",
            "url": "http://example.com",
            "priority": 1,
            "score": 0.85,
            "section": "第三条",
        }
    ]
    return r


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat.return_value = "缓考需要在考试前通过网上办事服务大厅申请。"
    llm.model = "mock-model"
    return llm


@pytest.fixture
def pipeline(mock_retriever, mock_llm):
    return RAGPipeline(
        retriever=mock_retriever,
        llm=mock_llm,
        settings=Settings(),
    )


class TestRAGPipelineAnswer:
    def test_empty_question(self, pipeline):
        r = pipeline.answer("")
        assert r["answer"] == "请输入您的问题。"
        assert r["risk_level"] == "low"
        assert r["sources"] == []
        assert r["debug"]["retrieval_count"] == 0

    def test_whitespace_question(self, pipeline):
        r = pipeline.answer("   ")
        assert r["answer"] == "请输入您的问题。"

    def test_normal_flow(self, pipeline, mock_retriever, mock_llm):
        r = pipeline.answer("缓考怎么申请？")
        assert r["question"] == "缓考怎么申请？"
        assert "缓考" in r["answer"]
        assert r["risk_level"] == "medium"
        assert r["need_human_confirm"] is True
        assert len(r["sources"]) == 1
        assert r["sources"][0]["title"] == "缓考管理规定"
        assert "latency" in r["debug"]
        assert r["debug"]["retrieval_count"] == 1

    def test_no_reliable_chunks_returns_refusal(self, pipeline, mock_retriever):
        mock_retriever.search.return_value = [
            {
                "chunk_id": "test-0001",
                "source_id": "test",
                "title": "某文档",
                "content": "无关内容",
                "url": "",
                "priority": 5,
                "score": 0.05,  # below MIN_RELIABLE_SCORE
                "section": "",
            }
        ]
        r = pipeline.answer("火星上怎么选课？")
        assert "抱歉" in r["answer"]

    def test_llm_error_fallback(self, pipeline, mock_llm):
        mock_llm.chat.side_effect = LLMError("API timeout")
        r = pipeline.answer("缓考怎么申请？")
        assert "系统暂时不可用" in r["answer"]
        assert r["need_human_confirm"] is True

    def test_llm_fallback_to_secondary(self, mock_retriever, mock_llm):
        """When primary fails and fallback is configured, use fallback."""
        fallback = MagicMock()
        fallback.chat.return_value = "来自回退模型的回答。"
        fallback.model = "fallback-model"

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm=mock_llm,
            fallback_llm=fallback,
            settings=Settings(),
        )
        mock_llm.chat.side_effect = LLMError("Primary timeout")
        r = pipeline.answer("缓考怎么申请？")
        assert r["answer"] == "来自回退模型的回答。"
        assert r["debug"]["llm_used"] == "fallback-model"

    def test_llm_used_tracks_primary(self, pipeline, mock_llm):
        """debug.llm_used should be set to primary model name."""
        mock_llm.model = "my-primary-model"
        r = pipeline.answer("缓考怎么申请？")
        assert r["debug"]["llm_used"] == "my-primary-model"

    def test_reranker_invoked_when_present(self, mock_retriever, mock_llm):
        """When reranker is configured and enabled, it should be called."""
        from app.config import Settings

        reranker = MagicMock()
        reranker.rerank.return_value = mock_retriever.search.return_value[:3]

        settings = Settings(enable_rerank=True, rerank_candidate_k=20, rerank_top_k=5)
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm=mock_llm,
            reranker=reranker,
            settings=settings,
        )
        r = pipeline.answer("缓考怎么申请？")
        assert reranker.rerank.called
        assert "缓考" in r["answer"]

    def test_high_risk_stricter_filter(self, pipeline, mock_retriever):
        """High-risk questions require HIGH_RISK_MIN_SCORE (0.25)."""
        mock_retriever.search.return_value = [
            {
                "chunk_id": "test-0001",
                "source_id": "test",
                "title": "处分规定",
                "content": "作弊将受到处分。",
                "url": "",
                "priority": 1,
                "score": 0.22,  # above MIN_RELIABLE (0.2) but below HIGH_RISK_MIN (0.25)
                "section": "",
            }
        ]
        r = pipeline.answer("我作弊了会不会被开除？")
        # Should get high-risk no-evidence response
        assert r["risk_level"] == "high"
        assert r["need_human_confirm"] is True
        assert "重大事项" in r["answer"] or "抱歉" in r["answer"]

    def test_high_risk_appends_notice(self, pipeline, mock_retriever, mock_llm):
        mock_retriever.search.return_value = [
            {
                "chunk_id": "test-0001",
                "source_id": "test",
                "title": "处分规定",
                "content": "考试作弊给予记过及以上处分。",
                "url": "",
                "priority": 1,
                "score": 0.85,
                "section": "",
            }
        ]
        r = pipeline.answer("我作弊了会不会被开除？")
        assert r["risk_level"] == "high"
        assert "提醒" in r["answer"] or "仅供参考" in r["answer"]


class TestRAGPipelineSteps:
    def test_filter_chunks_min_score(self, pipeline):
        chunks = [
            {"chunk_id": "c1", "score": 0.9},
            {"chunk_id": "c2", "score": 0.15},
            {"chunk_id": "c3", "score": 0.3},
        ]
        filtered = pipeline._filter_chunks(chunks, RiskLevel.LOW)
        assert len(filtered) == 2
        assert filtered[0]["chunk_id"] == "c1"

    def test_filter_chunks_high_risk_stricter(self, pipeline):
        chunks = [
            {"chunk_id": "c1", "score": 0.22},  # below HIGH_RISK_MIN_SCORE=0.25
            {"chunk_id": "c2", "score": 0.30},
        ]
        filtered = pipeline._filter_chunks(chunks, RiskLevel.HIGH)
        assert len(filtered) == 1
        assert filtered[0]["chunk_id"] == "c2"

    def test_build_context(self, pipeline):
        chunks = [
            {"title": "测试文档", "content": "内容A", "section": "第一条"},
            {"title": "测试文档2", "content": "内容B", "section": "第二条"},
        ]
        ctx = pipeline._build_context(chunks)
        assert "测试文档" in ctx
        assert "内容A" in ctx
        assert "内容B" in ctx
        assert "---" in ctx

    def test_build_context_empty(self, pipeline):
        assert "无参考资料" in pipeline._build_context([])

    def test_build_prompt_includes_high_risk_reminder(self, pipeline):
        chunks = [{"title": "T", "content": "C", "section": "S"}]
        msgs = pipeline._build_prompt("作弊怎么办", chunks, RiskLevel.HIGH)
        # Should have system + user + high-risk reminder
        assert len(msgs) == 3
        assert "高风险" in msgs[-1]["content"]

    def test_build_prompt_medium_has_no_extra_message(self, pipeline):
        chunks = [{"title": "T", "content": "C", "section": "S"}]
        msgs = pipeline._build_prompt("缓考怎么申请", chunks, RiskLevel.MEDIUM)
        assert len(msgs) == 2  # system + user only

    def test_extract_sources(self, pipeline):
        chunks = [
            {"chunk_id": "a-0001", "source_id": "a", "title": "T1", "url": "u1", "priority": 1},
            {"chunk_id": "b-0001", "source_id": "b", "title": "T2", "url": "", "priority": 5},
        ]
        sources = pipeline._extract_sources(chunks)
        assert len(sources) == 2
        assert sources[0]["source_id"] == "a"
        assert sources[1]["priority"] == 5


class TestTwoLayerRiskClassifier:
    """Tests for TwoLayerRiskClassifier (keyword + embedding)."""

    def test_layer1_only_works(self):
        from app.policy import TwoLayerRiskClassifier, RiskLevel
        c = TwoLayerRiskClassifier()
        r = c.classify("我作弊了会被开除吗")
        assert r.level == RiskLevel.HIGH
        r = c.classify("补考没过怎么办")
        assert r.level == RiskLevel.MEDIUM
        r = c.classify("仙林校区宿舍是几人间")
        assert r.level == RiskLevel.LOW

    def test_degree_info_downgrade(self):
        from app.policy import TwoLayerRiskClassifier, RiskLevel
        c = TwoLayerRiskClassifier()
        # "学位证" should be downgraded from HIGH to MEDIUM (informational)
        r = c.classify("学位证和毕业证有什么区别")
        assert r.level == RiskLevel.MEDIUM

    def test_classification_result_is_process(self):
        from app.policy import TwoLayerRiskClassifier
        c = TwoLayerRiskClassifier()
        r = c.classify("补考流程是什么")
        assert r.is_process is True
        r = c.classify("今天天气怎么样")
        assert r.is_process is False


class TestCitationVerification:
    """Tests for _verify_citations guardrail."""

    def test_supported_claims_no_warnings(self, pipeline):
        answer = "补考需要在考试后两周内通过教务系统申请。"
        chunks = [
            {"content": "学生应在考试结束后两周内通过教务系统提交补考申请。"},
        ]
        warnings = pipeline._verify_citations(answer, chunks)
        assert len(warnings) == 0

    def test_unsupported_claim_flagged(self, pipeline):
        # Source is about 补考流程, answer hallucinates about 宿舍电器
        answer = "根据校规，宿舍可以使用电饭煲和电磁炉。"
        chunks = [
            {"content": "补考由学生本人在教务系统中提交申请，补考不及格需要重修。"},
        ]
        warnings = pipeline._verify_citations(answer, chunks)
        # No overlap between "宿舍/电饭煲/电磁炉" and "补考/重修/教务系统"
        assert len(warnings) > 0

    def test_empty_chunks_no_warnings(self, pipeline):
        warnings = pipeline._verify_citations("任何回答内容", [])
        assert warnings == []
