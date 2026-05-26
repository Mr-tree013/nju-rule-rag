"""Tests for Settings dataclass and factory."""
import os

import pytest

from app.config import (
    DEFAULT_SYSTEM_PROMPT,
    RetrievalWeights,
    Settings,
    create_settings,
)


class TestRetrievalWeights:
    def test_default_weights_sum_to_one(self):
        w = RetrievalWeights()
        assert abs(w.bm25 + w.vector + w.priority - 1.0) < 0.01

    def test_validate_passes_for_valid_weights(self):
        w = RetrievalWeights(bm25=0.5, vector=0.3, priority=0.2)
        assert w.validate() == []

    def test_validate_warns_when_sum_not_one(self):
        w = RetrievalWeights(bm25=0.9, vector=0.3, priority=0.2)
        warnings = w.validate()
        assert len(warnings) == 1
        assert "权重之和" in warnings[0]

    def test_fallback_bm25_only(self):
        w = RetrievalWeights().fallback_bm25_only()
        assert w.bm25 == 0.80
        assert w.vector == 0.00
        assert w.priority == 0.20

    def test_fallback_vector_only(self):
        w = RetrievalWeights().fallback_vector_only()
        assert w.bm25 == 0.00
        assert w.vector == 0.80
        assert w.priority == 0.20


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.app_title == "NJU Rule RAG"
        assert s.bm25_top_k == 10
        assert s.hybrid_top_k == 5
        assert s.min_reliable_score == 0.2
        assert s.retry_count == 3
        assert s.max_answer_length == 600
        assert s.enable_vector is True

    def test_validate_warns_on_missing_llm_key(self):
        s = Settings(llm_api_key="", llm_model="")
        warnings = s.validate()
        assert any("LLM_API_KEY" in w for w in warnings)
        assert any("LLM_MODEL" in w for w in warnings)

    def test_validate_ok_when_configured(self):
        s = Settings(llm_api_key="sk-test", llm_model="deepseek-chat")
        warnings = s.validate()
        # May still warn about missing chunks file — that's fine
        assert not any("LLM_API_KEY" in w for w in warnings)

    def test_frozen_prevents_mutation(self):
        s = Settings()
        with pytest.raises(Exception):
            s.bm25_top_k = 999  # type: ignore

    def test_project_root_resolves(self):
        s = Settings()
        root = s.project_root
        assert root.name == "nju-rule-rag"

    def test_system_prompt_default(self):
        s = Settings()
        assert "南京大学" in s.system_prompt
        assert "参考资料" in s.system_prompt


class TestCreateSettings:
    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("APP_TITLE", "Test App")
        monkeypatch.setenv("BM25_TOP_K", "20")
        s = create_settings()
        assert s.app_title == "Test App"
        assert s.bm25_top_k == 20

    def test_bool_env_false_values(self, monkeypatch):
        for val in ("false", "0", "no"):
            monkeypatch.setenv("ENABLE_VECTOR", val)
            s = create_settings()
            assert s.enable_vector is False

    def test_invalid_int_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("BM25_TOP_K", "not_a_number")
        s = create_settings()
        assert s.bm25_top_k == 10


class TestBackwardCompat:
    def test_module_level_getattr(self):
        from app import config
        assert config.APP_TITLE == "NJU Rule RAG"
        assert config.BM25_TOP_K == 10
        assert config.HYBRID_TOP_K == 5
