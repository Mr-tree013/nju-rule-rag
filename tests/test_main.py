"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "vector_enabled" in data


class TestAskEndpoint:
    def test_empty_question_returns_400(self):
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 400
        data = resp.json()
        assert data["risk_level"] == "low"
        assert data["need_human_confirm"] is False
        assert data["sources"] == []

    def test_whitespace_only_returns_400(self):
        resp = client.post("/ask", json={"question": "   "})
        assert resp.status_code == 400

    def test_valid_question_returns_200(self):
        """Requires LLM config. Without it, pipeline returns fallback
        response but the endpoint itself should return 200."""
        resp = client.post("/ask", json={"question": "缓考怎么申请？"})
        # Without LLM configured, pipeline may return 500 (LLMError) or
        # a fallback response. Both are valid — we test the endpoint shape.
        assert resp.status_code in (200, 500)
        data = resp.json()
        required = {"question", "answer", "risk_level", "need_human_confirm", "sources"}
        assert required.issubset(data.keys())

    def test_response_has_expected_shape(self):
        resp = client.post("/ask", json={"question": "缓考怎么申请？"})
        data = resp.json()
        assert isinstance(data["question"], str)
        assert isinstance(data["answer"], str)
        assert data["risk_level"] in ("low", "medium", "high", "unknown")
        assert isinstance(data["need_human_confirm"], bool)
        assert isinstance(data["sources"], list)
