"""Tests for risk classification and response templates.

Covers both the new class-based API (RiskClassifier, ResponseTemplates)
and the backward-compatible module-level functions.
"""

import pytest

from app.policy import (
    ClassificationResult,
    ResponseTemplates,
    RiskClassifier,
    RiskLevel,
    build_high_risk_notice,
    classify_question,
    is_process_question,
    need_human_confirm,
    no_evidence_response,
)


# ── Reusable instances ───────────────────────────────────────────────

@pytest.fixture
def classifier():
    return RiskClassifier()


@pytest.fixture
def templates():
    return ResponseTemplates()


# =====================================================================
# classify_question (backward-compat)
# =====================================================================

def test_classify_medium_缓考怎么申请():
    assert classify_question("缓考怎么申请") == "medium"


def test_classify_medium_补考在哪里报名():
    assert classify_question("补考在哪里报名") == "medium"


def test_classify_high_作弊会不会被开除():
    assert classify_question("我作弊了会不会被开除") == "high"


def test_classify_high_我还能毕业吗():
    assert classify_question("我还能毕业吗") == "high"


def test_classify_low_学籍异动怎么办():
    assert classify_question("学籍异动怎么办") == "low"


def test_classify_low_校历在哪里看():
    assert classify_question("校历在哪里看") == "low"


def test_classify_high优先于medium():
    assert classify_question("作弊了还能补考吗") == "high"


def test_classify_empty_string():
    assert classify_question("") == "low"


def test_classify_whitespace():
    assert classify_question("   ") == "low"


def test_classify_no_match():
    assert classify_question("今天天气怎么样") == "low"


# =====================================================================
# is_process_question (backward-compat)
# =====================================================================

def test_is_process_缓考怎么申请():
    assert is_process_question("缓考怎么申请") is True


def test_is_process_补考在哪里报名():
    assert is_process_question("补考在哪里报名") is True


def test_is_process_学籍异动怎么办():
    assert is_process_question("学籍异动怎么办") is True


def test_is_process_校历在哪里看():
    assert is_process_question("校历在哪里看") is True


def test_is_process_flow_keywords():
    assert is_process_question("选课申请流程") is True
    assert is_process_question("补考截止日期") is True
    assert is_process_question("缓考需要什么材料") is True


def test_is_process_no_match():
    assert is_process_question("今天天气怎么样") is False


def test_is_process_empty():
    assert is_process_question("") is False


# =====================================================================
# need_human_confirm (backward-compat)
# =====================================================================

def test_confirm_high():
    assert need_human_confirm("我作弊了会不会被开除", "high") is True


def test_confirm_medium():
    assert need_human_confirm("缓考怎么申请", "medium") is True


def test_confirm_low():
    assert need_human_confirm("校历在哪里看", "low") is False


# =====================================================================
# no_evidence_response (backward-compat)
# =====================================================================

def test_no_evidence_returns_dict():
    result = no_evidence_response("某个问题")
    assert isinstance(result, dict)
    assert "question" in result
    assert "answer" in result
    assert "risk_level" in result
    assert "need_human_confirm" in result
    assert "sources" in result


def test_no_evidence_sources_empty():
    result = no_evidence_response("某个问题")
    assert result["sources"] == []


def test_no_evidence_includes_question():
    result = no_evidence_response("缓考怎么申请")
    assert result["question"] == "缓考怎么申请"


def test_no_evidence_risk_level_low():
    result = no_evidence_response("某个问题")
    assert result["risk_level"] == "low"


# =====================================================================
# build_high_risk_notice (backward-compat)
# =====================================================================

def test_high_risk_notice_not_empty():
    notice = build_high_risk_notice("我作弊了会不会被开除")
    assert isinstance(notice, str)
    assert len(notice) > 0


def test_high_risk_notice_mentions_contact():
    notice = build_high_risk_notice("作弊")
    assert "教务" in notice or "辅导员" in notice or "相关部门" in notice


# =====================================================================
# New class-based API
# =====================================================================

class TestRiskClassifier:
    def test_classify_returns_classification_result(self, classifier):
        result = classifier.classify("缓考怎么申请")
        assert isinstance(result, ClassificationResult)
        assert result.level == RiskLevel.MEDIUM

    def test_classify_high_overrides_medium(self, classifier):
        result = classifier.classify("作弊了还能补考吗")
        assert result.level == RiskLevel.HIGH

    def test_classify_empty(self, classifier):
        assert classifier.classify("").level == RiskLevel.LOW

    def test_needs_human_confirm_high(self, classifier):
        assert classifier.needs_human_confirm("作弊", RiskLevel.HIGH) is True

    def test_needs_human_confirm_medium(self, classifier):
        assert classifier.needs_human_confirm("缓考", RiskLevel.MEDIUM) is True

    def test_needs_human_confirm_low(self, classifier):
        assert classifier.needs_human_confirm("校历", RiskLevel.LOW) is False

    def test_is_process_detects_flow(self, classifier):
        assert classifier.is_process_question("缓考怎么申请") is True
        assert classifier.is_process_question("天气不错") is False

    def test_subclass_can_extend_keywords(self):
        class Custom(RiskClassifier):
            high_keywords = RiskClassifier.high_keywords + ("新违规词",)

        c = Custom()
        assert c.classify("新违规词").level == RiskLevel.HIGH
        # Original classifier unchanged
        orig = RiskClassifier()
        assert orig.classify("新违规词").level == RiskLevel.LOW


class TestResponseTemplates:
    def test_no_evidence_has_expected_keys(self, templates):
        r = templates.no_evidence("测试")
        assert r["sources"] == []
        assert r["need_human_confirm"] is True

    def test_high_risk_notice_nonempty(self, templates):
        assert len(templates.high_risk_notice("作弊")) > 0

    def test_high_risk_no_evidence(self, templates):
        r = templates.high_risk_no_evidence("作弊")
        assert r["risk_level"] == "high"
        assert r["need_human_confirm"] is True
        assert r["sources"] == []
