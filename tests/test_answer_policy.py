import pytest

from app.answer_policy import (
    build_high_risk_notice,
    classify_question,
    is_process_question,
    need_human_confirm,
    no_evidence_response,
)


# ============================================================
# classify_question
# ============================================================

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
    """同时命中 high 和 medium 关键词时，应返回 high。"""
    assert classify_question("作弊了还能补考吗") == "high"


def test_classify_empty_string():
    assert classify_question("") == "low"


def test_classify_whitespace():
    assert classify_question("   ") == "low"


def test_classify_no_match():
    assert classify_question("今天天气怎么样") == "low"


# ============================================================
# is_process_question
# ============================================================

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


# ============================================================
# need_human_confirm
# ============================================================

def test_confirm_high():
    assert need_human_confirm("我作弊了会不会被开除", "high") is True


def test_confirm_medium():
    assert need_human_confirm("缓考怎么申请", "medium") is True


def test_confirm_low():
    assert need_human_confirm("校历在哪里看", "low") is False


# ============================================================
# no_evidence_response
# ============================================================

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


# ============================================================
# build_high_risk_notice
# ============================================================

def test_high_risk_notice_not_empty():
    notice = build_high_risk_notice("我作弊了会不会被开除")
    assert isinstance(notice, str)
    assert len(notice) > 0


def test_high_risk_notice_mentions_contact():
    notice = build_high_risk_notice("作弊")
    assert "教务" in notice or "辅导员" in notice or "相关部门" in notice
