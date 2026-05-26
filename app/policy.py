"""
Risk classification and response templates.

Pluggable design: to add a custom risk classifier, subclass ``RiskClassifier``
and override the keyword tuples.  To change response wording, subclass
``ResponseTemplates`` and override the relevant method.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    level: RiskLevel
    is_process: bool = False


# ── Risk classifier ──────────────────────────────────────────────────


class RiskClassifier:
    """
    Keyword-based risk classifier.

    Keywords are immutable ``ClassVar`` tuples — override them in a subclass
    to add or remove terms without touching the matching logic.
    """

    high_keywords: ClassVar[tuple[str, ...]] = (
        "退学", "开除", "处分", "作弊", "代考", "替考",
        "劝退", "留级", "记过", "留校察看", "通报批评",
        "毕业资格", "毕业不了", "不能毕业", "无法毕业",
        "毕不了业", "还能毕业", "还能正常毕业",
        "没毕业", "能不能毕业", "毕业论文没过",
        "拿不到学位", "不给学位", "撤销学位",
        "学位被取消", "没有学位", "取消学位",
        "学籍取消", "严重警告", "取消考试资格",
        "会不会被开除", "会不会被退学", "被开除", "被退学",
        "学术不端", "论文抄袭", "被处分",
    )

    # Phrases containing "学位" that indicate a purely informational query.
    # When the ONLY high-keyword hit is a substring of "学位" and the
    # question matches one of these, we suppress the high classification.
    _degree_info_phrases: ClassVar[tuple[str, ...]] = (
        "学位证", "学位认证", "学位申请", "学位查询",
    )

    medium_keywords: ClassVar[tuple[str, ...]] = (
        "转专业", "辅修", "课程认定", "交换", "缓考",
        "补考", "重修", "学业预警", "绩点", "成绩", "选课",
        "挂科", "学位", "休学", "复学", "免修", "免听", "退课",
    )

    process_keywords: ClassVar[tuple[str, ...]] = (
        "怎么办", "在哪里", "入口", "流程", "材料",
        "申请", "截止", "什么时候", "找谁",
    )

    def classify(self, question: str) -> ClassificationResult:
        """
        Return a ``ClassificationResult`` for *question*.

        Matching order: high > medium > low.

        A bare "学位" match is downgraded to medium when the question is
        purely informational (e.g. "学位证和毕业证有什么区别").
        """
        if not question or not question.strip():
            return ClassificationResult(RiskLevel.LOW)

        text = question
        for kw in self.high_keywords:
            if kw in text:
                level = RiskLevel.HIGH
                # Downgrade: bare "学位" informational queries
                if self._is_degree_info_only(text, kw):
                    level = RiskLevel.MEDIUM
                return ClassificationResult(
                    level=level,
                    is_process=self._match_any(text, self.process_keywords),
                )

        for kw in self.medium_keywords:
            if kw in text:
                return ClassificationResult(
                    level=RiskLevel.MEDIUM,
                    is_process=self._match_any(text, self.process_keywords),
                )

        return ClassificationResult(
            level=RiskLevel.LOW,
            is_process=self._match_any(text, self.process_keywords),
        )

    def is_process_question(self, question: str) -> bool:
        """True when *question* contains process-related keywords."""
        if not question:
            return False
        return self._match_any(question, self.process_keywords)

    def needs_human_confirm(self, question: str, level: RiskLevel) -> bool:
        """True when a human should review the answer before showing it."""
        if level == RiskLevel.HIGH:
            return True
        if level == RiskLevel.MEDIUM:
            return True
        return False

    @staticmethod
    def _match_any(text: str, keywords: tuple[str, ...]) -> bool:
        for kw in keywords:
            if kw in text:
                return True
        return False

    @classmethod
    def _is_degree_info_only(cls, text: str, matched_kw: str) -> bool:
        """True when *matched_kw* is a bare 学位 hit on an informational query.

        E.g. "学位证和毕业证有什么区别" should not be high risk just because
        it happens to contain the substring "学位".
        """
        # Only apply to the bare "学位" keyword (2 chars), not longer phrases.
        if len(matched_kw) > 2 or matched_kw != "学位":
            return False
        return cls._match_any(text, cls._degree_info_phrases)


# ── Response templates ───────────────────────────────────────────────


class ResponseTemplates:
    """
    Overridable response templates for refusal and risk notices.

    Each method accepts the user's question and returns a dict or string
    matching the dev-contract ``/ask`` response shape.
    """

    def no_evidence(self, question: str) -> dict:
        return {
            "question": question,
            "answer": (
                "抱歉，目前没有找到与您问题相关的足够可靠的校规依据。\n\n"
                "建议你：\n"
                "1. 联系所在院系的教务员或辅导员获取准确信息；\n"
                "2. 访问南京大学本科生院官网 (jw.nju.edu.cn) 查询相关文件；\n"
                "3. 尝试换一种方式描述你的问题。"
            ),
            "risk_level": "low",
            "need_human_confirm": True,
            "sources": [],
        }

    def high_risk_notice(self, question: str) -> str:
        return (
            "需要提醒的是，以上信息仅供参考，不构成对个人情况的正式结论。"
            "涉及退学、开除、处分、作弊、学位等重大事项，"
            "请你务必第一时间联系所在院系教务员、辅导员或学校相关部门，获取正式处理意见。"
        )

    def high_risk_no_evidence(self, question: str) -> dict:
        return {
            "question": question,
            "answer": (
                "抱歉，根据现有资料无法对您的问题给出明确的校规依据，"
                "或检索到的相关规定不足以支撑可靠的回答。\n\n"
                "涉及可能影响学业或学籍的重大事项，"
                "请你务必第一时间联系所在院系教务员、辅导员或学校相关部门，获取正式处理意见。"
            ),
            "risk_level": "high",
            "need_human_confirm": True,
            "sources": [],
        }


# ── Default instances ────────────────────────────────────────────────

default_classifier = RiskClassifier()
default_templates = ResponseTemplates()


# ── Backward-compatible module-level functions ───────────────────────


def classify_question(question: str) -> str:
    """Return one of ``"low"``, ``"medium"``, ``"high"``."""
    return default_classifier.classify(question).level


def is_process_question(question: str) -> bool:
    return default_classifier.is_process_question(question)


def need_human_confirm(question: str, risk_level: str) -> bool:
    try:
        level = RiskLevel(risk_level)
    except ValueError:
        level = RiskLevel.LOW
    return default_classifier.needs_human_confirm(question, level)


def no_evidence_response(question: str) -> dict:
    return default_templates.no_evidence(question)


def build_high_risk_notice(question: str) -> str:
    return default_templates.high_risk_notice(question)
