"""
风险分级与回答策略。

职责：
1. 对用户问题做风险分级（low / medium / high）
2. 判断是否需要人工确认
3. 找不到依据时的标准拒答
4. 高风险问题的提示语

规则来源：docs/risk_policy.md
"""

HIGH_RISK_KEYWORDS = [
    "退学", "开除", "处分", "作弊", "学位",
    "毕业资格", "毕业不了", "学籍取消", "严重警告", "取消考试资格",
    "毕业",       # 覆盖 "我还能毕业吗" 等毕业资格类问题
]

MEDIUM_RISK_KEYWORDS = [
    "转专业", "辅修", "课程认定", "交换", "缓考",
    "补考", "重修", "学业预警", "绩点", "成绩", "选课",
]

PROCESS_KEYWORDS = [
    "怎么办", "在哪里", "入口", "流程", "材料",
    "申请", "截止", "什么时候", "找谁",
]


def classify_question(question: str) -> str:
    """
    对问题进行风险分级。

    返回 "high"、"medium" 或 "low"。

    规则：
    - 命中 HIGH_RISK_KEYWORDS → high
    - 未命中 high 但命中 MEDIUM_RISK_KEYWORDS → medium
    - 其他 → low
    """
    if not question or not question.strip():
        return "low"

    text = question

    for kw in HIGH_RISK_KEYWORDS:
        if kw in text:
            return "high"

    for kw in MEDIUM_RISK_KEYWORDS:
        if kw in text:
            return "medium"

    return "low"


def is_process_question(question: str) -> bool:
    """
    判断问题是否属于流程类查询。

    流程类问题通常包含 "怎么办"、"在哪里"、"申请" 等关键词。
    这些问题的风险较低，可以给出具体操作指引。
    """
    if not question:
        return False

    text = question
    for kw in PROCESS_KEYWORDS:
        if kw in text:
            return True
    return False


def need_human_confirm(question: str, risk_level: str) -> bool:
    """
    判断是否需要提示用户进一步人工确认。

    - high 风险 → 一定 True
    - medium 风险 → 默认 True
    - low 风险 → 默认 False
    """
    if risk_level == "high":
        return True
    if risk_level == "medium":
        return True
    return False


def no_evidence_response(question: str) -> dict:
    """
    找不到足够可靠依据时的标准拒答。

    返回与 /ask 一致的 dict 结构，来源为空。
    """
    return {
        "question": question,
        "answer": (
            "抱歉，目前没有找到与该问题相关的足够可靠的校规依据。"
            "建议你联系所在院系的教务员或辅导员获取准确信息。"
        ),
        "risk_level": "low",
        "need_human_confirm": True,
        "sources": [],
    }


def build_high_risk_notice(question: str) -> str:
    """
    高风险问题的提示语，追加在回答末尾。
    """
    return (
        "需要提醒的是，以上信息仅供参考，不构成对个人情况的正式结论。"
        "涉及退学、开除、处分、作弊、学位等重大事项，"
        "请你务必第一时间联系所在院系教务员、辅导员或学校相关部门，获取正式处理意见。"
    )
