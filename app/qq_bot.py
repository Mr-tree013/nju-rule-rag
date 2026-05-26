"""
QQ Bot adapter layer.

Responsibility boundary: receives messages → calls /ask → formats replies.
Contains zero RAG, retrieval, or risk-judgment logic.
"""

import requests

from app.config import get_settings


def _settings():
    return get_settings()


def ask_backend(question: str) -> dict | None:
    """Call the FastAPI ``/ask`` endpoint, return the full response dict."""
    s = _settings()
    try:
        resp = requests.post(
            f"{s.qq_bot_api_base_url}/ask",
            json={"question": question},
            timeout=s.qq_bot_request_timeout,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


def format_reply(question: str) -> str:
    """
    Convert a user question into a QQ-group-friendly reply.

    Format::

        【结论】
        ...（up to max_reply_length chars）

        【依据】
        1. 《来源标题》

        【提醒】
        ...（only for high-risk）
    """
    s = _settings()
    data = ask_backend(question)

    if data is None:
        return "系统暂时不可用，请稍后再试。"

    answer = data.get("answer", "").strip()
    sources = data.get("sources", [])
    risk_level = data.get("risk_level", "")

    if data.get("error") == "internal_error" or risk_level == "unknown":
        return "系统暂时不可用，请稍后再试。"

    lines = ["【结论】", answer, ""]

    if sources:
        lines.append("【依据】")
        for i, src in enumerate(sources, 1):
            title = src.get("title", "未知来源")
            lines.append(f"{i}. 《{title}》")
        lines.append("")

    if risk_level == "high":
        lines.append("【提醒】")
        lines.append(
            "以上信息仅供参考，不构成对个人情况的正式结论。"
            "涉及重大事项，请务必联系院系教务员或辅导员获取正式处理意见。"
        )
        lines.append("")

    reply = "\n".join(lines).strip()

    if len(reply) > s.qq_bot_max_reply_length:
        reply = reply[:s.qq_bot_max_reply_length] + "..."

    return reply


def handle_message(message: str) -> str:
    """
    QQ Bot message handler entry point.

    Recognises ``/问`` or ``/ask`` prefixes, extracts the question,
    and returns a formatted reply.  All other messages receive a usage hint.
    """
    msg = message.strip()

    for prefix in ("/问 ", "/ask ", "/问", "/ask"):
        if msg.startswith(prefix):
            question = msg[len(prefix):].strip()
            if not question:
                return "请输入问题。例如：/问 缓考怎么申请？"
            return format_reply(question)

    return (
        "欢迎使用 NJU Rule RAG Bot！\n"
        "发送 /问 + 你的问题 即可查询校规。\n"
        "例如：/问 缓考怎么申请？"
    )
