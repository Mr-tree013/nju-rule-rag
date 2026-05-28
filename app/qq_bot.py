"""
QQ Bot adapter layer.

Responsibility boundary: receives messages → calls pipeline → formats replies.
Contains zero RAG, retrieval, or risk-judgment logic.
"""

import re

from app.config import get_settings


def _settings():
    return get_settings()


def ask_backend(question: str) -> dict | None:
    """Call the RAG pipeline directly (same process, no HTTP round-trip)."""
    try:
        from app.pipeline import answer_question
        return answer_question(question)
    except Exception:
        return None


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting for plain-text QQ group display."""
    # Remove bold/italic markers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    # Remove heading markers
    text = re.sub(r"^#{1,4}\s+", "", text, flags=re.MULTILINE)
    # Remove backtick code
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_reply(question: str) -> str:
    """
    Convert a user question into a QQ-group-friendly plain-text reply.

    Format::

        结论
        ...（up to max_reply_length chars）

        依据
        1. 来源标题

        提醒
        ...（only for high-risk）
    """
    s = _settings()
    data = ask_backend(question)

    if data is None:
        return "系统暂时不可用，请稍后再试。"

    answer = _strip_markdown(data.get("answer", "").strip())
    sources = data.get("sources", [])
    risk_level = data.get("risk_level", "")

    if data.get("error") == "internal_error" or risk_level == "unknown":
        return "系统暂时不可用，请稍后再试。"

    lines = ["结论", answer, ""]

    if sources:
        lines.append("依据")
        for i, src in enumerate(sources, 1):
            title = src.get("title", "未知来源")
            lines.append(f"{i}. {title}")
        lines.append("")

    if risk_level == "high":
        lines.append("提醒")
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
    QQ Bot message handler.  Extracts the question from the message and
    returns a formatted reply.  Strips /ask and /问 command prefixes if
    present; otherwise treats the whole message as the question.
    """
    msg = message.strip()
    if not msg:
        return ""

    # Strip optional command prefix
    for prefix in ("/ask ", "/问 ", "/ask", "/问"):
        if msg.startswith(prefix):
            msg = msg[len(prefix):].strip()
            break

    if not msg:
        return ""

    return format_reply(msg)
