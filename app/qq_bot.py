"""
QQ Bot adapter layer.

Responsibility boundary: receives messages → calls pipeline → formats replies.
Contains zero RAG, retrieval, or risk-judgment logic.
"""

import re
import time
from collections import deque

from app.config import get_settings

# ── In-group feedback ──────────────────────────────────────────────
# Key: (group_id, user_id) → {question, request_id, ts}
_pending_requests: dict[tuple[int | str, int | str], dict] = {}
_PENDING_TTL = 300  # 5 minutes

FEEDBACK_SUFFIX = "\n━━━━━━\n👍 有用  |  👎 有问题"

# ── Conversation memory (multi-turn) ───────────────────────────────
# Key: (group_id, user_id) → deque of {question, answer, ts}
_conversations: dict[tuple[int | str, int | str], deque] = {}
_CONV_MAX_TURNS = 3
_CONV_TTL = 300  # 5 minutes


def _settings():
    return get_settings()


def _cleanup_expired():
    """Remove pending requests older than _PENDING_TTL."""
    now = time.time()
    expired = [k for k, v in _pending_requests.items() if now - v["ts"] > _PENDING_TTL]
    for k in expired:
        _pending_requests.pop(k, None)


def store_pending(group_id: int | str, user_id: int | str,
                  question: str, request_id: str):
    """Remember that a user just asked a question, so follow-up emoji can be
    interpreted as feedback."""
    _cleanup_expired()
    _pending_requests[(str(group_id), str(user_id))] = {
        "question": question,
        "request_id": request_id,
        "ts": time.time(),
    }


def get_history(group_id: int | str, user_id: int | str) -> list[dict] | None:
    """Return recent Q&A pairs for prompt injection, or None if expired/empty."""
    _cleanup_conversations()
    key = (str(group_id), str(user_id))
    turns = _conversations.get(key)
    if not turns:
        return None
    return list(turns)


def add_to_history(group_id: int | str, user_id: int | str,
                   question: str, answer: str):
    """Append a Q&A turn to the user's conversation memory."""
    _cleanup_conversations()
    key = (str(group_id), str(user_id))
    if key not in _conversations:
        _conversations[key] = deque(maxlen=_CONV_MAX_TURNS)
    _conversations[key].append({
        "question": question,
        "answer": answer,
        "ts": time.time(),
    })


def _cleanup_conversations():
    """Remove expired conversation entries."""
    now = time.time()
    expired = []
    for k, turns in _conversations.items():
        # Keep only turns within TTL
        while turns and now - turns[0]["ts"] > _CONV_TTL:
            turns.popleft()
        if not turns:
            expired.append(k)
    for k in expired:
        _conversations.pop(k, None)


# ── Feedback detection ─────────────────────────────────────────────

_FEEDBACK_UP = {
    "👍", "好", "好了", "赞", "有用", "有用了", "对的", "对了",
    "不错", "可以", "棒", "厉害", "靠谱", "行的", "好评",
    "好用", "好用的", "对", "谢谢", "牛", "行的", "好了",
    "yes", "y", "1",
}
_FEEDBACK_DOWN = {
    "👎", "差", "踩", "没用", "错的", "不对", "错了", "错误",
    "有问题", "不行", "不准", "胡说", "瞎说", "假的", "烂",
    "不好", "不太好", "算了吧",
    "no", "n", "2",
}
_RE_CQ = re.compile(r"\[CQ:\w+,.*?\]")


def check_feedback(text: str, group_id: int | str,
                   user_id: int | str) -> dict | None:
    """If *text* is a pure-feedback message from a user with a pending
    question, return {rating, question, request_id}.  Otherwise None."""
    stripped = _RE_CQ.sub("", text).strip()
    if not stripped:
        return None

    rating = None
    if stripped in _FEEDBACK_UP:
        rating = "up"
    elif stripped in _FEEDBACK_DOWN:
        rating = "down"
    else:
        return None

    key = (str(group_id), str(user_id))
    pending = _pending_requests.pop(key, None)
    if pending is None:
        return None  # user reacted but has no recent question

    _cleanup_expired()
    return {
        "rating": rating,
        "question": pending["question"],
        "request_id": pending["request_id"],
    }


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


def format_reply_from_data(question: str, data: dict | None,
                           group_id: int | str = "",
                           user_id: int | str = "",
                           request_id: str = "") -> str:
    """Same as format_reply but accepts pre-fetched pipeline data (avoids double call).

    When *group_id* + *user_id* + *request_id* are provided, a pending
    entry is stored so follow-up 👍/👎 messages can be matched as feedback.
    """
    s = _settings()

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

    # Feedback prompt
    lines.append(FEEDBACK_SUFFIX)

    reply = "\n".join(lines).strip()

    if len(reply) > s.qq_bot_max_reply_length:
        reply = reply[:s.qq_bot_max_reply_length] + "..."

    # Register pending so follow-up emoji maps to feedback
    if group_id and user_id and request_id:
        store_pending(group_id, user_id, question, request_id)

    return reply


def format_reply(question: str) -> str:
    """
    Convert a user question into a QQ-group-friendly plain-text reply.
    """
    data = ask_backend(question)
    return format_reply_from_data(question, data)


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
