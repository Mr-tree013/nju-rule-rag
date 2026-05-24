"""
RAG 完整问答流程。

串联：风险分类 → 混合检索 → 依据判断 → LLM 生成 → 结果校验

不负责：
- 检索索引构建（B 同学）
- 风险关键词维护（answer_policy.py）
- LLM 调用细节（llm_client.py）
- Web 接口（main.py）
"""

import time

from app.answer_policy import (
    build_high_risk_notice,
    classify_question,
    need_human_confirm,
    no_evidence_response,
)
from app.config import HIGH_RISK_MIN_SCORE
from app.llm_client import LLMError, chat
from app.retriever import HybridRetriever

# ---------- 常量 ----------

# 当所有检索 chunk 分数低于此值时，视为无可靠依据
MIN_RELIABLE_SCORE = 0.10

# 回答最大长度（适配 QQ 群展示）
MAX_ANSWER_LENGTH = 600


# ---------- 懒加载检索器 ----------

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# ---------- 提示词 ----------

SYSTEM_PROMPT = """你是一个南京大学本科新生校规与教务流程问答助手。

你必须严格遵守以下规则：

1. 只能依据下面提供的【参考资料片段】进行回答，不得使用任何外部知识。
2. 不得编造任何文件名、文件编号、具体日期、条款编号或链接。如果资料中没有，就说没有。
3. 如果资料片段不足以回答用户的问题，必须明确说：抱歉，目前没有找到与您问题相关的足够可靠的校规依据。
4. 对于涉及退学、开除、处分、作弊、学位等高风险问题，只提供校规中已有的客观规定描述，不得对用户个人情况做出判断或结论。
5. 回答要简洁直接，控制在 300 字以内，适合在 QQ 群里快速阅读。
6. 不要在回答中提及"根据参考资料"、"资料显示"等引用词，直接给出答案即可。
7. 如果用户问题与校规、教务流程完全无关，礼貌说明你只能回答校规相关问题。
8. 回答格式：先给出简短的直接结论（1-2句），再列出关键要点（如有必要）。不要添加客套话和无关内容。"""


def build_context(chunks):
    """
    将检索到的 chunks 拼接为 LLM 可读的参考资料。

    每个 chunk 格式:
        [来源: {title} | 条款: {section}]
        {content}
    """
    if not chunks:
        return "（无参考资料）"

    parts = []
    for i, c in enumerate(chunks):
        parts.append(
            f"[来源: {c['title']} | 条款: {c.get('section', '无')}]\n"
            f"{c['content']}"
        )
    return "\n\n---\n\n".join(parts)


def build_prompt(question, chunks, risk_level):
    """
    构建发送给 LLM 的消息列表。

    高风险问题的 user 消息会额外强调不要下个人结论。
    """
    context = build_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"【参考资料片段】\n\n{context}\n\n【用户问题】\n{question}",
        },
    ]

    # 高风险问题追加防止下结论的提醒
    if risk_level == "high":
        messages.append(
            {
                "role": "user",
                "content": "（注意：这是一个高风险问题。请只描述校规中已有的客观规定，"
                "不要对用户个人情况做任何判断或结论。）",
            }
        )

    return messages


def extract_sources(chunks):
    """
    从检索 chunks 中提取来源信息。

    返回与 /ask 契约一致的 sources 列表。
    """
    return [
        {
            "chunk_id": c["chunk_id"],
            "source_id": c.get("source_id", c["chunk_id"].rsplit("-", 1)[0]),
            "title": c["title"],
            "url": c.get("url", ""),
            "priority": c.get("priority", 5),
        }
        for c in chunks
    ]


# ---------- 主流程 ----------


def answer_question(question):
    """
    完整的 RAG 问答流程。

    参数:
        question: 用户问题字符串

    返回:
        {
            "question": str,
            "answer": str,
            "risk_level": "low"|"medium"|"high",
            "need_human_confirm": bool,
            "sources": [{chunk_id, source_id, title, url, priority}, ...],
            "debug": {"retrieval_count": int, "latency": float}
        }
    """
    t_start = time.time()

    # 1. 空问题直接返回
    if not question or not question.strip():
        return {
            "question": question or "",
            "answer": "请输入您的问题。",
            "risk_level": "low",
            "need_human_confirm": False,
            "sources": [],
            "debug": {"retrieval_count": 0, "latency": 0},
        }

    # 2. 风险分级（在任何可能抛异常的代码之前完成，确保后续 except 块可用）
    risk_level = "low"
    try:
        risk_level = classify_question(question)
    except Exception:
        pass

    # 3. 混合检索
    try:
        retriever = _get_retriever()
        chunks = retriever.search(question)
    except Exception:
        # 检索失败，返回服务器错误
        t_end = time.time()
        return {
            "question": question,
            "answer": "系统暂时不可用，请稍后再试。",
            "risk_level": risk_level,
            "need_human_confirm": True,
            "sources": [],
            "debug": {"retrieval_count": 0, "latency": round(t_end - t_start, 2)},
        }

    retrieval_count = len(chunks)

    # 4. 判断是否有足够依据
    reliable_chunks = [c for c in chunks if c["score"] >= MIN_RELIABLE_SCORE]
    if not reliable_chunks:
        t_end = time.time()
        result = no_evidence_response(question)
        # Preserve the actual risk level instead of always "low".
        result["risk_level"] = risk_level
        result["need_human_confirm"] = need_human_confirm(question, risk_level)
        result["debug"] = {"retrieval_count": retrieval_count, "latency": round(t_end - t_start, 2)}
        return result

    # 5. 高风险问题使用更严格的阈值
    if risk_level == "high":
        strong_chunks = [c for c in reliable_chunks if c["score"] >= HIGH_RISK_MIN_SCORE]
        if not strong_chunks:
            t_end = time.time()
            return {
                "question": question,
                "answer": (
                    "抱歉，根据现有资料无法对您的问题给出明确的校规依据，"
                    "或检索到的相关规定不足以支撑可靠的回答。\n\n"
                    "涉及可能影响学业或学籍的重大事项，"
                    "请你务必第一时间联系所在院系教务员、辅导员或学校相关部门，获取正式处理意见。"
                ),
                "risk_level": risk_level,
                "need_human_confirm": True,
                "sources": [],
                "debug": {"retrieval_count": retrieval_count, "latency": round(t_end - t_start, 2)},
            }

    # 6. 构建 Prompt 并调用 LLM
    messages = build_prompt(question, reliable_chunks, risk_level)

    try:
        answer = chat(messages, temperature=0.2)
    except LLMError:
        t_end = time.time()
        return {
            "question": question,
            "answer": "系统暂时不可用，请稍后再试。",
            "risk_level": risk_level,
            "need_human_confirm": need_human_confirm(question, risk_level),
            "sources": extract_sources(reliable_chunks[:5]),
            "debug": {"retrieval_count": retrieval_count, "latency": round(t_end - t_start, 2)},
        }

    # 7. 控制回答长度
    if len(answer) > MAX_ANSWER_LENGTH:
        answer = answer[:MAX_ANSWER_LENGTH] + "..."

    # 8. 高风险问题的末尾追加提醒
    if risk_level == "high":
        answer += "\n\n" + build_high_risk_notice(question)

    # 9. 提取来源
    sources = extract_sources(reliable_chunks[:5])

    # 10. 是否需要人工确认
    confirm = need_human_confirm(question, risk_level)

    t_end = time.time()

    return {
        "question": question,
        "answer": answer,
        "risk_level": risk_level,
        "need_human_confirm": confirm,
        "sources": sources,
        "debug": {
            "retrieval_count": retrieval_count,
            "latency": round(t_end - t_start, 2),
        },
    }
