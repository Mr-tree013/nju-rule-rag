"""
QQ Bot 适配层。

只负责：接收消息 → 调用 /ask → 格式化回复。
不包含任何 RAG、检索、风险判断逻辑。
"""

import requests

# 可通过环境变量配置，默认指向本地开发服务。
API_BASE_URL = "http://127.0.0.1:8000"

MAX_REPLY_LENGTH = 800
REQUEST_TIMEOUT = 30  # seconds


def ask_backend(question: str) -> dict | None:
    """
    调用 FastAPI /ask 接口，返回完整响应 dict。
    失败时返回 None。
    """
    try:
        resp = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


def format_reply(question: str) -> str:
    """
    将用户问题转换为 QQ 群回复文本。

    格式：
      【结论】
      ...（不超过 800 字）

      【依据】
      1. 《来源标题》
      2. 《来源标题》

      【提醒】
      ...（仅高风险问题显示）
    """
    data = ask_backend(question)

    # 后端不可用
    if data is None:
        return "系统暂时不可用，请稍后再试。"

    answer = data.get("answer", "").strip()
    sources = data.get("sources", [])
    risk_level = data.get("risk_level", "")

    # 后端返回了内部错误
    if data.get("error") == "internal_error" or risk_level == "unknown":
        return "系统暂时不可用，请稍后再试。"

    lines = []

    # 结论
    lines.append("【结论】")
    lines.append(answer)
    lines.append("")

    # 依据（有来源时才显示）
    if sources:
        lines.append("【依据】")
        for i, src in enumerate(sources, 1):
            title = src.get("title", "未知来源")
            lines.append(f"{i}. 《{title}》")
        lines.append("")

    # 提醒（高风险问题追加）
    if risk_level == "high":
        lines.append("【提醒】")
        lines.append(
            "以上信息仅供参考，不构成对个人情况的正式结论。"
            "涉及重大事项，请务必联系院系教务员或辅导员获取正式处理意见。"
        )
        lines.append("")

    reply = "\n".join(lines).strip()

    # 硬截断
    if len(reply) > MAX_REPLY_LENGTH:
        reply = reply[:MAX_REPLY_LENGTH] + "..."

    return reply


# ── 触发入口（供 QQ Bot 框架调用）───────────────────────────────

def handle_message(message: str) -> str:
    """
    QQ Bot 消息处理入口。

    识别 "/问 " 或 "/ask " 前缀，提取问题并返回格式化回复。
    其他消息返回使用说明。
    """
    msg = message.strip()

    # 识别命令前缀
    for prefix in ("/问 ", "/ask ", "/问", "/ask"):
        if msg.startswith(prefix):
            question = msg[len(prefix):].strip()
            if not question:
                return "请输入问题。例如：/问 缓考怎么申请？"
            return format_reply(question)

    # 非命令消息
    return (
        "欢迎使用 NJU Rule RAG Bot！\n"
        "发送 /问 + 你的问题 即可查询校规。\n"
        "例如：/问 缓考怎么申请？"
    )
