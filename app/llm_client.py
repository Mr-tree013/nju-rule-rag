"""
LLM 与 Embedding API 封装。

支持通过 .env 配置切换不同 provider（DeepSeek / OpenAI / 通义千问 / 智谱）。
所有 provider 使用 OpenAI 兼容的 API 格式。

不暴露：API Key 不出现在日志中。
"""

import time

import requests

from app.config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)

# ---------- 自定义异常 ----------


class LLMError(Exception):
    """LLM 调用异常，包含状态码和原始错误信息。"""

    def __init__(self, message, status_code=None, orig=None):
        super().__init__(message)
        self.status_code = status_code
        self.original = orig


# ---------- 内部工具 ----------


def _safe_key(key):
    """返回脱敏后的 Key，日志输出用。"""
    if not key or len(key) < 8:
        return "***"
    return key[:4] + "***" + key[-4:]


RETRY_COUNT = 3
RETRY_DELAYS = [1, 2, 4]  # 秒


def _retry_request(method, url, headers, json_body, timeout=60):
    """
    发送 HTTP 请求，带自动重试。

    - 网络错误 → 重试
    - 429（限流） → 重试
    - 5xx → 重试
    - 4xx（非 429） → 不重试，直接抛异常
    """
    last_exc = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            resp = requests.request(
                method, url, headers=headers, json=json_body, timeout=timeout
            )
            if resp.status_code < 400:
                return resp
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < RETRY_COUNT:
                    delay = RETRY_DELAYS[attempt]
                    print(f"[LLM] {resp.status_code} 错误，{delay}s 后重试 (第{attempt+1}次)")
                    time.sleep(delay)
                    continue
            # 非重试错误
            raise LLMError(
                f"LLM API 返回 {resp.status_code}: {resp.text[:300]}",
                status_code=resp.status_code,
            )
        except requests.RequestException as e:
            last_exc = e
            if attempt < RETRY_COUNT:
                delay = RETRY_DELAYS[attempt]
                print(f"[LLM] 网络错误: {e}，{delay}s 后重试 (第{attempt+1}次)")
                time.sleep(delay)
            else:
                raise LLMError(f"LLM 网络请求失败: {e}", orig=e) from e

    raise LLMError(f"LLM 调用失败（已重试 {RETRY_COUNT} 次）", orig=last_exc)


def _build_headers(api_key):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


# ---------- 公开接口 ----------


def chat(messages, temperature=0.2):
    """
    调用 LLM 聊天接口。

    参数:
        messages: [{"role": "system"|"user", "content": "..."}, ...]
        temperature: 生成温度，默认 0.2（低随机性，适合事实问答）

    返回:
        str: 模型回复文本

    异常:
        LLMError: 配置缺失或调用失败
    """
    if not LLM_API_KEY or not LLM_MODEL:
        raise LLMError("LLM 未配置: 请在 .env 中设置 LLM_API_KEY 和 LLM_MODEL")

    url = f"{LLM_BASE_URL}/chat/completions"
    headers = _build_headers(LLM_API_KEY)
    body = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    resp = _retry_request("POST", url, headers, body)
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def embed_texts(texts):
    """
    调用 Embedding 接口，将文本列表转为向量列表。

    参数:
        texts: ["文本1", "文本2", ...]

    返回:
        [[float, ...], ...]  每个文本对应一个向量

    异常:
        LLMError: 配置缺失或调用失败
    """
    if not EMBEDDING_API_KEY or not EMBEDDING_MODEL:
        raise LLMError(
            "Embedding 未配置: 请在 .env 中设置 EMBEDDING_API_KEY 和 EMBEDDING_MODEL"
        )

    url = f"{EMBEDDING_BASE_URL}/embeddings"
    headers = _build_headers(EMBEDDING_API_KEY)
    body = {
        "model": EMBEDDING_MODEL,
        "input": texts,
    }

    resp = _retry_request("POST", url, headers, body)
    data = resp.json()
    return [item["embedding"] for item in data["data"]]
