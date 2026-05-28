"""
LLM and Embedding API client.

Wraps an OpenAI-compatible chat + embeddings API with automatic retry
and key masking in logs.  Supports DeepSeek, Qwen, Zhipu, OpenAI, and
any other provider that speaks the same HTTP contract.
"""

import time
from typing import Any

import requests

from app.errors import LLMError


class LLMClient:
    """
    OpenAI-compatible chat + embeddings client.

    Parameters:
        api_key: Bearer token for the API.
        base_url: Base URL (e.g. ``https://api.deepseek.com``).
        model: Model name (e.g. ``deepseek-chat``).
        retry_count: Max retries on transient failures.
        retry_delays: Seconds to wait between retries (one per attempt).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        retry_count: int = 3,
        retry_delays: tuple[float, ...] = (1, 2, 4),
        timeout: int = 60,
    ):
        if not api_key:
            raise LLMError("LLM_API_KEY 未设置")
        if not model:
            raise LLMError("LLM_MODEL 未设置")

        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._retry_count = retry_count
        self._retry_delays = retry_delays
        self._timeout = timeout

    # ── Public API ───────────────────────────────────────────────

    def chat(self, messages: list[dict], temperature: float = 0.2) -> str:
        """Send a chat-completion request and return the reply text."""
        url = f"{self._base_url}/chat/completions"
        body = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        resp = self._request("POST", url, body)
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def chat_stream(self, messages: list[dict], temperature: float = 0.2):
        """Stream chat-completion tokens via SSE.

        Yields content deltas as they arrive.  Caller must iterate the
        generator to receive tokens.
        """
        url = f"{self._base_url}/chat/completions"
        body = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        resp = requests.post(
            url, headers=self._headers, json=body,
            timeout=self._timeout, stream=True,
        )
        if resp.status_code >= 400:
            raise LLMError(
                f"LLM stream API 返回 {resp.status_code}: {resp.text[:300]}",
                status_code=resp.status_code,
            )
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]  # strip "data: " prefix
            if data_str.strip() == "[DONE]":
                break
            try:
                import json
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Send an embeddings request and return the vector list."""
        url = f"{self._base_url}/embeddings"
        body = {"model": self._model, "input": texts}
        resp = self._request("POST", url, body)
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    # ── Internal ─────────────────────────────────────────────────

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, url: str, body: dict) -> requests.Response:
        """Send an HTTP request with retry logic."""
        last_exc: Exception | None = None

        for attempt in range(self._retry_count + 1):
            try:
                resp = requests.request(
                    method, url, headers=self._headers, json=body, timeout=self._timeout
                )
                if resp.status_code < 400:
                    return resp
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < self._retry_count:
                        delay = self._retry_delays[attempt]
                        print(
                            f"[LLM] {resp.status_code} 错误，"
                            f"{delay}s 后重试 (第{attempt + 1}次)"
                        )
                        time.sleep(delay)
                        continue
                raise LLMError(
                    f"LLM API 返回 {resp.status_code}: {resp.text[:300]}",
                    status_code=resp.status_code,
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self._retry_count:
                    delay = self._retry_delays[attempt]
                    print(f"[LLM] 网络错误: {exc}，{delay}s 后重试 (第{attempt + 1}次)")
                    time.sleep(delay)
                else:
                    raise LLMError(f"LLM 网络请求失败: {exc}") from exc

        raise LLMError(
            f"LLM 调用失败（已重试 {self._retry_count} 次）"
        ) from last_exc

    @staticmethod
    def mask_key(key: str) -> str:
        """Return a log-safe version of an API key."""
        if not key or len(key) < 8:
            return "***"
        return key[:4] + "***" + key[-4:]

    # ── Convenience ──────────────────────────────────────────────

    @property
    def model(self) -> str:
        return self._model

    @property
    def key_display(self) -> str:
        return self.mask_key(self._api_key)


# ── Embedding-specific client ────────────────────────────────────────


class EmbeddingClient:
    """OpenAI-compatible embeddings API client (separate from chat LLM)."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        retry_count: int = 3,
        retry_delays: tuple[float, ...] = (1, 2, 4),
        timeout: int = 60,
    ):
        if not api_key:
            raise LLMError("EMBEDDING_API_KEY 未设置")
        if not model:
            raise LLMError("EMBEDDING_MODEL 未设置")

        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._retry_count = retry_count
        self._retry_delays = retry_delays
        self._timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        url = f"{self._base_url}/embeddings"
        body = {"model": self._model, "input": texts}
        resp = self._request("POST", url, body)
        return [item["embedding"] for item in resp.json()["data"]]

    def _request(self, method: str, url: str, body: dict) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(self._retry_count + 1):
            try:
                resp = requests.request(
                    method, url, headers=self._headers, json=body, timeout=self._timeout
                )
                if resp.status_code < 400:
                    return resp
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < self._retry_count:
                        delay = self._retry_delays[attempt]
                        print(
                            f"[Embed] {resp.status_code} 错误，"
                            f"{delay}s 后重试 (第{attempt + 1}次)"
                        )
                        time.sleep(delay)
                        continue
                raise LLMError(
                    f"Embedding API 返回 {resp.status_code}: {resp.text[:300]}",
                    status_code=resp.status_code,
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self._retry_count:
                    delay = self._retry_delays[attempt]
                    print(
                        f"[Embed] 网络错误: {exc}，"
                        f"{delay}s 后重试 (第{attempt + 1}次)"
                    )
                    time.sleep(delay)
                else:
                    raise LLMError(f"Embedding 网络请求失败: {exc}") from exc

        raise LLMError(
            f"Embedding 调用失败（已重试 {self._retry_count} 次）"
        ) from last_exc


# ── Backward-compatible module-level functions ───────────────────────

_default_client: LLMClient | None = None


def _get_default() -> LLMClient:
    global _default_client
    if _default_client is None:
        from app.config import _get_settings
        s = _get_settings()
        _default_client = LLMClient(
            api_key=s.llm_api_key,
            base_url=s.llm_base_url,
            model=s.llm_model,
        )
    return _default_client


def chat(messages: list[dict], temperature: float = 0.2) -> str:
    """Backward-compatible shorthand.  Prefer ``LLMClient.chat()``."""
    return _get_default().chat(messages, temperature)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Backward-compatible shorthand.  Prefer ``LLMClient.embed()``."""
    return _get_default().embed(texts)
