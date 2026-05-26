"""Tests for LLM client."""
import pytest

from app.errors import LLMError
from app.llm_client import LLMClient


class TestLLMClientInit:
    def test_requires_api_key(self):
        with pytest.raises(LLMError) as exc:
            LLMClient(api_key="", base_url="https://api.example.com", model="m")
        assert "LLM_API_KEY" in str(exc.value)

    def test_requires_model(self):
        with pytest.raises(LLMError) as exc:
            LLMClient(api_key="sk-test", base_url="https://api.example.com", model="")
        assert "LLM_MODEL" in str(exc.value)

    def test_creates_with_valid_params(self):
        c = LLMClient(api_key="sk-test", base_url="https://api.example.com", model="gpt-4")
        assert c.model == "gpt-4"

    def test_default_retry_config(self):
        c = LLMClient(api_key="sk-test", base_url="https://api.example.com", model="m")
        assert c._retry_count == 3
        assert c._retry_delays == (1, 2, 4)
        assert c._timeout == 60

    def test_custom_retry_config(self):
        c = LLMClient(
            api_key="sk-test",
            base_url="https://api.example.com",
            model="m",
            retry_count=5,
            retry_delays=(2, 4, 8, 16, 32),
            timeout=30,
        )
        assert c._retry_count == 5
        assert c._timeout == 30


class TestKeyMasking:
    def test_masks_normal_key(self):
        masked = LLMClient.mask_key("sk-abcdefgh12345678")
        assert masked == "sk-a***5678"

    def test_handles_short_key(self):
        masked = LLMClient.mask_key("abc")
        assert masked == "***"

    def test_handles_empty_key(self):
        masked = LLMClient.mask_key("")
        assert masked == "***"

    def test_key_display_property(self):
        c = LLMClient(api_key="sk-1234567890abcdef", base_url="https://x.com", model="m")
        assert c.key_display == "sk-1***cdef"


class TestHeaders:
    def test_authorization_header(self):
        c = LLMClient(api_key="sk-test", base_url="https://api.example.com", model="m")
        headers = c._headers
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Content-Type"] == "application/json"
