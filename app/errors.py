"""
Shared exception hierarchy for the NJU Rule RAG application.

All application-level exceptions inherit from RAGError so callers can
catch a single base type when they need to handle any app error.
"""


class RAGError(Exception):
    """Base exception for all application errors."""


class ConfigError(RAGError):
    """Configuration is missing or invalid."""


class EmptyQuestionError(RAGError):
    """User submitted an empty or whitespace-only question."""


class LLMError(RAGError):
    """LLM API call failed (network, auth, rate-limit, or server error)."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class RetrievalError(RAGError):
    """Index loading or search failed."""
