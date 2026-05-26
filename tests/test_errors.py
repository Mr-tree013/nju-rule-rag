"""Tests for shared exception hierarchy."""
from app.errors import (
    ConfigError,
    EmptyQuestionError,
    LLMError,
    RAGError,
    RetrievalError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_rag_error(self):
        assert issubclass(ConfigError, RAGError)
        assert issubclass(LLMError, RAGError)
        assert issubclass(EmptyQuestionError, RAGError)
        assert issubclass(RetrievalError, RAGError)

    def test_llm_error_stores_status_code(self):
        e = LLMError("bad request", status_code=400)
        assert e.status_code == 400
        assert "bad request" in str(e)

    def test_llm_error_status_code_defaults_to_none(self):
        e = LLMError("timeout")
        assert e.status_code is None

    def test_empty_question_is_rag_error(self):
        assert issubclass(EmptyQuestionError, RAGError)

    def test_can_catch_all_with_rag_error(self):
        for exc_cls in [ConfigError, LLMError, EmptyQuestionError, RetrievalError]:
            try:
                raise exc_cls("test")
            except RAGError:
                pass  # should catch
            else:
                pytest.fail(f"RAGError should catch {exc_cls.__name__}")
