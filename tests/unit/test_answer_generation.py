"""Answer-generation schemas, prompt, and AnthropicAnswerGenerator error mapping (no API calls)."""

from typing import Any

import anthropic
import pytest
from instructor.core.exceptions import InstructorRetryException
from pydantic import ValidationError

from src.core.exceptions import AnswerGenerationError, LLMTimeoutError
from src.llm.anthropic_answer_generator import AnthropicAnswerGenerator
from src.llm.base import AnswerGenerator
from src.llm.prompts import ANSWER_SYSTEM_PROMPT, build_user_message
from src.llm.schemas import ALLOWED_IDS_CONTEXT_KEY, ContextChunk, GeneratedAnswer
from tests.fakes import FakeAnswerGenerator, load_llm_fixture

CHUNKS = [
    ContextChunk(chunk_id="P-0024.md:4", patient_id="P-0024", section="Allergies", text="Latex."),
    ContextChunk(chunk_id="P-0005.md:4", patient_id="P-0005", section="Allergies", text="<b>x</b>"),
]


# ------------------------------------------------------------------ schemas


def test_saved_real_response_validates_against_its_retrieved_set() -> None:
    fixture = load_llm_fixture()
    answer = GeneratedAnswer.model_validate(
        fixture["answer"],
        context={ALLOWED_IDS_CONTEXT_KEY: set(fixture["retrieved_chunk_ids"])},
    )
    assert answer.insufficient_context is False
    assert answer.cited_chunk_ids == ["P-0024.md:4"]


def test_cited_id_outside_allowed_set_rejected() -> None:
    with pytest.raises(ValidationError, match="not provided"):
        GeneratedAnswer.model_validate(
            {"answer": "x", "cited_chunk_ids": ["P-9999.md:0"], "insufficient_context": False},
            context={ALLOWED_IDS_CONTEXT_KEY: {"P-0024.md:4"}},
        )


def test_citations_deduplicated_and_trimmed() -> None:
    answer = GeneratedAnswer(
        answer="x", cited_chunk_ids=[" a:1", "a:1", "", "b:2"], insufficient_context=False
    )
    assert answer.cited_chunk_ids == ["a:1", "b:2"]


def test_answer_without_citations_must_be_marked_insufficient() -> None:
    with pytest.raises(ValidationError, match="must cite"):
        GeneratedAnswer(answer="x", cited_chunk_ids=[], insufficient_context=False)
    ok = GeneratedAnswer(answer="Not in records.", insufficient_context=True)
    assert ok.cited_chunk_ids == []


# ------------------------------------------------------------------ prompt


def test_system_prompt_rules() -> None:
    text = ANSWER_SYSTEM_PROMPT.lower()
    for phrase in ("only", "cite", "insufficient_context", "diagnosis", "treatment advice"):
        assert phrase in text


def test_user_message_contains_ids_and_escapes_text() -> None:
    message = build_user_message("Who has <latex> allergy?", CHUNKS)
    assert 'id="P-0024.md:4"' in message
    assert "&lt;b&gt;x&lt;/b&gt;" in message  # chunk text cannot inject tags
    assert "&lt;latex&gt;" in message
    assert message.index("<chunks>") < message.index("<question>")


# ------------------------------------------------------------------ generator


class _StubMessages:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.result


class _StubClient:
    def __init__(self, messages: _StubMessages) -> None:
        self.messages = messages


def _generator(messages: _StubMessages) -> AnthropicAnswerGenerator:
    generator = AnthropicAnswerGenerator("test-key-not-real", max_retries=2, max_tokens=321)
    generator._client = _StubClient(messages)  # type: ignore[assignment]
    return generator


def _exc(cls: type[Exception], **attrs: Any) -> Exception:
    exc = cls.__new__(cls)  # SDK exceptions need live httpx objects; attributes are enough here
    for key, value in attrs.items():
        setattr(exc, key, value)
    return exc


def test_generator_requires_key() -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicAnswerGenerator(None)


def test_generator_satisfies_protocol_and_fake_too() -> None:
    assert isinstance(AnthropicAnswerGenerator("test-key-not-real"), AnswerGenerator)
    assert isinstance(FakeAnswerGenerator(), AnswerGenerator)


async def test_generate_success_passes_context_and_limits() -> None:
    expected = GeneratedAnswer(**load_llm_fixture()["answer"])
    messages = _StubMessages(result=expected)
    answer = await _generator(messages).generate("Who is allergic to latex?", CHUNKS)
    assert answer == expected
    kwargs = messages.kwargs
    assert kwargs["model"] == "claude-haiku-5-5"
    assert kwargs["max_tokens"] == 321
    assert kwargs["max_retries"] == 2
    assert kwargs["response_model"] is GeneratedAnswer
    assert kwargs["system"] == ANSWER_SYSTEM_PROMPT
    assert kwargs["context"] == {ALLOWED_IDS_CONTEXT_KEY: {"P-0024.md:4", "P-0005.md:4"}}
    assert "P-0024.md:4" in kwargs["messages"][0]["content"]


@pytest.mark.parametrize(
    ("error", "expected", "match"),
    [
        (_exc(anthropic.APITimeoutError), LLMTimeoutError, "timed out"),
        (_exc(anthropic.APIStatusError, status_code=529), AnswerGenerationError, "529"),
        (_exc(anthropic.APIConnectionError), AnswerGenerationError, "upstream"),
        (
            InstructorRetryException("bad", n_attempts=3, total_usage=0),
            AnswerGenerationError,
            "validation",
        ),
    ],
)
async def test_sdk_errors_mapped(error: Exception, expected: type[Exception], match: str) -> None:
    with pytest.raises(expected, match=match):
        await _generator(_StubMessages(error=error)).generate("q?", CHUNKS)


async def test_retry_exception_caused_by_timeout_maps_to_timeout() -> None:
    retry = InstructorRetryException("t", n_attempts=1, total_usage=0)
    retry.__cause__ = _exc(anthropic.APITimeoutError)
    with pytest.raises(LLMTimeoutError):
        await _generator(_StubMessages(error=retry)).generate("q?", CHUNKS)


async def test_unexpected_result_type_rejected() -> None:
    with pytest.raises(AnswerGenerationError, match="unexpected"):
        await _generator(_StubMessages(result={"answer": "raw dict"})).generate("q?", CHUNKS)


def test_on_attempt_hook_fires_per_attempt() -> None:
    attempts: list[int] = []
    generator = AnthropicAnswerGenerator("test-key-not-real", on_attempt=lambda: attempts.append(1))
    generator._client.hooks.emit_completion_arguments(model="x")
    generator._client.hooks.emit_completion_arguments(model="x")
    assert len(attempts) == 2
