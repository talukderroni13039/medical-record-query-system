"""Anthropic (Claude) implementation of `AnswerGenerator` using Instructor + `AsyncAnthropic`.

NOT on-premise: the question and retrieved chunk text are sent to the Anthropic API.

Call accounting: the SDK's own retries are disabled (`max_retries=0` on `AsyncAnthropic`), so every
Instructor attempt is exactly one API call. Instructor makes `max_retries + 1` attempts, re-asking
only when the output fails validation (e.g. a cited chunk ID that was not provided).
"""

import logging
from collections.abc import Callable
from typing import Any

import anthropic
import instructor
from instructor.core.exceptions import InstructorRetryException

from src.core.exceptions import AnswerGenerationError, LLMTimeoutError
from src.llm.prompts import ANSWER_SYSTEM_PROMPT, build_user_message
from src.llm.schemas import ALLOWED_IDS_CONTEXT_KEY, ContextChunk, GeneratedAnswer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-5-5"


class AnthropicAnswerGenerator:
    def __init__(
        self,
        api_key: str | None,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1000,
        max_retries: int = 2,
        timeout_seconds: float = 30.0,
        on_attempt: Callable[[], None] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.model_name = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        raw_client = anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout_seconds, max_retries=0
        )
        self._client = instructor.from_anthropic(raw_client, mode=instructor.Mode.ANTHROPIC_TOOLS)
        if on_attempt is not None:
            # Fires once per API attempt (incl. validation retries). Used by the live-test budget.
            self._client.on("completion:kwargs", lambda *_args, **_kwargs: on_attempt())

    async def generate(self, question: str, chunks: list[ContextChunk]) -> GeneratedAnswer:
        allowed_ids = {c.chunk_id for c in chunks}
        try:
            result: Any = await self._client.messages.create(
                model=self.model_name,
                max_tokens=self._max_tokens,
                system=ANSWER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_message(question, chunks)}],
                response_model=GeneratedAnswer,
                max_retries=self._max_retries,
                context={ALLOWED_IDS_CONTEXT_KEY: allowed_ids},
            )
        except anthropic.APITimeoutError as exc:
            logger.warning("Answer generation timed out (model=%s)", self.model_name)
            raise LLMTimeoutError("Answer generation timed out") from exc
        except InstructorRetryException as exc:
            if isinstance(exc.__cause__, anthropic.APITimeoutError):
                raise LLMTimeoutError("Answer generation timed out") from exc
            logger.warning(
                "Answer failed validation after %s attempts (model=%s)",
                getattr(exc, "n_attempts", "?"),
                self.model_name,
            )
            raise AnswerGenerationError(
                "Answer generation failed: the model output did not pass validation"
            ) from exc
        except anthropic.APIStatusError as exc:
            logger.warning("Anthropic API error %s (model=%s)", exc.status_code, self.model_name)
            raise AnswerGenerationError(
                f"Answer generation failed: upstream API error ({exc.status_code})"
            ) from exc
        except anthropic.APIError as exc:
            logger.warning("Anthropic API error %s (model=%s)", type(exc).__name__, self.model_name)
            raise AnswerGenerationError("Answer generation failed: upstream API error") from exc

        if not isinstance(result, GeneratedAnswer):
            raise AnswerGenerationError("Answer generation failed: unexpected response type")
        return result
