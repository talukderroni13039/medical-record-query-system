"""Test doubles shared by unit and integration tests (no model, no network)."""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from src.llm.schemas import ContextChunk, GeneratedAnswer

_WORD_RE = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Deterministic 8-dim normalized bag-of-words hash embedding.

    Texts that share words get similar vectors, so retrieval ordering in tests is meaningful
    without loading a real model.
    """

    model_name = "fake-embedder"
    dimensions = 8

    def __init__(self) -> None:
        self.document_calls = 0
        self.embedded_texts: list[str] = []
        self.queries: list[str] = []

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for word in _WORD_RE.findall(text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vec[digest[0] % self.dimensions] += 1.0 + digest[1] / 255.0
        if not any(vec):
            vec[0] = 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        self.embedded_texts.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)


# ---------------------------------------------------------------- answer generation

LLM_FIXTURE = Path(__file__).parent / "fixtures" / "llm_answer.json"


def load_llm_fixture() -> dict[str, Any]:
    """The real Claude response saved by the first live test (reused by mocked tests)."""
    return dict(json.loads(LLM_FIXTURE.read_text(encoding="utf-8")))


class FakeAnswerGenerator:
    """AnswerGenerator double: returns a fixed answer, raises, or cites the first chunk."""

    model_name = "fake-llm"

    def __init__(
        self, answer: GeneratedAnswer | None = None, error: Exception | None = None
    ) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[tuple[str, list[ContextChunk]]] = []

    async def generate(self, question: str, chunks: list[ContextChunk]) -> GeneratedAnswer:
        self.calls.append((question, chunks))
        if self.error is not None:
            raise self.error
        if self.answer is not None:
            return self.answer
        first = chunks[0].chunk_id
        return GeneratedAnswer(
            answer=f"According to the records: {chunks[0].text} [{first}]",
            cited_chunk_ids=[first],
            insufficient_context=False,
        )
