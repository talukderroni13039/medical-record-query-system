from typing import Protocol, runtime_checkable

from src.llm.schemas import ContextChunk, GeneratedAnswer


@runtime_checkable
class AnswerGenerator(Protocol):
    """Writes a grounded, cited answer from retrieved chunks. Services depend on this only."""

    model_name: str

    async def generate(self, question: str, chunks: list[ContextChunk]) -> GeneratedAnswer: ...
