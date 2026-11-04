"""Create heavy objects ONCE at startup and keep them on `app.state`.

The `build_*` functions are also used by `scripts/` so the CLI and API share one construction path.
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.embeddings.base import Embedder
from src.llm.base import AnswerGenerator
from src.utils.env import env_float, env_int, env_str
from src.vectorstore.base import VectorStore
from src.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def build_embedder() -> Embedder:
    # Imported lazily: loading torch/sentence-transformers is slow and not needed by tests.
    from src.embeddings.bge_embedder import BgeEmbedder  # noqa: PLC0415

    return BgeEmbedder(
        model_name=env_str("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        device=env_str("EMBEDDING_DEVICE", "cpu"),
        batch_size=env_int("EMBEDDING_BATCH_SIZE", 16),
        query_instruction=env_str("QUERY_INSTRUCTION", DEFAULT_QUERY_INSTRUCTION),
    )


def build_vector_store() -> VectorStore:
    return ChromaVectorStore.persistent(
        path=env_str("CHROMA_PATH", "./data/chroma"),
        collection_name=env_str("CHROMA_COLLECTION", "patient_records"),
    )


ANSWER_PROVIDERS = ("none", "anthropic")
ANTHROPIC_PRIVACY_WARNING = (
    "ANSWER_PROVIDER=anthropic: answer generation sends retrieved record text to the Anthropic "
    "API. This mode is NOT on-premise and must be used with synthetic data only."
)


def answer_provider() -> str:
    return env_str("ANSWER_PROVIDER", "none").strip().lower()


def build_answer_generator() -> AnswerGenerator | None:
    """None when ANSWER_PROVIDER=none (default). Fails fast on unknown values or a missing key."""
    provider = answer_provider()
    if provider not in ANSWER_PROVIDERS:
        raise ValueError(f"ANSWER_PROVIDER must be one of {ANSWER_PROVIDERS}, got {provider!r}")
    if provider == "none":
        return None

    from src.llm.anthropic_answer_generator import (  # noqa: PLC0415
        DEFAULT_MODEL,
        AnthropicAnswerGenerator,
    )

    logger.warning(ANTHROPIC_PRIVACY_WARNING)
    return AnthropicAnswerGenerator(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model=env_str("LLM_MODEL", DEFAULT_MODEL),
        max_tokens=env_int("LLM_MAX_TOKENS", 1000),
        max_retries=env_int("LLM_MAX_RETRIES", 2),
        timeout_seconds=env_float("LLM_TIMEOUT_SECONDS", 30.0),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.embedder = None
    app.state.vector_store = None
    # Misconfiguration (unknown provider / missing key) should stop startup, not degrade silently.
    app.state.answer_generator = build_answer_generator()

    try:
        app.state.vector_store = build_vector_store()
        logger.info("Vector store ready")
    except Exception:
        logger.exception("Vector store failed to open; /health will report 503")

    try:
        app.state.embedder = await asyncio.to_thread(build_embedder)
    except Exception:
        logger.exception("Embedding model failed to load; /health will report 503")

    yield

    app.state.embedder = None
    app.state.vector_store = None
    app.state.answer_generator = None
