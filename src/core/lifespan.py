"""Create heavy objects ONCE at startup and keep them on `app.state`.

The `build_*` functions are also used by `scripts/` so the CLI and API share one construction path.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.embeddings.base import Embedder
from src.utils.env import env_int, env_str
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.embedder = None
    app.state.vector_store = None

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
