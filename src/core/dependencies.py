"""ALL dependency providers (FastAPI `Depends` only, no DI container).

Heavy objects are created once in `core/lifespan.py` and read here from `app.state`.
Tests replace `get_embedder` / `get_vector_store` via `app.dependency_overrides`.
"""

import threading
from pathlib import Path
from typing import Annotated

from fastapi import Depends, Request

from src.core.exceptions import ModelNotReadyError, VectorStoreError
from src.domains.patient_records.services.ingestion_service import IngestionService
from src.domains.patient_records.services.query_service import QueryService
from src.embeddings.base import Embedder
from src.llm.base import AnswerGenerator
from src.utils.env import env_bool, env_int, env_str
from src.vectorstore.base import VectorStore

# One process-wide lock: services are built per request, the lock must be shared.
_INGEST_LOCK = threading.Lock()


def get_embedder(request: Request) -> Embedder:
    embedder: Embedder | None = getattr(request.app.state, "embedder", None)
    if embedder is None:
        raise ModelNotReadyError("Embedding model is not loaded")
    return embedder


def get_vector_store(request: Request) -> VectorStore:
    store: VectorStore | None = getattr(request.app.state, "vector_store", None)
    if store is None:
        raise VectorStoreError("Vector store is not available")
    return store


def get_answer_generator(request: Request) -> AnswerGenerator | None:
    """None when ANSWER_PROVIDER=none: the query service then rejects generate_answer=true."""
    generator: AnswerGenerator | None = getattr(request.app.state, "answer_generator", None)
    return generator


def get_ingest_lock() -> threading.Lock:
    return _INGEST_LOCK


def get_records_dir() -> Path:
    return Path(env_str("RECORDS_DIR", "./data/records"))


def get_ingestion_service(
    embedder: Annotated[Embedder, Depends(get_embedder)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
    lock: Annotated[threading.Lock, Depends(get_ingest_lock)],
    records_dir: Annotated[Path, Depends(get_records_dir)],
) -> IngestionService:
    return IngestionService(
        embedder,
        store,
        records_dir,
        lock,
        chunk_size=env_int("CHUNK_SIZE", 256),
        chunk_overlap=env_int("CHUNK_OVERLAP", 32),
        section_prefix=env_bool("SECTION_PREFIX", True),
    )


def get_query_service(
    embedder: Annotated[Embedder, Depends(get_embedder)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
    answer_generator: Annotated[AnswerGenerator | None, Depends(get_answer_generator)],
) -> QueryService:
    return QueryService(
        embedder,
        store,
        answer_generator,
        max_answer_chunks=env_int("LLM_MAX_CONTEXT_CHUNKS", 8),
    )
