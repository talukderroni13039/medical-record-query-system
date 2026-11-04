"""ChromaDB implementation of `VectorStore` (embedded, no server; cosine HNSW; telemetry off)."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings

from src.core.exceptions import AppError, VectorStoreError
from src.vectorstore.base import Metadata, SearchHit, StoreStats, WhereFilter

logger = logging.getLogger(__name__)

_PAGE_SIZE = 1000


@contextmanager
def _guard(operation: str) -> Iterator[None]:
    """Map any Chroma failure to VectorStoreError (503). Details go to the log, not the client."""
    try:
        yield
    except AppError:
        raise
    except Exception as exc:
        logger.exception("Vector store operation failed: %s", operation)
        raise VectorStoreError(f"Vector store operation failed: {operation}") from exc


class ChromaVectorStore:
    def __init__(self, client: ClientAPI, collection_name: str) -> None:
        self._client = client
        self.collection_name = collection_name
        with _guard("open collection"):
            # Chroma >= 1.0: HNSW space is set via `configuration` at creation (immutable after).
            # embedding_function=None: vectors always come from our Embedder.
            self._collection = client.get_or_create_collection(
                name=collection_name,
                configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None,
            )

    @classmethod
    def persistent(cls, path: str, collection_name: str) -> "ChromaVectorStore":
        with _guard("open persistent client"):
            client = chromadb.PersistentClient(
                path=path, settings=Settings(anonymized_telemetry=False)
            )
        return cls(client, collection_name)

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[Metadata],
    ) -> None:
        if not ids:
            return
        with _guard("upsert"):
            batch = max(1, min(self._client.get_max_batch_size(), _PAGE_SIZE))
            for start in range(0, len(ids), batch):
                end = start + batch
                self._collection.upsert(
                    ids=ids[start:end],
                    embeddings=embeddings[start:end],  # type: ignore[arg-type]
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],  # type: ignore[arg-type]
                )

    def delete_by_file(self, file_name: str) -> None:
        with _guard("delete"):
            self._collection.delete(where={"file_name": file_name})

    def get_file_hash(self, file_name: str) -> str | None:
        with _guard("hash lookup"):
            result = self._collection.get(
                where={"file_name": file_name}, limit=1, include=["metadatas"]
            )
        metadatas = result.get("metadatas") or []
        if not metadatas or metadatas[0] is None:
            return None
        value = metadatas[0].get("content_hash")
        return str(value) if value is not None else None

    def query(
        self, embedding: list[float], top_k: int, where: WhereFilter | None = None
    ) -> list[SearchHit]:
        with _guard("query"):
            result: Any = self._collection.query(
                query_embeddings=[embedding],  # type: ignore[arg-type]
                n_results=top_k,
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        ids = result["ids"][0] if result.get("ids") else []
        documents = result["documents"][0] if result.get("documents") else []
        metadatas = result["metadatas"][0] if result.get("metadatas") else []
        distances = result["distances"][0] if result.get("distances") else []
        return [
            SearchHit(
                id=hit_id,
                text=documents[i] or "",
                metadata=dict(metadatas[i] or {}),
                distance=float(distances[i]),
            )
            for i, hit_id in enumerate(ids)
        ]

    def count(self) -> int:
        with _guard("count"):
            return int(self._collection.count())

    def stats(self) -> StoreStats:
        files: set[str] = set()
        with _guard("stats"):
            total = int(self._collection.count())
            for offset in range(0, total, _PAGE_SIZE):
                page = self._collection.get(include=["metadatas"], limit=_PAGE_SIZE, offset=offset)
                for md in page.get("metadatas") or []:
                    if md and "file_name" in md:
                        files.add(str(md["file_name"]))
        return StoreStats(collection=self.collection_name, chunk_count=total, file_count=len(files))

    def ping(self) -> bool:
        with _guard("heartbeat"):
            self._client.heartbeat()
            self._collection.count()
        return True
