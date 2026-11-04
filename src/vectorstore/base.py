from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]
# Chroma-style filter: {"field": value} or {"$and": [{"a": x}, {"b": y}]}.
# Other store implementations translate it to their own syntax.
WhereFilter = dict[str, Any]


@dataclass(frozen=True)
class SearchHit:
    id: str
    text: str
    metadata: Metadata
    distance: float  # cosine distance: 0 = identical direction


@dataclass(frozen=True)
class StoreStats:
    collection: str
    chunk_count: int
    file_count: int


@runtime_checkable
class VectorStore(Protocol):
    """Chunk-vector storage + similarity search. Services depend on this, not on a library."""

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[Metadata],
    ) -> None: ...

    def delete_by_file(self, file_name: str) -> None: ...

    def get_file_hash(self, file_name: str) -> str | None: ...

    def query(
        self, embedding: list[float], top_k: int, where: WhereFilter | None = None
    ) -> list[SearchHit]: ...

    def count(self) -> int: ...

    def stats(self) -> StoreStats: ...

    def ping(self) -> bool: ...
