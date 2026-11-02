"""Query orchestration: validate filters → embed question → search → map to QueryResult."""

import asyncio
import logging
import re
import time

from src.core.exceptions import CollectionEmptyError, InvalidInputError
from src.domains.patient_records.chunking.record_parser import KNOWN_SECTIONS, canonical_section
from src.domains.patient_records.schemas import (
    ChunkSource,
    QueryFilters,
    QueryRequest,
    QueryResult,
    RetrievedChunk,
    StatsResult,
)
from src.embeddings.base import Embedder
from src.vectorstore.base import SearchHit, VectorStore, WhereFilter

logger = logging.getLogger(__name__)

_PATIENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def build_where(filters: QueryFilters) -> WhereFilter | None:
    """Validate filters and translate them to a store filter (Chroma-style)."""
    conditions: list[WhereFilter] = []
    if filters.patient_id is not None:
        patient_id = filters.patient_id.strip()
        if not _PATIENT_ID_RE.fullmatch(patient_id):
            raise InvalidInputError("patient_id filter must look like an ID, e.g. 'P-0007'.")
        conditions.append({"patient_id": patient_id})
    if filters.section is not None:
        section = canonical_section(filters.section)
        if section is None:
            raise InvalidInputError(
                f"Unknown section '{filters.section}'. Known sections: {', '.join(KNOWN_SECTIONS)}."
            )
        conditions.append({"section": section})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def to_retrieved_chunk(hit: SearchHit) -> RetrievedChunk:
    md = hit.metadata
    return RetrievedChunk(
        chunk_id=hit.id,
        text=hit.text,
        score=round(max(-1.0, min(1.0, 1.0 - hit.distance)), 4),
        source=ChunkSource(
            patient_id=str(md.get("patient_id", "unknown")),
            file_name=str(md.get("file_name", "unknown")),
            section=str(md.get("section", "unknown")),
            record_date=str(md.get("record_date", "unknown")),
            chunk_index=int(md.get("chunk_index", -1)),
        ),
    )


class QueryService:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def _retrieve(self, request: QueryRequest, where: WhereFilter | None) -> list[SearchHit]:
        if self._store.count() == 0:
            raise CollectionEmptyError("No records are indexed yet. Run ingest first.")
        vector = self._embedder.embed_query(request.question)
        return self._store.query(vector, request.top_k, where)

    async def query(self, request: QueryRequest) -> QueryResult:
        started = time.perf_counter()
        where = build_where(request.filters)
        # Embedding + search are CPU-bound / blocking: keep them off the event loop.
        hits = await asyncio.to_thread(self._retrieve, request, where)
        results = [to_retrieved_chunk(h) for h in hits]
        logger.info(
            "Query done in %.0fms: top_k=%d filters=%s hits=%d",
            (time.perf_counter() - started) * 1000,
            request.top_k,
            where or {},
            len(results),
        )
        return QueryResult(question=request.question, results=results)

    def stats(self) -> StatsResult:
        store_stats = self._store.stats()
        return StatsResult(
            collection=store_stats.collection,
            chunk_count=store_stats.chunk_count,
            file_count=store_stats.file_count,
            embedding_model=self._embedder.model_name,
            dimensions=self._embedder.dimensions,
        )
