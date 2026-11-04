"""Query orchestration: validate filters → embed question → search → map to QueryResult."""

import asyncio
import logging
import re
import time

from src.core.exceptions import (
    AnswerGenerationDisabledError,
    AnswerGenerationError,
    CollectionEmptyError,
    InvalidInputError,
)
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
from src.llm.base import AnswerGenerator
from src.llm.schemas import ContextChunk, GeneratedAnswer
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


NO_RESULTS_ANSWER = GeneratedAnswer(
    answer="No matching record chunks were found, so this cannot be answered from the records.",
    cited_chunk_ids=[],
    insufficient_context=True,
)


class QueryService:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        answer_generator: AnswerGenerator | None = None,
        *,
        max_answer_chunks: int = 8,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._answer_generator = answer_generator
        self._max_answer_chunks = max_answer_chunks

    def _retrieve(self, request: QueryRequest, where: WhereFilter | None) -> list[SearchHit]:
        if self._store.count() == 0:
            raise CollectionEmptyError("No records are indexed yet. Run ingest first.")
        vector = self._embedder.embed_query(request.question)
        return self._store.query(vector, request.top_k, where)

    async def query(self, request: QueryRequest) -> QueryResult:
        started = time.perf_counter()
        if request.generate_answer and self._answer_generator is None:
            raise AnswerGenerationDisabledError(
                "Answer generation is disabled (ANSWER_PROVIDER=none). "
                "Retry with generate_answer=false for retrieval-only results."
            )
        where = build_where(request.filters)
        # Embedding + search are CPU-bound / blocking: keep them off the event loop.
        hits = await asyncio.to_thread(self._retrieve, request, where)
        results = [to_retrieved_chunk(h) for h in hits]
        answer = await self._answer(request, results) if request.generate_answer else None
        logger.info(
            "Query done in %.0fms: top_k=%d filters=%s hits=%d answer=%s",
            (time.perf_counter() - started) * 1000,
            request.top_k,
            where or {},
            len(results),
            answer is not None,
        )
        return QueryResult(question=request.question, results=results, answer=answer)

    async def _answer(
        self, request: QueryRequest, results: list[RetrievedChunk]
    ) -> GeneratedAnswer:
        if self._answer_generator is None:  # pragma: no cover - guarded at the top of query()
            raise AnswerGenerationDisabledError("Answer generation is disabled")
        if not results:
            return NO_RESULTS_ANSWER  # nothing to ground on: no LLM call
        context = [
            ContextChunk(
                chunk_id=r.chunk_id,
                patient_id=r.source.patient_id,
                section=r.source.section,
                text=r.text,
            )
            for r in results[: self._max_answer_chunks]
        ]
        answer = await self._answer_generator.generate(request.question, context)
        # Defence in depth: the generator validates too, but never trust a provider blindly.
        allowed = {c.chunk_id for c in context}
        unknown = [cid for cid in answer.cited_chunk_ids if cid not in allowed]
        if unknown:
            logger.warning("Answer cited %d chunk ID(s) outside the retrieved set", len(unknown))
            raise AnswerGenerationError(
                "Answer generation failed: the answer cited chunks that were not retrieved"
            )
        return answer

    def stats(self) -> StatsResult:
        store_stats = self._store.stats()
        return StatsResult(
            collection=store_stats.collection,
            chunk_count=store_stats.chunk_count,
            file_count=store_stats.file_count,
            embedding_model=self._embedder.model_name,
            dimensions=self._embedder.dimensions,
        )
