import threading
from pathlib import Path

import pytest

from src.core.exceptions import (
    AnswerGenerationDisabledError,
    AnswerGenerationError,
    CollectionEmptyError,
    InvalidInputError,
    LLMTimeoutError,
)
from src.domains.patient_records.schemas import QueryFilters, QueryRequest
from src.domains.patient_records.services.ingestion_service import IngestionService
from src.domains.patient_records.services.query_service import QueryService, build_where
from src.llm.schemas import GeneratedAnswer
from src.vectorstore.base import SearchHit
from src.vectorstore.chroma_store import ChromaVectorStore
from tests.fakes import FakeAnswerGenerator, FakeEmbedder, load_llm_fixture

FIXTURES = Path(__file__).parent.parent / "fixtures" / "records"


@pytest.fixture
def indexed_store(
    fake_embedder: FakeEmbedder, chroma_store: ChromaVectorStore
) -> ChromaVectorStore:
    IngestionService(fake_embedder, chroma_store, FIXTURES, threading.Lock()).ingest()
    return chroma_store


@pytest.fixture
def service(fake_embedder: FakeEmbedder, indexed_store: ChromaVectorStore) -> QueryService:
    return QueryService(fake_embedder, indexed_store)


async def test_results_mapped_with_score_and_source(service: QueryService) -> None:
    result = await service.query(QueryRequest(question="penicillin rash allergy", top_k=3))
    assert result.question == "penicillin rash allergy"
    assert len(result.results) == 3
    top = result.results[0]
    assert top.text == "Allergies: Penicillin (rash)."
    assert top.chunk_id == f"P-0101.md:{top.source.chunk_index}"
    assert top.source.patient_id == "P-0101"
    assert top.source.section == "Allergies"
    assert top.source.record_date == "2025-03-14"
    assert -1.0 <= top.score <= 1.0
    scores = [r.score for r in result.results]
    assert scores == sorted(scores, reverse=True)


async def test_query_with_patient_filter(service: QueryService) -> None:
    request = QueryRequest(question="allergy", top_k=10, filters={"patient_id": "P-0102"})
    result = await service.query(request)
    assert result.results
    assert {r.source.patient_id for r in result.results} == {"P-0102"}


async def test_query_with_section_filter_case_insensitive(service: QueryService) -> None:
    request = QueryRequest(question="allergy", top_k=10, filters={"section": "allergies"})
    result = await service.query(request)
    assert {r.source.section for r in result.results} == {"Allergies"}
    assert {r.source.patient_id for r in result.results} == {"P-0101", "P-0102"}


async def test_query_with_both_filters(service: QueryService) -> None:
    filters = {"patient_id": "P-0102", "section": "Allergies"}
    result = await service.query(QueryRequest(question="allergy", top_k=10, filters=filters))
    assert [r.text for r in result.results] == ["Allergies: Latex (contact dermatitis)."]


async def test_unknown_section_raises_invalid_input(service: QueryService) -> None:
    with pytest.raises(InvalidInputError, match="Unknown section"):
        await service.query(QueryRequest(question="x y z", filters={"section": "Diet"}))


async def test_malformed_patient_id_raises_invalid_input(service: QueryService) -> None:
    with pytest.raises(InvalidInputError, match="patient_id"):
        await service.query(QueryRequest(question="x y z", filters={"patient_id": "P 1; drop"}))


async def test_empty_collection_raises(
    fake_embedder: FakeEmbedder, chroma_store: ChromaVectorStore
) -> None:
    with pytest.raises(CollectionEmptyError):
        await QueryService(fake_embedder, chroma_store).query(QueryRequest(question="anything"))


def test_build_where_shapes() -> None:
    assert build_where(QueryFilters()) is None
    assert build_where(QueryFilters(patient_id=" P-1 ")) == {"patient_id": "P-1"}
    assert build_where(QueryFilters(section="LAB RESULTS")) == {"section": "Lab Results"}
    assert build_where(QueryFilters(patient_id="P-1", section="Plan")) == {
        "$and": [{"patient_id": "P-1"}, {"section": "Plan"}]
    }


def test_score_is_one_minus_distance_and_clamped() -> None:
    from src.domains.patient_records.services.query_service import to_retrieved_chunk

    hit = SearchHit(id="x:0", text="t", metadata={}, distance=0.25)
    chunk = to_retrieved_chunk(hit)
    assert chunk.score == 0.75
    assert chunk.source.patient_id == "unknown"
    assert to_retrieved_chunk(SearchHit(id="x", text="t", metadata={}, distance=2.5)).score == -1.0


def test_stats(service: QueryService) -> None:
    stats = service.stats()
    assert stats.chunk_count == 11
    assert stats.file_count == 2
    assert stats.embedding_model == "fake-embedder"
    assert stats.dimensions == 8


# ------------------------------------------------------------------ answer generation


def _answer_service(
    fake_embedder: FakeEmbedder, store: ChromaVectorStore, generator: FakeAnswerGenerator
) -> QueryService:
    return QueryService(fake_embedder, store, generator, max_answer_chunks=2)


async def test_generate_answer_with_provider_none_rejected(service: QueryService) -> None:
    with pytest.raises(AnswerGenerationDisabledError) as info:
        await service.query(QueryRequest(question="penicillin", generate_answer=True))
    assert info.value.code == "ANSWER_GENERATION_DISABLED"


async def test_answer_present_with_fake_generator(
    fake_embedder: FakeEmbedder, indexed_store: ChromaVectorStore
) -> None:
    generator = FakeAnswerGenerator()
    service = _answer_service(fake_embedder, indexed_store, generator)
    result = await service.query(
        QueryRequest(question="penicillin rash allergy", top_k=5, generate_answer=True)
    )
    assert result.answer is not None
    assert result.answer.cited_chunk_ids == [result.results[0].chunk_id]
    # Only the question + top `max_answer_chunks` retrieved chunks are handed to the generator.
    question, chunks = generator.calls[0]
    assert question == "penicillin rash allergy"
    assert [c.chunk_id for c in chunks] == [r.chunk_id for r in result.results[:2]]
    assert chunks[0].patient_id == result.results[0].source.patient_id


async def test_answer_not_requested_means_no_generator_call(
    fake_embedder: FakeEmbedder, indexed_store: ChromaVectorStore
) -> None:
    generator = FakeAnswerGenerator()
    result = await _answer_service(fake_embedder, indexed_store, generator).query(
        QueryRequest(question="penicillin")
    )
    assert result.answer is None
    assert generator.calls == []


async def test_cited_id_not_in_retrieved_set_rejected(
    fake_embedder: FakeEmbedder, indexed_store: ChromaVectorStore
) -> None:
    # The saved real answer cites P-0024.md:4, which is not in this test index.
    rogue = GeneratedAnswer(**load_llm_fixture()["answer"])
    service = _answer_service(fake_embedder, indexed_store, FakeAnswerGenerator(answer=rogue))
    with pytest.raises(AnswerGenerationError, match="not retrieved"):
        await service.query(QueryRequest(question="latex allergy", generate_answer=True))


async def test_no_hits_returns_insufficient_without_llm_call(
    fake_embedder: FakeEmbedder, indexed_store: ChromaVectorStore
) -> None:
    generator = FakeAnswerGenerator()
    request = QueryRequest(
        question="anything", filters={"patient_id": "P-NOPE"}, generate_answer=True
    )
    result = await _answer_service(fake_embedder, indexed_store, generator).query(request)
    assert result.results == []
    assert result.answer is not None
    assert result.answer.insufficient_context is True
    assert generator.calls == []


async def test_generator_errors_propagate(
    fake_embedder: FakeEmbedder, indexed_store: ChromaVectorStore
) -> None:
    generator = FakeAnswerGenerator(error=LLMTimeoutError("slow"))
    with pytest.raises(LLMTimeoutError):
        await _answer_service(fake_embedder, indexed_store, generator).query(
            QueryRequest(question="penicillin", generate_answer=True)
        )
