import threading
from pathlib import Path

import pytest

from src.core.exceptions import CollectionEmptyError, InvalidInputError
from src.domains.patient_records.schemas import QueryFilters, QueryRequest
from src.domains.patient_records.services.ingestion_service import IngestionService
from src.domains.patient_records.services.query_service import QueryService, build_where
from src.vectorstore.base import SearchHit
from src.vectorstore.chroma_store import ChromaVectorStore
from tests.fakes import FakeEmbedder

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
