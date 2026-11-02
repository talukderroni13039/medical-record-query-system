"""ChromaVectorStore against an in-memory `chromadb.EphemeralClient()` with tiny fake vectors."""

from pathlib import Path
from typing import Any

import pytest

from src.core.exceptions import VectorStoreError
from src.vectorstore.base import Metadata, VectorStore
from src.vectorstore.chroma_store import ChromaVectorStore


def _md(file_name: str, patient_id: str, section: str, idx: int, h: str = "h1") -> Metadata:
    return {
        "file_name": file_name,
        "patient_id": patient_id,
        "section": section,
        "chunk_index": idx,
        "content_hash": h,
    }


@pytest.fixture
def seeded(chroma_store: ChromaVectorStore) -> ChromaVectorStore:
    chroma_store.upsert(
        ids=["A.md:0", "A.md:1", "B.md:0", "B.md:1"],
        embeddings=[[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0], [0, 0, 1]],
        documents=["Allergies: penicillin", "Plan: recheck", "Allergies: latex", "Plan: rest"],
        metadatas=[
            _md("A.md", "P-A", "Allergies", 0, "hashA"),
            _md("A.md", "P-A", "Plan", 1, "hashA"),
            _md("B.md", "P-B", "Allergies", 0, "hashB"),
            _md("B.md", "P-B", "Plan", 1, "hashB"),
        ],
    )
    return chroma_store


def test_satisfies_protocol(chroma_store: ChromaVectorStore) -> None:
    assert isinstance(chroma_store, VectorStore)
    assert chroma_store.ping() is True


def test_collection_uses_cosine(chroma_store: ChromaVectorStore) -> None:
    assert chroma_store._collection.configuration["hnsw"]["space"] == "cosine"


def test_upsert_and_count_and_idempotent_ids(seeded: ChromaVectorStore) -> None:
    assert seeded.count() == 4
    seeded.upsert(
        ["A.md:0"],
        [[1, 0, 0]],
        ["Allergies: penicillin (rash)"],
        [_md("A.md", "P-A", "Allergies", 0)],
    )
    assert seeded.count() == 4  # same ID overwrites
    seeded.upsert([], [], [], [])
    assert seeded.count() == 4


def test_query_no_filter_ranked_by_distance(seeded: ChromaVectorStore) -> None:
    hits = seeded.query([1, 0, 0.2], top_k=3)
    assert [h.id for h in hits] == ["A.md:0", "B.md:0", "B.md:1"]
    assert 0 <= hits[0].distance < hits[1].distance < hits[2].distance
    assert hits[0].text == "Allergies: penicillin"
    assert hits[0].metadata["patient_id"] == "P-A"
    assert hits[0].metadata["chunk_index"] == 0


def test_query_filter_patient_id(seeded: ChromaVectorStore) -> None:
    hits = seeded.query([1, 0, 0], top_k=5, where={"patient_id": "P-B"})
    assert {h.metadata["patient_id"] for h in hits} == {"P-B"}
    assert len(hits) == 2


def test_query_filter_section(seeded: ChromaVectorStore) -> None:
    hits = seeded.query([0, 0, 1], top_k=5, where={"section": "Plan"})
    assert [h.id for h in hits] == ["B.md:1", "A.md:1"]


def test_query_filter_both(seeded: ChromaVectorStore) -> None:
    where = {"$and": [{"patient_id": "P-A"}, {"section": "Allergies"}]}
    hits = seeded.query([0, 0, 1], top_k=5, where=where)
    assert [h.id for h in hits] == ["A.md:0"]


def test_query_no_match_returns_empty(seeded: ChromaVectorStore) -> None:
    assert seeded.query([1, 0, 0], top_k=5, where={"patient_id": "P-NONE"}) == []


def test_hash_lookup(seeded: ChromaVectorStore) -> None:
    assert seeded.get_file_hash("A.md") == "hashA"
    assert seeded.get_file_hash("missing.md") is None


def test_delete_by_file(seeded: ChromaVectorStore) -> None:
    seeded.delete_by_file("A.md")
    assert seeded.count() == 2
    assert seeded.get_file_hash("A.md") is None
    assert {h.metadata["file_name"] for h in seeded.query([1, 0, 0], top_k=5)} == {"B.md"}


def test_stats(seeded: ChromaVectorStore, chroma_store_name: str) -> None:
    stats = seeded.stats()
    assert (stats.collection, stats.chunk_count, stats.file_count) == (chroma_store_name, 4, 2)


def test_empty_stats(chroma_store: ChromaVectorStore) -> None:
    stats = chroma_store.stats()
    assert (stats.chunk_count, stats.file_count) == (0, 0)


def test_persistent_store_survives_reopen(tmp_path: Path) -> None:
    path = str(tmp_path / "chroma")
    store = ChromaVectorStore.persistent(path, "persist_test")
    store.upsert(["x:0"], [[1, 0]], ["doc"], [_md("x", "P", "Plan", 0)])
    reopened = ChromaVectorStore.persistent(path, "persist_test")
    assert reopened.count() == 1


def test_failures_map_to_vector_store_error(chroma_store: ChromaVectorStore) -> None:
    class Broken:
        def __getattr__(self, name: str) -> Any:
            def fail(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("disk on fire")

            return fail

    chroma_store._collection = Broken()
    with pytest.raises(VectorStoreError, match="query"):
        chroma_store.query([1.0], top_k=1)
    with pytest.raises(VectorStoreError):
        chroma_store.count()
