import shutil
import threading
from pathlib import Path

import pytest

from src.core.exceptions import IngestionInProgressError, RecordsDirectoryError
from src.domains.patient_records.services.ingestion_service import IngestionService
from src.vectorstore.chroma_store import ChromaVectorStore
from tests.fakes import FakeEmbedder

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def records_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "records"
    directory.mkdir()
    shutil.copy(FIXTURES / "records" / "P-0101.md", directory)
    shutil.copy(FIXTURES / "records" / "P-0102.txt", directory)
    return directory


@pytest.fixture
def service(
    fake_embedder: FakeEmbedder, chroma_store: ChromaVectorStore, records_dir: Path
) -> IngestionService:
    return IngestionService(
        fake_embedder, chroma_store, records_dir, threading.Lock(), chunk_size=64, chunk_overlap=8
    )


def test_new_files_ingested(service: IngestionService, chroma_store: ChromaVectorStore) -> None:
    result = service.ingest()
    assert result.files_scanned == 2
    assert result.files_ingested == 2
    assert result.files_skipped_unchanged == 0
    assert result.files_rejected == []
    # P-0101 has 8 short sections, P-0102 has 3 -> one chunk each.
    assert result.chunks_upserted == 11
    assert chroma_store.count() == 11
    hit = chroma_store.query([1.0] + [0.0] * 7, top_k=20, where={"patient_id": "P-0101"})[0]
    md = hit.metadata
    assert md["file_name"] == "P-0101.md"
    assert (md["age_band"], md["sex"], md["record_date"]) == ("40-49", "F", "2025-03-14")
    assert len(str(md["content_hash"])) == 64
    assert hit.id == f"P-0101.md:{md['chunk_index']}"
    assert hit.text.startswith(f"{md['section']}: ")


def test_unchanged_files_skipped(
    service: IngestionService, fake_embedder: FakeEmbedder, chroma_store: ChromaVectorStore
) -> None:
    service.ingest()
    calls_after_first = fake_embedder.document_calls
    second = service.ingest()
    assert second.files_skipped_unchanged == 2
    assert second.files_ingested == 0
    assert second.chunks_upserted == 0
    assert fake_embedder.document_calls == calls_after_first  # nothing re-embedded
    assert chroma_store.count() == 11


def test_changed_file_replaced_old_chunks_gone(
    service: IngestionService, records_dir: Path, chroma_store: ChromaVectorStore
) -> None:
    service.ingest()
    old_hash = chroma_store.get_file_hash("P-0101.md")
    (records_dir / "P-0101.md").write_text(
        "# Patient Record: P-0101\n- Age band: 40-49\n\n## Allergies\nSulfa drugs (hives).\n",
        encoding="utf-8",
    )
    result = service.ingest()
    assert result.files_ingested == 1
    assert result.files_skipped_unchanged == 1
    assert chroma_store.get_file_hash("P-0101.md") != old_hash
    hits = chroma_store.query([1.0] + [0.0] * 7, top_k=20, where={"file_name": "P-0101.md"})
    assert [h.text for h in hits] == ["Allergies: Sulfa drugs (hives)."]
    assert chroma_store.count() == 1 + 3


def test_force_reingest_re_embeds_everything(
    service: IngestionService, fake_embedder: FakeEmbedder, chroma_store: ChromaVectorStore
) -> None:
    service.ingest()
    before = fake_embedder.document_calls
    result = service.ingest(force_reingest=True)
    assert result.files_ingested == 2
    assert result.files_skipped_unchanged == 0
    assert fake_embedder.document_calls == before + 2
    assert chroma_store.count() == 11  # replaced, not duplicated


def test_pii_file_rejected_and_reported(
    service: IngestionService, records_dir: Path, chroma_store: ChromaVectorStore
) -> None:
    shutil.copy(FIXTURES / "pii" / "P-9001.md", records_dir)
    shutil.copy(FIXTURES / "pii" / "P-9002.md", records_dir)
    result = service.ingest()
    assert result.files_scanned == 4
    assert result.files_ingested == 2
    reasons = {r.file_name: r.reason for r in result.files_rejected}
    assert reasons == {"P-9001.md": "PII detected: email", "P-9002.md": "PII detected: phone"}
    assert chroma_store.get_file_hash("P-9001.md") is None


def test_file_that_gains_pii_is_removed_from_index(
    service: IngestionService, records_dir: Path, chroma_store: ChromaVectorStore
) -> None:
    service.ingest()
    path = records_dir / "P-0102.txt"
    path.write_text(path.read_text(encoding="utf-8") + "\nSSN 123-45-6789\n", encoding="utf-8")
    result = service.ingest()
    assert [r.file_name for r in result.files_rejected] == ["P-0102.txt"]
    assert chroma_store.get_file_hash("P-0102.txt") is None
    assert chroma_store.count() == 8


def test_file_without_sections_rejected(service: IngestionService, records_dir: Path) -> None:
    (records_dir / "P-0200.md").write_text("# Patient Record: P-0200\n- Sex: M\n", encoding="utf-8")
    result = service.ingest()
    assert [(r.file_name, r.reason) for r in result.files_rejected] == [
        ("P-0200.md", "No content sections found")
    ]


def test_missing_records_dir_raises(
    fake_embedder: FakeEmbedder, chroma_store: ChromaVectorStore, tmp_path: Path
) -> None:
    service = IngestionService(fake_embedder, chroma_store, tmp_path / "nope", threading.Lock())
    with pytest.raises(RecordsDirectoryError, match="RECORDS_DIR"):
        service.ingest()


def test_concurrent_ingest_rejected(
    fake_embedder: FakeEmbedder, chroma_store: ChromaVectorStore, records_dir: Path
) -> None:
    lock = threading.Lock()
    service = IngestionService(fake_embedder, chroma_store, records_dir, lock)
    lock.acquire()  # simulate a run in progress (same lock shared across service instances)
    try:
        with pytest.raises(IngestionInProgressError):
            service.ingest()
    finally:
        lock.release()
    assert service.ingest().files_ingested == 2  # lock released -> works again


def test_lock_released_after_failure(
    fake_embedder: FakeEmbedder, chroma_store: ChromaVectorStore, tmp_path: Path
) -> None:
    lock = threading.Lock()
    service = IngestionService(fake_embedder, chroma_store, tmp_path / "nope", lock)
    with pytest.raises(RecordsDirectoryError):
        service.ingest()
    assert not lock.locked()


def test_section_prefix_toggle(
    fake_embedder: FakeEmbedder, chroma_store: ChromaVectorStore, records_dir: Path
) -> None:
    service = IngestionService(
        fake_embedder, chroma_store, records_dir, threading.Lock(), section_prefix=False
    )
    service.ingest()
    assert "Penicillin (rash)." in fake_embedder.embedded_texts
