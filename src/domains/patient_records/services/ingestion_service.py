"""Ingest orchestration: load → PII guard → skip/replace → parse → chunk → embed → upsert."""

import logging
import threading
import time
from pathlib import Path

from src.core.exceptions import IngestionInProgressError, RecordsDirectoryError
from src.domains.patient_records.chunking.record_parser import parse_record
from src.domains.patient_records.chunking.section_chunker import chunk_sections
from src.domains.patient_records.schemas import IngestionResult, RejectedFile
from src.embeddings.base import Embedder
from src.utils.file_loader import list_record_files, load_file
from src.utils.pii_guard import detect_pii
from src.vectorstore.base import Metadata, VectorStore

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        records_dir: Path,
        lock: threading.Lock,
        *,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        section_prefix: bool = True,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._records_dir = records_dir
        self._lock = lock
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._section_prefix = section_prefix

    def ingest(self, force_reingest: bool = False) -> IngestionResult:
        """Run one ingest. Raises IngestionInProgressError if another run holds the lock."""
        if not self._lock.acquire(blocking=False):
            raise IngestionInProgressError("An ingestion is already running. Try again later.")
        try:
            return self._run(force_reingest)
        finally:
            self._lock.release()

    def _run(self, force_reingest: bool) -> IngestionResult:
        started = time.perf_counter()
        try:
            files = list_record_files(self._records_dir)
        except OSError as exc:
            raise RecordsDirectoryError(
                f"RECORDS_DIR is missing or unreadable: {self._records_dir}"
            ) from exc

        ingested = skipped = chunks_upserted = 0
        rejected: list[RejectedFile] = []

        for path in files:
            try:
                loaded = load_file(path)
            except OSError:
                logger.warning("Unreadable record file skipped: %s", path.name)
                rejected.append(RejectedFile(file_name=path.name, reason="Unreadable file"))
                continue

            pii_types = detect_pii(loaded.text)
            if pii_types:
                logger.warning("Rejected %s: PII detected (%s)", path.name, ", ".join(pii_types))
                # A previously clean version may be indexed; make sure no version stays searchable.
                self._store.delete_by_file(loaded.file_name)
                rejected.append(
                    RejectedFile(
                        file_name=loaded.file_name,
                        reason=f"PII detected: {', '.join(pii_types)}",
                    )
                )
                continue

            existing_hash = self._store.get_file_hash(loaded.file_name)
            if existing_hash == loaded.content_hash and not force_reingest:
                skipped += 1
                continue

            record = parse_record(loaded.text, fallback_patient_id=path.stem)
            chunks = chunk_sections(
                record.sections,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                section_prefix=self._section_prefix,
            )
            if not chunks:
                rejected.append(
                    RejectedFile(file_name=loaded.file_name, reason="No content sections found")
                )
                continue

            texts = [c.text for c in chunks]
            embeddings = self._embedder.embed_documents(texts)
            metadatas: list[Metadata] = [
                {
                    "patient_id": record.patient_id,
                    "file_name": loaded.file_name,
                    "section": c.section,
                    "age_band": record.age_band,
                    "sex": record.sex,
                    "record_date": record.record_date,
                    "chunk_index": c.chunk_index,
                    "content_hash": loaded.content_hash,
                }
                for c in chunks
            ]
            ids = [f"{loaded.file_name}:{c.chunk_index}" for c in chunks]

            # Replace: remove every old chunk of this file (it may have had more chunks before).
            self._store.delete_by_file(loaded.file_name)
            self._store.upsert(ids, embeddings, texts, metadatas)
            ingested += 1
            chunks_upserted += len(chunks)

        result = IngestionResult(
            files_scanned=len(files),
            files_ingested=ingested,
            files_skipped_unchanged=skipped,
            files_rejected=rejected,
            chunks_upserted=chunks_upserted,
        )
        logger.info(
            "Ingest done in %.1fs: scanned=%d ingested=%d skipped=%d rejected=%d chunks=%d",
            time.perf_counter() - started,
            result.files_scanned,
            result.files_ingested,
            result.files_skipped_unchanged,
            len(result.files_rejected),
            result.chunks_upserted,
        )
        return result
