"""API tests: every endpoint returns the envelope; every error code maps correctly.

No model, no network: `get_embedder` / `get_vector_store` / `get_records_dir` are overridden with
fakes and an in-memory Chroma collection. The lifespan does not run (TestClient without `with`).
"""

import shutil
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core import dependencies
from src.main import create_app
from src.vectorstore.chroma_store import ChromaVectorStore
from tests.fakes import FakeEmbedder

FIXTURES = Path(__file__).parent.parent / "fixtures"
BASE = "/api/v1/patient-records"


@pytest.fixture
def records_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "records"
    directory.mkdir()
    for name in ("P-0101.md", "P-0102.txt"):
        shutil.copy(FIXTURES / "records" / name, directory)
    shutil.copy(FIXTURES / "pii" / "P-9001.md", directory)
    return directory


@pytest.fixture
def app(
    fake_embedder: FakeEmbedder, chroma_store: ChromaVectorStore, records_dir: Path
) -> Iterator[FastAPI]:
    app = create_app()
    app.dependency_overrides[dependencies.get_embedder] = lambda: fake_embedder
    app.dependency_overrides[dependencies.get_vector_store] = lambda: chroma_store
    app.dependency_overrides[dependencies.get_records_dir] = lambda: records_dir
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def ingested(client: TestClient) -> TestClient:
    assert client.post(f"{BASE}/ingest", json={}).status_code == 200
    return client


def assert_success(response: Any) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    assert set(body) == {"success", "data", "meta"}
    assert body["meta"]["request_id"] == response.headers["X-Request-ID"]
    assert isinstance(body["meta"]["processing_time_ms"], int)
    return dict(body["data"])


def assert_error(response: Any, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["success"] is False
    assert set(body) == {"success", "error", "meta"}
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    assert body["meta"]["request_id"] == response.headers["X-Request-ID"]
    return dict(body["error"])


# ------------------------------------------------------------------ health


def test_health_ok(client: TestClient) -> None:
    data = assert_success(client.get("/health"))
    assert data == {
        "status": "ok",
        "environment": "test",
        "model_loaded": True,
        "vector_store": "ok",
        "answer_provider": "none",
    }


def test_health_503_when_model_not_loaded(chroma_store: ChromaVectorStore) -> None:
    app = create_app()  # no overrides, no lifespan -> app.state has no embedder
    app.dependency_overrides[dependencies.get_vector_store] = lambda: chroma_store
    client = TestClient(app, raise_server_exceptions=False)
    assert_error(client.get("/health"), 503, "MODEL_NOT_READY")


def test_health_503_when_store_missing(fake_embedder: FakeEmbedder) -> None:
    app = create_app()
    app.dependency_overrides[dependencies.get_embedder] = lambda: fake_embedder
    client = TestClient(app, raise_server_exceptions=False)
    assert_error(client.get("/health"), 503, "VECTOR_STORE_ERROR")


# ------------------------------------------------------------------ ingest


def test_ingest_returns_envelope_and_reports_pii(client: TestClient) -> None:
    data = assert_success(client.post(f"{BASE}/ingest", json={"force_reingest": False}))
    assert data["files_scanned"] == 3
    assert data["files_ingested"] == 2
    assert data["files_skipped_unchanged"] == 0
    assert data["files_rejected"] == [{"file_name": "P-9001.md", "reason": "PII detected: email"}]
    assert data["chunks_upserted"] == 11


def test_ingest_without_body_and_idempotent(client: TestClient) -> None:
    assert_success(client.post(f"{BASE}/ingest"))
    data = assert_success(client.post(f"{BASE}/ingest"))
    assert data["files_skipped_unchanged"] == 2
    forced = assert_success(client.post(f"{BASE}/ingest", json={"force_reingest": True}))
    assert forced["files_ingested"] == 2


def test_ingest_in_progress_409(client: TestClient) -> None:
    lock = dependencies.get_ingest_lock()
    lock.acquire()
    try:
        assert_error(client.post(f"{BASE}/ingest", json={}), 409, "INGESTION_IN_PROGRESS")
    finally:
        lock.release()


def test_ingest_concurrent_requests_one_wins(app: FastAPI, client: TestClient) -> None:
    release = threading.Event()
    lock = threading.Lock()

    class SlowService:
        def ingest(self, force_reingest: bool = False) -> Any:
            if not lock.acquire(blocking=False):
                from src.core.exceptions import IngestionInProgressError

                raise IngestionInProgressError("busy")
            try:
                release.wait(timeout=5)
                from src.domains.patient_records.schemas import IngestionResult

                return IngestionResult(
                    files_scanned=0,
                    files_ingested=0,
                    files_skipped_unchanged=0,
                    files_rejected=[],
                    chunks_upserted=0,
                )
            finally:
                lock.release()

    service = SlowService()
    app.dependency_overrides[dependencies.get_ingestion_service] = lambda: service
    results: list[int] = []
    first = threading.Thread(
        target=lambda: results.append(client.post(f"{BASE}/ingest").status_code)
    )
    first.start()
    while not lock.locked():
        pass
    second = client.post(f"{BASE}/ingest")
    release.set()
    first.join()
    assert_error(second, 409, "INGESTION_IN_PROGRESS")
    assert results == [200]


def test_ingest_missing_records_dir_500(app: FastAPI, client: TestClient, tmp_path: Path) -> None:
    app.dependency_overrides[dependencies.get_records_dir] = lambda: tmp_path / "missing"
    assert_error(client.post(f"{BASE}/ingest", json={}), 500, "RECORDS_DIR_ERROR")


def test_ingest_validation_error(client: TestClient) -> None:
    assert_error(
        client.post(f"{BASE}/ingest", json={"force_reingest": "maybe"}), 422, "VALIDATION_ERROR"
    )


# ------------------------------------------------------------------ query


def test_query_returns_envelope(ingested: TestClient) -> None:
    payload = {"question": "penicillin allergy rash", "top_k": 3}
    data = assert_success(ingested.post(f"{BASE}/query", json=payload))
    assert data["question"] == "penicillin allergy rash"
    assert len(data["results"]) == 3
    first = data["results"][0]
    assert set(first) == {"chunk_id", "text", "score", "source"}
    assert set(first["source"]) == {
        "patient_id",
        "file_name",
        "section",
        "record_date",
        "chunk_index",
    }


def test_query_with_both_filters(ingested: TestClient) -> None:
    payload = {
        "question": "what allergies",
        "filters": {"patient_id": "P-0102", "section": "Allergies"},
    }
    data = assert_success(ingested.post(f"{BASE}/query", json=payload))
    assert [r["text"] for r in data["results"]] == ["Allergies: Latex (contact dermatitis)."]


def test_query_default_top_k(ingested: TestClient) -> None:
    data = assert_success(ingested.post(f"{BASE}/query", json={"question": "allergy"}))
    assert len(data["results"]) == 5


def test_query_unknown_section_400(ingested: TestClient) -> None:
    payload = {"question": "anything", "filters": {"section": "Diet"}}
    error = assert_error(ingested.post(f"{BASE}/query", json=payload), 400, "INVALID_INPUT")
    assert "Allergies" in error["message"]


def test_query_empty_collection_409(client: TestClient) -> None:
    assert_error(client.post(f"{BASE}/query", json={"question": "abc"}), 409, "COLLECTION_EMPTY")


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "hi"},
        {"question": "   x  "},
        {"question": "x" * 501},
        {"question": "valid question", "top_k": 0},
        {"question": "valid question", "top_k": 21},
        {"question": "valid question", "unexpected": 1},
        {"question": "valid question", "filters": {"ward": "A"}},
        {},
    ],
)
def test_query_validation_errors_use_error_envelope(client: TestClient, payload: Any) -> None:
    assert_error(client.post(f"{BASE}/query", json=payload), 422, "VALIDATION_ERROR")


def test_query_malformed_json_422(client: TestClient) -> None:
    response = client.post(
        f"{BASE}/query", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert_error(response, 422, "VALIDATION_ERROR")


def test_query_model_not_ready_503(app: FastAPI, client: TestClient) -> None:
    del app.dependency_overrides[dependencies.get_embedder]
    assert_error(client.post(f"{BASE}/query", json={"question": "abc"}), 503, "MODEL_NOT_READY")


def test_query_vector_store_failure_503(
    ingested: TestClient, chroma_store: ChromaVectorStore
) -> None:
    class Broken:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("chroma exploded")

    chroma_store._collection = Broken()
    error = assert_error(
        ingested.post(f"{BASE}/query", json={"question": "abc"}), 503, "VECTOR_STORE_ERROR"
    )
    assert "exploded" not in error["message"]


# ------------------------------------------------------------------ stats


def test_stats_returns_envelope(ingested: TestClient) -> None:
    data = assert_success(ingested.get(f"{BASE}/stats"))
    assert data == {
        "collection": data["collection"],
        "chunk_count": 11,
        "file_count": 2,
        "embedding_model": "fake-embedder",
        "dimensions": 8,
    }


# ------------------------------------------------------------------ cross-cutting


def test_unhandled_error_500_without_stack_trace(app: FastAPI, client: TestClient) -> None:
    class Exploding:
        def stats(self) -> None:
            raise RuntimeError("very secret stack detail")

    app.dependency_overrides[dependencies.get_query_service] = Exploding
    response = client.get(f"{BASE}/stats")
    error = assert_error(response, 500, "INTERNAL_ERROR")
    assert error["message"] == "Unexpected server error"
    assert "very secret" not in response.text
    assert "Traceback" not in response.text


def test_unknown_route_404_still_has_request_id(client: TestClient) -> None:
    response = client.get("/api/v1/nope")
    assert response.status_code == 404
    assert response.headers["X-Request-ID"]


def test_openapi_lists_all_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", f"{BASE}/ingest", f"{BASE}/query", f"{BASE}/stats"} <= set(paths)
