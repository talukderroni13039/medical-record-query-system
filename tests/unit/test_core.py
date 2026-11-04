"""Envelope, exception handler, and request-ID middleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from src.core.exceptions import (
    AnswerGenerationDisabledError,
    AnswerGenerationError,
    AppError,
    CollectionEmptyError,
    IngestionInProgressError,
    InvalidInputError,
    LLMTimeoutError,
    ModelNotReadyError,
    RecordsDirectoryError,
    VectorStoreError,
)
from src.main import create_app


class _Body(BaseModel):
    name: str = Field(min_length=3)


ERROR_CASES = [
    (InvalidInputError, 400, "INVALID_INPUT"),
    (AnswerGenerationDisabledError, 400, "ANSWER_GENERATION_DISABLED"),
    (IngestionInProgressError, 409, "INGESTION_IN_PROGRESS"),
    (CollectionEmptyError, 409, "COLLECTION_EMPTY"),
    (RecordsDirectoryError, 500, "RECORDS_DIR_ERROR"),
    (AnswerGenerationError, 502, "ANSWER_GENERATION_FAILED"),
    (ModelNotReadyError, 503, "MODEL_NOT_READY"),
    (VectorStoreError, 503, "VECTOR_STORE_ERROR"),
    (LLMTimeoutError, 504, "LLM_TIMEOUT"),
]


@pytest.fixture
def app() -> FastAPI:
    app = create_app()

    @app.get("/_raise/{name}")
    def raise_error(name: str) -> None:
        for cls, _, _ in ERROR_CASES:
            if cls.__name__ == name:
                raise cls("boom")
        raise RuntimeError("secret internal detail")

    @app.post("/_validate")
    def validate(body: _Body) -> dict[str, str]:
        return {"name": body.name}

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(("cls", "status", "code"), ERROR_CASES)
def test_app_errors_render_error_envelope(
    client: TestClient, cls: type[AppError], status: int, code: str
) -> None:
    response = client.get(f"/_raise/{cls.__name__}")
    assert response.status_code == status
    body = response.json()
    assert body["success"] is False
    assert body["error"] == {"code": code, "message": "boom"}
    assert body["meta"]["request_id"] == response.headers["X-Request-ID"]
    assert isinstance(body["meta"]["processing_time_ms"], int)


def test_validation_error_uses_error_envelope(client: TestClient) -> None:
    response = client.post("/_validate", json={"name": "x"})
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert "name" in body["error"]["message"]
    assert "x" not in body["error"]["message"].split(":")[0]  # input value not echoed as field
    assert body["meta"]["request_id"]


def test_unhandled_error_returns_500_without_stack_trace(client: TestClient) -> None:
    response = client.get("/_raise/Unknown")
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == {"code": "INTERNAL_ERROR", "message": "Unexpected server error"}
    assert "Traceback" not in response.text
    assert "secret internal detail" not in response.text
    assert response.headers.get("X-Request-ID")


def test_request_id_header_present_and_propagated(client: TestClient) -> None:
    generated = client.get("/health")
    assert len(generated.headers["X-Request-ID"]) == 32

    echoed = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert echoed.headers["X-Request-ID"] == "abc-123"


def test_base_app_error_defaults() -> None:
    err = AppError("x")
    assert (err.status_code, err.code, err.message) == (500, "APP_ERROR", "x")
