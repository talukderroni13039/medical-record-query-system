"""Lifespan: objects created once at startup; answer provider selection; startup WARNING."""

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.core import lifespan as lifespan_module
from src.llm.anthropic_answer_generator import AnthropicAnswerGenerator
from src.main import create_app
from src.vectorstore.chroma_store import ChromaVectorStore
from tests.fakes import FakeEmbedder


def test_provider_none_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANSWER_PROVIDER", raising=False)
    assert lifespan_module.answer_provider() == "none"
    assert lifespan_module.build_answer_generator() is None


def test_unknown_provider_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANSWER_PROVIDER", "openai")
    with pytest.raises(ValueError, match="ANSWER_PROVIDER"):
        lifespan_module.build_answer_generator()


def test_anthropic_without_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANSWER_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        lifespan_module.build_answer_generator()


def test_anthropic_provider_logs_privacy_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ANSWER_PROVIDER", " Anthropic ")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-5-5")
    with caplog.at_level(logging.WARNING, logger="src.core.lifespan"):
        generator = lifespan_module.build_answer_generator()
    assert isinstance(generator, AnthropicAnswerGenerator)
    assert generator.model_name == "claude-haiku-5-5"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "NOT on-premise" in warnings[0].getMessage()
    assert "synthetic data only" in warnings[0].getMessage()
    assert "test-key-not-real" not in caplog.text


def test_lifespan_creates_objects_once_and_health_ok(
    monkeypatch: pytest.MonkeyPatch, chroma_store: ChromaVectorStore
) -> None:
    calls: dict[str, int] = {"embedder": 0, "store": 0}

    def fake_embedder() -> Any:
        calls["embedder"] += 1
        return FakeEmbedder()

    def fake_store() -> Any:
        calls["store"] += 1
        return chroma_store

    monkeypatch.setattr(lifespan_module, "build_embedder", fake_embedder)
    monkeypatch.setattr(lifespan_module, "build_vector_store", fake_store)
    monkeypatch.setenv("ANSWER_PROVIDER", "none")
    app = create_app()
    with TestClient(app) as client:
        for _ in range(3):
            body = client.get("/health").json()
            assert body["data"]["model_loaded"] is True
        assert app.state.answer_generator is None
    assert calls == {"embedder": 1, "store": 1}
    assert app.state.embedder is None  # released on shutdown


def test_lifespan_model_failure_reports_503(
    monkeypatch: pytest.MonkeyPatch, chroma_store: ChromaVectorStore
) -> None:
    def broken() -> Any:
        raise OSError("model files missing")

    monkeypatch.setattr(lifespan_module, "build_embedder", broken)
    monkeypatch.setattr(lifespan_module, "build_vector_store", broken)
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_READY"


def test_build_vector_store_uses_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma_env"))
    monkeypatch.setenv("CHROMA_COLLECTION", "env_collection")
    store = lifespan_module.build_vector_store()
    assert store.stats().collection == "env_collection"
    assert (tmp_path / "chroma_env").exists()
