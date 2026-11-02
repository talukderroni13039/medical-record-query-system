"""Embedder Protocol conformance + BgeEmbedder logic with a stubbed SentenceTransformer."""

import math
from typing import Any

import numpy as np
import pytest

from src.embeddings import bge_embedder
from src.embeddings.base import Embedder
from tests.fakes import FakeEmbedder


class _StubSentenceTransformer:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def get_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.calls.append((texts, kwargs))
        return np.array([[0.5, 0.5, 0.5, 0.5] for _ in texts], dtype=np.float32)


@pytest.fixture
def stub_bge(monkeypatch: pytest.MonkeyPatch) -> bge_embedder.BgeEmbedder:
    monkeypatch.setattr(bge_embedder, "SentenceTransformer", _StubSentenceTransformer)
    return bge_embedder.BgeEmbedder(model_name="stub/model", batch_size=7, query_instruction="Q: ")


def test_fake_embedder_satisfies_protocol_and_is_normalized() -> None:
    fake = FakeEmbedder()
    assert isinstance(fake, Embedder)
    for vec in [*fake.embed_documents(["Allergies: penicillin", ""]), fake.embed_query("x")]:
        assert len(vec) == fake.dimensions
        assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0)


def test_fake_embedder_similar_texts_are_closer() -> None:
    fake = FakeEmbedder()
    a, b, c = fake.embed_documents(["penicillin allergy rash", "penicillin rash", "knee x-ray"])
    q = fake.embed_query("penicillin")
    assert sum(x * y for x, y in zip(q, a, strict=True)) > sum(
        x * y for x, y in zip(q, c, strict=True)
    )
    assert a != b


def test_bge_embedder_metadata(stub_bge: bge_embedder.BgeEmbedder) -> None:
    assert isinstance(stub_bge, Embedder)
    assert stub_bge.model_name == "stub/model"
    assert stub_bge.dimensions == 4


def test_bge_documents_are_normalized_batched_and_unprefixed(
    stub_bge: bge_embedder.BgeEmbedder,
) -> None:
    vectors = stub_bge.embed_documents(["a", "b"])
    assert vectors == [[0.5] * 4, [0.5] * 4]
    texts, kwargs = stub_bge._model.calls[-1]
    assert texts == ["a", "b"]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == 7
    assert stub_bge.embed_documents([]) == []


def test_bge_query_gets_instruction(stub_bge: bge_embedder.BgeEmbedder) -> None:
    assert stub_bge.embed_query("penicillin") == [0.5] * 4
    texts, _ = stub_bge._model.calls[-1]
    assert texts == ["Q: penicillin"]
