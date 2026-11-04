"""Real-model smoke test. Excluded by default; run once with: uv run pytest -m model --no-cov"""

import math
import os

import pytest

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def embedder():  # type: ignore[no-untyped-def]
    from src.embeddings.bge_embedder import DEFAULT_MODEL, BgeEmbedder

    return BgeEmbedder(model_name=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL))


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_vectors_are_1024_dim_and_normalized(embedder) -> None:  # type: ignore[no-untyped-def]
    from src.embeddings.base import Embedder

    assert isinstance(embedder, Embedder)
    assert embedder.dimensions == 1024
    docs = embedder.embed_documents(["Allergies: Penicillin (rash).", "Plan: recheck in 4 weeks."])
    query = embedder.embed_query("penicillin allergy")
    for vec in [*docs, query]:
        assert len(vec) == 1024
        assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, abs_tol=1e-3)
    assert embedder.embed_documents([]) == []


def test_semantic_sanity(embedder) -> None:  # type: ignore[no-untyped-def]
    hypertension, high_bp, broken_arm = embedder.embed_documents(
        ["hypertension", "high blood pressure", "broken arm"]
    )
    assert _cos(hypertension, high_bp) > _cos(hypertension, broken_arm)
