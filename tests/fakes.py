"""Test doubles shared by unit and integration tests (no model, no network)."""

import hashlib
import math
import re

_WORD_RE = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Deterministic 8-dim normalized bag-of-words hash embedding.

    Texts that share words get similar vectors, so retrieval ordering in tests is meaningful
    without loading a real model.
    """

    model_name = "fake-embedder"
    dimensions = 8

    def __init__(self) -> None:
        self.document_calls = 0
        self.embedded_texts: list[str] = []
        self.queries: list[str] = []

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for word in _WORD_RE.findall(text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vec[digest[0] % self.dimensions] += 1.0 + digest[1] / 255.0
        if not any(vec):
            vec[0] = 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        self.embedded_texts.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)
