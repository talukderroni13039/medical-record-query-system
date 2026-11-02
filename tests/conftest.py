"""Shared test configuration and fixtures.

`APP_ENV=test` is set before `src.main` is imported, so `.env.test` (which does not exist) is
loaded instead of the developer's `.env.dev`. Default tests never see real secrets, and an autouse
guard blocks every outbound network connection unless the test is marked `live` or `model`.
"""

import os

os.environ["APP_ENV"] = "test"
os.environ["ANSWER_PROVIDER"] = "none"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ.pop("ANTHROPIC_API_KEY", None)  # live tests load it explicitly from .env.dev

import socket
import uuid
from collections.abc import Iterator
from typing import Any

import chromadb
import pytest
from chromadb.config import Settings

from src.vectorstore.chroma_store import ChromaVectorStore
from tests.fakes import FakeEmbedder

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest) -> Iterator[None]:
    """Default tests: zero model downloads and zero API calls, enforced, not just intended."""
    if request.node.get_closest_marker("live") or request.node.get_closest_marker("model"):
        yield
        return

    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOCAL_HOSTS:
            raise RuntimeError(f"Network access blocked in default tests (tried {host!r})")
        return real_connect(self, address)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def chroma_store_name() -> str:
    # EphemeralClient shares in-process state, so every test gets its own collection.
    return f"test_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def chroma_store(chroma_store_name: str) -> ChromaVectorStore:
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    return ChromaVectorStore(client, chroma_store_name)
