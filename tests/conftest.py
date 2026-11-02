"""Shared test configuration and fixtures.

`APP_ENV=test` is set before `src.main` is imported, so `.env.test` (which does not exist) is
loaded instead of the developer's `.env.dev`. Default tests never see real secrets.
"""

import os

os.environ["APP_ENV"] = "test"
os.environ.setdefault("ANSWER_PROVIDER", "none")

import uuid

import chromadb
import pytest
from chromadb.config import Settings

from src.vectorstore.chroma_store import ChromaVectorStore
from tests.fakes import FakeEmbedder


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
