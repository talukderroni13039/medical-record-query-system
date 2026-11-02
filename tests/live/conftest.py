"""Live-test fixtures: real Claude calls with a HARD 50-call budget for the whole build.

Every API attempt (Instructor validation retries included) is recorded in
`tests/.live_call_budget.json` (gitignored). A test that could push the total past the cap fails
with "live API budget exhausted" before any call is made.

Run with:  uv run pytest -m live --no-cov
"""

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
import pytest
from chromadb.config import Settings
from dotenv import load_dotenv

BUDGET_FILE = Path(__file__).parent.parent / ".live_call_budget.json"
BUDGET_CAP = 50
LIVE_MAX_RETRIES = 1  # Instructor: max_retries + 1 attempts per test at most
LIVE_TOP_K = 5  # at most 5 retrieved chunks sent to the API
WORST_CASE_ATTEMPTS = LIVE_MAX_RETRIES + 1
RECORDS_DIR = Path(__file__).parent.parent.parent / "data" / "records"


class LiveCallBudget:
    def __init__(self, path: Path, cap: int) -> None:
        self.path = path
        self.cap = cap
        self._lock = threading.Lock()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            return dict(json.loads(self.path.read_text(encoding="utf-8")))
        return {"cap": self.cap, "calls": 0, "attempts": []}

    @property
    def calls(self) -> int:
        return int(self._load()["calls"])

    def ensure_available(self, needed: int) -> None:
        used = self.calls
        if used >= self.cap or used + needed > self.cap:
            pytest.fail(f"live API budget exhausted ({used}/{self.cap} calls used)")

    def record(self, test_name: str) -> None:
        with self._lock:
            data = self._load()
            if data["calls"] >= self.cap:
                raise RuntimeError("live API budget exhausted")
            data["calls"] += 1
            data["attempts"].append({"test": test_name, "at": datetime.now(UTC).isoformat()})
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@pytest.fixture(scope="session")
def live_budget() -> LiveCallBudget:
    return LiveCallBudget(BUDGET_FILE, BUDGET_CAP)


@pytest.fixture(scope="session")
def live_index() -> Any:
    """Real bge embeddings of data/records in an in-memory Chroma collection (built once)."""
    from src.domains.patient_records.services.ingestion_service import IngestionService
    from src.embeddings.bge_embedder import BgeEmbedder
    from src.vectorstore.chroma_store import ChromaVectorStore

    embedder = BgeEmbedder()
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    store = ChromaVectorStore(client, "live_patient_records")
    IngestionService(embedder, store, RECORDS_DIR, threading.Lock()).ingest()
    return embedder, store


@pytest.fixture
def live_query_service(
    request: pytest.FixtureRequest, live_budget: LiveCallBudget, live_index: Any
) -> Any:
    from src.domains.patient_records.services.query_service import QueryService
    from src.llm.anthropic_answer_generator import AnthropicAnswerGenerator

    load_dotenv(".env.dev")  # only to make ANTHROPIC_API_KEY available; never printed
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    live_budget.ensure_available(WORST_CASE_ATTEMPTS)
    generator = AnthropicAnswerGenerator(
        api_key,
        model=os.getenv("LLM_MODEL", "claude-haiku-5-5"),
        max_tokens=1000,
        max_retries=LIVE_MAX_RETRIES,
        timeout_seconds=60,
        on_attempt=lambda: live_budget.record(request.node.name),
    )
    embedder, store = live_index
    return QueryService(embedder, store, generator, max_answer_chunks=LIVE_TOP_K)
