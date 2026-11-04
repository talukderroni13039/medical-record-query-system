"""Retrieval evaluation: golden questions → hit@1, hit@5, MRR across chunking/embedding configs.

Uses the REAL embedding model (local, offline after the first download) and an in-memory Chroma
collection per config, so the persistent index in CHROMA_PATH is never touched. No LLM calls.

Usage (PowerShell):
    uv run python scripts/evaluate_retrieval.py                 # all configs
    uv run python scripts/evaluate_retrieval.py --configs default
    uv run python scripts/evaluate_retrieval.py --output data/eval/last_results.json
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class EvalConfig:
    name: str
    chunk_size: int = 256
    chunk_overlap: int = 32
    section_prefix: bool = True
    query_instruction: bool = True


CONFIGS = [
    EvalConfig("default"),
    EvalConfig("chunk_128", chunk_size=128, chunk_overlap=16),
    EvalConfig("chunk_512", chunk_size=512, chunk_overlap=64),
    EvalConfig("no_section_prefix", section_prefix=False),
    EvalConfig("no_query_instruction", query_instruction=False),
]


def is_hit(result: dict[str, Any], item: dict[str, Any]) -> bool:
    if result["patient_id"] not in item["expected_patient_ids"]:
        return False
    expected_section = item.get("expected_section")
    return expected_section is None or result["section"] == expected_section


def score(ranked: list[dict[str, Any]], item: dict[str, Any]) -> tuple[bool, bool, float]:
    """(hit@1, hit@5, reciprocal rank) for one question's ranked results."""
    first_hit = next((i for i, r in enumerate(ranked, start=1) if is_hit(r, item)), None)
    return (
        first_hit == 1,
        first_hit is not None and first_hit <= 5,
        1.0 / first_hit if first_hit else 0.0,
    )


class CachingEmbedder:
    """Wraps the real embedder: identical document texts are embedded once across configs."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.model_name = inner.model_name
        self.dimensions = inner.dimensions
        self.use_instruction = True
        self._cache: dict[str, list[float]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for text, vec in zip(missing, self.inner.embed_documents(missing), strict=True):
                self._cache[text] = vec
        return [self._cache[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.use_instruction:
            return list(self.inner.embed_query(text))
        return list(self.inner.embed_documents([text])[0])


async def run_config(
    config: EvalConfig, embedder: CachingEmbedder, golden: list[dict[str, Any]], k: int
) -> dict[str, Any]:
    import chromadb
    from chromadb.config import Settings

    from src.domains.patient_records.schemas import QueryRequest
    from src.domains.patient_records.services.ingestion_service import IngestionService
    from src.domains.patient_records.services.query_service import QueryService
    from src.vectorstore.chroma_store import ChromaVectorStore

    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    store = ChromaVectorStore(client, f"eval_{config.name}_{uuid.uuid4().hex[:6]}")
    started = time.perf_counter()
    ingest = IngestionService(
        embedder,
        store,
        ROOT / "data" / "records",
        threading.Lock(),
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        section_prefix=config.section_prefix,
    ).ingest()
    ingest_s = time.perf_counter() - started

    embedder.use_instruction = config.query_instruction
    service = QueryService(embedder, store)
    hits1 = hits5 = 0
    rr_total = 0.0
    misses: list[str] = []
    latencies: list[float] = []
    for item in golden:
        t0 = time.perf_counter()
        result = await service.query(QueryRequest(question=item["question"], top_k=k))
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = [
            {"patient_id": r.source.patient_id, "section": r.source.section} for r in result.results
        ]
        h1, h5, rr = score(ranked, item)
        hits1 += h1
        hits5 += h5
        rr_total += rr
        if not h5:
            misses.append(item["id"])
    embedder.use_instruction = True

    n = len(golden)
    latencies.sort()
    return {
        "config": asdict(config),
        "chunks": ingest.chunks_upserted,
        "hit@1": round(hits1 / n, 3),
        "hit@5": round(hits5 / n, 3),
        "mrr": round(rr_total / n, 3),
        "misses@5": misses,
        "query_p50_ms": round(latencies[n // 2], 1),
        "query_p95_ms": round(latencies[min(n - 1, int(n * 0.95))], 1),
        "ingest_seconds": round(ingest_s, 1),
    }


def to_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Config | Chunk size / overlap | Section prefix | Query instruction | Chunks "
        "| hit@1 | hit@5 | MRR | Misses@5 | Query p50 / p95 (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        c = r["config"]
        lines.append(
            f"| {c['name']} | {c['chunk_size']} / {c['chunk_overlap']} "
            f"| {'yes' if c['section_prefix'] else 'no'} "
            f"| {'yes' if c['query_instruction'] else 'no'} | {r['chunks']} "
            f"| {r['hit@1']:.2f} | {r['hit@5']:.2f} | {r['mrr']:.2f} "
            f"| {', '.join(r['misses@5']) or '-'} | {r['query_p50_ms']} / {r['query_p95_ms']} |"
        )
    return "\n".join(lines)


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / f".env.{os.getenv('APP_ENV', 'dev')}")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

    parser = argparse.ArgumentParser(description="Evaluate retrieval on the golden set.")
    parser.add_argument("--golden", type=Path, default=ROOT / "data" / "eval" / "golden_set.json")
    parser.add_argument("--configs", nargs="*", default=[c.name for c in CONFIGS])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None, help="write JSON results here")
    args = parser.parse_args()

    from src.core.lifespan import build_embedder

    golden = json.loads(args.golden.read_text(encoding="utf-8"))["questions"]
    embedder = CachingEmbedder(build_embedder())
    selected = [c for c in CONFIGS if c.name in set(args.configs)]
    results = []
    for config in selected:
        print(f"Running config {config.name} ...", file=sys.stderr)
        results.append(asyncio.run(run_config(config, embedder, golden, args.k)))

    print(f"\nGolden questions: {len(golden)}, k={args.k}, model={embedder.model_name}\n")
    print(to_markdown(results))
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
