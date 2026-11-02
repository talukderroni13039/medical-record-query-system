# Local Medical Record Query System — Architecture & Build Workflow

An on-premise semantic search API over anonymized patient history files. Records are chunked, embedded locally with `bge-large-en-v1.5`, stored in a persistent ChromaDB index, and queried through FastAPI. No data and no API calls leave the machine.

> Scope: **retrieval only** (question → most relevant record chunks with sources). LLM answer generation is a stretch goal (section 11) and, if added, must also run locally to keep the system on-premise.

---

## 1. How It Works (plain language)

| Concept | Analogy |
|---|---|
| **Embedding** | GPS coordinates for meaning. Similar sentences get nearby coordinates, even with different words ("high blood pressure" ≈ "hypertension"). |
| **Chunking** | Cutting a long record into index cards so search returns the exact paragraph, not the whole file. |
| **Vector DB (ChromaDB)** | A library catalog sorted by meaning instead of title. |
| **HNSW index** | Express lanes through the catalog, so search doesn't check every card. |
| **Metadata filter** | "Only search cards from patient P-0007, Allergies section." |
| **Cosine similarity** | How closely two arrows point the same way; 1.0 = same meaning. |

**Two pipelines:**
```
INGEST:  records/*.md → load → PII check → split by section → chunk → embed (bge) → upsert into Chroma
QUERY:   question → embed (bge, query instruction) → Chroma similarity search (+ filters) → top-k chunks + sources
```

---

## 2. Tech Stack

| Purpose | Choice |
|---|---|
| Language | Python 3.12 |
| API | FastAPI (`fastapi[standard]`) |
| Embeddings | `sentence-transformers` + `BAAI/bge-large-en-v1.5` (1024 dims, runs locally) |
| Vector DB | ChromaDB, `PersistentClient` (embedded, no server) |
| Chunking | `llama-index-core` `SentenceSplitter` (after section split) |
| Synthetic data | `Faker` + templates (no LLM, no cost) |
| Env loading | `python-dotenv` |
| Package manager | `uv` |
| Quality | `ruff`, `mypy`, `pytest`, `pytest-cov`, `pre-commit` |
| Container | Docker (CPU-only torch) |

**Cost: $0.** No API key. The model downloads once (~1.3 GB) and then runs offline.

---

## 3. Requirements

### 3.1 Functional

| ID | Requirement |
|---|---|
| FR1 | Ingest `.md` / `.txt` patient records from `RECORDS_DIR` |
| FR2 | Parse record header (patient ID, age band, sex, record date) into metadata |
| FR3 | Section-aware chunking: split by `##` sections, then `SentenceSplitter` within each section |
| FR4 | Embed chunks locally with `bge-large-en-v1.5` (normalized vectors, cosine distance) |
| FR5 | Store in persistent Chroma collection; ingestion is idempotent (unchanged files skipped, changed files replaced) |
| FR6 | PII guard: files containing obvious direct identifiers (email, phone, SSN-like numbers) are rejected and reported, not indexed |
| FR7 | Query endpoint: question + `top_k` + optional filters (`patient_id`, `section`) → ranked chunks with score and source |
| FR8 | Stats endpoint: chunk count, file count, model name, dimensions |
| FR9 | Health endpoint: model loaded, collection reachable |
| FR10 | Every response uses the shared `ApiResponse` / `ErrorResponse` envelope (same as Week 1) |
| FR11 | Retrieval evaluation script: golden question set → hit@k and MRR |

### 3.2 Non-Functional

| Area | Requirement |
|---|---|
| Privacy / on-prem | No network calls at runtime after first model download (`HF_HUB_OFFLINE=1`); Chroma telemetry disabled; record text never written to logs |
| Data | Synthetic, pseudonymous records only; no real patient data in the repo |
| Performance | Query p95 < 1s on CPU after warm-up; ingestion embeds in batches |
| Startup | Model and Chroma client loaded once at startup (FastAPI lifespan), not per request |
| Reliability | Idempotent ingest; concurrent ingest requests rejected (`409`) |
| Testability | `Embedder` and `VectorStore` Protocols; tests use a fake embedder + in-memory Chroma; no model download in tests or CI |
| Maintainability | Same domain-based structure and conventions as Week 1; ruff + mypy clean |
| Portability | Docker with volumes for the Chroma index and model cache |
| Extensibility | Swap Chroma → Qdrant / PGVector by adding one `VectorStore` implementation |
| Safety | API and README state the system retrieves records only and gives no medical advice |

---

## 4. Architecture

Same conventions as Week 1: domain-based layout, `core/` for cross-cutting code, shared response envelope, one global exception handler, all DI in `core/dependencies.py`, no `config.py`.

### 4.1 Folder Structure

```
medical-record-query/
├── src/
│   ├── __init__.py
│   ├── main.py                        # env loading, app creation, lifespan, router + handler + middleware
│   ├── api_router.py                  # central /api/v1 router
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── dependencies.py            # ALL DI providers (embedder, vector store, services)
│   │   ├── lifespan.py                # load model + open Chroma once at startup
│   │   ├── exceptions.py              # AppError + subclasses
│   │   ├── exception_handlers.py      # ONE global handler
│   │   ├── middleware.py              # request ID + start time
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── api_response.py        # ApiResponse[T], ErrorResponse, ResponseMeta
│   │
│   ├── domains/
│   │   ├── __init__.py
│   │   └── patient_records/
│   │       ├── __init__.py
│   │       ├── routers/
│   │       │   ├── __init__.py
│   │       │   └── api_endpoints.py           # ingest, query, stats
│   │       ├── schemas/
│   │       │   ├── __init__.py                # re-exports
│   │       │   ├── ingestion_schemas.py       # IngestRequest, IngestionResult, RejectedFile
│   │       │   └── query_schemas.py           # QueryRequest, QueryFilters, QueryResult, RetrievedChunk, ChunkSource
│   │       ├── services/
│   │       │   ├── __init__.py
│   │       │   ├── ingestion_service.py       # load → guard → chunk → embed → upsert
│   │       │   └── query_service.py           # embed question → search → map results
│   │       └── chunking/
│   │           ├── __init__.py
│   │           ├── record_parser.py           # header → metadata, body → sections
│   │           └── section_chunker.py         # sections → chunks (SentenceSplitter)
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py                    # Embedder Protocol
│   │   └── bge_embedder.py            # sentence-transformers implementation
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── base.py                    # VectorStore Protocol
│   │   └── chroma_store.py            # ChromaDB implementation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── file_loader.py             # read .md/.txt, content hash
│       └── pii_guard.py               # regex checks for direct identifiers
│
├── scripts/
│   ├── generate_records.py            # synthetic records → data/records/
│   ├── ingest.py                      # CLI ingest (same service as the API)
│   └── evaluate_retrieval.py          # golden set → hit@k, MRR
│
├── data/
│   ├── records/                       # synthetic .md records (committed)
│   ├── eval/golden_set.json           # question → expected patient_id/section
│   └── chroma/                        # persisted index (gitignored)
│
├── tests/
│   ├── conftest.py                    # FakeEmbedder, in-memory Chroma, TestClient overrides
│   ├── fixtures/                      # tiny sample records incl. one with PII
│   ├── unit/
│   │   ├── test_record_parser.py
│   │   ├── test_section_chunker.py
│   │   ├── test_pii_guard.py
│   │   ├── test_chroma_store.py
│   │   ├── test_ingestion_service.py
│   │   └── test_query_service.py
│   └── integration/
│       └── test_patient_records_api.py
│
├── docs/
│   ├── DECISIONS.md
│   └── EVALUATION.md                  # chunk size experiments + metrics
├── .env.example
├── .env.dev
├── .env.uat
├── .gitignore
├── .dockerignore
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── CLAUDE.md
└── README.md
```

### 4.2 Why `embeddings/` and `vectorstore/` are top-level

They are infrastructure, like `llm/` in Week 1: any future domain (e.g. lab reports, discharge summaries) reuses them. Each has a Protocol so the services never import `chromadb` or `sentence_transformers` directly.

```python
# embeddings/base.py
class Embedder(Protocol):
    model_name: str
    dimensions: int
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

# vectorstore/base.py
class VectorStore(Protocol):
    def upsert(self, ids: list[str], embeddings: list[list[float]],
               documents: list[str], metadatas: list[dict]) -> None: ...
    def delete_by_file(self, file_name: str) -> None: ...
    def get_file_hash(self, file_name: str) -> str | None: ...
    def query(self, embedding: list[float], top_k: int,
              where: dict | None = None) -> list[SearchHit]: ...
    def stats(self) -> StoreStats: ...
```

### 4.3 Responsibility of Each Module

| Module | Does | Does NOT |
|---|---|---|
| `routers/api_endpoints.py` | Receives requests, calls services, wraps result in `ApiResponse` | Chunk, embed, or touch Chroma |
| `services/ingestion_service.py` | Orchestrates ingest; skip/replace logic; collects counts + rejections | Know HTTP or Chroma API details |
| `services/query_service.py` | Validates filters, embeds question, searches, maps to `QueryResult` | Know HTTP or Chroma API details |
| `chunking/record_parser.py` | Header → metadata dict; body → `(section, text)` list | Embed |
| `chunking/section_chunker.py` | Section text → chunks with `chunk_index` | Know about files or storage |
| `embeddings/bge_embedder.py` | Loads model once; batch embeds; adds query instruction; normalizes | Store vectors |
| `vectorstore/chroma_store.py` | Persistent collection (cosine), upsert, delete, filtered query, stats | Embed text |
| `utils/pii_guard.py` | Returns list of detected identifier types | Decide what to do about them |
| `core/lifespan.py` | Creates embedder + store at startup, stores on `app.state` | Business logic |
| `core/dependencies.py` | Exposes `app.state` objects and services via `Depends` | Create heavy objects per request |

### 4.4 Request Flows

**Ingest**
```
POST /api/v1/patient-records/ingest
 → middleware (request_id, start_time)
 → api_endpoints.ingest → Depends(get_ingestion_service)
 → ingestion_service (holds a lock; 409 if already running)
     for each file in RECORDS_DIR:
       file_loader → text + content_hash
       pii_guard → if hits: add to rejected, continue
       vector_store.get_file_hash(file) == content_hash and not force → skip
       record_parser → metadata + sections
       section_chunker → chunks
       embedder.embed_documents (batched)
       vector_store.delete_by_file(file); vector_store.upsert(...)
 → IngestionResult → ApiResponse
```

**Query**
```
POST /api/v1/patient-records/query
 → api_endpoints.query → Depends(get_query_service)
 → query_service
     embedder.embed_query(question)          # with bge query instruction
     vector_store.query(vector, top_k, where) # where built from filters
     map hits → RetrievedChunk (text, score = 1 - distance, source)
 → QueryResult → ApiResponse
```

Ingestion and embedding are CPU-bound: endpoints that call them are plain `def` (FastAPI runs them in a threadpool) or use `run_in_threadpool`, so the event loop is never blocked.

---

## 5. Data Specification

### 5.1 Synthetic Record Format (`data/records/P-0007.md`)

```markdown
# Patient Record: P-0007
- Age band: 40-49
- Sex: F
- Record date: 2025-03-14

## Chief Complaint
Recurring headaches for three weeks.

## History of Present Illness
...

## Past Medical History
Hypertension diagnosed 2019.

## Medications
Lisinopril 10 mg daily.

## Allergies
Penicillin (rash).

## Lab Results
...

## Assessment
...

## Plan
...
```

Rules for `scripts/generate_records.py`:
- 25–30 records, pseudonymous IDs `P-0001`…; **no names, addresses, phone numbers, emails, or real dates of birth**
- Age as a band, not exact age
- Vary conditions, medications, allergies, and labs from template lists so queries have meaningful answers
- Seeded randomness (`Faker.seed(42)`) so output is reproducible

### 5.2 Chunk Metadata

| Field | Example | Use |
|---|---|---|
| `patient_id` | `P-0007` | filter, source |
| `file_name` | `P-0007.md` | replace on re-ingest, source |
| `section` | `Allergies` | filter, source |
| `age_band` | `40-49` | source (future filter) |
| `sex` | `F` | source (future filter) |
| `record_date` | `2025-03-14` | source |
| `chunk_index` | `2` | ordering, ID |
| `content_hash` | `9f2c…` (SHA-256 of file) | skip unchanged files |

Chunk ID: `"{file_name}:{chunk_index}"` — deterministic, so re-ingest overwrites instead of duplicating.

### 5.3 Chunking Defaults (tune in Step 10)

| Setting | Default | Reason |
|---|---|---|
| Strategy | split by `##` section → `SentenceSplitter` | Medical sections are natural semantic units; section becomes filterable metadata |
| `CHUNK_SIZE` | 256 tokens | Record sections are short; bge max input is 512 tokens |
| `CHUNK_OVERLAP` | 32 tokens | Keeps sentence context across chunk edges |

Short sections stay as one chunk. Each chunk's text is prefixed with `"{section}: "` before embedding so the vector carries section context.

---

## 6. API Contract

All endpoints use the shared envelope from Week 1:
`{"success": true, "data": {...}, "meta": {"request_id": "...", "processing_time_ms": 0}}`
`{"success": false, "error": {"code": "...", "message": "..."}, "meta": {...}}`

### 6.1 Ingest — `POST /api/v1/patient-records/ingest`

Request:
```json
{ "force_reingest": false }
```

`data` — `IngestionResult`:
```json
{
  "files_scanned": 28,
  "files_ingested": 26,
  "files_skipped_unchanged": 0,
  "files_rejected": [
    { "file_name": "P-0031.md", "reason": "PII detected: email" }
  ],
  "chunks_upserted": 214
}
```

### 6.2 Query — `POST /api/v1/patient-records/query`

Request:
```json
{
  "question": "Which patients are allergic to penicillin?",
  "top_k": 5,
  "filters": { "patient_id": null, "section": "Allergies" }
}
```
Validation: `question` 3–500 chars; `top_k` 1–`MAX_TOP_K`; `section` must be one of the known section names.

`data` — `QueryResult`:
```json
{
  "question": "Which patients are allergic to penicillin?",
  "results": [
    {
      "text": "Allergies: Penicillin (rash).",
      "score": 0.83,
      "source": {
        "patient_id": "P-0007",
        "file_name": "P-0007.md",
        "section": "Allergies",
        "record_date": "2025-03-14",
        "chunk_index": 5
      }
    }
  ]
}
```

### 6.3 Stats — `GET /api/v1/patient-records/stats`

`data`: `{ "collection": "patient_records", "chunk_count": 214, "file_count": 26, "embedding_model": "BAAI/bge-large-en-v1.5", "dimensions": 1024 }`

### 6.4 Health — `GET /health`

`{ "status": "ok", "environment": "dev", "model_loaded": true, "vector_store": "ok" }` — returns `503` if the model or store is not ready.

### 6.5 Errors

| Exception | HTTP | `code` | When |
|---|---|---|---|
| `InvalidInputError` | 400 | `INVALID_INPUT` | Unknown section filter, malformed filter |
| *(request validation)* | 422 | `VALIDATION_ERROR` | Pydantic body errors — same global handler, same envelope |
| `IngestionInProgressError` | 409 | `INGESTION_IN_PROGRESS` | Second ingest while one is running |
| `CollectionEmptyError` | 409 | `COLLECTION_EMPTY` | Query before any ingest |
| `RecordsDirectoryError` | 500 | `RECORDS_DIR_ERROR` | `RECORDS_DIR` missing or unreadable |
| `ModelNotReadyError` | 503 | `MODEL_NOT_READY` | Embedding model failed to load |
| `VectorStoreError` | 503 | `VECTOR_STORE_ERROR` | Chroma read/write failure |
| *(anything else)* | 500 | `INTERNAL_ERROR` | Logged with traceback; generic message to client |

Register the single `global_exception_handler` for `AppError`, `RequestValidationError`, and `Exception`, so every error — including FastAPI's own validation errors — uses `ErrorResponse`.

---

## 7. Environment Files

No secrets in this project. `main.py` loads `.env.{APP_ENV}`; every value has a safe default in code via `os.getenv`.

```dotenv
# .env.example / .env.dev
APP_ENV=dev
LOG_LEVEL=DEBUG

EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=16
QUERY_INSTRUCTION=Represent this sentence for searching relevant passages: 
HF_HUB_OFFLINE=0            # set to 1 after the first successful model download

CHROMA_PATH=./data/chroma
CHROMA_COLLECTION=patient_records
RECORDS_DIR=./data/records

CHUNK_SIZE=256
CHUNK_OVERLAP=32
DEFAULT_TOP_K=5
MAX_TOP_K=20
```

`.env.uat`: same keys, `APP_ENV=uat`, `LOG_LEVEL=INFO`, `HF_HUB_OFFLINE=1`, separate `CHROMA_PATH` (e.g. `./data/chroma_uat`).

`.gitignore` additions: `data/chroma*/`, `.env.dev`, `.env.uat`, model cache folder if you set `HF_HOME` inside the repo.

---

## 8. Key Implementation Notes

- **Load once:** `core/lifespan.py` creates `BgeEmbedder` and `ChromaVectorStore` at startup and stores them on `app.state`; `core/dependencies.py` reads them from `request.app.state`. Tests replace them via `app.dependency_overrides`.
- **Normalize + cosine:** encode with `normalize_embeddings=True`; create the Chroma collection with cosine distance (configure the HNSW space at creation; syntax depends on Chroma version, check its docs). Score returned to clients = `1 - distance`.
- **Query instruction:** prepend `QUERY_INSTRUCTION` to the question only (never to documents). It's recommended for short queries against longer passages with bge models.
- **Chroma filters:** one condition → `{"section": "Allergies"}`; two → `{"$and": [{"patient_id": "P-0007"}, {"section": "Allergies"}]}`.
- **Telemetry off:** `chromadb.PersistentClient(path=..., settings=Settings(anonymized_telemetry=False))`.
- **Offline:** after the first download, set `HF_HUB_OFFLINE=1` so the app never reaches the internet at runtime.
- **Logging:** log counts, IDs, durations, filters. Never log chunk text or questions at INFO level or above.
- **Ingest lock:** a `threading.Lock` in the ingestion service; `acquire(blocking=False)` failing → `IngestionInProgressError`.
- **CLI and API share the service:** `scripts/ingest.py` builds the same embedder/store/service and calls `ingest()`; no duplicated logic.

---

## 9. Testing Strategy

| Layer | Approach | Model download? |
|---|---|---|
| Parser, chunker, PII guard | Pure unit tests on fixture records | No |
| Chroma store | `chromadb.EphemeralClient()` (in-memory) + small fake vectors | No |
| Services | `FakeEmbedder` (deterministic hash → 8-dim normalized vector) + in-memory store | No |
| API | `TestClient` + `dependency_overrides` | No |
| Real model smoke test | `@pytest.mark.model`, excluded by default; checks 1024 dims, normalized, and "hypertension" closer to "high blood pressure" than to "broken arm" | Yes (once) |
| Retrieval quality | `scripts/evaluate_retrieval.py` on the real index | Yes |

Minimum cases: header parsing; section split; chunk boundaries and overlap; PII file rejected; unchanged file skipped; changed file replaced (old chunks gone); `force_reingest` re-embeds all; query with no filters; with `patient_id`; with `section`; with both; unknown section → 400; empty collection → 409; concurrent ingest → 409; validation error uses envelope; unhandled error → 500 without stack trace.

---

## 10. Build Workflow (SDLC) — One Step at a Time

Branch: `feature/medical-record-query`. Commit after each step: `step N: <name>`. After each step run:
```powershell
uv run ruff check .
uv run mypy src
uv run pytest
```

### Phase 1 — Requirements & Design
**Step 1. README, API contract, data spec**
- Copy sections 1, 3, 5, 6 into `README.md`; create `CLAUDE.md` with structure rules (same style as Week 1) and `docs/DECISIONS.md`.
- ✅ Done when: ingest and query requests/responses are defined without code.

### Phase 2 — Setup
**Step 2. Skeleton**
```powershell
uv init medical-record-query
cd medical-record-query
uv add "fastapi[standard]" chromadb sentence-transformers llama-index-core python-dotenv
uv add --dev pytest pytest-asyncio pytest-cov ruff mypy pre-commit faker
```
- Folders from 4.1, `.env.example`, `.env.dev`, `.gitignore`, `/health` returning `{"status": "ok"}`.
- ✅ Done when: `uv run uvicorn src.main:app --reload` serves `/docs`.

**Step 3. Core (reuse Week 1)**
- `core/schemas/api_response.py`, `core/exceptions.py` (section 6.5), `core/exception_handlers.py` (also registered for `RequestValidationError`), `core/middleware.py`.
- ✅ Done when: a bad request returns `ErrorResponse` with `request_id`.

### Phase 3 — Data
**Step 4. Synthetic records**
- `scripts/generate_records.py` per 5.1; generate 25–30 files into `data/records/`; add 1–2 deliberately bad files into `tests/fixtures/` (with a fake email/phone) for the PII guard.
- ✅ Done when: records are readable, varied, and contain no direct identifiers.

### Phase 4 — Implementation (bottom-up)
**Step 5. Loader, PII guard, parser, chunker**
- `utils/file_loader.py`, `utils/pii_guard.py`, `chunking/record_parser.py`, `chunking/section_chunker.py` + unit tests.
- ✅ Done when: one record → correct metadata + section-tagged chunks in tests.

**Step 6. Embedder**
- `embeddings/base.py`, `embeddings/bge_embedder.py`; `FakeEmbedder` in `tests/conftest.py`.
- Run the `model` smoke test once (downloads the model), then set `HF_HUB_OFFLINE=1`.
- ✅ Done when: 1024-dim normalized vectors; semantic sanity check passes.

**Step 7. Vector store**
- `vectorstore/base.py`, `vectorstore/chroma_store.py` (persistent, cosine, telemetry off) + tests with `EphemeralClient`.
- ✅ Done when: upsert, delete-by-file, filtered query, hash lookup, and stats pass.

**Step 8. Services**
- `services/ingestion_service.py` (lock, skip/replace, PII rejection, batching), `services/query_service.py` (filters → `where`, score mapping) + tests with fakes.
- ✅ Done when: all service cases in section 9 pass without the real model.

**Step 9. Lifespan, DI, routers, CLI**
- `core/lifespan.py`, `core/dependencies.py`, `routers/api_endpoints.py`, `api_router.py`, `scripts/ingest.py`.
- Manual check:
```powershell
uv run python scripts/ingest.py
curl.exe -X POST http://localhost:8000/api/v1/patient-records/query -H "Content-Type: application/json" -d '{\"question\": \"penicillin allergy\", \"top_k\": 3}'
```
- ✅ Done when: ingest + query work from Swagger, curl, and the CLI.

### Phase 5 — Evaluation & Testing
**Step 10. Retrieval evaluation**
- `data/eval/golden_set.json`: 15–20 questions, each with expected `patient_id` and/or `section`.
- `scripts/evaluate_retrieval.py`: hit@1, hit@5, MRR.
- Compare at least 3 configs (e.g. chunk 128/256/512; with vs without section prefix; with vs without query instruction). Record the table and chosen defaults in `docs/EVALUATION.md`.
- ✅ Done when: hit@5 ≥ 0.8 on the golden set, or the shortfall is explained.

**Step 11. Integration tests + coverage**
- `tests/integration/test_patient_records_api.py` covering every endpoint and every error code.
- ✅ Done when: `uv run pytest --cov=src` ≥ 80%, no model download needed.

### Phase 6 — Deployment
**Step 12. Docker**
- Multi-stage `Dockerfile`, `python:3.12-slim`, non-root user. Install **CPU-only torch** (from the PyTorch CPU wheel index) to avoid a multi-GB CUDA image.
- Volumes: Chroma index and Hugging Face cache, so the model isn't baked in or re-downloaded.
```powershell
docker build -t medical-record-query .
docker run --env-file .env.uat -p 8000:8000 `
  -v "${PWD}/data/chroma_uat:/app/data/chroma_uat" `
  -v "${HOME}/.cache/huggingface:/home/app/.cache/huggingface" `
  medical-record-query
```
- ✅ Done when: `/health` shows `"model_loaded": true` in UAT and a query returns results.

### Phase 7 — Documentation
**Step 13. Finalize**
- README: setup, ingest, query examples, filters, evaluation results, privacy notes, "not medical advice" statement.
- `CLAUDE.md`: add the "how to add a new record type / vector store" checklist.
- ✅ Done when: a teammate can clone, generate data, ingest, and query in 10 minutes.

---

## 11. Stretch Goals (only after Step 13)

| Goal | Week-2 skill it practices |
|---|---|
| `QdrantVectorStore` or `PgVectorStore` implementing the same Protocol, switchable via env | Vector DB comparison, payload filtering |
| Compare HNSW settings (`M`, `ef_construction`, `ef_search`) on latency vs hit@k | Indexing algorithms |
| Semantic chunking vs section chunking in `docs/EVALUATION.md` | Chunking strategies (Kamradt levels) |
| Reranking with a local cross-encoder (`bge-reranker-base`) | Retrieval quality |
| Local answer generation via Ollama, citing chunk sources | Full on-prem RAG |
| Date-range filter on `record_date` | Metadata filtering |
