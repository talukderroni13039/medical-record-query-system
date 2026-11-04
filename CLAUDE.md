# CLAUDE.md — Local Medical Record Query System

## 1. Summary
On-premise semantic search over synthetic patient records: FastAPI + ChromaDB (`PersistentClient`) + local `BAAI/bge-large-en-v1.5`.
The default (`ANSWER_PROVIDER=none`) is retrieval only, with no API key and no runtime network access.
Optional `ANSWER_PROVIDER=anthropic` adds a cited Claude answer on `/query`. It sends chunk text to Anthropic, so it is **not on-prem** and is for synthetic data only.

## 2. Commands (PowerShell)
```powershell
uv sync                                              # install (Python 3.12, CPU torch)
uv run python scripts/generate_records.py            # synthetic records -> data/records/
uv run python scripts/ingest.py [--force]            # CLI ingest (same service as API)
Invoke-RestMethod -Method Post http://localhost:8000/api/v1/patient-records/ingest -ContentType application/json -Body '{"force_reingest": false}'
$env:APP_ENV="dev"; uv run uvicorn src.main:app --reload   # dev  -> http://localhost:8000/docs
$env:APP_ENV="uat"; uv run uvicorn src.main:app            # uat
uv run pytest                                        # mocked tests + coverage (no model, no API)
uv run pytest -m model --no-cov                      # real embedding model (downloads once)
uv run pytest -m live --no-cov                       # real Claude calls (budget-capped, max 5 cases)
uv run python scripts/evaluate_retrieval.py          # golden set -> hit@1/hit@5/MRR (docs/EVALUATION.md)
$env:RECORDS_DIR="./samples/records"; $env:CHROMA_PATH="./data/chroma_samples"   # 5-record demo (docs/RUNNING_AND_SWAGGER.md)
uv run ruff check .; uv run ruff format .            # lint / format
uv run mypy src                                      # strict type check
docker build -t medical-record-query .
docker run --env-file .env.uat -p 8000:8000 -v "${PWD}/data/chroma_uat:/app/data/chroma_uat" -v "${HOME}/.cache/huggingface:/home/app/.cache/huggingface" medical-record-query
```

## 3. Architecture rules
- `src/core/`: cross-cutting code only: `dependencies.py` (ALL DI providers), `lifespan.py` (creates heavy objects once),
  `exceptions.py`, `exception_handlers.py` (ONE handler), `middleware.py`, `schemas/` (envelope, health).
- `src/domains/<domain>/`
  - `routers/api_endpoints.py`: HTTP only. Call a service, wrap the result in `ApiResponse`.
  - `schemas/*_schemas.py`: Pydantic request/result models, re-exported in `schemas/__init__.py`.
  - `services/*_service.py`: orchestration and business rules. Depends on Protocols only.
  - `chunking/`: pure parsing/chunking functions. No I/O to stores and no embedding.
- `src/embeddings/`: `Embedder` Protocol (`base.py`) + implementations (`bge_embedder.py`).
- `src/vectorstore/`: `VectorStore` Protocol (`base.py`) + implementations (`chroma_store.py`).
- `src/llm/`: `AnswerGenerator` Protocol (`base.py`), `schemas.py`, `anthropic_answer_generator.py`, `prompts.py`.
- `src/utils/`: small stateless helpers (`file_loader.py`, `pii_guard.py`).
- `src/api_router.py`: the central `/api/v1` router. Every domain router is included here.
- **Layer rule:** router → service → Protocols. Services never import `chromadb`, `sentence_transformers`, `anthropic`, or `instructor`.
  Only `core/lifespan.py`, `scripts/`, and the implementation modules construct concrete classes.

## 4. Naming conventions
- snake_case modules with role suffixes: `*_service.py`, `*_schemas.py`, `api_endpoints.py`, `*_embedder.py`, `*_store.py`.
- PascalCase classes. Protocols are nouns (`Embedder`, `VectorStore`, `AnswerGenerator`). Implementations are prefixed (`BgeEmbedder`, `ChromaVectorStore`).
- Domain data placed in `ApiResponse.data` is named `*Result` (`IngestionResult`, `QueryResult`, `StatsResult`). Requests are named `*Request`.
- Errors are named `*Error`, subclass `AppError`, and carry `status_code` and an UPPER_SNAKE `code`.

## 5. Response contract
- Every endpoint declares `response_model=ApiResponse[<Result>]` and returns `ApiResponse(data=..., meta=ResponseMeta.from_request(request))`.
- Errors: raise an `AppError` subclass. `global_exception_handler` renders `ErrorResponse` for `AppError`,
  `RequestValidationError` (422 `VALIDATION_ERROR`), and `Exception` (500 `INTERNAL_ERROR`, with no stack trace sent to the client).
- Never return raw dicts or a hand-built `JSONResponse` from routers.

## 6. Adding a new domain or record type (e.g. `lab_reports`)
1. Create `src/domains/lab_reports/{routers,schemas,services,chunking}/` with `__init__.py` in each.
2. `schemas/`: `*Request` and `*Result` models, re-exported in `schemas/__init__.py`.
3. `chunking/`: parser (header → metadata, body → sections) + chunker. Pure functions with unit tests.
4. `services/`: a service class taking `Embedder` / `VectorStore` (and a separate collection name if needed).
5. `core/dependencies.py`: add a `get_<x>_service` provider built from existing providers.
6. `routers/api_endpoints.py`: an `APIRouter(prefix="/lab-reports")`, included in `src/api_router.py`.
7. `core/exceptions.py`: new `AppError` subclasses if needed (status + code), plus a row in the README error table.
8. Tests: `tests/unit/test_<x>_*.py` with fakes, and `tests/integration/test_<x>_api.py` (envelope + error codes).

## 7. Adding a vector store (Qdrant / PGVector)
1. `src/vectorstore/qdrant_store.py`: implement every `VectorStore` method (upsert, delete_by_file, get_file_hash, query, stats, ping).
2. Translate the Chroma-style `where` filters (`{"k": v}` / `{"$and": [...]}`) into the store's filter syntax.
3. Select it in `core/lifespan.py` through an env var (`VECTOR_STORE=chroma|qdrant`, default `chroma`), with defaults in code.
4. Add `tests/unit/test_qdrant_store.py` mirroring `test_chroma_store.py` (use an in-memory or local mode, no network in CI).

## 8. Adding a database later
- `src/domains/<d>/models/`: SQLAlchemy ORM models (persistence shape). `schemas/` stays Pydantic (API shape). Never mix them.
- `src/domains/<d>/repositories/`: data access classes. Services depend on repository Protocols.
- `src/core/database.py`: engine/session factory, created in `lifespan.py`, exposed via `get_db_session` in `dependencies.py`.
- Migrations: Alembic (`alembic/`, `alembic.ini`). Never call `create_all` in production code.

## 9. Testing rules
- Default `uv run pytest`: `FakeEmbedder` (8-dim, deterministic), `chromadb.EphemeralClient()`, and `FakeAnswerGenerator`
  (`tests/fakes.py`). **Zero model downloads and zero API calls.** An autouse fixture blocks non-localhost sockets. Coverage of `src/` must be ≥ 80% (enforced in `addopts`).
- `@pytest.mark.model`: real bge model. Excluded by default.
- `@pytest.mark.live`: real Claude. Excluded by default, **max 5 cases**, `max_retries=1`, ≤ 5 chunks.
  Every attempt is counted in `tests/.live_call_budget.json` (gitignored). At **50** the fixture fails with
  "live API budget exhausted" instead of calling.
- Mocked LLM tests reuse the real response saved in `tests/fixtures/llm_answer.json`.
- Use `app.dependency_overrides` in API tests. Never construct real heavy objects in default tests.

## 10. Privacy, security, cost
- Synthetic, pseudonymous data only. The PII guard rejects files with email/phone/SSN-like strings.
- Never log record text or questions at INFO or above. Log counts, IDs, durations, and filters.
- `ANTHROPIC_API_KEY` is read only via `os.getenv`. Never print it, echo it, or commit it. `.env.*` (except `.env.example`) is gitignored.
- After the first model download, set `HF_HUB_OFFLINE=1`. Chroma telemetry is always off.
- Anthropic mode logs a startup WARNING (not on-prem). Claude's SDK retries are off (`max_retries=0`), so each Instructor attempt = 1 call.

## 11. Do not
- Add a settings class / `config.py` or a DI container library. Use `os.getenv` with defaults and FastAPI `Depends`.
- Put logic in routers, or return raw dicts.
- Import `chromadb`, `sentence_transformers`, `anthropic`, or `instructor` from services or routers.
- Log record text or questions. Load the model, Chroma client, or Anthropic client per request.
- Call the real model or real APIs in default tests, or commit `.env.dev` / `.env.uat` / `data/chroma*`.
