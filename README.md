# Local Medical Record Query System

On-premise semantic search API over **synthetic, pseudonymous** patient history files.
Records are split by section, chunked, embedded locally with `BAAI/bge-large-en-v1.5`,
stored in a persistent ChromaDB index, and queried through FastAPI.

- **Default mode (`ANSWER_PROVIDER=none`)**: retrieval only. No API key, and no network calls at runtime after the one-time model download.
- **Optional mode (`ANSWER_PROVIDER=anthropic`)**: adds a short answer written by Claude from the retrieved chunks, with chunk citations. **This mode is not on-premise:** the retrieved record text is sent to the Anthropic API. Use it **with synthetic data only**.

> **Not medical advice.** This system retrieves text from records. It does not diagnose, recommend treatment, or replace a clinician. Generated answers only restate what is in the retrieved chunks.

---

## 1. How it works

| Concept | Analogy |
|---|---|
| **Embedding** | GPS coordinates for meaning. Similar sentences get nearby coordinates even when they use different words ("high blood pressure" ≈ "hypertension"). |
| **Chunking** | Cutting a long record into index cards so search returns the exact paragraph, not the whole file. |
| **Vector DB (ChromaDB)** | A library catalog sorted by meaning instead of title. |
| **HNSW index** | Express lanes through the catalog, so search doesn't check every card. |
| **Metadata filter** | "Only search cards from patient P-0007, Allergies section." |
| **Cosine similarity** | How closely two arrows point the same way. 1.0 = same meaning. |

```
INGEST:  records/*.md → load → PII check → split by ## section → chunk → embed (bge) → upsert into Chroma
QUERY:   question → embed (bge + query instruction) → Chroma search (+ filters) → top-k chunks + sources
         [optional] → Claude answers from those chunks only, citing chunk IDs
```

---

## 2. Quick start (Windows PowerShell)

```powershell
# 1. Install (Python 3.12 is fetched by uv if missing)
uv sync

# 2. Generate the synthetic records (already committed in data/records/, so this is optional)
uv run python scripts/generate_records.py

# 3. Build the index (the first run downloads the model, ~1.3 GB)
uv run python scripts/ingest.py

# 4. Run the API (dev)
$env:APP_ENV = "dev"
uv run uvicorn src.main:app --reload
# Open http://localhost:8000/docs
```

After the first successful model download, add `HF_HUB_OFFLINE=1` to `.env.dev` so the app never goes online at runtime.

For a step-by-step Swagger walkthrough with 5 sample files and test questions, see **[docs/RUNNING_AND_SWAGGER.md](docs/RUNNING_AND_SWAGGER.md)**.

---

## 3. Requirements

### 3.1 Functional

| ID | Requirement |
|---|---|
| FR1 | Ingest `.md` / `.txt` patient records from `RECORDS_DIR` |
| FR2 | Parse the record header (patient ID, age band, sex, record date) into metadata |
| FR3 | Section-aware chunking: split by `##` section, then sentence-aware splitting within each section |
| FR4 | Embed chunks locally with `bge-large-en-v1.5` (normalized vectors, cosine distance) |
| FR5 | Persistent Chroma collection. Ingest is idempotent: unchanged files are skipped and changed files are replaced |
| FR6 | PII guard: files with direct identifiers (email, phone, SSN-like numbers) are rejected and reported, not indexed |
| FR7 | Query: question + `top_k` + optional filters (`patient_id`, `section`) → ranked chunks with score and source |
| FR8 | Stats: chunk count, file count, model name, dimensions |
| FR9 | Health: model loaded, collection reachable |
| FR10 | Every response uses the shared `ApiResponse` / `ErrorResponse` envelope |
| FR11 | Retrieval evaluation script: golden question set → hit@k and MRR |
| FR12 | Optional: `generate_answer=true` returns a cited answer from Claude (only when `ANSWER_PROVIDER=anthropic`) |

### 3.2 Non-functional

| Area | Requirement |
|---|---|
| Privacy / on-prem | No runtime network calls in default mode (`HF_HUB_OFFLINE=1`). Chroma telemetry is off. Record text and questions are never logged at INFO or above |
| Data | Synthetic, pseudonymous records only |
| Performance | Query p95 < 1 s on CPU after warm-up. Ingest embeds in batches |
| Startup | Model, Chroma client, and the optional answer generator are created once (FastAPI lifespan) |
| Reliability | Idempotent ingest. A second ingest while one is running is rejected (`409`) |
| Testability | `Embedder`, `VectorStore`, and `AnswerGenerator` Protocols. Default tests use fakes and in-memory Chroma: no model download, no API calls |

---

## 4. Data specification

### 4.1 Record format (`data/records/P-0007.md`)

```markdown
# Patient Record: P-0007
- Age band: 40-49
- Sex: F
- Record date: 2025-03-14

## Chief Complaint
Recurring headaches for three weeks.

## Past Medical History
Hypertension diagnosed 2019.

## Medications
Lisinopril 10 mg daily.

## Allergies
Penicillin (rash).
```

Known sections (usable as the `section` filter, case-insensitive): `Chief Complaint`, `History of Present Illness`,
`Past Medical History`, `Medications`, `Allergies`, `Lab Results`, `Assessment`, `Plan`.

### 4.2 Chunk metadata

| Field | Example | Use |
|---|---|---|
| `patient_id` | `P-0007` | filter, source |
| `file_name` | `P-0007.md` | replace on re-ingest, source |
| `section` | `Allergies` | filter, source |
| `age_band` / `sex` / `record_date` | `40-49` / `F` / `2025-03-14` | source |
| `chunk_index` | `2` | ordering, ID |
| `content_hash` | SHA-256 of the file | skip unchanged files |

Chunk ID: `"{file_name}:{chunk_index}"`. It is deterministic, so a re-ingest overwrites chunks instead of duplicating them.

### 4.3 Chunking defaults

| Setting | Default | Why |
|---|---|---|
| Strategy | split by `##` section, then sentence-aware packing | Sections are natural semantic units and become filterable metadata |
| `CHUNK_SIZE` | 256 tokens | Sections are short. bge's maximum input is 512 tokens |
| `CHUNK_OVERLAP` | 32 tokens | Keeps sentence context across chunk edges |

Each chunk is stored and embedded as `"{section}: {text}"`. See `docs/EVALUATION.md` for the experiments behind these defaults.

---

## 5. API contract

Every response uses one envelope:

```json
{"success": true,  "data": { ... }, "meta": {"request_id": "...", "processing_time_ms": 12}}
{"success": false, "error": {"code": "...", "message": "..."}, "meta": {"request_id": "...", "processing_time_ms": 3}}
```

Every response also includes an `X-Request-ID` header.

### 5.1 `POST /api/v1/patient-records/ingest`

```json
{ "force_reingest": false }
```
`data` (`IngestionResult`):
```json
{
  "files_scanned": 28, "files_ingested": 26, "files_skipped_unchanged": 0,
  "files_rejected": [{ "file_name": "P-0031.md", "reason": "PII detected: email" }],
  "chunks_upserted": 214
}
```

### 5.2 `POST /api/v1/patient-records/query`

```json
{
  "question": "Which patients are allergic to penicillin?",
  "top_k": 5,
  "filters": { "patient_id": null, "section": "Allergies" },
  "generate_answer": false
}
```
Validation: `question` must be 3–500 characters, `top_k` must be 1–`MAX_TOP_K` (default 20), and `section` must be a known section.

`data` (`QueryResult`):
```json
{
  "question": "Which patients are allergic to penicillin?",
  "results": [{
    "chunk_id": "P-0007.md:5",
    "text": "Allergies: Penicillin (rash).",
    "score": 0.83,
    "source": { "patient_id": "P-0007", "file_name": "P-0007.md", "section": "Allergies",
                "record_date": "2025-03-14", "chunk_index": 5 }
  }],
  "answer": null
}
```

When `generate_answer` is `true` and `ANSWER_PROVIDER=anthropic`, `answer` is filled in:
```json
"answer": { "answer": "P-0007 is allergic to penicillin (rash) [P-0007.md:5].",
            "cited_chunk_ids": ["P-0007.md:5"], "insufficient_context": false }
```

### 5.3 `GET /api/v1/patient-records/stats`

`data`: `{"collection": "patient_records", "chunk_count": 214, "file_count": 26, "embedding_model": "BAAI/bge-large-en-v1.5", "dimensions": 1024}`

### 5.4 `GET /health`

`data`: `{"status": "ok", "environment": "dev", "model_loaded": true, "vector_store": "ok", "answer_provider": "none"}`.
Returns `503` (`MODEL_NOT_READY` / `VECTOR_STORE_ERROR`) when the app is not ready.

### 5.5 Errors

| HTTP | `code` | When |
|---|---|---|
| 400 | `INVALID_INPUT` | Unknown section filter, malformed filter |
| 400 | `ANSWER_GENERATION_DISABLED` | `generate_answer=true` while `ANSWER_PROVIDER=none` |
| 422 | `VALIDATION_ERROR` | Request body fails validation |
| 409 | `INGESTION_IN_PROGRESS` | Second ingest while one is running |
| 409 | `COLLECTION_EMPTY` | Query before any ingest |
| 500 | `RECORDS_DIR_ERROR` | `RECORDS_DIR` missing or unreadable |
| 502 | `ANSWER_GENERATION_FAILED` | Claude API error, invalid output, or citation of a chunk not in the retrieved set |
| 503 | `MODEL_NOT_READY` | Embedding model not loaded |
| 503 | `VECTOR_STORE_ERROR` | Chroma read/write failure |
| 504 | `LLM_TIMEOUT` | Claude call exceeded `LLM_TIMEOUT_SECONDS` |
| 500 | `INTERNAL_ERROR` | Anything else. Logged with a traceback; the client gets a generic message |

---

## 6. Configuration

`src/main.py` loads `.env.{APP_ENV}` (`APP_ENV` defaults to `dev`). Variables that are already set in the shell or in Docker win.
Every non-secret value has a default in code. See [`.env.example`](.env.example) for the full list.

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | |
| `EMBEDDING_DEVICE` / `EMBEDDING_BATCH_SIZE` | `cpu` / `16` | |
| `HF_HUB_OFFLINE` | `0` | Set to `1` after the first download |
| `CHROMA_PATH` / `CHROMA_COLLECTION` | `./data/chroma` / `patient_records` | |
| `RECORDS_DIR` | `./data/records` | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `256` / `32` | |
| `DEFAULT_TOP_K` / `MAX_TOP_K` | `5` / `20` | |
| `ANSWER_PROVIDER` | `none` | `none` or `anthropic` |
| `ANTHROPIC_API_KEY` | — | Only needed for `anthropic`. Keep it in `.env.dev` only, never commit it |
| `LLM_MODEL` / `LLM_MAX_TOKENS` / `LLM_MAX_RETRIES` / `LLM_TIMEOUT_SECONDS` | `claude-haiku-5-5` / `1000` / `2` / `30` | |

---

## 7. Privacy and safety

- **Synthetic data only.** `data/records/` is produced by `scripts/generate_records.py` and uses pseudonymous IDs and age bands. It contains no names, contact details, or dates of birth.
- **PII guard.** Files that contain an email address, a phone number, or an SSN-like number are rejected at ingest and reported in `files_rejected`.
- **No record text in logs.** Logs contain counts, IDs, durations, and filters only.
- **On-premise by default.** With `ANSWER_PROVIDER=none` the app makes no outbound calls (set `HF_HUB_OFFLINE=1`, and Chroma telemetry is off).
- **Anthropic mode is not on-premise.** With `ANSWER_PROVIDER=anthropic`, the question and the retrieved chunks are sent to the Anthropic API. The app logs a WARNING about this at startup. Never use this mode with real patient data.
- **Not medical advice.** The answer prompt forbids diagnosis or treatment advice and allows only what the chunks contain.

---

## 8. Evaluation

20 paraphrased golden questions over the 28 synthetic records (`data/eval/golden_set.json`):

| Defaults (section chunking 256/32, section prefix, BGE query instruction) | hit@1 | hit@5 | MRR |
|---|---|---|---|
| `BAAI/bge-large-en-v1.5`, Chroma cosine | 0.95 | **1.00** | 0.97 |

Run it with `uv run python scripts/evaluate_retrieval.py`. The config comparison and its limitations are in [`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## 9. Testing

| Command | What runs | Network / cost |
|---|---|---|
| `uv run pytest` | Mocked unit + API tests, coverage gate ≥ 80% (currently ~98%) | **None.** Outbound sockets are blocked by a fixture; uses `FakeEmbedder`, in-memory Chroma, and `FakeAnswerGenerator` |
| `uv run pytest -m model --no-cov` | Real bge model: 1024-dim normalized vectors and a semantic sanity check | Model download once |
| `uv run pytest -m live --no-cov` | 5 real Claude answer tests | Real API calls, capped at **50 for the whole build** via `tests/.live_call_budget.json` |

---

## 10. Docker (UAT)

Multi-stage image: `python:3.12-slim`, uv with a frozen lock, CPU-only torch, non-root user (`app`, uid 10001). No `.env*` files are copied into the image. The model and the index are volumes, not baked in.

`.env.uat` (create it locally; it is gitignored and needs no secrets):

```dotenv
APP_ENV=uat
LOG_LEVEL=INFO
HF_HUB_OFFLINE=1
CHROMA_PATH=./data/chroma_uat
RECORDS_DIR=./data/records
ANSWER_PROVIDER=none
```

```powershell
docker build -t medical-record-query .
docker run --env-file .env.uat -p 8000:8000 `
  -v "${PWD}/data/chroma_uat:/app/data/chroma_uat" `
  -v "${HOME}/.cache/huggingface:/home/app/.cache/huggingface" `
  medical-record-query
# then: GET http://127.0.0.1:8000/health  ->  "model_loaded": true, "environment": "uat"
#       POST /api/v1/patient-records/ingest, then /query
```

---

## 11. Development

See [`CLAUDE.md`](CLAUDE.md) for commands, architecture rules, and conventions,
[`docs/DECISIONS.md`](docs/DECISIONS.md) for deviations from the architecture doc,
[`docs/EVALUATION.md`](docs/EVALUATION.md) for retrieval metrics, and
[`docs/RUNNING_AND_SWAGGER.md`](docs/RUNNING_AND_SWAGGER.md) for a hands-on walkthrough with the 5 sample records in
`samples/records/` and the test questions in `samples/QUESTIONS.md`.
