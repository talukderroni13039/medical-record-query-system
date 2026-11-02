# Running the app and testing it in Swagger (Windows PowerShell)

This guide gets you from a fresh clone to queries in Swagger in about 10 minutes. It uses the **5 sample records**
in `samples/records/` and the test questions in [`samples/QUESTIONS.md`](../samples/QUESTIONS.md).

> Synthetic data only. Results are record excerpts, **not medical advice**.

---

## 1. One-time setup

```powershell
cd D:\Techjays\Projects\AI-Learning\medical-record-query-system
uv sync                                   # Python 3.12 + locked deps (CPU-only torch)
```

The project reads `.env.dev` when `APP_ENV=dev` (the default). Every setting has a default in code, so `.env.dev` only needs
what you want to change. Recommended `.env.dev` lines:

```dotenv
HF_HUB_OFFLINE=1          # after the model has been downloaded once (it already has been on this machine)
ANSWER_PROVIDER=none      # on-prem default; set to anthropic only for the optional Claude answers
```

The first model load downloads `BAAI/bge-large-en-v1.5` (~1.3 GB) into `%USERPROFILE%\.cache\huggingface`. While it
downloads, leave `HF_HUB_OFFLINE` at `0`.

---

## 2. Start the API with the 5 sample records

Variables set in the shell override `.env.dev`. This points the app at the sample folder and a separate index, so
the main index in `data/chroma` is not touched:

```powershell
$env:APP_ENV     = "dev"
$env:RECORDS_DIR = "./samples/records"
$env:CHROMA_PATH = "./data/chroma_samples"
uv run uvicorn src.main:app --reload --port 8000
```

Wait for `Application startup complete.` (the model takes about 5 s to load), then open **http://127.0.0.1:8000/docs**.

> Use `127.0.0.1` rather than `localhost` in PowerShell / `Invoke-RestMethod`. Windows PowerShell 5.1 tries IPv6 first
> and can hang.

To use the full synthetic set (28 records) instead, leave `RECORDS_DIR` and `CHROMA_PATH` unset.

---

## 3. Test in Swagger, step by step

In Swagger, click an endpoint, then **Try it out**, edit the body, and click **Execute**. Every response has the same shape:

```json
{"success": true, "data": {...}, "meta": {"request_id": "…", "processing_time_ms": 12}}
{"success": false, "error": {"code": "…", "message": "…"}, "meta": {...}}
```

The response headers also include `X-Request-ID`.

### Step 1 — `GET /health`
Expect `data.model_loaded: true`, `data.vector_store: "ok"`, `data.answer_provider: "none"`.

### Step 2 — `POST /api/v1/patient-records/query` before ingesting (optional error check)
Body `{"question": "allergies"}` → **409 `COLLECTION_EMPTY`** on a fresh `CHROMA_PATH`.

### Step 3 — `POST /api/v1/patient-records/ingest`
Body:
```json
{ "force_reingest": false }
```
Expect `files_scanned: 5, files_ingested: 5, chunks_upserted: 40` (8 sections × 5 records).
Run it again: `files_skipped_unchanged: 5` (idempotent). Use `"force_reingest": true` to re-embed everything.

### Step 4 — `GET /api/v1/patient-records/stats`
Expect `chunk_count: 40, file_count: 5, embedding_model: "BAAI/bge-large-en-v1.5", dimensions: 1024`.

### Step 5 — `POST /api/v1/patient-records/query`
Paste the bodies from [`samples/QUESTIONS.md`](../samples/QUESTIONS.md) section A, for example:

```json
{ "question": "Which patient is allergic to penicillin?", "top_k": 3 }
```
→ first result `chunk_id: "P-1001.md:4"`, `text: "Allergies: Penicillin (hives)."`, `score ≈ 0.76`.

```json
{ "question": "What medication is used for chest pain?", "top_k": 3,
  "filters": { "patient_id": null, "section": "Medications" } }
```
→ `P-1003.md:3` (nitroglycerin spray).

```json
{ "question": "What were the lab results?", "top_k": 5, "filters": { "patient_id": "P-1005" } }
```
→ only P-1005 chunks, Lab Results first.

### Step 6 — Error cases
Use section C of `samples/QUESTIONS.md`: short question → 422 `VALIDATION_ERROR`; `top_k: 50` → 422;
section `"Diet"` → 400 `INVALID_INPUT`; `"generate_answer": true` with the default provider → 400 `ANSWER_GENERATION_DISABLED`.

### Step 7 — PII guard demo (optional)
```powershell
Copy-Item tests\fixtures\pii\P-9001.md samples\records\
```
Run ingest again → `files_rejected: [{"file_name": "P-9001.md", "reason": "PII detected: email"}]`. The file is not indexed.
Clean up afterwards:
```powershell
Remove-Item samples\records\P-9001.md
```

---

## 4. Optional: Claude answers (`generate_answer: true`)

> **Not on-premise.** The question and retrieved chunk text are sent to the Anthropic API. Use synthetic data only.
> On startup the app logs a WARNING saying exactly this.

1. Make sure `.env.dev` contains your key (`ANTHROPIC_API_KEY=…`; it already does on this machine) and add:
   ```dotenv
   ANSWER_PROVIDER=anthropic
   ```
   (Optional: `LLM_MODEL=claude-haiku-5-5`, `LLM_MAX_TOKENS=1000`, `LLM_MAX_RETRIES=2`, `LLM_TIMEOUT_SECONDS=30`.)
2. Restart uvicorn. `GET /health` now shows `answer_provider: "anthropic"`.
3. Query with `"generate_answer": true` (section B of `samples/QUESTIONS.md`):
   ```json
   { "question": "Which patient is allergic to sulfa drugs?", "top_k": 5, "generate_answer": true }
   ```
   `data.answer` looks like this (example; exact wording varies) →
   ```json
   { "answer": "Patient P-1002 is allergic to sulfa drugs (rash) [P-1002.md:4].",
     "cited_chunk_ids": ["P-1002.md:4"], "insufficient_context": false }
   ```
   Every cited ID is guaranteed to be one of the retrieved `results[].chunk_id`.
   Errors: 504 `LLM_TIMEOUT`, 502 `ANSWER_GENERATION_FAILED`.
4. Switch back with `ANSWER_PROVIDER=none` (or remove the line) and restart.

Cost: Claude Haiku 5.5, `LLM_MAX_TOKENS=1000`, and at most 8 chunks per request. Each question is 1 call, or up to 3 if
the output fails validation.

---

## 5. CLI alternatives (no Swagger)

```powershell
uv run python scripts/ingest.py            # same service as the API
uv run python scripts/ingest.py --force

Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/v1/patient-records/query `
  -ContentType application/json `
  -Body '{"question": "Which patient is allergic to penicillin?", "top_k": 3}' | ConvertTo-Json -Depth 6

curl.exe -X POST http://127.0.0.1:8000/api/v1/patient-records/query -H "Content-Type: application/json" `
  -d '{\"question\": \"penicillin allergy\", \"top_k\": 3}'
```

---

## 6. Docker (UAT)

Create `.env.uat` (see the README for the exact lines; it needs no secrets), then:

```powershell
docker build -t medical-record-query .
New-Item -ItemType Directory -Force data\chroma_uat | Out-Null
docker run --rm --env-file .env.uat -p 8000:8000 `
  -v "${PWD}\data\chroma_uat:/app/data/chroma_uat" `
  -v "${HOME}\.cache\huggingface:/home/app/.cache/huggingface" `
  medical-record-query
```
Open http://127.0.0.1:8000/docs. `GET /health` should show `"model_loaded": true` and `"environment": "uat"`, then run
ingest and query as above. The image contains the 28 synthetic records in `/app/data/records`.

> Docker Desktop keeps its disk image on C:. The image is about 1.8 GB and the build cache about 3.4 GB, so keep at
> least ~8 GB free on C: before building.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` → 503 `MODEL_NOT_READY` | The model failed to load. With `HF_HUB_OFFLINE=1` the model must already be in the HF cache: run once with `HF_HUB_OFFLINE=0` |
| Query → 409 `COLLECTION_EMPTY` | Run ingest first (Swagger step 3 or `scripts/ingest.py`) |
| Ingest → 500 `RECORDS_DIR_ERROR` | `RECORDS_DIR` points to a missing folder |
| Ingest → 409 `INGESTION_IN_PROGRESS` | Another ingest is running. Wait and retry |
| Startup error `ANSWER_PROVIDER must be one of…` / `ANTHROPIC_API_KEY is not set` | Fix the value in `.env.dev` (`none` or `anthropic`). The key is needed only for `anthropic` |
| `Invoke-RestMethod` hangs | Use `127.0.0.1` instead of `localhost` |
