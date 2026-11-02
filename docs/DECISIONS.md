# Decisions and deviations from `medical-record-query-architecture.md`

One line per deviation: **what** — *why*.

- **Optional Claude answer step (`ANSWER_PROVIDER=anthropic`, `src/llm/`)** — requested explicitly by the owner. It overrides the doc's "answer generation must be local" stretch note. The default stays `none`, so the system is still fully on-prem by default.
- **Project lives in the repo root, not `uv init medical-record-query/`** — the repo already exists, so a nested project folder would only add a level.
- **Repo `README.md` (1-line placeholder) replaced with the full README** — required by Step 1. The old version is in git history (`main`).
- **CPU-only torch via a `[tool.uv.sources]` index pin (`download.pytorch.org/whl/cpu`)** — the standard way to keep `uv.lock` and the Docker image CUDA-free on every platform.
- **`ruff` excludes `*.md`** — ruff 0.17 formats Python blocks inside Markdown, and it rewrote the owner's architecture doc. The doc was restored to its original text.
- **Own sentence-aware splitter (`section_chunker.py`) instead of llama-index `SentenceSplitter`** — `SentenceSplitter` tokenises with tiktoken, which downloads its encoding from the internet at first use (this breaks the on-prem/offline rule). It also pulls in nltk and other heavy dependencies. The replacement does the same thing (pack sentences to `CHUNK_SIZE` with `CHUNK_OVERLAP`) and is fully tested.
- **Token counts are approximated (`\w+|[^\w\s]`)** — this avoids loading a tokenizer inside the pure chunking layer. It is close to WordPiece counts for this text, and the 256-token default leaves wide headroom under bge's 512-token limit.
- **Missing header fields stored as `"unknown"`** — Chroma metadata values cannot be `None`. A missing title falls back to the file stem as `patient_id`.
- **Section names canonicalised case-insensitively** — `"allergies"` in a file or a filter maps to `Allergies`. Unknown section names are still indexed under their own name.
- **`/health` uses the `ApiResponse` envelope; unhealthy returns 503 `ErrorResponse`** — the binding rule says every endpoint uses the envelope. The doc's flat `{status, model_loaded, ...}` shape now sits in `data`, plus `answer_provider`.
- **`RetrievedChunk.chunk_id` added** — answers cite chunk IDs, so clients need the same IDs. It is also useful on its own as a stable reference.
- **`StatsResult` lives in `ingestion_schemas.py`; `QueryService.stats()` serves it** — this avoids an extra schema module or a stats service just for one endpoint.
- **Ingest lock is module-level in `core/dependencies.py`** — services are built per request, so a lock held inside the service would not be shared. The CLI is a separate process and uses its own lock.
- **`QueryService.query` is async; embedding and search run via `asyncio.to_thread`** — the answer step is async I/O, while retrieval is CPU-bound and must not block the event loop. Ingest stays a plain `def` endpoint (threadpool).
- **Malformed `patient_id` filter → 400 `INVALID_INPUT`; body shape errors → 422** — as in the doc's error table. Unknown body/filter keys are rejected (`extra="forbid"`) so typos don't silently disable a filter.
- **PII guard also deletes previously indexed chunks of a file that now contains PII** — otherwise a stale clean version would stay searchable.
- **`ANONYMIZED_TELEMETRY=False` is also set in the process env** — belt and braces next to `Settings(anonymized_telemetry=False)`.
- **`GeneratedAnswer` and `ContextChunk` live in `src/llm/schemas.py`** — the `AnswerGenerator` Protocol is infrastructure (like `embeddings/`) and must not import a domain. The domain re-exports `GeneratedAnswer` for `QueryResult.answer`.
- **Cited-ID validation happens twice** — first in the Instructor `response_model` validator (via validation `context`, so Claude gets a corrective re-ask), then again in `QueryService` (defence in depth for any provider). A rogue ID returns 502 `ANSWER_GENERATION_FAILED`.
- **Answers that are not `insufficient_context` must cite at least one chunk** — enforces grounding; an uncited answer triggers a re-ask.
- **Anthropic SDK `max_retries=0`; retries come only from Instructor** — this makes every attempt exactly one billable call, so the 50-call budget is exact. Note: Instructor 1.17 makes `max_retries + 1` attempts, so `LLM_MAX_RETRIES=2` means up to 3 calls per question.
- **Instructor `Mode.ANTHROPIC_TOOLS` with `claude-haiku-5-5` and default (adaptive) thinking** — Haiku 5.5 accepts a forced tool call with thinking on (the call simply skips thinking), so no extra parameters are needed.
- **Startup fails fast on an unknown `ANSWER_PROVIDER` or a missing key in `anthropic` mode** — misconfiguration should not silently downgrade to "answers disabled".
- **No hits + `generate_answer=true` → canned `insufficient_context` answer with no LLM call** — this saves calls and avoids sending an empty context.
- **At most `LLM_MAX_CONTEXT_CHUNKS` (default 8) retrieved chunks go to Claude** — this caps prompt size and data exposure independently of `top_k`.
- **Chunk text is XML-escaped inside `<chunk>` tags, and the prompt says chunk text is data, not instructions** — basic prompt-injection hygiene for record content.
- **Health and `main.py` read `ANSWER_PROVIDER` through `core.lifespan.answer_provider()`** — one normalisation point (trimmed, lower-cased).
