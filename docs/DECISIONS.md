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
