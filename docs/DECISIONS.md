# Decisions and deviations from `medical-record-query-architecture.md`

One line per deviation: **what** — *why*.

- **Optional Claude answer step (`ANSWER_PROVIDER=anthropic`, `src/llm/`)** — requested explicitly by the owner. It overrides the doc's "answer generation must be local" stretch note. The default stays `none`, so the system is still fully on-prem by default.
- **Project lives in the repo root, not `uv init medical-record-query/`** — the repo already exists, so a nested project folder would only add a level.
- **Repo `README.md` (1-line placeholder) replaced with the full README** — required by Step 1. The old version is in git history (`main`).
