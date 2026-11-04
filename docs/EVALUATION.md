# Retrieval Evaluation

**Goal:** hit@5 ≥ 0.8 on the golden set. **Result: hit@5 = 1.00, hit@1 = 0.95, MRR = 0.97 with the chosen defaults.**

## Setup

| Item | Value |
|---|---|
| Corpus | `data/records/` — 28 synthetic records (seed 42), 8 sections each, 224 chunks |
| Golden set | `data/eval/golden_set.json` — 20 paraphrased questions (e.g. "seafood allergy / EpiPen" → *Shellfish (anaphylaxis). Carries an epinephrine auto-injector.*) |
| Hit rule | A result is a hit if its `patient_id` is expected **and**, when the question names a section, the section matches |
| Metrics | hit@1, hit@5, MRR (reciprocal rank of the first hit) |
| Model / store | `BAAI/bge-large-en-v1.5` on CPU (local, offline) + an in-memory Chroma collection per config (cosine) |
| Command | `uv run python scripts/evaluate_retrieval.py --output data/eval/last_results.json` |
| LLM calls | None (retrieval only) |

## Results (k = 5)

| Config | Chunk size / overlap | Section prefix | Query instruction | Chunks | hit@1 | hit@5 | MRR | Misses@5 | Query p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| **default** | **256 / 32** | **yes** | **yes** | 224 | 0.95 | **1.00** | 0.97 | - | 147.8 / 310.8 |
| chunk_128 | 128 / 16 | yes | yes | 224 | 0.95 | 1.00 | 0.97 | - | 247.6 / 426.0 |
| chunk_512 | 512 / 64 | yes | yes | 224 | 0.95 | 1.00 | 0.97 | - | 174.0 / 323.9 |
| no_section_prefix | 256 / 32 | no | yes | 224 | 0.95 | 1.00 | 0.97 | - | 840.9 / 1027.1 |
| no_query_instruction | 256 / 32 | yes | no | 224 | 1.00 | 1.00 | 1.00 | - | 608.6 / 851.7 |

Raw numbers are in `data/eval/last_results.json`. Latency was measured on a laptop CPU while the same process was also embedding the next config, so treat the p50/p95 columns as noisy. The default config's p95 (≈ 0.3 s) meets the < 1 s target.

## What the numbers say

1. **Chunk size has no effect on this corpus.** Every section of a generated record is shorter than 128 tokens, so section-first chunking already yields one chunk per section at 128, 256, and 512 tokens (224 chunks in every config). The section split, not `CHUNK_SIZE`, sets the unit of retrieval here. Chunk size only matters for longer free-text sections (a long HPI or discharge note). **256 / 32** stays the default: it is well under bge's 512-token limit and leaves room for longer real-world sections.
2. **The section prefix is neutral on this set.** Prefixing `"Allergies: "` etc. did not change hit@k, because questions here are distinguishable by content alone. It is kept on: it costs nothing, makes each chunk self-describing in results and in LLM context, and helps generic questions ("what are this patient's allergies?") where the content words alone ("Penicillin (rash).") carry no section signal.
3. **The query instruction is within noise.** Without the BGE retrieval instruction, one more question ranked its target first (hit@1 1.00 vs 0.95; hit@5 identical). With 20 questions, one question is not a meaningful difference. The instruction is kept because the model card recommends it for short queries against passages. **Revisit this when the golden set grows** (switch it with `QUERY_INSTRUCTION=""`).
4. **Limitation: the golden set is saturated.** All configs reach hit@5 = 1.00, so this set cannot rank the configs. Templated synthetic records are easier than real notes. Next steps: add harder questions (multi-hop, negation such as "no known allergies", numeric thresholds), add longer free-text sections to exercise chunk size, and add near-duplicate patients (same condition, different allergies).

## Chosen defaults

| Setting | Value |
|---|---|
| Strategy | split by `##` section → sentence-aware packing |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 256 / 32 |
| Section prefix | on (`"{section}: "`) |
| Query instruction | on (`Represent this sentence for searching relevant passages: `) |
| `DEFAULT_TOP_K` | 5 |

## Answer generation (optional mode)

The answer step was checked with 5 live Claude tests (`tests/live/`, run once, **5 API calls in total**):

| Case | Result |
|---|---|
| Allergy question ("Which patient is allergic to latex?") | cites `P-0024.md:4` (Allergies) ✅ |
| Medication question ("Which patient takes sertraline, and at what dose?") | cites `P-0008.md:3` (Medications), mentions 50 mg ✅ |
| Question not covered by records (blood type) | `insufficient_context = true` ✅ |
| `patient_id = P-0011` filter + answer | cites only `P-0011.md:*` ✅ |
| Request for a prescribing decision | declines to give medical advice; cites only that patient's chunks ✅ |
