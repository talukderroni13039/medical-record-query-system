# Sample questions for the 5 sample records

The records in `samples/records/` (P-1001 … P-1005) are **synthetic**. Each one has 8 sections, which become chunks in this order:

| `chunk_index` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Section | Chief Complaint | History of Present Illness | Past Medical History | Medications | Allergies | Lab Results | Assessment | Plan |

So `P-1003.md:3` is P-1003's Medications chunk.

| Patient | Summary |
|---|---|
| P-1001 | F 30-39 · asthma flare after a cold · **penicillin** allergy · prednisolone 40 mg × 5 days |
| P-1002 | M 60-69 · diabetic foot neuropathy · **sulfa** allergy · gabapentin, insulin glargine |
| P-1003 | F 50-59 · stable angina · **no known drug allergies** · aspirin, atorvastatin, nitroglycerin |
| P-1004 | M 20-29 · hay fever + eczema · **cat dander** allergy · cetirizine, IgE 410 |
| P-1005 | F 70-79 · mild cognitive impairment · **codeine** allergy · donepezil, low B12 |

Paste each JSON body into **`POST /api/v1/patient-records/query`** in Swagger (`/docs`).

## A. Retrieval (works with the default `ANSWER_PROVIDER=none`, offline)

| # | Request body | Expected top result |
|---|---|---|
| 1 | `{"question": "Which patient is allergic to penicillin?", "top_k": 3}` | `P-1001.md:4` (Allergies: Penicillin (hives)) |
| 2 | `{"question": "Who has nerve damage in the feet from diabetes?", "top_k": 3}` | a P-1002 chunk (Assessment / Chief Complaint) |
| 3 | `{"question": "What medication is used for chest pain?", "top_k": 3, "filters": {"section": "Medications"}}` | `P-1003.md:3` (nitroglycerin spray) |
| 4 | `{"question": "hay fever and itchy eyes", "top_k": 3}` | a P-1004 chunk |
| 5 | `{"question": "memory loss in an older patient", "top_k": 3}` | a P-1005 chunk |
| 6 | `{"question": "What were the lab results?", "top_k": 5, "filters": {"patient_id": "P-1005"}}` | only P-1005 chunks; Lab Results first (MoCA 22/30, B12 low) |
| 7 | `{"question": "allergies", "top_k": 5, "filters": {"patient_id": "P-1003", "section": "allergies"}}` | exactly one chunk: `P-1003.md:4` "No known drug allergies." (section names are case-insensitive) |
| 8 | `{"question": "Which patients use an inhaler?", "top_k": 3, "filters": {"section": "Medications"}}` | `P-1001.md:3` |

## B. Answer generation (needs `ANSWER_PROVIDER=anthropic`; sends chunk text to Anthropic, synthetic data only)

Add `"generate_answer": true` to the body. Each request uses 1 Claude call (up to 3 if the answer fails validation).

| # | Request body | What to look for in `data.answer` |
|---|---|---|
| 9 | `{"question": "Which patient is allergic to sulfa drugs?", "top_k": 5, "generate_answer": true}` | names P-1002, `cited_chunk_ids` contains `P-1002.md:4` |
| 10 | `{"question": "What dose of gabapentin was started?", "top_k": 5, "generate_answer": true}` | "300 mg at night", cites a P-1002 chunk (Plan `:7`) |
| 11 | `{"question": "What is this patient's blood type?", "top_k": 5, "filters": {"patient_id": "P-1004"}, "generate_answer": true}` | `insufficient_context: true` |
| 12 | `{"question": "Should I increase her donepezil dose?", "top_k": 5, "filters": {"patient_id": "P-1005"}, "generate_answer": true}` | declines to give medical advice; may quote the documented Plan, citing P-1005 only |

With the default `ANSWER_PROVIDER=none`, requests 9–12 return **400 `ANSWER_GENERATION_DISABLED`**. That is the expected on-prem behaviour.

## C. Error handling (every error uses the same envelope)

| # | Request | Expected |
|---|---|---|
| 13 | `{"question": "hi"}` | 422 `VALIDATION_ERROR` (question needs 3–500 characters) |
| 14 | `{"question": "allergies", "top_k": 50}` | 422 `VALIDATION_ERROR` (`top_k` maximum is 20) |
| 15 | `{"question": "allergies", "filters": {"section": "Diet"}}` | 400 `INVALID_INPUT`, lists the valid sections |
| 16 | `{"question": "allergies", "filters": {"patient_id": "P 1; drop"}}` | 400 `INVALID_INPUT` |
| 17 | any query **before** running ingest (fresh `CHROMA_PATH`) | 409 `COLLECTION_EMPTY` |
| 18 | Copy `tests/fixtures/pii/P-9001.md` into `samples/records/`, then ingest | `files_rejected: [{"file_name": "P-9001.md", "reason": "PII detected: email"}]`. Delete the copy afterwards. |
