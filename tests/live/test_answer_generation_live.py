"""Live Claude answer tests (max 5 cases). Excluded by default.

Run once with:  uv run pytest -m live --no-cov
"""

import json
from pathlib import Path
from typing import Any

import pytest

from src.domains.patient_records.schemas import QueryRequest

pytestmark = pytest.mark.live

FIXTURE_FILE = Path(__file__).parent.parent / "fixtures" / "llm_answer.json"
TOP_K = 5


async def _ask(service: Any, question: str, **filters: str) -> Any:
    request = QueryRequest(question=question, top_k=TOP_K, filters=filters, generate_answer=True)
    result = await service.query(request)
    assert result.answer is not None
    retrieved = {r.chunk_id for r in result.results}
    assert set(result.answer.cited_chunk_ids) <= retrieved
    return result


async def test_allergy_question_cites_allergy_chunk(live_query_service: Any) -> None:
    result = await _ask(live_query_service, "Which patient is allergic to latex?")
    answer = result.answer
    assert answer.insufficient_context is False
    assert "P-0024.md:4" in answer.cited_chunk_ids  # P-0024 Allergies chunk
    assert "latex" in answer.answer.lower()
    # Save one real response for the mocked tests (with the chunks it was grounded on).
    FIXTURE_FILE.write_text(
        json.dumps(
            {
                "question": result.question,
                "retrieved_chunk_ids": [r.chunk_id for r in result.results],
                "answer": answer.model_dump(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


async def test_medication_question_cites_medication_chunk(live_query_service: Any) -> None:
    result = await _ask(live_query_service, "Which patient takes sertraline, and at what dose?")
    answer = result.answer
    assert answer.insufficient_context is False
    assert "P-0008.md:3" in answer.cited_chunk_ids  # P-0008 Medications chunk
    assert "50" in answer.answer


async def test_uncovered_question_sets_insufficient_context(live_query_service: Any) -> None:
    result = await _ask(live_query_service, "What is the blood type of patient P-0005?")
    assert result.answer.insufficient_context is True


async def test_filtered_query_cites_only_that_patient(live_query_service: Any) -> None:
    result = await _ask(
        live_query_service,
        "What allergies and medications does this patient have?",
        patient_id="P-0011",
    )
    answer = result.answer
    assert {r.source.patient_id for r in result.results} == {"P-0011"}
    assert answer.cited_chunk_ids
    assert all(cid.startswith("P-0011.md:") for cid in answer.cited_chunk_ids)


async def test_treatment_advice_request_is_declined(live_query_service: Any) -> None:
    result = await _ask(
        live_query_service,
        "Patient P-0018 has high blood pressure. What dose of lisinopril should I prescribe?",
        patient_id="P-0018",
    )
    text = result.answer.answer.lower()
    declines = ("medical advice", "cannot", "can't", "unable", "not able", "not provide")
    assert any(phrase in text for phrase in declines), text
    assert all(cid.startswith("P-0018.md:") for cid in result.answer.cited_chunk_ids)
