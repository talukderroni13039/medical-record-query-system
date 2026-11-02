"""Metric logic of scripts/evaluate_retrieval.py and golden-set integrity (no model needed)."""

import json
from pathlib import Path

import pytest

from scripts.evaluate_retrieval import CachingEmbedder, is_hit, score, to_markdown
from src.domains.patient_records.chunking.record_parser import KNOWN_SECTIONS
from tests.fakes import FakeEmbedder

ROOT = Path(__file__).parent.parent.parent
ITEM = {"expected_patient_ids": ["P-1", "P-2"], "expected_section": "Allergies"}


def test_is_hit_requires_patient_and_section() -> None:
    assert is_hit({"patient_id": "P-1", "section": "Allergies"}, ITEM)
    assert not is_hit({"patient_id": "P-1", "section": "Plan"}, ITEM)
    assert not is_hit({"patient_id": "P-9", "section": "Allergies"}, ITEM)
    assert is_hit({"patient_id": "P-2", "section": "Plan"}, {**ITEM, "expected_section": None})


@pytest.mark.parametrize(
    ("ranked_patients", "expected"),
    [
        (["P-1", "P-9"], (True, True, 1.0)),
        (["P-9", "P-9", "P-2"], (False, True, pytest.approx(1 / 3))),
        (["P-9"] * 5 + ["P-1"], (False, False, pytest.approx(1 / 6))),
        (["P-9"] * 3, (False, False, 0.0)),
    ],
)
def test_score(ranked_patients: list[str], expected: tuple[bool, bool, float]) -> None:
    ranked = [{"patient_id": p, "section": "Allergies"} for p in ranked_patients]
    assert score(ranked, ITEM) == expected


def test_caching_embedder_embeds_each_text_once() -> None:
    inner = FakeEmbedder()
    cached = CachingEmbedder(inner)
    first = cached.embed_documents(["a b", "c d", "a b"])
    second = cached.embed_documents(["c d"])
    assert first[0] == first[2]
    assert second[0] == first[1]
    assert inner.embedded_texts == ["a b", "c d"]
    cached.use_instruction = False
    assert cached.embed_query("a b") == first[0]


def test_markdown_table_has_row_per_config() -> None:
    row = {
        "config": {
            "name": "x",
            "chunk_size": 1,
            "chunk_overlap": 0,
            "section_prefix": True,
            "query_instruction": False,
        },
        "chunks": 3,
        "hit@1": 0.5,
        "hit@5": 1.0,
        "mrr": 0.75,
        "misses@5": [],
        "query_p50_ms": 1.0,
        "query_p95_ms": 2.0,
    }
    table = to_markdown([row, row])
    assert table.count("\n| x |") == 2


def test_golden_set_is_consistent_with_records() -> None:
    golden = json.loads((ROOT / "data" / "eval" / "golden_set.json").read_text(encoding="utf-8"))
    questions = golden["questions"]
    assert 15 <= len(questions) <= 20
    assert len({q["id"] for q in questions}) == len(questions)
    records = {p.stem for p in (ROOT / "data" / "records").glob("*.md")}
    for q in questions:
        assert set(q["expected_patient_ids"]) <= records, q["id"]
        assert q["expected_section"] in (None, *KNOWN_SECTIONS), q["id"]
