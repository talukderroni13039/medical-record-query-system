import itertools

import pytest

from src.domains.patient_records.chunking.record_parser import RecordSection
from src.domains.patient_records.chunking.section_chunker import (
    chunk_sections,
    count_tokens,
    split_sentences,
    split_text,
)


def _sentences(n: int) -> str:
    # Each sentence is exactly 6 tokens: "Sentence", "<n>", "is", "about", "stuff", "."
    return " ".join(f"Sentence {i} is about stuff." for i in range(n))


def test_count_tokens_counts_words_and_punctuation() -> None:
    assert count_tokens("Penicillin (rash).") == 5
    assert count_tokens("") == 0


def test_split_sentences_handles_bullets_and_decimals() -> None:
    text = "- Colchicine 0.6 mg twice daily\n- Uric acid 9.2 mg/dL. Recheck soon."
    assert split_sentences(text) == [
        "- Colchicine 0.6 mg twice daily",
        "- Uric acid 9.2 mg/dL.",
        "Recheck soon.",
    ]


def test_short_section_is_one_chunk() -> None:
    chunks = chunk_sections([RecordSection("Allergies", "Penicillin (rash).")])
    assert len(chunks) == 1
    assert chunks[0].text == "Allergies: Penicillin (rash)."
    assert chunks[0].section == "Allergies"
    assert chunks[0].chunk_index == 0


def test_long_section_split_with_overlap() -> None:
    pieces = split_text(_sentences(10), chunk_size=20, chunk_overlap=6)
    assert len(pieces) > 1
    for piece in pieces:
        assert count_tokens(piece) <= 20
    # Consecutive chunks share the trailing sentence of the previous chunk.
    for prev, nxt in itertools.pairwise(pieces):
        last_sentence = prev.split(". ")[-1].rstrip(".") + "."
        assert nxt.startswith(last_sentence)
    # Every sentence appears somewhere.
    joined = " ".join(pieces)
    for i in range(10):
        assert f"Sentence {i} " in joined


def test_no_overlap_when_zero() -> None:
    pieces = split_text(_sentences(6), chunk_size=12, chunk_overlap=0)
    assert pieces == [
        "Sentence 0 is about stuff. Sentence 1 is about stuff.",
        "Sentence 2 is about stuff. Sentence 3 is about stuff.",
        "Sentence 4 is about stuff. Sentence 5 is about stuff.",
    ]


def test_over_long_sentence_split_on_words() -> None:
    long_sentence = " ".join(["word"] * 50)
    pieces = split_text(long_sentence, chunk_size=20, chunk_overlap=5)
    assert all(count_tokens(p) <= 20 for p in pieces)
    assert sum(count_tokens(p) for p in pieces) >= 50


def test_section_prefix_applied_and_within_budget() -> None:
    chunks = chunk_sections(
        [RecordSection("History of Present Illness", _sentences(20))],
        chunk_size=40,
        chunk_overlap=8,
    )
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.startswith("History of Present Illness: ")
        assert count_tokens(c.text) <= 40


def test_section_prefix_can_be_disabled() -> None:
    chunks = chunk_sections([RecordSection("Plan", "Recheck.")], section_prefix=False)
    assert chunks[0].text == "Recheck."


def test_chunk_index_sequential_across_sections() -> None:
    sections = [
        RecordSection("Chief Complaint", "Headache."),
        RecordSection("History of Present Illness", _sentences(20)),
        RecordSection("Allergies", "None."),
    ]
    chunks = chunk_sections(sections, chunk_size=30, chunk_overlap=6)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].section == "Chief Complaint"
    assert chunks[-1].section == "Allergies"


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_settings_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="chunk_"):
        split_text("Text.", chunk_size=size, chunk_overlap=overlap)


def test_empty_section_text_produces_no_chunks() -> None:
    assert chunk_sections([RecordSection("Plan", "   \n  ")]) == []
