from pathlib import Path

from src.domains.patient_records.chunking.record_parser import (
    KNOWN_SECTIONS,
    UNKNOWN,
    canonical_section,
    parse_record,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "records"


def test_header_fields_extracted() -> None:
    record = parse_record((FIXTURES / "P-0101.md").read_text(encoding="utf-8"), "fallback")
    assert record.patient_id == "P-0101"
    assert record.age_band == "40-49"
    assert record.sex == "F"
    assert record.record_date == "2025-03-14"


def test_sections_split_correctly() -> None:
    record = parse_record((FIXTURES / "P-0101.md").read_text(encoding="utf-8"), "fallback")
    assert [s.name for s in record.sections] == list(KNOWN_SECTIONS)
    allergies = next(s for s in record.sections if s.name == "Allergies")
    assert allergies.text == "Penicillin (rash)."
    meds = next(s for s in record.sections if s.name == "Medications")
    assert "Lisinopril 10 mg" in meds.text
    assert "##" not in meds.text


def test_missing_header_fields_become_unknown() -> None:
    record = parse_record((FIXTURES / "P-0102.txt").read_text(encoding="utf-8"), "fallback")
    assert record.patient_id == "P-0102"
    assert record.age_band == "60-69"
    assert record.sex == UNKNOWN
    assert record.record_date == UNKNOWN
    assert [s.name for s in record.sections] == ["Chief Complaint", "Medications", "Allergies"]


def test_missing_title_uses_fallback_and_empty_sections_skipped() -> None:
    text = (
        "Some preamble\r\n\r\n## allergies\r\nLatex.\r\n\r\n"
        "## Plan\r\n\r\n## Social History\r\nSmoker."
    )
    record = parse_record(text, "P-0999")
    assert record.patient_id == "P-0999"
    assert record.age_band == UNKNOWN
    # Known names are canonicalised; unknown names are kept as written; empty "Plan" is skipped.
    assert [(s.name, s.text) for s in record.sections] == [
        ("Allergies", "Latex."),
        ("Social History", "Smoker."),
    ]


def test_canonical_section() -> None:
    assert canonical_section("lab   results") == "Lab Results"
    assert canonical_section("ALLERGIES") == "Allergies"
    assert canonical_section("Diet") is None


def test_all_generated_records_parse_with_full_header() -> None:
    records_dir = Path(__file__).parent.parent.parent / "data" / "records"
    files = sorted(records_dir.glob("*.md"))
    assert len(files) >= 25
    for path in files:
        record = parse_record(path.read_text(encoding="utf-8"), path.stem)
        assert record.patient_id == path.stem
        assert UNKNOWN not in (record.age_band, record.sex, record.record_date)
        assert [s.name for s in record.sections] == list(KNOWN_SECTIONS)
