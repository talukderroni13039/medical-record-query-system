from pathlib import Path

import pytest

from src.utils.pii_guard import detect_pii

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reach me at jane.roe+test@mail.example.org today", ["email"]),
        ("Call (555) 123-4567 after 5pm", ["phone"]),
        ("Call 555-123-4567", ["phone"]),
        ("Call +1 555.123.4567", ["phone"]),
        ("Call 5551234567", ["phone"]),
        ("SSN 123-45-6789", ["ssn"]),
        ("mail a@b.co or 555 123 4567, SSN 987-65-4321", ["email", "phone", "ssn"]),
    ],
)
def test_identifiers_detected(text: str, expected: list[str]) -> None:
    assert detect_pii(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Record date: 2025-03-14",
        "Blood pressure 152/96 mmHg, eGFR 41 mL/min/1.73m2",
        "HbA1c 8.4%, DEXA T-score -2.8, 30 pack-years",
        "Colchicine 0.6 mg twice daily; CHA2DS2-VASc score 3",
    ],
)
def test_clinical_text_is_clean(text: str) -> None:
    assert detect_pii(text) == []


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [("P-9001.md", ["email"]), ("P-9002.md", ["phone"]), ("P-9003.md", ["ssn"])],
)
def test_pii_fixture_files_detected(file_name: str, expected: list[str]) -> None:
    assert detect_pii((FIXTURES / "pii" / file_name).read_text(encoding="utf-8")) == expected


def test_clean_fixture_and_all_generated_records_pass() -> None:
    files = [FIXTURES / "records" / "P-0101.md"]
    files += sorted((Path(__file__).parent.parent.parent / "data" / "records").glob("*.md"))
    for path in files:
        assert detect_pii(path.read_text(encoding="utf-8")) == [], path.name
