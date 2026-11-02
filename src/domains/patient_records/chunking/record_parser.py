"""Record header → metadata; record body → `##` sections."""

import re
from dataclasses import dataclass

KNOWN_SECTIONS: tuple[str, ...] = (
    "Chief Complaint",
    "History of Present Illness",
    "Past Medical History",
    "Medications",
    "Allergies",
    "Lab Results",
    "Assessment",
    "Plan",
)
_CANONICAL = {s.lower(): s for s in KNOWN_SECTIONS}

UNKNOWN = "unknown"

_TITLE_RE = re.compile(r"^#\s+Patient Record:\s*(?P<pid>\S+)\s*$", re.IGNORECASE | re.MULTILINE)
_HEADER_FIELD_RE = re.compile(
    r"^\s*[-*]\s*(?P<key>Age band|Sex|Record date)\s*:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SECTION_RE = re.compile(r"^##\s+(?P<name>.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class RecordSection:
    name: str
    text: str


@dataclass(frozen=True)
class ParsedRecord:
    patient_id: str
    age_band: str
    sex: str
    record_date: str
    sections: list[RecordSection]


def canonical_section(name: str) -> str | None:
    """Case-insensitive match against KNOWN_SECTIONS. Returns the canonical spelling or None."""
    return _CANONICAL.get(" ".join(name.split()).lower())


def parse_record(text: str, fallback_patient_id: str) -> ParsedRecord:
    """Parse one record. Missing header fields become `"unknown"`, and a missing title falls back
    to `fallback_patient_id` (usually the file stem)."""
    text = text.replace("\r\n", "\n")
    first_section = _SECTION_RE.search(text)
    header = text[: first_section.start()] if first_section else text

    title = _TITLE_RE.search(header)
    fields = {m["key"].lower(): m["value"] for m in _HEADER_FIELD_RE.finditer(header)}

    sections: list[RecordSection] = []
    matches = list(_SECTION_RE.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if not body:
            continue
        raw_name = match["name"]
        sections.append(RecordSection(name=canonical_section(raw_name) or raw_name, text=body))

    return ParsedRecord(
        patient_id=title["pid"] if title else fallback_patient_id,
        age_band=fields.get("age band", UNKNOWN),
        sex=fields.get("sex", UNKNOWN),
        record_date=fields.get("record date", UNKNOWN),
        sections=sections,
    )
