"""Sections → chunks.

Each `##` section is chunked on its own (sections are natural semantic units). Inside a section,
sentences are packed greedily up to `chunk_size` tokens, and consecutive chunks share up to
`chunk_overlap` tokens of trailing sentences. Each chunk's text is prefixed with `"{section}: "`
so the embedding carries section context.

Token counts use a dependency-free approximation (words + punctuation), which tracks WordPiece
token counts closely for this kind of clinical prose. See docs/DECISIONS.md.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.domains.patient_records.chunking.record_parser import RecordSection

TokenCounter = Callable[[str], int]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'-])")


@dataclass(frozen=True)
class Chunk:
    section: str
    text: str
    chunk_index: int


def count_tokens(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    """Lines (bullets) first, then sentence boundaries within each line."""
    sentences: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            sentences.extend(s.strip() for s in _SENTENCE_END_RE.split(line) if s.strip())
    return sentences


def _split_long(sentence: str, limit: int, counter: TokenCounter) -> list[str]:
    """Split a single over-long sentence on word boundaries into pieces of ≤ `limit` tokens."""
    pieces: list[str] = []
    current: list[str] = []
    tokens = 0
    for word in sentence.split():
        word_tokens = counter(word)
        if current and tokens + word_tokens > limit:
            pieces.append(" ".join(current))
            current, tokens = [], 0
        current.append(word)
        tokens += word_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_text(
    text: str, chunk_size: int, chunk_overlap: int, counter: TokenCounter = count_tokens
) -> list[str]:
    """Pack sentences into chunks of ≤ `chunk_size` tokens with sentence-level overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    units: list[str] = []
    for sentence in split_sentences(text):
        if counter(sentence) > chunk_size:
            units.extend(_split_long(sentence, chunk_size, counter))
        else:
            units.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = counter(unit)
        if current and current_tokens + unit_tokens > chunk_size:
            chunks.append(" ".join(current))
            # Carry trailing sentences (never the whole previous chunk) as overlap.
            overlap: list[str] = []
            overlap_tokens = 0
            for prev in reversed(current[1:]):
                prev_tokens = counter(prev)
                if overlap_tokens + prev_tokens > chunk_overlap:
                    break
                overlap.insert(0, prev)
                overlap_tokens += prev_tokens
            while overlap and overlap_tokens + unit_tokens > chunk_size:
                overlap_tokens -= counter(overlap.pop(0))
            current, current_tokens = overlap, overlap_tokens
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_sections(
    sections: Sequence[RecordSection],
    chunk_size: int = 256,
    chunk_overlap: int = 32,
    section_prefix: bool = True,
    counter: TokenCounter = count_tokens,
) -> list[Chunk]:
    """Chunk every section. `chunk_index` is sequential across the whole record (0..n-1)."""
    chunks: list[Chunk] = []
    for section in sections:
        prefix = f"{section.name}: " if section_prefix else ""
        # Reserve room for the prefix so the final text stays within chunk_size.
        budget = max(chunk_size - counter(prefix), 8)
        overlap = min(chunk_overlap, budget - 1)
        for piece in split_text(section.text, budget, overlap, counter):
            chunks.append(
                Chunk(section=section.name, text=f"{prefix}{piece}", chunk_index=len(chunks))
            )
    return chunks
