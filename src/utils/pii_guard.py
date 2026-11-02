"""Regex checks for obvious direct identifiers.

Reports which identifier types were found. It does not decide what to do about them (the
ingestion service does). It never returns or logs the matched values.
"""

import re

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    # 3-3-4 digit groups with separators (optional +1 / parenthesised area code) or 10 bare digits.
    # Does not match ISO dates (4-2-2), SSNs (3-2-4), or lab values.
    "phone": re.compile(
        r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\d)"
        r"|(?<![\d.-])\d{10}(?![\d.-])"
    ),
    "ssn": re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])"),
}


def detect_pii(text: str) -> list[str]:
    """Return the sorted identifier types present in `text` (empty list = clean)."""
    return sorted(name for name, pattern in PII_PATTERNS.items() if pattern.search(text))
