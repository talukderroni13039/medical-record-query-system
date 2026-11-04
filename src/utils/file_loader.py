"""Read `.md` / `.txt` record files and compute a content hash."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

ALLOWED_EXTENSIONS = frozenset({".md", ".txt"})


@dataclass(frozen=True)
class LoadedFile:
    file_name: str
    text: str
    content_hash: str


def list_record_files(directory: Path) -> list[Path]:
    """Sorted record files directly inside `directory`.

    Raises FileNotFoundError / NotADirectoryError / PermissionError (callers map these).
    """
    if not directory.exists():
        raise FileNotFoundError(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )


def decode_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def load_file(path: Path) -> LoadedFile:
    raw = path.read_bytes()
    return LoadedFile(
        file_name=path.name,
        text=decode_text(raw),
        content_hash=hashlib.sha256(raw).hexdigest(),
    )
