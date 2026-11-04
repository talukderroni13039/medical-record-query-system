import hashlib
from pathlib import Path

import pytest

from src.utils.file_loader import decode_text, list_record_files, load_file


def test_list_record_files_filters_and_sorts(tmp_path: Path) -> None:
    for name in ["b.md", "a.txt", "c.pdf", "notes.MD"]:
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "sub.md").mkdir()
    assert [p.name for p in list_record_files(tmp_path)] == ["a.txt", "b.md", "notes.MD"]


def test_list_record_files_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_record_files(tmp_path / "missing")
    file_path = tmp_path / "f.md"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        list_record_files(file_path)


def test_load_file_hash_and_text(tmp_path: Path) -> None:
    path = tmp_path / "P-1.md"
    path.write_bytes("# Patient Record: P-1\nCafé".encode())
    loaded = load_file(path)
    assert loaded.file_name == "P-1.md"
    assert loaded.text.endswith("Café")
    assert loaded.content_hash == hashlib.sha256(path.read_bytes()).hexdigest()


def test_decode_text_fallbacks() -> None:
    assert decode_text("﻿hello".encode()) == "hello"  # BOM stripped
    assert decode_text(b"caf\xe9") == "café"  # latin-1 fallback
