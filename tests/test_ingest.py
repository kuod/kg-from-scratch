"""Tests for the markdown ingestion module."""
from pathlib import Path
import tempfile
import pytest

from src.ingest import load_paper, _sliding_window, _split_into_sections, _clean_header


SAMPLE_MD = """\
# Test Paper

## Introduction

This is the introduction section with some content about biomedical topics.
It has multiple sentences and discusses various findings in the field.

## Methods

**Cell culture**

HEK293T cells were cultured in DMEM supplemented with 10% FBS.

## Results

The western blot results showed significant upregulation of TP53.
MDM2 expression was reduced following treatment with Nutlin-3.

## Discussion

These findings suggest a novel mechanism of p53 regulation.
"""


def _write_temp_md(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8")
    f.write(content)
    f.close()
    return Path(f.name)


def test_load_paper_returns_chunks():
    path = _write_temp_md(SAMPLE_MD)
    chunks = load_paper(path)
    assert len(chunks) >= 3
    sections = [c.section for c in chunks]
    assert any("Introduction" in s for s in sections)
    assert any("Methods" in s for s in sections)
    assert any("Results" in s for s in sections)


def test_chunk_has_correct_paper_id():
    path = _write_temp_md(SAMPLE_MD)
    chunks = load_paper(path)
    for chunk in chunks:
        assert chunk.paper_id == path.stem


def test_chunk_word_count_computed():
    path = _write_temp_md(SAMPLE_MD)
    chunks = load_paper(path)
    for chunk in chunks:
        assert chunk.word_count == len(chunk.text.split())


def test_bold_header_detected():
    md = "# Paper\n\n**Cell culture**\n\nSome text about cells."
    path = _write_temp_md(md)
    chunks = load_paper(path)
    sections = [c.section for c in chunks]
    assert any("Cell culture" in s for s in sections)


def test_sliding_window_produces_overlapping_windows():
    words = list(range(1000))
    windows = list(_sliding_window([str(w) for w in words]))
    assert len(windows) > 1
    # Check overlap: last N words of window[0] == first N words of window[1]
    w0_words = windows[0].split()
    w1_words = windows[1].split()
    from src.config import CHUNK_OVERLAP_WORDS
    assert w0_words[-CHUNK_OVERLAP_WORDS:] == w1_words[:CHUNK_OVERLAP_WORDS]


def test_clean_header_strips_atx():
    assert _clean_header("## Results and Discussion") == "Results and Discussion"
    assert _clean_header("### Methods") == "Methods"


def test_clean_header_strips_bold():
    assert _clean_header("**Cell culture**") == "Cell culture"


def test_empty_sections_skipped():
    md = "# Paper\n\n## Empty\n\n## Has Content\n\nSome text here."
    path = _write_temp_md(md)
    chunks = load_paper(path)
    sections = [c.section for c in chunks]
    assert "Empty" not in sections
    assert "Has Content" in sections
