"""
Markdown → Chunk dataclasses.

Splits a paper's markdown into sections using ATX headers (##) and
standalone bold lines (**...**), then applies a sliding window for
sections longer than CHUNK_MAX_WORDS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.config import (
    CHUNK_MAX_WORDS,
    CHUNK_OVERLAP_WORDS,
    CHUNK_WINDOW_WORDS,
    PAPERS_DIR,
)


@dataclass
class Chunk:
    paper_id: str
    section: str
    chunk_index: int
    text: str
    char_offset: int
    word_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.word_count = len(self.text.split())


# Match ATX headers (## or deeper) and standalone **bold lines**
_HEADER_RE = re.compile(
    r"^(#{2,6}\s+.+|(?:\*\*[^*\n]+\*\*)\s*)$",
    re.MULTILINE,
)


def load_paper(path: Path | str) -> list[Chunk]:
    """Parse a markdown file into a list of Chunk objects."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    paper_id = path.stem
    return list(_chunk_paper(paper_id, text))


def load_all_papers(papers_dir: Path | str | None = None) -> list[Chunk]:
    """Load and chunk all .md files in papers_dir."""
    papers_dir = Path(papers_dir) if papers_dir else PAPERS_DIR
    chunks: list[Chunk] = []
    for md_file in sorted(papers_dir.glob("*.md")):
        chunks.extend(load_paper(md_file))
    return chunks


def _chunk_paper(paper_id: str, text: str) -> Iterator[Chunk]:
    sections = _split_into_sections(text)
    chunk_index = 0
    for section_title, section_text, char_offset in sections:
        words = section_text.split()
        if len(words) <= CHUNK_MAX_WORDS:
            yield Chunk(
                paper_id=paper_id,
                section=section_title,
                chunk_index=chunk_index,
                text=section_text.strip(),
                char_offset=char_offset,
            )
            chunk_index += 1
        else:
            for part_num, window_text in enumerate(_sliding_window(words)):
                yield Chunk(
                    paper_id=paper_id,
                    section=f"{section_title}:part_{part_num}",
                    chunk_index=chunk_index,
                    text=window_text.strip(),
                    char_offset=char_offset,
                )
                chunk_index += 1


def _split_into_sections(text: str) -> list[tuple[str, str, int]]:
    """Return list of (header, body, char_offset) tuples."""
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [("Introduction", text, 0)]

    sections: list[tuple[str, str, int]] = []
    # Text before the first header
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("Preamble", preamble, 0))

    for i, m in enumerate(matches):
        header = _clean_header(m.group(0))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((header, body, m.start()))

    return sections


def _clean_header(raw: str) -> str:
    """Strip markdown syntax from a header line."""
    h = raw.strip()
    h = re.sub(r"^#{2,6}\s+", "", h)
    h = re.sub(r"^\*\*(.+)\*\*$", r"\1", h)
    return h.strip()


def _sliding_window(words: list[str]) -> Iterator[str]:
    """Yield overlapping windows of CHUNK_WINDOW_WORDS words."""
    step = CHUNK_WINDOW_WORDS - CHUNK_OVERLAP_WORDS
    start = 0
    while start < len(words):
        window = words[start : start + CHUNK_WINDOW_WORDS]
        yield " ".join(window)
        if start + CHUNK_WINDOW_WORDS >= len(words):
            break
        start += step
