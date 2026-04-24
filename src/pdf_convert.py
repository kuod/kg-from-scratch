"""
PDF → Markdown conversion.

Primary: pymupdf4llm (layout-aware, handles multi-column and tables)
Fallback: pdfminer.six (plain text extraction wrapped in minimal markdown)

Usage:
    from src.pdf_convert import pdf_to_markdown
    md = pdf_to_markdown(Path("data/pdfs/paper.pdf"))
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from src.config import PAPERS_DIR


def pdf_to_markdown(pdf_path: Path | str, out_path: Path | str | None = None) -> str:
    """Convert a PDF to clean markdown.

    Writes the result to out_path (defaults to data/papers/<stem>.md).
    Returns the markdown string.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    md = _try_pymupdf4llm(pdf_path)
    if len(md.strip()) < 200:
        md = _try_pdfminer(pdf_path)

    md = _post_process(md, pdf_path)

    dest = Path(out_path) if out_path else PAPERS_DIR / (pdf_path.stem + ".md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md, encoding="utf-8")
    return md


def _try_pymupdf4llm(pdf_path: Path) -> str:
    try:
        import pymupdf4llm  # type: ignore
        return pymupdf4llm.to_markdown(str(pdf_path))
    except Exception:
        return ""


def _try_pdfminer(pdf_path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        text = extract_text(str(pdf_path))
        lines = text.splitlines()
        md_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                md_lines.append("")
                continue
            # Heuristic: short ALL-CAPS lines are section headers
            if len(stripped) < 80 and stripped.isupper():
                md_lines.append(f"## {stripped.title()}")
            else:
                md_lines.append(stripped)
        return "\n".join(md_lines)
    except Exception as exc:
        raise RuntimeError(f"Both pdf converters failed for {pdf_path}") from exc


def _post_process(md: str, pdf_path: Path) -> str:
    md = _strip_repeated_headers_footers(md)
    md = _normalize_figure_captions(md)
    md = _collapse_blank_lines(md)
    md = _add_paper_title_header(md, pdf_path.stem)
    return md


def _strip_repeated_headers_footers(md: str) -> str:
    """Remove lines that appear verbatim on many pages (running headers/footers)."""
    lines = md.splitlines()
    counts = Counter(ln.strip() for ln in lines if ln.strip())
    # A line appearing > 3 times and shorter than 120 chars is likely a header/footer
    noise = {text for text, n in counts.items() if n > 3 and len(text) < 120}
    # Never remove section headers or long sentences
    noise = {t for t in noise if not t.startswith("#") and len(t.split()) < 15}
    return "\n".join(ln for ln in lines if ln.strip() not in noise)


def _normalize_figure_captions(md: str) -> str:
    """Rewrite 'Figure N:' / 'Fig. N.' lines as blockquotes."""
    pattern = re.compile(
        r"^(Fig(?:ure)?\.?\s*\d+[a-zA-Z]?[.:–\-]?\s*.{0,200})$",
        re.IGNORECASE | re.MULTILINE,
    )
    return pattern.sub(r"> \1", md)


def _collapse_blank_lines(md: str) -> str:
    """Replace 3+ consecutive blank lines with two blank lines."""
    return re.sub(r"\n{3,}", "\n\n", md)


def _add_paper_title_header(md: str, stem: str) -> str:
    """Prepend a top-level header with the filename stem if none exists."""
    stripped = md.lstrip()
    if stripped.startswith("# "):
        return md
    title = stem.replace("_", " ").replace("-", " ").title()
    return f"# {title}\n\n{md}"
