"""
Batch PDF → Markdown conversion.

Usage:
    python scripts/convert_pdfs.py [--pdf PATH] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PDFS_DIR, PAPERS_DIR
from src.pdf_convert import pdf_to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDFs to markdown")
    parser.add_argument("--pdf", help="Convert a single PDF file")
    parser.add_argument("--out-dir", help="Output directory (default: data/papers/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else PAPERS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        pdf_files = [Path(args.pdf)]
    else:
        pdf_files = sorted(PDFS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {PDFS_DIR}")
        return

    for pdf_path in pdf_files:
        out_path = out_dir / (pdf_path.stem + ".md")
        print(f"Converting {pdf_path.name} → {out_path.name} ... ", end="", flush=True)
        try:
            md = pdf_to_markdown(pdf_path, out_path)
            print(f"ok ({len(md)} chars)")
        except Exception as exc:
            print(f"FAILED: {exc}")


if __name__ == "__main__":
    main()
