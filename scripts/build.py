"""
Full pipeline runner: PDF → Markdown → Graph → Resolve → Embed.

Usage:
    python scripts/build.py [--clear] [--skip-pdf] [--skip-ner] [--skip-embed]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the biomedical knowledge graph")
    parser.add_argument("--clear", action="store_true", help="Wipe graph before building")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF → Markdown conversion")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding step")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed chunks")
    parser.add_argument("--paper", help="Process only one paper (filename stem)")
    args = parser.parse_args()

    # Step 1: PDF conversion
    if not args.skip_pdf:
        print("\n=== Step 1: Converting PDFs ===")
        from src.config import PDFS_DIR
        from src.pdf_convert import pdf_to_markdown
        from src.config import PAPERS_DIR
        pdf_files = sorted(PDFS_DIR.glob("*.pdf"))
        for pdf in pdf_files:
            out = PAPERS_DIR / (pdf.stem + ".md")
            if out.exists():
                print(f"  Skipping {pdf.name} (already converted)")
                continue
            print(f"  Converting {pdf.name}...")
            pdf_to_markdown(pdf, out)

    # Step 2: Build graph
    print("\n=== Step 2: Building knowledge graph ===")
    from src.build_graph import build
    build(clear=args.clear, resume=args.resume, paper_id=args.paper)

    # Step 3: Entity resolution
    print("\n=== Step 3: Entity resolution ===")
    from src.graph import GraphDB
    from src.resolve import run_resolution
    from src.clean_graph import clean_graph
    with GraphDB() as db:
        run_resolution(db)
        cleanup_stats = clean_graph(db)
        print(f"  Cleanup: {cleanup_stats}")

    # Step 4: Embedding
    if not args.skip_embed:
        print("\n=== Step 4: Embedding graph nodes ===")
        from src.embed import embed_all_nodes
        with GraphDB() as db:
            n = embed_all_nodes(db, verbose=True)
        print(f"  Embedded {n} nodes.")

    print("\nPipeline complete. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
