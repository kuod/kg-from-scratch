"""
Full pipeline orchestrator: markdown files → Neo4j knowledge graph.

Usage:
    python -m src.build_graph [--clear] [--resume] [--paper PAPER_ID]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from src.config import PAPERS_DIR
from src.graph import GraphDB
from src.ingest import load_paper, Chunk
from src.ner import extract_entities_ner
from src.extract import extract_from_chunk
from src.confidence import compute_edge_confidences


def build(
    papers_dir: Path = PAPERS_DIR,
    clear: bool = False,
    resume: bool = False,
    paper_id: str | None = None,
) -> None:
    md_files = sorted(papers_dir.glob("*.md"))
    if paper_id:
        md_files = [f for f in md_files if f.stem == paper_id]

    if not md_files:
        print("No markdown files found in", papers_dir)
        return

    with GraphDB() as db:
        db.setup_constraints()

        if clear:
            print("Clearing graph...")
            db.run("MATCH (n) DETACH DELETE n")

        for md_file in tqdm(md_files, desc="Papers"):
            _process_paper(md_file, db, resume=resume)

        print("Computing edge confidence intervals...")
        n_edges = compute_edge_confidences(db)
        print(f"Updated {n_edges} edges with Wilson CI bounds.")

    print("Done.")


def _process_paper(md_file: Path, db: GraphDB, resume: bool) -> None:
    paper_id = md_file.stem
    db.upsert_paper(paper_id, title=_extract_title(md_file))

    chunks = load_paper(md_file)
    for chunk in tqdm(chunks, desc=paper_id, leave=False):
        chunk_id = f"{paper_id}__{chunk.chunk_index:04d}"

        if resume and _chunk_already_processed(db, chunk_id):
            continue

        db.upsert_chunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            section=chunk.section,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
        )

        ner_entities = extract_entities_ner(chunk.text)
        extraction = extract_from_chunk(chunk, ner_entities)

        # Upsert entities
        for ent in extraction.get("entities", []):
            db.upsert_entity(
                label=ent["type"],
                name=ent["name"],
                aliases=ent.get("aliases", []),
            )
            db.upsert_mention(ent["type"], ent["name"], chunk_id)

        # Upsert relationships
        for rel in extraction.get("relationships", []):
            db.upsert_entity(label=rel["source_type"], name=rel["source_name"])
            db.upsert_entity(label=rel["target_type"], name=rel["target_name"])
            db.upsert_relationship(
                src_label=rel["source_type"],
                src_name=rel["source_name"],
                rel_type=rel["relationship"],
                tgt_label=rel["target_type"],
                tgt_name=rel["target_name"],
                chunk_id=chunk_id,
            )


def _chunk_already_processed(db: GraphDB, chunk_id: str) -> bool:
    rows = db.run(
        "MATCH (c:Chunk {id: $id})-[:MENTIONED_IN|FROM_PAPER]-() RETURN count(*) AS cnt",
        id=chunk_id,
    )
    return bool(rows and rows[0]["cnt"] > 0)


def _extract_title(md_file: Path) -> str:
    """Read the first # heading from a markdown file as the paper title."""
    import re
    text = md_file.read_text(encoding="utf-8")
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else md_file.stem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Delete all graph data before building")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed chunks")
    parser.add_argument("--paper", help="Process only a single paper by ID (filename stem)")
    args = parser.parse_args()
    build(clear=args.clear, resume=args.resume, paper_id=args.paper)
