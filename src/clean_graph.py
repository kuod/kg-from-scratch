"""
Post-extraction graph cleanup.

Removes low-quality nodes: very short names, purely numeric names,
stopword-only names, and nodes with no edges.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph import GraphDB

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "but",
    "for", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "we", "our", "they",
    "not", "also", "such", "both", "all", "many", "some", "than", "more",
    "most", "well", "only", "other", "further", "however", "therefore",
    "figure", "table", "fig", "et", "al", "supplementary", "data",
}


def clean_graph(db: "GraphDB") -> dict[str, int]:
    """Run all cleanup passes. Returns dict of removed node/rel counts."""
    stats: dict[str, int] = {}

    # 1. Remove isolated nodes (no edges at all)
    rows = db.run(
        "MATCH (n) WHERE NOT (n)--() AND NOT n:Paper AND NOT n:Chunk "
        "WITH n LIMIT 5000 "
        "DETACH DELETE n RETURN count(n) AS cnt"
    )
    stats["isolated_nodes"] = rows[0]["cnt"] if rows else 0

    # 2. Remove entities whose name is a single character or purely numeric
    rows = db.run(
        "MATCH (n) WHERE size(n.name) <= 1 OR n.name =~ '^[0-9.]+$' "
        "AND NOT n:Paper AND NOT n:Chunk "
        "WITH n LIMIT 2000 "
        "DETACH DELETE n RETURN count(n) AS cnt"
    )
    stats["trivial_name_nodes"] = rows[0]["cnt"] if rows else 0

    # 3. Remove entities whose name is only stopwords
    from src.graph import ENTITY_LABELS
    removed_stopword = 0
    for label in ENTITY_LABELS:
        nodes = db.run(f"MATCH (n:{label}) RETURN n.name AS name")
        for node in nodes:
            words = set(node["name"].lower().split())
            if words and words.issubset(_STOPWORDS):
                db.run(f"MATCH (n:{label} {{name: $name}}) DETACH DELETE n", name=node["name"])
                removed_stopword += 1
    stats["stopword_nodes"] = removed_stopword

    # 4. Remove self-loops
    rows = db.run(
        "MATCH (n)-[r]->(n) DELETE r RETURN count(r) AS cnt"
    )
    stats["self_loops"] = rows[0]["cnt"] if rows else 0

    return stats
