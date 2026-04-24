"""
Build structured text representations of graph nodes for embedding.

Each node type has a template that incorporates N-hop neighborhood context
so the resulting embedding captures functional relationships, not just the name.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from src.config import NEIGHBOR_CONTEXT_LIMIT

if TYPE_CHECKING:
    from src.graph import GraphDB

TEMPLATES: dict[str, str] = {
    "Gene": "Gene {name}. Also known as: {aliases}. {relationships}",
    "Protein": "Protein {name}. Also known as: {aliases}. {relationships}",
    "Drug": "Drug {name}. Also known as: {aliases}. {relationships}",
    "Disease": "Disease {name}. Also known as: {aliases}. {relationships}",
    "Pathway": "Biological pathway {name}. Also known as: {aliases}. {relationships}",
    "CellType": "Cell type {name}. Also known as: {aliases}. {relationships}",
    "Organism": "Organism {name}. Also known as: {aliases}. {relationships}",
    "Mechanism": "Biological mechanism {name}. Also known as: {aliases}. {relationships}",
    "Chunk": "{section} (from paper {paper_id}): {text}",
}


def build_node_text(node: dict[str, Any], label: str, db: "GraphDB") -> str:
    """Render a node's structured text representation including neighborhood context."""
    if label == "Chunk":
        text = node.get("text", "")
        # Truncate to ~512 tokens (~1800 chars as proxy)
        return TEMPLATES["Chunk"].format(
            section=node.get("section", ""),
            paper_id=node.get("paper_id", ""),
            text=text[:1800],
        )

    name = node.get("name", "")
    aliases = node.get("aliases", [])
    alias_str = ", ".join(aliases[:5]) if aliases else "none"
    rel_str = _build_relationship_context(label, name, db)

    template = TEMPLATES.get(label, TEMPLATES["Mechanism"])
    return template.format(name=name, aliases=alias_str, relationships=rel_str)


def _build_relationship_context(label: str, name: str, db: "GraphDB") -> str:
    """Fetch N-hop relationships and format as prose for the embedding context."""
    rows = db.run(
        f"MATCH (n:{label} {{name: $name}})-[r]->(m) "
        "RETURN type(r) AS rel, labels(m)[0] AS tgt_label, m.name AS tgt_name "
        f"LIMIT {NEIGHBOR_CONTEXT_LIMIT * 2}",
        name=name,
    )
    incoming = db.run(
        f"MATCH (n:{label} {{name: $name}})<-[r]-(m) "
        "RETURN type(r) AS rel, labels(m)[0] AS tgt_label, m.name AS tgt_name "
        f"LIMIT {NEIGHBOR_CONTEXT_LIMIT}",
        name=name,
    )

    parts: list[str] = []
    for row in rows[:NEIGHBOR_CONTEXT_LIMIT]:
        parts.append(f"{_humanize_rel(row['rel'])} {row['tgt_name']} ({row['tgt_label']})")
    for row in incoming[:NEIGHBOR_CONTEXT_LIMIT]:
        parts.append(f"{row['tgt_name']} ({row['tgt_label']}) {_humanize_rel(row['rel'])} this")

    return ". ".join(parts) + "." if parts else "No known relationships in graph."


def _humanize_rel(rel_type: str) -> str:
    mapping = {
        "REGULATES": "regulates",
        "INHIBITS": "inhibits",
        "ACTIVATES": "activates",
        "TARGETS": "targets",
        "BINDS": "binds to",
        "ASSOCIATED_WITH": "is associated with",
        "PROMOTES": "promotes",
        "SUPPRESSES": "suppresses",
        "INVOLVES": "involves",
        "EXPRESSED_IN": "is expressed in",
        "MUTATED_IN": "is mutated in",
        "MENTIONED_IN": "mentioned in",
        "FROM_PAPER": "from paper",
    }
    return mapping.get(rel_type, rel_type.lower().replace("_", " "))


def build_all_node_texts(db: "GraphDB") -> list[tuple[str, str, str, dict[str, Any]]]:
    """Yield (label, identifier, text, node) for all nodes that need embedding.

    identifier is node.name for entities or node.id for chunks.
    """
    from src.graph import ENTITY_LABELS

    results = []
    for label in ENTITY_LABELS:
        nodes = db.run(f"MATCH (n:{label}) RETURN n.name AS name, n.aliases AS aliases LIMIT 50000")
        for node in nodes:
            text = build_node_text(node, label, db)
            results.append((label, node["name"], text, node))

    chunks = db.run("MATCH (c:Chunk) RETURN c.id AS id, c.paper_id AS paper_id, c.section AS section, c.text AS text")
    for chunk in chunks:
        text = build_node_text(chunk, "Chunk", db)
        results.append(("Chunk", chunk["id"], text, chunk))

    return results
