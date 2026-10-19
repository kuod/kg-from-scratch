"""
Neo4j driver wrapper and graph schema helpers.

Provides the GraphDB context manager and all upsert operations.
All write operations are idempotent (MERGE-based).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from neo4j import GraphDatabase, Driver  # type: ignore

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

ENTITY_LABELS = ["Gene", "Protein", "Drug", "Disease", "Pathway", "CellType", "Organism", "Mechanism"]

RELATIONSHIP_TYPES = [
    "REGULATES", "INHIBITS", "ACTIVATES", "TARGETS", "BINDS",
    "ASSOCIATED_WITH", "PROMOTES", "SUPPRESSES", "INVOLVES",
    "EXPRESSED_IN", "MUTATED_IN",
]


def check_label(label: str) -> str:
    """Return label if it is a known entity label, else raise ValueError.

    Labels and relationship types cannot be passed as Cypher parameters, so they
    are interpolated into queries; they must therefore come from an allow-list.
    """
    if label not in ENTITY_LABELS:
        raise ValueError(f"Unknown entity label: {label!r}")
    return label


def check_rel_type(rel_type: str) -> str:
    if rel_type not in RELATIONSHIP_TYPES:
        raise ValueError(f"Unknown relationship type: {rel_type!r}")
    return rel_type


class GraphDB:
    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ) -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "GraphDB":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------ schema

    def setup_constraints(self) -> None:
        """Create uniqueness constraints and vector indexes (idempotent)."""
        # Paper
        self.run("CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
        # Chunk
        self.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        # Entity labels
        for label in ENTITY_LABELS:
            self.run(
                f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
            )

    def setup_vector_indexes(self, dim: int = 768) -> None:
        """Create vector indexes for Chunk and entity embeddings."""
        indexes = [("chunk_embedding_idx", "Chunk"), ("entity_embedding_idx", "Gene")]
        for idx_name, label in indexes:
            try:
                self.run(
                    f"CREATE VECTOR INDEX {idx_name} IF NOT EXISTS "
                    f"FOR (n:{label}) ON n.embedding "
                    f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}"
                )
            except Exception:
                pass  # index may already exist with different config

    # ------------------------------------------------------------------ upserts

    def upsert_paper(self, paper_id: str, title: str = "", doi: str = "", year: int | None = None) -> None:
        self.run(
            "MERGE (p:Paper {id: $id}) "
            "ON CREATE SET p.title=$title, p.doi=$doi, p.year=$year "
            "ON MATCH SET p.title=CASE WHEN $title <> '' THEN $title ELSE p.title END",
            id=paper_id, title=title, doi=doi, year=year,
        )

    def upsert_chunk(self, chunk_id: str, paper_id: str, section: str, chunk_index: int, text: str) -> None:
        self.run(
            "MERGE (c:Chunk {id: $id}) "
            "SET c.paper_id=$paper_id, c.section=$section, c.chunk_index=$chunk_index, c.text=$text "
            "WITH c "
            "MATCH (p:Paper {id: $paper_id}) "
            "MERGE (c)-[:FROM_PAPER]->(p)",
            id=chunk_id, paper_id=paper_id, section=section, chunk_index=chunk_index, text=text,
        )

    def mark_chunk_processed(self, chunk_id: str) -> None:
        """Flag a chunk once its entities/relationships have been fully written."""
        self.run("MATCH (c:Chunk {id: $id}) SET c.processed=true", id=chunk_id)

    def is_chunk_processed(self, chunk_id: str) -> bool:
        rows = self.run(
            "MATCH (c:Chunk {id: $id}) RETURN coalesce(c.processed, false) AS done", id=chunk_id
        )
        return bool(rows and rows[0]["done"])

    def upsert_entity(
        self,
        label: str,
        name: str,
        aliases: list[str] | None = None,
        umls_cui: str | None = None,
        **extra_props: Any,
    ) -> None:
        if label not in ENTITY_LABELS:
            label = "Mechanism"
        props: dict[str, Any] = {"name": name, "aliases": aliases or []}
        if umls_cui:
            props["umls_cui"] = umls_cui
        props.update(extra_props)
        # Aliases are merged (union) rather than overwritten, so a later call
        # without aliases (e.g. relationship endpoints) never wipes them.
        sets = [
            "n.aliases = reduce(acc = [], x IN coalesce(n.aliases, []) + $aliases | "
            "CASE WHEN x IN acc THEN acc ELSE acc + [x] END)"
        ]
        sets += [f"n.{k}=${k}" for k in props if k not in ("name", "aliases")]
        self.run(
            f"MERGE (n:{label} {{name: $name}}) SET {', '.join(sets)}",
            **props,
        )

    def upsert_mention(self, entity_label: str, entity_name: str, chunk_id: str) -> None:
        """Link an entity to the chunk that mentions it."""
        check_label(entity_label)
        self.run(
            f"MATCH (e:{entity_label} {{name: $name}}) "
            "MATCH (c:Chunk {id: $chunk_id}) "
            "MERGE (e)-[r:MENTIONED_IN]->(c) "
            "ON CREATE SET r.count=1",
            name=entity_name, chunk_id=chunk_id,
        )

    def upsert_relationship(
        self,
        src_label: str,
        src_name: str,
        rel_type: str,
        tgt_label: str,
        tgt_name: str,
        chunk_id: str,
        **props: Any,
    ) -> None:
        """Upsert a relationship between two entities, tracking evidence."""
        if rel_type not in RELATIONSHIP_TYPES:
            rel_type = "ASSOCIATED_WITH"
        check_label(src_label)
        check_label(tgt_label)
        self.run(
            f"MATCH (s:{src_label} {{name: $src}}) "
            f"MATCH (t:{tgt_label} {{name: $tgt}}) "
            f"MERGE (s)-[r:{rel_type}]->(t) "
            "ON CREATE SET r.evidence_count=1, r.source_chunk_ids=[$chunk_id] "
            "ON MATCH SET "
            "  r.evidence_count=CASE WHEN $chunk_id IN r.source_chunk_ids "
            "    THEN r.evidence_count ELSE r.evidence_count+1 END, "
            "  r.source_chunk_ids=CASE WHEN NOT $chunk_id IN r.source_chunk_ids "
            "    THEN r.source_chunk_ids + [$chunk_id] "
            "    ELSE r.source_chunk_ids END",
            src=src_name, tgt=tgt_name, chunk_id=chunk_id, **props,
        )

    def upsert_relationship_with_confidence(
        self,
        src_label: str,
        src_name: str,
        rel_type: str,
        tgt_label: str,
        tgt_name: str,
        confidence_lower: float,
        confidence_upper: float,
        evidence_count: int,
        source_chunk_ids: list[str],
    ) -> None:
        if rel_type not in RELATIONSHIP_TYPES:
            rel_type = "ASSOCIATED_WITH"
        check_label(src_label)
        check_label(tgt_label)
        self.run(
            f"MATCH (s:{src_label} {{name: $src}}) "
            f"MATCH (t:{tgt_label} {{name: $tgt}}) "
            f"MERGE (s)-[r:{rel_type}]->(t) "
            "SET r.confidence_lower=$cl, r.confidence_upper=$cu, "
            "    r.evidence_count=$ec, r.source_chunk_ids=$ids",
            src=src_name, tgt=tgt_name,
            cl=confidence_lower, cu=confidence_upper,
            ec=evidence_count, ids=source_chunk_ids,
        )

    def set_embedding(self, label: str, name_or_id: str, embedding: list[float], is_chunk: bool = False) -> None:
        if is_chunk:
            self.run("MATCH (c:Chunk {id: $id}) SET c.embedding=$emb", id=name_or_id, emb=embedding)
        else:
            check_label(label)
            self.run(f"MATCH (n:{label} {{name: $name}}) SET n.embedding=$emb", name=name_or_id, emb=embedding)

    # ------------------------------------------------------------------ queries

    def get_entity_relationships(self, label: str, name: str) -> list[dict[str, Any]]:
        check_label(label)
        return self.run(
            f"MATCH (s:{label} {{name: $name}})-[r]->(t) "
            "RETURN type(r) AS rel_type, labels(t)[0] AS tgt_label, t.name AS tgt_name, "
            "r.evidence_count AS evidence_count, "
            "r.confidence_lower AS confidence_lower, r.confidence_upper AS confidence_upper "
            "UNION "
            f"MATCH (src)-[r]->(n:{label} {{name: $name}}) "
            "RETURN type(r) AS rel_type, labels(src)[0] AS tgt_label, src.name AS tgt_name, "
            "r.evidence_count AS evidence_count, "
            "r.confidence_lower AS confidence_lower, r.confidence_upper AS confidence_upper",
            name=name,
        )

    def keyword_search_chunks(self, keyword: str, limit: int = 10) -> list[dict[str, Any]]:
        return self.run(
            "MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS toLower($kw) "
            "RETURN c.id AS id, c.paper_id AS paper_id, c.section AS section, "
            "c.text AS text LIMIT $limit",
            kw=keyword, limit=limit,
        )

    def find_path(self, src_label: str, src_name: str, tgt_label: str, tgt_name: str, max_hops: int = 4) -> list[dict]:
        check_label(src_label)
        check_label(tgt_label)
        max_hops = max(1, min(int(max_hops), 6))
        return self.run(
            f"MATCH p=shortestPath((s:{src_label} {{name: $src}})-[*1..{max_hops}]-(t:{tgt_label} {{name: $tgt}})) "
            "RETURN [n IN nodes(p) | n.name] AS node_names, "
            "[r IN relationships(p) | type(r)] AS rel_types",
            src=src_name, tgt=tgt_name,
        )

    def get_edge_confidence(
        self, src_label: str, src_name: str, rel_type: str, tgt_label: str, tgt_name: str
    ) -> dict[str, Any]:
        check_label(src_label)
        check_label(tgt_label)
        check_rel_type(rel_type)
        rows = self.run(
            f"MATCH (s:{src_label} {{name: $src}})-[r:{rel_type}]->(t:{tgt_label} {{name: $tgt}}) "
            "RETURN r.confidence_lower AS lower, r.confidence_upper AS upper, "
            "r.evidence_count AS evidence_count",
            src=src_name, tgt=tgt_name,
        )
        return rows[0] if rows else {"lower": None, "upper": None, "evidence_count": 0}

    def get_graph_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        for label in ["Paper", "Chunk"] + ENTITY_LABELS:
            rows = self.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
            stats[label] = rows[0]["cnt"] if rows else 0
        edge_rows = self.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC")
        stats["relationships"] = {r["rel"]: r["cnt"] for r in edge_rows}
        return stats
