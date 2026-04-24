"""
Encode graph nodes and chunks as sentence embeddings; store in Neo4j.

Model: pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb (768-dim)
"""
from __future__ import annotations

import functools
from typing import Any, TYPE_CHECKING

from tqdm import tqdm  # type: ignore

from src.config import EMBED_MODEL, EMBED_BATCH_SIZE, EMBED_MAX_TOKENS
from src.tokenize_graph import build_all_node_texts

if TYPE_CHECKING:
    from src.graph import GraphDB


@functools.lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer  # type: ignore
    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = EMBED_MAX_TOKENS
    return model


def encode_texts(texts: list[str]) -> list[list[float]]:
    """Encode a list of strings and return embeddings as list of float lists."""
    model = _load_model()
    embeddings = model.encode(texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False)
    return embeddings.tolist()


def embed_all_nodes(db: "GraphDB", verbose: bool = True) -> int:
    """Encode all graph nodes and write embeddings to Neo4j.

    Returns number of nodes embedded.
    """
    all_nodes = build_all_node_texts(db)
    if not all_nodes:
        return 0

    texts = [text for _, _, text, _ in all_nodes]
    if verbose:
        print(f"Encoding {len(texts)} nodes with {EMBED_MODEL}...")

    # Batch encode
    model = _load_model()
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=verbose,
        convert_to_numpy=True,
    )

    # Write to Neo4j
    written = 0
    for i, (label, identifier, _, node) in enumerate(tqdm(all_nodes, desc="Writing embeddings", disable=not verbose)):
        emb: list[float] = embeddings[i].tolist()
        is_chunk = label == "Chunk"
        db.set_embedding(label, identifier, emb, is_chunk=is_chunk)
        written += 1

    # Create vector indexes after all embeddings are written
    dim = embeddings.shape[1] if len(embeddings) > 0 else 768
    _ensure_vector_indexes(db, dim)

    return written


def _ensure_vector_indexes(db: "GraphDB", dim: int) -> None:
    """Create Neo4j vector indexes for Chunk and entity embeddings."""
    index_specs = [
        ("chunk_embedding_idx", "Chunk", "embedding"),
        ("gene_embedding_idx", "Gene", "embedding"),
        ("drug_embedding_idx", "Drug", "embedding"),
        ("disease_embedding_idx", "Disease", "embedding"),
    ]
    for idx_name, label, prop in index_specs:
        try:
            db.run(
                f"CREATE VECTOR INDEX {idx_name} IF NOT EXISTS "
                f"FOR (n:{label}) ON n.{prop} "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}"
            )
        except Exception:
            pass


def query_similar_chunks(db: "GraphDB", query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Return the top_k most semantically similar chunks to query."""
    model = _load_model()
    query_emb = model.encode([query])[0].tolist()
    return db.run(
        "CALL db.index.vector.queryNodes('chunk_embedding_idx', $k, $emb) "
        "YIELD node AS c, score "
        "RETURN c.id AS id, c.paper_id AS paper_id, c.section AS section, "
        "c.text AS text, score ORDER BY score DESC",
        k=top_k, emb=query_emb,
    )


def query_similar_entities(db: "GraphDB", query: str, label: str = "Gene", top_k: int = 5) -> list[dict[str, Any]]:
    """Return the top_k most semantically similar entities of a given label."""
    model = _load_model()
    query_emb = model.encode([query])[0].tolist()
    idx_map = {"Gene": "gene_embedding_idx", "Drug": "drug_embedding_idx", "Disease": "disease_embedding_idx"}
    index = idx_map.get(label, "gene_embedding_idx")
    return db.run(
        f"CALL db.index.vector.queryNodes('{index}', $k, $emb) "
        "YIELD node AS n, score "
        "RETURN n.name AS name, labels(n)[0] AS label, score ORDER BY score DESC",
        k=top_k, emb=query_emb,
    )
