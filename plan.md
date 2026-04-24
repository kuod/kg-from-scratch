# Biomedical Knowledge Graph Pipeline — Plan

## Overview

This pipeline ingests markdown-format biomedical papers, constructs a persistent Neo4j knowledge graph annotated with biomedical entities and their relationships, embeds all nodes and text chunks for semantic retrieval, and exposes a Streamlit chatbot interface that answers user questions with grounded citations and calibrated confidence intervals. A preprocessing utility converts raw PDFs to markdown so the ingestion stage has a single, consistent input format.

---

## Directory Structure

```
kg-from-scratch/
├── .env.example                  # API keys and connection strings template
├── .gitignore
├── docker-compose.yml            # Neo4j 5.x with APOC + GDS plugins
├── pyproject.toml                # Python 3.11+ package manifest
├── requirements.txt              # Pinned dependencies
│
├── data/
│   ├── papers/                   # PRIMARY INPUT: drop .md files here
│   ├── pdfs/                     # Raw PDFs before conversion
│   └── cache/                    # LLM extraction cache (JSON, gitignored)
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Env vars, paths, model names
│   ├── pdf_convert.py            # PDF → Markdown converter
│   ├── ingest.py                 # Markdown → Chunk dataclasses
│   ├── ner.py                    # scispaCy NER + entity type mapping
│   ├── extract.py                # LLM-based relation extraction per chunk
│   ├── graph.py                  # Neo4j driver wrapper + schema helpers
│   ├── build_graph.py            # Orchestrator: ingest → NER → extract → graph
│   ├── resolve.py                # Entity resolution (synonyms + normalization)
│   ├── tokenize_graph.py         # Tokenize node/edge text for embedding
│   ├── embed.py                  # Encode chunks + nodes; store vectors in Neo4j
│   ├── agent.py                  # RAG agent with tool-use loop + confidence
│   ├── confidence.py             # Confidence interval computation module
│   └── clean_graph.py            # Remove noise nodes/rels post-extraction
│
├── app.py                        # Streamlit chatbot entry point
│
├── scripts/
│   ├── convert_pdfs.py           # Batch PDF → Markdown conversion
│   ├── build.py                  # Full pipeline runner (CLI)
│   └── eval.py                   # Evaluation harness
│
└── tests/
    ├── test_ingest.py
    ├── test_ner.py
    ├── test_extract.py
    ├── test_confidence.py
    └── test_agent.py
```

---

## Component Breakdown

### 1. `src/config.py`

Central configuration loaded from `.env`. Exposes:

- `PAPERS_DIR` — absolute path to `data/papers/`
- `PDFS_DIR` — absolute path to `data/pdfs/`
- `CACHE_DIR` — extraction cache directory
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `LLM_MODEL` — extraction model (default: `llama-3.3-70b-versatile` via Groq)
- `AGENT_MODEL` — QA model (default: `claude-sonnet-4-6`)
- `EMBED_MODEL` — sentence-transformer model name
- `SCISPACY_MODEL` — spaCy model name

### 2. `src/pdf_convert.py`

Converts PDFs to clean markdown. Wraps `pymupdf4llm` as primary with `pdfminer.six` fallback.

Key function:

```python
def pdf_to_markdown(pdf_path: Path, out_path: Path | None = None) -> str:
    """Convert a PDF to markdown. Returns the markdown string."""
```

Internal logic:
1. Attempt `pymupdf4llm.to_markdown(str(pdf_path))` — layout-aware markdown preserving tables.
2. If output < 200 characters or raises, fall back to `pdfminer.six`.
3. Post-process: strip running headers/footers via line-repetition detection, normalize figure captions to blockquotes, collapse excessive blank lines.
4. Write to `data/papers/<stem>.md` (or caller-specified `out_path`).

### 3. `src/ingest.py`

Parses markdown files into `Chunk` dataclasses:

```python
@dataclass
class Chunk:
    paper_id: str       # filename stem
    section: str        # ATX header or bold-line header text
    chunk_index: int    # ordinal within paper
    text: str           # section body
    char_offset: int    # byte offset for provenance
    word_count: int
```

Splitting strategy:
- Primary split on `##` ATX headers and `**Bold standalone lines**`.
- Secondary split: sections > 1,200 words get a 600-word sliding window with 150-word overlap.

### 4. `src/ner.py`

Fast offline biomedical NER using scispaCy `en_core_sci_lg` (UMLS semantic types) supplemented with the `en_ner_bc5cdr_md` linker for drug/disease UMLS CUIs.

**Why scispaCy over BioBERT NER:** scispaCy runs fully offline with no GPU requirement. BioBERT fine-tuned NER has higher precision but requires GPU for reasonable throughput. The NER pass is used as a hint-provider for the LLM extraction step, not as the sole authority.

### 5. `src/extract.py`

LLM-based relation extraction (Groq `llama-3.3-70b-versatile`) with NER hints injected into the prompt to reduce hallucination. Results are cached to `data/cache/<paper_id>__<chunk_index>.json`.

Entity types: `Gene`, `Protein`, `Drug`, `Disease`, `Pathway`, `CellType`, `Organism`, `Mechanism`.

Relationship types: `REGULATES`, `INHIBITS`, `ACTIVATES`, `TARGETS`, `BINDS`, `ASSOCIATED_WITH`, `PROMOTES`, `SUPPRESSES`, `INVOLVES`, `EXPRESSED_IN`, `MUTATED_IN`.

### 6. `src/graph.py`

Neo4j driver wrapper with:
- `GraphDB` context manager
- `setup_constraints()` — idempotent uniqueness constraints
- `upsert_paper()`, `upsert_chunk()`, `upsert_entity()`, `upsert_relationship()`
- `upsert_relationship_with_confidence()` — stores `evidence_count`, `source_chunk_ids`, `confidence_lower`, `confidence_upper` on edges

**Graph schema:**

Nodes: `Paper`, `Chunk`, `Gene`, `Protein`, `Drug`, `Disease`, `Pathway`, `CellType`, `Mechanism` — each with `name`, `aliases[]`, relevant ontology IDs, `embedding: float[]`.

Edges: carry `confidence_lower`, `confidence_upper`, `evidence_count`, `source_chunk_ids[]`.

**Why Neo4j over NetworkX or RDFLib:**
- Neo4j supports native vector indexes alongside property graph storage — no separate vector database needed.
- APOC enables node merging during entity resolution.
- Cypher is more reliable for LLM-generated tool calls than SPARQL.
- NetworkX has no persistent storage, no vector index, and no graph query language.

### 7. `src/build_graph.py`

Orchestrator that drives: ingest → NER → extract → graph upsert for all `.md` files in `data/papers/`.

Flags: `--clear`, `--resume` (skip already-processed chunks), `--paper <id>`.

### 8. `src/resolve.py`

Entity resolution in two passes:

**Pass 1 — Synonym merge:** Curated `SYNONYM_GROUPS` list maps aliases to canonical names (e.g. `p53 → TP53`, `HER2 → ERBB2`). Uses APOC `refactor.mergeNodes`.

**Pass 2 — Normalization merge:** Nodes whose normalized names (lowercase, stripped punctuation) are identical are merged. The node with the highest `MENTIONED_IN` edge count survives as canonical.

String normalization only — no embedding similarity for entity-to-entity merging, because biomedically distinct entities (e.g. `CXCL1`/`CXCL2`) can be near-identical to sentence-transformer models.

### 9. `src/tokenize_graph.py`

Builds a structured text representation per node for embedding:

```python
TEMPLATES = {
    "Gene": "Gene {name}. Also known as: {aliases}. Involved in: {relationships}.",
    "Drug": "Drug {name} (also: {aliases}). Targets: {targets}. Inhibits: {inhibits}.",
    "Disease": "Disease {name}. Associated genes: {genes}. Pathways: {pathways}.",
    "Pathway": "Pathway {name}. Key members: {members}. Regulated by: {regulators}.",
    "Chunk": "{section} from {paper_id}: {text[:512]}",
}
```

Up to 5 neighbors per direction are fetched from Neo4j and appended so the embedding captures graph neighborhood context.

**Why not raw name tokenization:** Raw name encoding (just "TP53") gives poor retrieval recall. Including neighborhood context yields embeddings that cluster functionally related nodes correctly.

### 10. `src/embed.py`

Encodes chunks and entity nodes using `pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb` (768-dim).

**Why PubMedBERT over all-MiniLM-L6-v2:** PubMedBERT fine-tuned on biomedical STS datasets clusters biomedical synonyms and functionally related terms far better. `all-MiniLM-L6-v2` conflates biomedically distinct but linguistically similar terms.

Embeddings stored as `float[]` on Neo4j nodes; vector indexes created for `Chunk.embedding` and entity nodes.

### 11. `src/confidence.py`

**Two-level confidence system:**

**Level 1 — Edge-level structural confidence (build time):**

Wilson score interval on evidence count:

```python
def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    k: chunks asserting this relationship
    n: total chunks mentioning both endpoint entities
    Returns (lower, upper) 95% CI on the evidence proportion.
    """
```

**Level 2 — Answer-level epistemic confidence (query time):**

LLM-elicited CI via a calibration call presenting the question, answer, and source passages. The model returns `{"score": 0-100, "label": "High|Medium|Low", "lower_bound": 0.0-1.0, "upper_bound": 0.0-1.0, "rationale": "...", "evidence_count": N, "contradictions": "..."}`.

Optional enhancement: log-probability CI behind `USE_LOGPROB_CI` flag (requires provider support).

**Why not Monte Carlo:** N=30 inference calls is cost-prohibitive for real-time chatbot. Log-prob CI is provider-dependent. LLM-elicited CI works across all providers and is interpretable to non-statisticians.

### 12. `src/agent.py`

Tool-using RAG agent with seven tools:

| Tool | Description |
|---|---|
| `rank_papers` | Rank papers by semantic similarity to query |
| `search_entity` | Keyword + alias search for entities by type |
| `get_entity_relationships` | All edges for a named entity with confidence bounds |
| `search_chunks` | Hybrid keyword + vector search with graph-walk expansion |
| `get_chunk_text` | Full text of a specific chunk |
| `find_path` | Shortest path between two named entities (Cypher shortestPath) |
| `get_edge_confidence` | Wilson CI bounds for a specific relationship |

Returns `(answer, chunk_ids, confidence_dict, tool_trace)`.

### 13. `app.py`

Streamlit application with four tabs:

1. **Ask (chatbot):** Chat input, tool-call streaming, answer + citations, confidence badge `[lower, upper]`, PyVis knowledge subgraph
2. **Graph Explorer:** Entity search → N-hop neighborhood, relationship table with CI column
3. **Papers:** Browse chunks per paper with entity type tags
4. **Graph Stats:** Node/edge counts by type, embedding index status

---

## Data Flow

```
[PDF files in data/pdfs/]
    │
    ▼ src/pdf_convert.py (pymupdf4llm + pdfminer.six fallback)
    │
[data/papers/*.md]
    │
    ▼ src/ingest.py  (ATX header split + sliding window for long sections)
    │
[list[Chunk]]
    ├──▶ src/ner.py  (scispaCy en_core_sci_lg + BC5CDR linker)
    │         → list[NEREntity]
    │
    └──▶ src/extract.py  (Groq llama-3.3-70b, NER hints, disk cache)
              → {entities, relationships}
    │
    ▼ src/graph.py  (Neo4j upsert Paper/Chunk/Entity/Relationship nodes+edges)
    │
    ▼ src/resolve.py  (synonym merge + normalization merge via APOC)
    │
    ▼ src/tokenize_graph.py  (structured text per node with N-hop context)
    │
    ▼ src/embed.py  (PubMedBERT encode + store float[] + create vector indexes)
    │
    ▼ src/confidence.py  (Wilson CI per edge at build time)
    │
    ─────────────── runtime ───────────────
    │
    ▼ src/agent.py  (tool-use loop, hybrid retrieval, answer + citations)
    │
    ▼ src/confidence.py  (LLM-elicited CI at query time)
    │
    ▼ app.py (Streamlit: answer + citations + confidence + PyVis subgraph)
```

---

## Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| **Graph database** | Neo4j 5.x (Docker) | Native vector index + property graph + APOC merge = no separate vector DB |
| **NER library** | scispaCy `en_core_sci_lg` + BC5CDR linker | Offline, fast, biomedical-trained, UMLS type system |
| **Relation extraction** | Groq `llama-3.3-70b-versatile` | Strong JSON output, fast, low cost, disk-cached |
| **Agent QA model** | `claude-sonnet-4-6` | Excels at citation-grounded biomedical reasoning |
| **Sentence embeddings** | `pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb` | 768-dim, fine-tuned on biomedical STS |
| **PDF conversion** | `pymupdf4llm` (primary), `pdfminer.six` (fallback) | Preserves multi-column layout and tables as structured markdown |
| **Confidence intervals** | Wilson CI (structural) + LLM-elicited CI (epistemic) | Wilson CI is mathematically grounded; LLM CI works across all providers |
| **Visualization** | PyVis | Force-directed interactive graph in Streamlit |

---

## Implementation Phases

### Phase 1 — Infrastructure (Day 1)
- `docker-compose.yml` with Neo4j 5.x + APOC + GDS
- `pyproject.toml`, `.env.example`, `.gitignore`
- `src/config.py`, `src/graph.py`

### Phase 2 — PDF Conversion (Day 1-2)
- `src/pdf_convert.py` with header/footer stripping
- `scripts/convert_pdfs.py` batch runner

### Phase 3 — Ingestion + NER (Day 2)
- `src/ingest.py` with sliding-window chunking
- `src/ner.py` with scispaCy + BC5CDR linker

### Phase 4 — LLM Extraction + Graph Population (Day 2-3)
- `src/extract.py` with NER-hint injection + disk cache
- `src/build_graph.py` orchestrator
- `src/clean_graph.py` noise removal

### Phase 5 — Entity Resolution (Day 3)
- `src/resolve.py` with APOC synonym merge + normalization merge

### Phase 6 — Tokenization + Embedding (Day 3-4)
- `src/tokenize_graph.py` with N-hop neighborhood templates
- `src/embed.py` with PubMedBERT + Neo4j vector indexes

### Phase 7 — Confidence Intervals (Day 4)
- `src/confidence.py` with Wilson CI + LLM-elicited CI

### Phase 8 — Agent (Day 4-5)
- `src/agent.py` with seven tools + `ask_with_confidence()`

### Phase 9 — Streamlit App (Day 5)
- `app.py` with four tabs: Ask, Graph Explorer, Papers, Stats

### Phase 10 — Evaluation + Polish (Day 5-6)
- `scripts/eval.py`, `scripts/build.py`
- `CLAUDE.md`, `README.md`, pinned `requirements.txt`
