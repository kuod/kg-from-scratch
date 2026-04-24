# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Quick Start

**Install dependencies:**
```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

**Environment setup:**
```bash
cp .env.example .env
# Fill in GROQ_API_KEY and ANTHROPIC_API_KEY
docker compose up -d  # Start Neo4j
```

**Common commands:**
```bash
pytest tests/                    # Run all tests
black src/ scripts/ app.py       # Format code
ruff check src/                  # Lint
mypy src/                        # Type check
python scripts/build.py          # Build knowledge graph from papers in data/papers/
streamlit run app.py             # Launch chatbot UI at http://localhost:8501
```

---

## System Architecture

### High-Level Data Flow

The system is a **document-to-knowledge-graph pipeline** with a Q&A agent frontend:

```
Input Papers (PDF or Markdown)
    ↓
PDF Conversion (pdf_convert.py) → Markdown
    ↓
Chunking (ingest.py) → Markdown split into sections
    ↓
Named Entity Recognition (ner.py) → Biomedical entity extraction via scispaCy
    ↓
LLM Relation Extraction (extract.py) → Groq llama-3.3 extracts relationships
    ↓
Graph Construction (graph.py) → Neo4j schema: Paper → Chunk → Entity → Relationship
    ↓
Entity Resolution (resolve.py) → Merge aliases to canonical names (e.g., p53 → TP53)
    ↓
Embedding (embed.py) → PubMedBERT 768-dim vectors for semantic search
    ↓
Confidence Intervals (confidence.py) → Wilson score CI on edge evidence counts
    ↓
Chatbot (agent.py) → Claude agent with 7 tools for RAG, reports CI bounds
    ↓
UI (app.py) → Streamlit with 4 tabs: Ask, Graph Explorer, Papers, Stats
```

### Core Modules

#### **1. Input & Conversion Layer**

- **`src/pdf_convert.py`** — Converts PDFs to Markdown
  - Primary: `pymupdf4llm` (layout-aware, tables, multi-column)
  - Fallback: `pdfminer.six` (plain text extraction)
  - Post-processing: removes headers/footers, normalizes figure captions
  - Entry point: `pdf_to_markdown(pdf_path, out_path) → str`

- **`src/ingest.py`** — Parses Markdown into chunks
  - Detects sections via ATX headers (`##`) and bold lines (`**...**`)
  - Applies sliding window for long sections (max 1200 words)
  - Returns `Chunk` dataclass: `(paper_id, section, chunk_index, text, char_offset)`
  - Entry point: `load_paper(path: Path) → list[Chunk]`

#### **2. Entity Extraction Layer**

- **`src/ner.py`** — Biomedical Named Entity Recognition
  - Uses scispaCy with two models:
    - `en_core_sci_lg` — General biomedical NER (UMLS semantic types: Gene, Disease, Drug, etc.)
    - `en_ner_bc5cdr_md` — Specialized disease/drug detection with UMLS CUI linking
  - Returns `NEREntity` dataclass with `(text, label, raw_label, umls_cui, score)`
  - Maps UMLS semantic types to 8 canonical entity types: Gene, Protein, Drug, Disease, Pathway, CellType, Organism, Mechanism
  - Entry point: `extract_entities_ner(text: str) → list[NEREntity]`

- **`src/extract.py`** — LLM-based Relation Extraction
  - Calls Groq `llama-3.3-70b-versatile` (configurable via `LLM_MODEL` env var)
  - Prompt injects NER results to guide extraction
  - Returns JSON: `{entities: [...], relationships: [...]}`
  - **Caching**: Results stored in `data/cache/<paper_id>__<chunk_index>.json` (disk cache makes re-runs free)
  - Fallback: If Groq fails, uses Anthropic SDK
  - Entry point: `extract_from_chunk(chunk: Chunk, ner_entities: list[NEREntity]) → dict`

#### **3. Graph Storage Layer**

- **`src/graph.py`** — Neo4j driver wrapper and schema
  - Manages connection pooling and idempotent MERGE-based upserts
  - Schema constraints: unique `Paper.id`, `Chunk.id`, entity names per type
  - Methods for upserting:
    - `upsert_paper(paper_id, title, doi, year)`
    - `upsert_chunk(chunk_id, paper_id, section, chunk_index, text)`
    - `upsert_entity(label, name, aliases, umls_cui, **props)`
    - `upsert_relationship(src_label, src_name, rel_type, tgt_label, tgt_name, chunk_id, **props)`
  - Query methods:
    - `get_entity_relationships(label, name) → list[dict]`
    - `keyword_search_chunks(keyword, limit) → list[dict]`
    - `find_path(src_label, src_name, tgt_label, tgt_name, max_hops) → list[dict]`
    - `get_graph_stats() → dict[label → count]`
  - Uses context manager: `with GraphDB() as db: ...`
  - 8 entity types: Gene, Protein, Drug, Disease, Pathway, CellType, Organism, Mechanism
  - 11 relationship types: REGULATES, INHIBITS, ACTIVATES, TARGETS, BINDS, ASSOCIATED_WITH, PROMOTES, SUPPRESSES, INVOLVES, EXPRESSED_IN, MUTATED_IN

#### **4. Orchestration & Post-Processing**

- **`src/build_graph.py`** — Full pipeline orchestrator
  - Loads papers from `data/papers/`, processes chunks in sequence
  - For each chunk: NER → extract → upsert entities & relationships
  - Supports `--resume` flag to skip already-processed chunks (idempotent)
  - Calls `compute_edge_confidences(db)` after graph is built
  - Entry point: `build(papers_dir, clear=False, resume=False, paper_id=None) → None`

- **`src/resolve.py`** — Entity resolution (synonym merging)
  - Pass 1: Curated `SYNONYM_GROUPS` dict (p53→TP53, her2→ERBB2, etc.)
  - Pass 2: String normalization (lowercase, strip punctuation) via APOC
  - Entry point: `run_resolution(db) → None`

- **`src/clean_graph.py`** — Post-extraction noise removal
  - Removes isolated nodes (no edges)
  - Removes trivial names (single char, purely numeric)
  - Removes stopword-only nodes
  - Entry point: `clean_graph(db) → dict[str, int]` (returns removal counts)

#### **5. Embedding & Semantic Search**

- **`src/embed.py`** — Node and chunk embedding
  - Uses PubMedBERT (`pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb`)
  - 768-dimensional embeddings
  - Entry points:
    - `embed_all_nodes(db, verbose=True) → int` (embeds all nodes, writes to Neo4j)
    - `encode_texts(texts: list[str]) → list[list[float]]` (encode arbitrary strings)
    - `query_similar_chunks(query: str, db: GraphDB, top_k: int) → list[dict]` (semantic search)
  - Lazy-loads model on first call (cached in `~/.cache/huggingface/`)

- **`src/tokenize_graph.py`** — Node text templating for embedding
  - Builds structured text for each node type (e.g., "Gene TP53. Also known as: p53. Relationships: REGULATES KRAS...")
  - Incorporates N-hop neighborhood context so embeddings capture functional relationships
  - Entry points:
    - `build_node_text(node: dict, label: str, db: GraphDB) → str`
    - `build_all_node_texts(db) → list[tuple]` (all nodes as (label, identifier, text, node))

#### **6. Confidence Intervals**

- **`src/confidence.py`** — Wilson score CI computation
  - **Level 1 (Build-time)**: Wilson score CI on per-edge evidence counts
    - For each relationship: `ci = wilson_ci(k=evidence_count, n=co_mention_count)`
    - Stored as `confidence_lower` and `confidence_upper` on edges
    - Entry point: `compute_edge_confidences(db) → int` (updates edges, returns count)
  - **Level 2 (Query-time)**: LLM-elicited epistemic CI from source passages
    - Entry point: `calibrate_answer_confidence(answer, chunk_texts, db) → dict`
  - **Level 3 (Optional)**: Log-probability CI (if provider exposes logprobs)
    - Controlled by `USE_LOGPROB_CI` env var
  - Wilson formula implements 95% CI with continuity correction

#### **7. Query Interface (RAG Agent)**

- **`src/agent.py`** — Tool-using Claude agent
  - Uses `claude-sonnet-4-6` (configurable via `AGENT_MODEL` env var)
  - Max 8 tool calls per query to avoid loops
  - **7 tools**:
    1. `rank_papers(query, top_k)` — Rank papers by semantic similarity
    2. `search_entity(name, entity_type)` — Find entities by name/type
    3. `get_entity_relationships(entity_type, entity_name)` — Get entity edges with CI bounds
    4. `search_chunks(query, top_k, expand_graph)` — Hybrid keyword + vector search
    5. `get_chunk_text(chunk_id)` — Retrieve chunk full text
    6. `find_path(src_type, src_name, tgt_type, tgt_name, max_hops)` — Shortest path between entities
    7. `get_edge_confidence(src_type, src_name, rel_type, tgt_type, tgt_name)` — CI for specific edge
  - Returns: `(answer_text, chunk_ids, confidence_dict, tool_trace)`
  - Entry point: `ask_with_confidence(question: str, db: GraphDB, max_turns: int = 8) → tuple`

#### **8. Frontend**

- **`app.py`** — Streamlit chatbot
  - **Tab 1: Ask** — Chat interface with citations, CI display, tool trace, knowledge subgraph
  - **Tab 2: Graph Explorer** — Entity search, relationship table with CI bounds
  - **Tab 3: Papers** — Browse chunks per paper with entity type annotations
  - **Tab 4: Stats** — Node/edge counts, relationship distribution chart
  - Uses PyVis for interactive network visualization
  - Caches GraphDB connection as `@st.cache_resource`

#### **9. Configuration**

- **`src/config.py`** — Centralized env var loading
  - Neo4j: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
  - LLM: `LLM_MODEL` (default: `llama-3.3-70b-versatile`), `GROQ_API_KEY`
  - Agent: `AGENT_MODEL` (default: `claude-sonnet-4-6`), `ANTHROPIC_API_KEY`
  - Embedding: `EMBED_MODEL` (default: PubMedBERT), `SCISPACY_MODEL` (default: `en_core_sci_lg`)
  - Chunking: `CHUNK_MAX_WORDS=1200`, `CHUNK_WINDOW_WORDS=600`, `CHUNK_OVERLAP_WORDS=150`
  - Embedding: `EMBED_BATCH_SIZE=64`, `EMBED_MAX_TOKENS=512`
  - Paths: `PAPERS_DIR`, `PDFS_DIR`, `CACHE_DIR` (auto-created)

### Critical Data Structures

**Chunk** (from `ingest.py`):
```python
@dataclass
class Chunk:
    paper_id: str        # filename stem
    section: str         # section header text
    chunk_index: int     # ordinal within paper
    text: str            # section body
    char_offset: int     # byte offset for provenance
    word_count: int      # computed field
```

**NEREntity** (from `ner.py`):
```python
@dataclass
class NEREntity:
    text: str            # extracted entity text
    label: str           # normalized type (Gene, Drug, Disease, etc.)
    raw_label: str       # original model label
    start_char: int      # char position in text
    end_char: int        # char position in text
    umls_cui: str | None # UMLS identifier
    canonical_name: str | None
    score: float = 1.0   # confidence
```

### Key Design Patterns

1. **Idempotency**: All graph operations use `MERGE` (not `CREATE`) so re-runs are safe
2. **Context Managers**: GraphDB uses `__enter__`/`__exit__` for connection pooling
3. **Caching**: LLM extraction results cached to disk in `data/cache/`; models cached via `@functools.lru_cache(maxsize=1)`
4. **Lazy Loading**: Sentence transformers and scispaCy models loaded on first use
5. **Composition**: `build_graph.py` orchestrates NER, extraction, and graph updates without duplicating logic
6. **Tool-Use Loop**: Agent makes up to 8 tool calls per query; Claude decides when to stop

---

## Development Workflow

### Building the Graph

```bash
# Full pipeline (includes PDF conversion, NER, extraction, resolution, embedding)
python scripts/build.py

# Options:
python scripts/build.py --clear              # Wipe graph before building
python scripts/build.py --resume             # Skip already-processed chunks (safe for incremental builds)
python scripts/build.py --paper my_paper_id  # Process only one paper
python scripts/build.py --skip-embed         # Skip embedding step (faster, no semantic search)

# Just convert PDFs to Markdown
python scripts/convert_pdfs.py
python scripts/convert_pdfs.py --pdf path/to/paper.pdf
```

### Testing

All tests run **without Neo4j or API keys** (they mock external dependencies):

```bash
pytest tests/                           # Run all tests
pytest tests/test_ingest.py             # Test chunking
pytest tests/test_ner.py                # Test NER entity mapping
pytest tests/test_confidence.py         # Test Wilson CI computation
pytest tests/test_ingest.py::test_load_paper_returns_chunks -v  # Single test
```

Test files:
- `tests/test_ingest.py` — Markdown chunking with various header formats
- `tests/test_ner.py` — Entity type normalization from UMLS labels
- `tests/test_confidence.py` — Wilson score CI edge cases (k=0, k=n, partial evidence)

### Linting & Formatting

```bash
black src/ scripts/ app.py tests/      # Format with Black (100 char line length)
ruff check src/                        # Lint with Ruff
mypy src/                              # Type check (strict mode)
```

Code style:
- Line length: 100 characters (configured in `pyproject.toml`)
- Python 3.11+
- Type hints required on function signatures

---

## Common Troubleshooting

**PDF conversion fails:** Try `ocrmypdf input.pdf output.pdf` for scanned PDFs, then retry.

**Neo4j connection errors:** Ensure `docker compose up -d` is running and `NEO4J_PASSWORD` in `.env` matches `docker-compose.yml`.

**LLM extraction cache:** Delete `data/cache/` to force re-extraction of all chunks, or delete individual `<paper_id>__<chunk_index>.json` files.

**Embedding model downloads (first run only):** PubMedBERT (~400 MB) downloads to `~/.cache/huggingface/`. Neo4j plugins (~200 MB) download to Docker volume `neo4j_plugins`.

---

## Data Directory Structure

```
data/
├── papers/          # Input: .md files (chunked and ingested by build.py)
├── pdfs/            # Input: .pdf files (converted to markdown by scripts/convert_pdfs.py)
└── cache/           # LLM extraction results cached as JSON (gitignored, safe to delete)
```

**Paper ID format:** Filename stem (without `.md` or `.pdf`) becomes the `paper_id` used throughout (e.g., `smith_2024.md` → `paper_id: smith_2024`).

---

## Dependencies & External Services

**Python packages** (installed via `pip install -e ".[dev]"`):
- Core: `neo4j`, `streamlit`, `anthropic`, `groq`, `sentence-transformers`, `pyvis`
- NER: `scispacy` (+ two S3-hosted models installed separately)
- PDF: `pymupdf4llm`, `pdfminer.six`
- Stats: `scipy`, `numpy`, `pandas`, `tqdm`
- Dev: `pytest`, `pytest-asyncio`, `black`, `ruff`, `mypy`

**External services** (require API keys in `.env`):
- **Groq** (`GROQ_API_KEY`) — Extraction model (`llama-3.3-70b-versatile`)
- **Anthropic** (`ANTHROPIC_API_KEY`) — Agent model (`claude-sonnet-4-6`)
- **Google** (`GOOGLE_API_KEY`, optional) — Fallback model

**Docker services**:
- **Neo4j 5.20** (community edition) with APOC and Graph Data Science plugins
- Ports: 7474 (HTTP/browser), 7687 (Bolt/driver)
- Heap: 1–4 GB (configured in `docker-compose.yml`)

---

## Graph Schema Reference

**Node labels:**
- `Paper` — metadata: `id`, `title`, `doi`, `year`
- `Chunk` — metadata: `id`, `paper_id`, `section`, `chunk_index`, `text`, `embedding`
- `Gene`, `Protein`, `Drug`, `Disease`, `Pathway`, `CellType`, `Organism`, `Mechanism` — metadata: `name`, `aliases`, `umls_cui`, `embedding`

**Edge types:**
- `FROM_PAPER` — Chunk → Paper
- `MENTIONED_IN` — Entity → Chunk (with `.count`)
- `REGULATES`, `INHIBITS`, `ACTIVATES`, `TARGETS`, `BINDS`, `ASSOCIATED_WITH`, `PROMOTES`, `SUPPRESSES`, `INVOLVES`, `EXPRESSED_IN`, `MUTATED_IN` — Entity → Entity (with `.evidence_count`, `.confidence_lower`, `.confidence_upper`, `.source_chunk_ids`)

**Indexes:**
- Uniqueness constraints on `Paper.id`, `Chunk.id`, and entity names per type
- Vector indexes for semantic search (created by `db.setup_vector_indexes()`)

---

## When to Read Each File

- **Architecture overview:** README.md, plan.md
- **Setup & usage:** get-started.md, `.env.example`
- **Building the graph:** `src/build_graph.py`, `scripts/build.py`
- **Understanding NER:** `src/ner.py`, `src/extract.py`
- **Understanding Q&A:** `src/agent.py`, `src/confidence.py`
- **Debugging graph issues:** `src/graph.py`, `src/resolve.py`, `src/clean_graph.py`
- **Tuning embedding:** `src/embed.py`, `src/tokenize_graph.py`
- **Testing without APIs:** `tests/`

