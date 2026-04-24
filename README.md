    __               ____                                                __       __  
   / /______ _      / __/________  ____ ___        ___________________ _/ /______/ /_ 
  / //_/ __ `/_____/ /_/ ___/ __ \/ __ `__ \______/ ___/ ___/ ___/ __ `/ __/ ___/ __ \
 / ,< / /_/ /_____/ __/ /  / /_/ / / / / / /_____(__  ) /__/ /  / /_/ / /_/ /__/ / / /
/_/|_|\__, /     /_/ /_/   \____/_/ /_/ /_/     /____/\___/_/   \__,_/\__/\___/_/ /_/ 
     /____/                                                                           

# Biomedical Knowledge Graph from Scratch

A pipeline that ingests biomedical papers (PDF or Markdown), extracts entities and relationships using LLMs, builds a Neo4j knowledge graph, and exposes a Streamlit chatbot that answers questions with **95% confidence intervals**.

---

## How It Works

```
PDF / Markdown
     │
     ▼  pdf_convert.py
Markdown chunks
     │
     ├─▶ NER (scispaCy)   ─┐
     └─▶ LLM extraction    ├─▶ Neo4j graph ─▶ Entity resolution
          (Groq llama-3.3) ┘       │
                                   ▼
                          PubMedBERT embeddings
                          + vector indexes
                                   │
                                   ▼
                          Streamlit chatbot
                          (Claude agent + Wilson CI)
```

---

## Quick Start

```bash
# 1. Install
python3.11 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

# 2. Configure
cp .env.example .env          # fill in GROQ_API_KEY and ANTHROPIC_API_KEY

# 3. Start database
docker compose up -d

# 4. Drop papers in data/papers/ (or PDFs in data/pdfs/)

# 5. Build graph
python scripts/build.py

# 6. Launch app
streamlit run app.py           # → http://localhost:8501
```

See [get-started.md](get-started.md) for the full step-by-step walkthrough.

---

## Dependencies

### System Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.11 | Required by pyproject.toml |
| Docker + Compose | any recent | Runs Neo4j |
| RAM | ≥ 4 GB free | Neo4j heap configured to 4 GB max |

### Python Packages

Installed via `pip install -e ".[dev]"`:

| Package | Version | Purpose |
|---------|---------|---------|
| `neo4j` | ≥ 5.18 | Neo4j Python driver |
| `python-dotenv` | ≥ 1.0 | Load `.env` into environment |
| `streamlit` | ≥ 1.35 | Web UI framework |
| `sentence-transformers` | ≥ 3.0 | PubMedBERT embeddings |
| `pyvis` | ≥ 0.3 | Interactive graph visualization |
| `pymupdf4llm` | ≥ 0.0.17 | PDF → Markdown (primary converter) |
| `pdfminer.six` | ≥ 20221105 | PDF → text (fallback converter) |
| `scispacy` | ≥ 0.5 | Biomedical NER framework |
| `anthropic` | ≥ 0.30 | Claude API (agent + confidence calibration) |
| `groq` | ≥ 0.9 | Groq API (relation extraction LLM) |
| `openai` | ≥ 1.35 | OpenAI-compatible fallback |
| `tqdm` | ≥ 4.66 | Progress bars |
| `scipy` | ≥ 1.13 | Wilson score CI (normal distribution) |
| `numpy` | ≥ 1.26 | Embedding arrays |
| `pandas` | ≥ 2.2 | DataFrame operations in Streamlit |

**Dev only** (`pip install -e ".[dev]"`):

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ≥ 8 | Test runner |
| `pytest-asyncio` | ≥ 0.23 | Async test support |
| `black` | ≥ 24 | Code formatter |
| `ruff` | ≥ 0.4 | Linter |
| `mypy` | ≥ 1.10 | Static type checker |

### scispaCy Models

These are **not on PyPI** — install from the release URLs:

| Model | URL | Purpose |
|-------|-----|---------|
| `en_core_sci_lg` | `https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz` | General biomedical NER (UMLS semantic types) |
| `en_ner_bc5cdr_md` | `https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz` | Disease + drug NER with UMLS CUI linking |

### External Services

| Service | Key variable | Used for | Required |
|---------|-------------|----------|----------|
| [Groq](https://console.groq.com) | `GROQ_API_KEY` | `llama-3.3-70b-versatile` relation extraction | Yes |
| [Anthropic](https://console.anthropic.com) | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` agent + confidence calibration | Yes |
| Google | `GOOGLE_API_KEY` | Optional model fallback | No |

### Downloaded at Runtime

| Asset | Size | Cached at |
|-------|------|-----------|
| PubMedBERT sentence-transformer | ~400 MB | `~/.cache/huggingface/` |
| Neo4j APOC plugin | ~50 MB | Docker volume `neo4j_plugins` |
| Neo4j GDS plugin | ~150 MB | Docker volume `neo4j_plugins` |

---

## Project Structure

```
kg-from-scratch/
├── app.py                    # Streamlit chatbot (4 tabs)
├── docker-compose.yml        # Neo4j 5.x + APOC + GDS
├── pyproject.toml            # Package manifest + dependencies
├── plan.md                   # Full architecture & rationale
├── get-started.md            # Step-by-step onboarding guide
│
├── data/
│   ├── papers/               # Input: .md files
│   ├── pdfs/                 # Input: .pdf files
│   └── cache/                # LLM extraction cache (gitignored)
│
├── src/
│   ├── config.py             # Env vars, paths, constants
│   ├── pdf_convert.py        # PDF → Markdown
│   ├── ingest.py             # Markdown → Chunk dataclasses
│   ├── ner.py                # scispaCy NER
│   ├── extract.py            # LLM relation extraction
│   ├── graph.py              # Neo4j driver + schema
│   ├── build_graph.py        # Per-paper pipeline orchestrator
│   ├── resolve.py            # Entity deduplication
│   ├── tokenize_graph.py     # Node text templates for embedding
│   ├── embed.py              # PubMedBERT + vector indexes
│   ├── confidence.py         # Wilson CI + LLM-elicited CI
│   ├── agent.py              # 7-tool RAG agent
│   └── clean_graph.py        # Post-extraction noise removal
│
├── scripts/
│   ├── build.py              # Full pipeline CLI
│   └── convert_pdfs.py       # Batch PDF → Markdown
│
└── tests/
    ├── test_ingest.py
    ├── test_ner.py
    └── test_confidence.py
```

---

## App Features

**Ask tab** — Chatbot with citations and confidence intervals
- Agent makes up to 8 tool calls (entity lookup, graph path, semantic search)
- Every claim cited as `[paper_id § section]`
- 95% CI displayed as `[lower, upper]` with High / Medium / Low label
- Expandable tool trace and knowledge subgraph

**Graph Explorer tab** — Browse entities and relationships
- Search by name and entity type
- Relationship table includes Wilson CI bounds and evidence counts

**Papers tab** — Browse source text
- Per-paper chunk viewer with entity type annotations

**Stats tab** — Graph health overview
- Node and edge counts by type
- Relationship distribution chart

---

## FAQ

**How do I add new papers to an existing graph without reprocessing everything?**

Drop the new files into `data/papers/` (or `data/pdfs/`) and run:

```bash
python scripts/build.py --resume
```

`--resume` skips any chunk that already has edges in Neo4j, so only the new papers are processed. LLM extraction results for new chunks are cached automatically, making any future re-runs free.

**How do I completely rebuild the graph from scratch?**

```bash
python scripts/build.py --clear
```

This wipes all nodes and edges before rebuilding. Cached extraction results in `data/cache/` are preserved, so the LLM step is still fast.

**Can I process a single paper without touching the rest of the graph?**

```bash
python scripts/build.py --paper my_paper_id --resume
```

The `--paper` flag limits processing to one file (matched by filename stem). Combine with `--resume` to leave existing nodes untouched.

**Why is the first build slow?**

Two one-time downloads happen on first run: the PubMedBERT sentence-transformer model (~400 MB, cached in `~/.cache/huggingface/`) and the Neo4j APOC + GDS plugins (~200 MB total, cached in the Docker volume). Subsequent builds skip both.

**Can I skip the embedding step to build faster?**

Yes — use `--skip-embed`. The graph and keyword search will work, but semantic (vector) search in the chatbot won't. You can run embedding separately any time:

```bash
python scripts/build.py --skip-embed    # build graph only
# later...
python -c "from src.graph import GraphDB; from src.embed import embed_all_nodes; embed_all_nodes(GraphDB())"
```

**How do I swap the extraction model?**

Set `LLM_MODEL` in `.env` to any model supported by Groq (e.g. `llama-3.1-8b-instant` for faster/cheaper runs, `llama-3.3-70b-versatile` for higher quality). The Anthropic SDK is used as a silent fallback if Groq fails.

**What happens if a PDF fails to convert?**

The converter tries `pymupdf4llm` first, then falls back to `pdfminer.six`. If both fail (encrypted PDF or image-only scan), run the file through OCR first — `ocrmypdf input.pdf output.pdf` — then retry conversion.

**Where are LLM extraction results cached?**

In `data/cache/` as `<paper_id>__<chunk_index>.json`. Delete a file there to force re-extraction of that chunk, or delete the whole folder to re-extract everything.

**How do I view the raw graph?**

Open the Neo4j browser at **http://localhost:7474** (login: `neo4j` / `password`) while `docker compose up -d` is running. Try `MATCH (n) RETURN n LIMIT 50` to explore.

---

## Running Tests

```bash
pytest tests/
```

Tests for `ingest`, `ner`, and `confidence` run without Neo4j or API keys.

---

## Configuration Reference

Copy `.env.example` to `.env` and set these values:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

GROQ_API_KEY=           # required
ANTHROPIC_API_KEY=      # required
GOOGLE_API_KEY=         # optional

LLM_MODEL=llama-3.3-70b-versatile
AGENT_MODEL=claude-sonnet-4-6
EMBED_MODEL=pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb
SCISPACY_MODEL=en_core_sci_lg
USE_LOGPROB_CI=false
```
