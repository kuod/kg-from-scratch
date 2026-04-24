# Get Started

This guide takes you from a fresh checkout to a running Streamlit chatbot that can answer biomedical questions about your paper corpus with confidence intervals.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python ≥ 3.11** | Check with `python3 --version` |
| **Docker + Docker Compose** | Required for the Neo4j graph database |
| **4 GB+ free RAM** | Neo4j heap is configured up to 4 GB |
| **Groq API key** | Free tier at [console.groq.com](https://console.groq.com) — used for relation extraction |
| **Anthropic API key** | At [console.anthropic.com](https://console.anthropic.com) — used for the chatbot agent |
| **Internet connection** | First build downloads the PubMedBERT embedding model (~400 MB) |

---

## Installation

```bash
# 1. Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install Python dependencies
pip install -e ".[dev]"

# 3. Install scispaCy biomedical NER models
#    These are not on PyPI — install from the S3 release URLs directly
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

- `en_core_sci_lg` — general biomedical NER using UMLS semantic types (genes, diseases, chemicals, etc.)
- `en_ner_bc5cdr_md` — specialized disease and drug detection with UMLS CUI linking

---

## Configure Environment

```bash
cp .env.example .env
```

Open `.env` and fill in at minimum:

```
GROQ_API_KEY=your_groq_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

All other values have working defaults. If you change `NEO4J_PASSWORD`, update it to match in `docker-compose.yml` under `NEO4J_AUTH: neo4j/<password>`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEO4J_URI` | `bolt://localhost:7687` | Database connection |
| `NEO4J_USER` | `neo4j` | Database user |
| `NEO4J_PASSWORD` | `password` | Database password |
| `GROQ_API_KEY` | *(required)* | Extraction LLM (llama-3.3-70b) |
| `ANTHROPIC_API_KEY` | *(required)* | Agent QA (claude-sonnet-4-6) |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Relation extraction model |
| `AGENT_MODEL` | `claude-sonnet-4-6` | Chatbot agent model |
| `EMBED_MODEL` | `pritamdeka/PubMedBERT-...` | Sentence embedding model |
| `USE_LOGPROB_CI` | `false` | Enable log-probability confidence intervals |

---

## Start Neo4j

```bash
docker compose up -d
```

- First startup takes ~30–60 seconds while APOC and GDS plugins download.
- Monitor progress: `docker compose logs -f neo4j`
- Neo4j browser UI: **http://localhost:7474** (login: `neo4j` / `password`)
- Bolt port for the driver: `7687`

To stop the database: `docker compose down`
To wipe all data: `docker compose down -v`

---

## Add Your Papers

The pipeline accepts two input formats — pick whichever applies.

### Option A — PDF files

```bash
# Copy your PDFs into the input folder
cp /path/to/your/papers/*.pdf data/pdfs/

# Convert to markdown (outputs to data/papers/)
python scripts/convert_pdfs.py

# Convert a single file
python scripts/convert_pdfs.py --pdf data/pdfs/my_paper.pdf
```

The converter uses `pymupdf4llm` (layout-aware, handles multi-column and tables) with a `pdfminer.six` fallback. Encrypted or image-only (scanned) PDFs will fail — use OCR to produce a text PDF first.

### Option B — Markdown files directly

```bash
cp /path/to/your/papers/*.md data/papers/
```

The filename stem becomes the `paper_id` used in the graph and in all citations (e.g., `smith_2024.md` → `paper_id: smith_2024`). Use short, descriptive, underscore-separated names.

---

## Build the Knowledge Graph

```bash
python scripts/build.py
```

This runs four steps in sequence:

| Step | What happens |
|------|-------------|
| **1. PDF → Markdown** | Converts any PDFs in `data/pdfs/` (skipped if `--skip-pdf`) |
| **2. Graph population** | Chunks each paper, runs scispaCy NER, calls Groq LLM for entity/relation extraction, upserts everything into Neo4j. LLM results are cached in `data/cache/` — re-runs are free. |
| **3. Entity resolution** | Merges aliases to canonical names (e.g., `p53 → TP53`) and deduplicates by string normalization via APOC |
| **4. Embedding** | Encodes all nodes with PubMedBERT (768-dim) and creates vector indexes for semantic search |

**Useful flags:**

| Flag | Effect |
|------|--------|
| `--clear` | Delete entire graph before building |
| `--resume` | Skip chunks already stored in Neo4j (safe to append) |
| `--paper <id>` | Process only one paper (e.g., `--paper smith_2024`) |
| `--skip-pdf` | Skip the PDF conversion step |
| `--skip-embed` | Skip embedding — faster, but semantic search won't work |

**Expected time:** ~2–5 minutes per paper, depending on chunk count and LLM latency. The cache makes subsequent runs much faster.

---

## Launch the Streamlit App

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

Keep Neo4j running (`docker compose up -d`) while the app is active.

---

## Using the App

### Ask tab
Type any biomedical question. The agent runs up to 10 tool calls against the knowledge graph, then synthesizes an answer with:
- **Citations** in the form `[paper_id § section]`
- **Confidence badge**: High / Medium / Low label
- **95% confidence interval**: `[lower, upper]` — a Wilson score CI on the graph evidence, plus an LLM-elicited epistemic CI on the answer quality
- **Evidence chunks**: expandable source passages
- **Tool trace**: the full input/output of each tool call for transparency
- **Knowledge subgraph**: interactive PyVis visualization of entities mentioned in the evidence

Example questions:
- *What does TP53 regulate?*
- *Which drugs target EGFR and what is the evidence confidence?*
- *Is there a relationship between MDM2 and apoptosis?*
- *What pathways are associated with KRAS mutations in colorectal cancer?*

### Graph Explorer tab
Search any entity by name (partial match supported) and filter by type. The results table shows all relationships with their **Wilson CI** bounds and evidence counts.

### Papers tab
Browse chunks from any ingested paper. Each chunk shows its entity mentions tagged by type (Gene, Drug, Disease, etc.).

### Stats tab
Live counts of nodes and edges by type, plus a bar chart of relationship type distribution.

---

## Re-running and Updating

To add more papers to an existing graph without reprocessing everything:

```bash
cp new_paper.md data/papers/
python scripts/build.py --resume
```

`--resume` skips any chunk that already has edges in Neo4j. LLM extraction for new chunks will be cached automatically.

To rebuild from scratch:

```bash
python scripts/build.py --clear
```

---

## Troubleshooting

**`RuntimeError: scispaCy model not found`**
The scispaCy model URLs were not pip-installed. Re-run the two `pip install https://...` commands from the Installation section with your virtualenv active.

**`Neo4j connection failed` in the Streamlit sidebar**
- Check Docker is running: `docker compose ps`
- Check the password matches: compare `NEO4J_PASSWORD` in `.env` with `NEO4J_AUTH` in `docker-compose.yml`
- Check the port is free: `lsof -i :7687`

**`No markdown files found in data/papers/`**
Your `.md` files need to be in `data/papers/`, not in a subdirectory. Run `ls data/papers/` to verify.

**`GROQ_API_KEY not set` / `ANTHROPIC_API_KEY not set`**
You created `.env.example` but not `.env`. Run `cp .env.example .env` and fill in your keys.

**`Both pdf converters failed`**
The PDF is likely encrypted or image-only (scanned without OCR). Run the PDF through an OCR tool first (e.g., Adobe Acrobat, `ocrmypdf`) to produce a text-layer PDF.

**First build is very slow**
The PubMedBERT sentence-transformer model downloads ~400 MB on first use. It is cached by `sentence-transformers` in `~/.cache/` and will not re-download on subsequent runs.

**Embeddings step hangs**
If you have no GPU, encoding thousands of nodes on CPU can take several minutes. Use `--skip-embed` to test the rest of the pipeline first, then run embedding separately: `python -c "from src.graph import GraphDB; from src.embed import embed_all_nodes; embed_all_nodes(GraphDB())"`.

---

## Architecture Reference

For the full rationale and data flow diagram see `plan.md`. Quick reference:

| File | Purpose |
|------|---------|
| `src/pdf_convert.py` | PDF → Markdown (pymupdf4llm + pdfminer fallback) |
| `src/ingest.py` | Markdown → `Chunk` dataclasses (header split + sliding window) |
| `src/ner.py` | scispaCy offline biomedical NER |
| `src/extract.py` | Groq LLM relation extraction with NER hints; disk-cached |
| `src/graph.py` | Neo4j driver, schema setup, all upsert/query methods |
| `src/build_graph.py` | Per-paper pipeline orchestrator |
| `src/resolve.py` | Synonym merge + string normalization deduplication |
| `src/tokenize_graph.py` | N-hop neighborhood text templates for each node type |
| `src/embed.py` | PubMedBERT encoding + Neo4j vector index creation |
| `src/confidence.py` | Wilson score CI (build time) + LLM-elicited CI (query time) |
| `src/agent.py` | 7-tool RAG agent returning answer + CI + tool trace |
| `src/clean_graph.py` | Post-extraction noise removal |
| `src/resolve.py` | Synonym + normalization entity deduplication |
| `app.py` | Streamlit 4-tab UI |
| `scripts/build.py` | Full pipeline CLI |
| `scripts/convert_pdfs.py` | Batch PDF → Markdown CLI |
| `docker-compose.yml` | Neo4j 5.x + APOC + GDS |
