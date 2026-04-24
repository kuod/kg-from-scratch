from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ROOT = Path(__file__).parent.parent
PAPERS_DIR = ROOT / "data" / "papers"
PDFS_DIR = ROOT / "data" / "pdfs"
CACHE_DIR = ROOT / "data" / "cache"

for _d in (PAPERS_DIR, PDFS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-4-6")
EMBED_MODEL = os.getenv("EMBED_MODEL", "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb")
SCISPACY_MODEL = os.getenv("SCISPACY_MODEL", "en_core_sci_lg")

USE_LOGPROB_CI = os.getenv("USE_LOGPROB_CI", "false").lower() == "true"

CHUNK_MAX_WORDS = 1200
CHUNK_WINDOW_WORDS = 600
CHUNK_OVERLAP_WORDS = 150

EMBED_BATCH_SIZE = 64
EMBED_MAX_TOKENS = 512
NEIGHBOR_CONTEXT_LIMIT = 5
