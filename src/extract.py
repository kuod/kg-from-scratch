"""
LLM-based relation extraction per chunk.

Uses Groq (llama-3.3-70b-versatile) with NER hints injected into the prompt.
Results are disk-cached in data/cache/ to make re-runs free.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from src.config import CACHE_DIR, LLM_MODEL
from src.ingest import Chunk
from src.ner import NEREntity, format_ner_hints

ENTITY_TYPES = ["Gene", "Protein", "Drug", "Disease", "Pathway", "CellType", "Organism", "Mechanism"]
RELATIONSHIP_TYPES = [
    "REGULATES", "INHIBITS", "ACTIVATES", "TARGETS", "BINDS",
    "ASSOCIATED_WITH", "PROMOTES", "SUPPRESSES", "INVOLVES",
    "EXPRESSED_IN", "MUTATED_IN",
]

_SYSTEM_PROMPT = """\
You are a biomedical knowledge extraction assistant. Extract entities and \
relationships from the provided text and return ONLY valid JSON.

Entity types: Gene, Protein, Drug, Disease, Pathway, CellType, Organism, Mechanism
Relationship types: REGULATES, INHIBITS, ACTIVATES, TARGETS, BINDS, ASSOCIATED_WITH,
  PROMOTES, SUPPRESSES, INVOLVES, EXPRESSED_IN, MUTATED_IN

Output format:
{
  "entities": [
    {"name": "TP53", "type": "Gene", "aliases": ["p53", "tumor protein p53"]}
  ],
  "relationships": [
    {
      "source_name": "MDM2", "source_type": "Gene",
      "relationship": "INHIBITS",
      "target_name": "TP53", "target_type": "Gene",
      "evidence": "direct quote from text supporting this relationship"
    }
  ]
}

Rules:
- Only extract relationships explicitly stated or strongly implied by the text.
- Prefer canonical gene/protein names (HGNC symbols, UniProt names).
- If unsure of entity type, use Mechanism.
- evidence must be a short quote (< 100 words) from the text.
- Return an empty list for entities or relationships if none are found.
- Return ONLY JSON, no markdown fences, no explanation.
"""


def _cache_path(paper_id: str, chunk_index: int) -> Path:
    return CACHE_DIR / f"{paper_id}__{chunk_index:04d}.json"


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def extract_from_chunk(chunk: Chunk, ner_entities: list[NEREntity]) -> dict[str, Any]:
    """Extract entities and relationships from a single chunk.

    Returns a dict with keys 'entities' and 'relationships'.
    Results are read from cache if available.
    """
    cache_file = _cache_path(chunk.paper_id, chunk.chunk_index)
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            cached = None  # truncated/corrupt cache entry: re-extract
        if isinstance(cached, dict) and cached.get("_text_key") == _cache_key(chunk.text):
            return cached

    result = _call_llm(chunk.text, ner_entities)
    result["_text_key"] = _cache_key(chunk.text)
    tmp = cache_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(cache_file)  # atomic: an interrupted run never leaves a partial file
    return result


def _call_llm(text: str, ner_entities: list[NEREntity]) -> dict[str, Any]:
    ner_hints = format_ner_hints(ner_entities)
    user_message = (
        f"Pre-identified entity hints from automated NER:\n{ner_hints}\n\n"
        f"Text to extract from:\n{text}"
    )

    try:
        return _call_groq(user_message)
    except Exception:
        return _call_anthropic(user_message)


def _call_groq(user_message: str) -> dict[str, Any]:
    from groq import Groq  # type: ignore
    import os

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    return _parse_json(raw)


def _call_anthropic(user_message: str) -> dict[str, Any]:
    import anthropic  # type: ignore

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = response.content[0].text if response.content else "{}"
    return _parse_json(raw)


def _parse_json(raw: str) -> dict[str, Any]:
    import re

    data: Any = None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Extract JSON from fenced block if model added markdown
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        data = {}

    return {
        "entities": _validate_entities(_as_list(data.get("entities"))),
        "relationships": _validate_relationships(_as_list(data.get("relationships"))),
    }


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _validate_entities(raw: list[Any]) -> list[dict[str, Any]]:
    clean = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = _str(item.get("name"))
        etype = item.get("type", "Mechanism")
        if name and etype in ENTITY_TYPES:
            clean.append({
                "name": name,
                "type": etype,
                "aliases": [a for a in _as_list(item.get("aliases")) if isinstance(a, str)],
            })
    return clean


def _validate_relationships(raw: list[Any]) -> list[dict[str, Any]]:
    clean = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        rel = _str(item.get("relationship")).upper()
        src = _str(item.get("source_name"))
        tgt = _str(item.get("target_name"))
        src_type = item.get("source_type", "Mechanism")
        tgt_type = item.get("target_type", "Mechanism")
        if src and tgt and rel in RELATIONSHIP_TYPES:
            clean.append({
                "source_name": src,
                "source_type": src_type if src_type in ENTITY_TYPES else "Mechanism",
                "relationship": rel,
                "target_name": tgt,
                "target_type": tgt_type if tgt_type in ENTITY_TYPES else "Mechanism",
                "evidence": str(item.get("evidence", ""))[:300],
            })
    return clean
