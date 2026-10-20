"""
LLM-based relation extraction per chunk.

Uses Groq (llama-3.3-70b-versatile) with NER hints injected into the prompt.
Results are disk-cached in data/cache/ to make re-runs free.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import logging
from pathlib import Path
from typing import Any

from src.config import CACHE_DIR, LLM_FALLBACK_MODEL, LLM_MODEL
from src.ingest import Chunk
from src.ner import NEREntity, format_ner_hints

log = logging.getLogger(__name__)

# Bump when the user-message template or call settings below change in a way the
# system prompt text alone does not capture. Any bump invalidates every cache entry.
PROMPT_VERSION = "2"
_MAX_TOKENS = 2048
_TEMPERATURE = 0.0

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


def _config_key() -> str:
    """Fingerprint of everything except the chunk: models, prompt, call settings."""
    cfg = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt": _SYSTEM_PROMPT,
        "model": LLM_MODEL,
        "fallback_model": LLM_FALLBACK_MODEL,
        "max_tokens": _MAX_TOKENS,
        "temperature": _TEMPERATURE,
    }
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def _cache_key(text: str, ner_entities: list[NEREntity] | None = None) -> str:
    """Key over (model, prompt, settings, NER hints, chunk text)."""
    payload = json.dumps(
        [_config_key(), format_ner_hints(ner_entities or []), text], sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_entry(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def extract_from_chunk(chunk: Chunk, ner_entities: list[NEREntity]) -> dict[str, Any]:
    """Extract entities and relationships from a single chunk.

    Returns a dict with keys 'entities' and 'relationships'.
    Results are read from cache if the entry's key matches the current
    model, prompt, settings, NER hints and chunk text; otherwise re-extracted.
    """
    cache_file = _cache_path(chunk.paper_id, chunk.chunk_index)
    key = _cache_key(chunk.text, ner_entities)
    if cache_file.exists():
        cached = _load_entry(cache_file)  # None: truncated/corrupt, re-extract
        if cached is not None and cached.get("_cache_key") == key:
            return cached
        if cached is not None and "_cache_key" not in cached:
            log.warning(
                "Legacy cache entry %s (no model/prompt key): treating as a miss and "
                "re-calling the LLM", cache_file.name,
            )

    result = _call_llm(chunk.text, ner_entities)
    result["_cache_key"] = key
    result["_config_key"] = _config_key()
    tmp = cache_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(cache_file)  # atomic: an interrupted run never leaves a partial file
    return result


def cache_status(purge: bool = False) -> dict[str, int]:
    """Count cache entries by freshness; optionally delete the stale ones.

    stale = legacy (no key), or produced under a different model/prompt/settings.
    Unreadable files count as 'corrupt' and are also removed when purging.
    Entries whose only difference is chunk text/NER hints are not detectable
    here (they self-heal on the next build) and count as current.
    """
    counts = {"current": 0, "stale": 0, "legacy": 0, "corrupt": 0}
    cfg = _config_key()
    for path in sorted(CACHE_DIR.glob("*.json")):
        entry = _load_entry(path)
        if entry is None:
            kind = "corrupt"
        elif "_cache_key" not in entry:
            kind = "legacy"
        elif entry.get("_config_key") != cfg:
            kind = "stale"
        else:
            kind = "current"
        counts[kind] += 1
        if purge and kind != "current":
            path.unlink(missing_ok=True)
    return counts


def _main() -> None:
    ap = argparse.ArgumentParser(description="Inspect or purge the LLM extraction cache")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cache-status", action="store_true", help="Report current/stale entries")
    g.add_argument("--purge-stale", action="store_true", help="Delete stale/legacy/corrupt entries")
    args = ap.parse_args()
    counts = cache_status(purge=args.purge_stale)
    verb = "purged" if args.purge_stale else "found"
    stale = counts["stale"] + counts["legacy"] + counts["corrupt"]
    print(f"Cache {CACHE_DIR}: current={counts['current']} stale={counts['stale']} "
          f"legacy={counts['legacy']} corrupt={counts['corrupt']}")
    print(f"{stale} non-current entries {verb}; each will be re-extracted (LLM spend) on next build.")


if __name__ == "__main__":
    _main()


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
        temperature=_TEMPERATURE,
        max_tokens=_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    return _parse_json(raw)


def _call_anthropic(user_message: str) -> dict[str, Any]:
    import anthropic  # type: ignore

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=LLM_FALLBACK_MODEL,
        max_tokens=_MAX_TOKENS,
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
