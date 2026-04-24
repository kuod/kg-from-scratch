"""
Biomedical named entity recognition using scispaCy.

Primary model:  en_core_sci_lg   (UMLS semantic types)
Linker:         en_ner_bc5cdr_md  (BC5CDR drug/disease → UMLS CUI)

Install:
    pip install scispacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field

UMLS_LABEL_MAP: dict[str, str] = {
    "GENE_OR_GENE_PRODUCT": "Gene",
    "SIMPLE_CHEMICAL": "Drug",
    "DISEASE_OR_PHENOTYPIC_FEATURE": "Disease",
    "CELL": "CellType",
    "ORGANISM": "Organism",
    "CELLULAR_COMPONENT": "Mechanism",
    "MOLECULAR_FUNCTION": "Mechanism",
    "BIOLOGICAL_PROCESS": "Pathway",
    "CANCER": "Disease",
    "CHEMICAL": "Drug",
}

# BC5CDR entity types
_BC5CDR_MAP = {
    "DISEASE": "Disease",
    "CHEMICAL": "Drug",
}


@dataclass
class NEREntity:
    text: str
    label: str          # normalized biomedical type (e.g. "Gene", "Drug")
    raw_label: str      # original model label
    start_char: int
    end_char: int
    umls_cui: str | None = None
    canonical_name: str | None = None
    score: float = 1.0


@functools.lru_cache(maxsize=1)
def _load_nlp():
    """Load and cache spaCy pipeline (expensive; called once per process)."""
    import spacy  # type: ignore
    try:
        nlp = spacy.load("en_core_sci_lg")
    except OSError:
        raise RuntimeError(
            "scispaCy model not found. Install with:\n"
            "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
        )
    try:
        from scispacy.linking import EntityLinker  # type: ignore
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    except Exception:
        pass  # linker is optional; proceed without UMLS CUIs
    return nlp


@functools.lru_cache(maxsize=1)
def _load_bc5cdr_nlp():
    """Load BC5CDR model for high-recall drug/disease detection."""
    import spacy  # type: ignore
    try:
        return spacy.load("en_ner_bc5cdr_md")
    except OSError:
        return None


def extract_entities_ner(text: str) -> list[NEREntity]:
    """Return biomedical entities from text using scispaCy + BC5CDR."""
    nlp = _load_nlp()
    doc = nlp(text)

    entities: list[NEREntity] = []
    seen: set[tuple[int, int]] = set()

    for ent in doc.ents:
        label = UMLS_LABEL_MAP.get(ent.label_, ent.label_)
        umls_cui: str | None = None
        canonical: str | None = None

        # Extract UMLS link if linker is loaded
        if hasattr(ent._, "kb_ents") and ent._.kb_ents:
            top_kb = ent._.kb_ents[0]
            umls_cui = top_kb[0]
            score = top_kb[1]
            try:
                linker = nlp.get_pipe("scispacy_linker")
                entity_data = linker.kb.cui_to_entity.get(umls_cui)
                if entity_data:
                    canonical = entity_data.canonical_name
            except Exception:
                pass
        else:
            score = 1.0

        entities.append(NEREntity(
            text=ent.text,
            label=label,
            raw_label=ent.label_,
            start_char=ent.start_char,
            end_char=ent.end_char,
            umls_cui=umls_cui,
            canonical_name=canonical,
            score=score if isinstance(score, float) else 1.0,
        ))
        seen.add((ent.start_char, ent.end_char))

    # Supplement with BC5CDR for drug/disease
    bc5 = _load_bc5cdr_nlp()
    if bc5:
        bc5_doc = bc5(text)
        for ent in bc5_doc.ents:
            span = (ent.start_char, ent.end_char)
            if span not in seen:
                label = _BC5CDR_MAP.get(ent.label_, ent.label_)
                entities.append(NEREntity(
                    text=ent.text,
                    label=label,
                    raw_label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                ))
                seen.add(span)

    return entities


def format_ner_hints(entities: list[NEREntity]) -> str:
    """Format NER results as a compact hint string for LLM prompts."""
    if not entities:
        return "(none detected)"
    lines = []
    for ent in entities:
        cui_part = f" [UMLS:{ent.umls_cui}]" if ent.umls_cui else ""
        canonical_part = f" → {ent.canonical_name}" if ent.canonical_name else ""
        lines.append(f"  - {ent.text} ({ent.label}){canonical_part}{cui_part}")
    return "\n".join(lines)
