"""
Entity resolution: collapse aliases and case variants to canonical nodes.

Pass 1: APOC synonym merge using curated SYNONYM_GROUPS.
Pass 2: String normalization merge (lowercase + strip punctuation).
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.graph import GraphDB

# Curated alias -> canonical name maps, keyed per entity label so that an alias
# is only ever applied to nodes of a compatible type (a Gene called "MS" must
# not become a disease). Deliberately excluded:
#   * ambiguous short abbreviations (ms, ad, pd, tam, neu) that mean different
#     things in different fields;
#   * aliases that collapse distinct entities (pi3k, akt, erk, jnk, pdgf,
#     vegf165, "braf v600e", "alk fusion"), because they drop isoform/mutation
#     or family information.
_GENE_SYNONYMS: dict[str, str] = {
    "p53": "TP53",
    "tumor protein p53": "TP53",
    "trp53": "TP53",
    "her2": "ERBB2",
    "her-2": "ERBB2",
    "c-erbb2": "ERBB2",
    "kras2": "KRAS",
    "k-ras": "KRAS",
    "egfr": "EGFR",
    "her1": "EGFR",
    "erbb1": "EGFR",
    "rb1": "RB1",
    "pten": "PTEN",
    "mmac1": "PTEN",
    "brca-1": "BRCA1",
    "brca-2": "BRCA2",
    "vegf": "VEGFA",
    "vegf-a": "VEGFA",
    "tgfb": "TGFB1",
    "tgf-beta": "TGFB1",
    "tgf-β": "TGFB1",
    "nf-kb": "NFKB1",
    "nfkb": "NFKB1",
    "nf-κb": "NFKB1",
    "mtor": "MTOR",
    "m-tor": "MTOR",
    "erk2": "MAPK1",
    "pkb": "AKT1",
    "akt serine/threonine kinase 1": "AKT1",
}

_DISEASE_SYNONYMS: dict[str, str] = {
    "nsclc": "Non-small-cell lung carcinoma",
    "non-small cell lung cancer": "Non-small-cell lung carcinoma",
    "non small cell lung cancer": "Non-small-cell lung carcinoma",
    "sclc": "Small cell lung carcinoma",
    "small cell lung cancer": "Small cell lung carcinoma",
    "crc": "Colorectal carcinoma",
    "colorectal cancer": "Colorectal carcinoma",
    "hcc": "Hepatocellular carcinoma",
    "hepatocellular cancer": "Hepatocellular carcinoma",
    "glioblastoma multiforme": "Glioblastoma",
    "gbm": "Glioblastoma",
    "acute myeloid leukaemia": "Acute myeloid leukemia",
    "aml": "Acute myeloid leukemia",
    "cll": "Chronic lymphocytic leukemia",
    "cml": "Chronic myelogenous leukemia",
    "t2d": "Type 2 diabetes mellitus",
    "type 2 diabetes": "Type 2 diabetes mellitus",
    "t1d": "Type 1 diabetes mellitus",
    "type 1 diabetes": "Type 1 diabetes mellitus",
    "alzheimer": "Alzheimer's disease",
    "parkinson": "Parkinson's disease",
}

_DRUG_SYNONYMS: dict[str, str] = {
    "tamoxifen": "Tamoxifen",
    "cisplatin": "Cisplatin",
    "cddp": "Cisplatin",
    "doxorubicin": "Doxorubicin",
    "adriamycin": "Doxorubicin",
    "5-fu": "Fluorouracil",
    "5fu": "Fluorouracil",
    "fluorouracil": "Fluorouracil",
    "imatinib": "Imatinib",
    "gleevec": "Imatinib",
    "glivec": "Imatinib",
    "erlotinib": "Erlotinib",
    "tarceva": "Erlotinib",
    "gefitinib": "Gefitinib",
    "iressa": "Gefitinib",
    "bevacizumab": "Bevacizumab",
    "avastin": "Bevacizumab",
    "pembrolizumab": "Pembrolizumab",
    "keytruda": "Pembrolizumab",
    "nivolumab": "Nivolumab",
    "opdivo": "Nivolumab",
    "metformin": "Metformin",
    "glucophage": "Metformin",
}

_PATHWAY_SYNONYMS: dict[str, str] = {
    "wnt signaling": "Wnt signaling pathway",
    "wnt pathway": "Wnt signaling pathway",
    "mapk pathway": "MAPK signaling pathway",
    "ras/mapk": "MAPK signaling pathway",
    "pi3k/akt": "PI3K-Akt signaling pathway",
    "pi3k/akt/mtor": "PI3K-Akt signaling pathway",
    "p53 pathway": "TP53 signaling pathway",
    "notch signaling": "Notch signaling pathway",
    "hedgehog signaling": "Hedgehog signaling pathway",
    "hh signaling": "Hedgehog signaling pathway",
    "jak/stat": "JAK-STAT signaling pathway",
    "jak-stat": "JAK-STAT signaling pathway",
}

SYNONYM_GROUPS: dict[tuple[str, str], str] = {
    **{("Gene", k): v for k, v in _GENE_SYNONYMS.items()},
    **{("Protein", k): v for k, v in _GENE_SYNONYMS.items()},
    **{("Disease", k): v for k, v in _DISEASE_SYNONYMS.items()},
    **{("Drug", k): v for k, v in _DRUG_SYNONYMS.items()},
    **{("Pathway", k): v for k, v in _PATHWAY_SYNONYMS.items()},
}


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# APOC merge config. Relationship/node properties are combined only for the
# list-valued ones we know how to de-duplicate; everything else keeps the target
# node's value. evidence_count is discarded here and recomputed afterwards from
# source_chunk_ids, so 'combine' can never leave it as an array.
_MERGE_CONFIG = (
    "{properties: {`*`: 'discard', source_chunk_ids: 'combine', "
    "aliases: 'combine', evidence_count: 'discard'}, mergeRels: true}"
)


def _merge_nodes(db: "GraphDB", label: str, alias: str, canon: str) -> bool:
    """Merge `alias` into `canon` via APOC. Returns True on success.

    Failures (including APOC not being installed) are logged, never swallowed.
    """
    try:
        db.run(
            f"MATCH (alias:{label} {{name: $alias}}) "
            f"MATCH (canon:{label} {{name: $canon}}) "
            f"CALL apoc.refactor.mergeNodes([canon, alias], {_MERGE_CONFIG}) YIELD node "
            "SET node.name = $canon, "
            "    node.aliases = reduce(acc = [], x IN coalesce(node.aliases, []) + [$alias] | "
            "      CASE WHEN x IN acc OR x = $canon THEN acc ELSE acc + [x] END) "
            "RETURN node",
            alias=alias, canon=canon,
        )
        # Re-derive evidence from the de-duplicated chunk ids on touched edges.
        db.run(
            f"MATCH (n:{label} {{name: $canon}})-[r]-() "
            "WHERE r.source_chunk_ids IS NOT NULL "
            "WITH DISTINCT r, apoc.coll.toSet(apoc.coll.flatten([r.source_chunk_ids])) AS ids "
            "SET r.source_chunk_ids = ids, r.evidence_count = size(ids)",
            canon=canon,
        )
        return True
    except Exception:
        logger.exception("Failed to merge %s %r into %r", label, alias, canon)
        return False


def apoc_available(db: "GraphDB") -> bool:
    """True if the APOC procedures needed for merging are installed."""
    try:
        db.run("RETURN apoc.version() AS v")
        return True
    except Exception:
        logger.exception("APOC is not available; node merges will fail")
        return False


def resolve_synonyms(db: "GraphDB") -> int:
    """Pass 1: merge nodes whose name matches a (label, alias) SYNONYM_GROUPS key.

    Returns count of merges performed. Failed merges are logged.
    """
    from src.graph import ENTITY_LABELS

    merged = 0
    failed = 0
    for label in ENTITY_LABELS:
        nodes = db.run(f"MATCH (n:{label}) RETURN n.name AS name")
        for node in nodes:
            name_lower = node["name"].lower().strip()
            canonical = SYNONYM_GROUPS.get((label, name_lower))
            if canonical and canonical != node["name"]:
                canon_rows = db.run(
                    f"MATCH (c:{label} {{name: $canon}}) RETURN c.name AS name",
                    canon=canonical,
                )
                if canon_rows:
                    if _merge_nodes(db, label, node["name"], canonical):
                        merged += 1
                    else:
                        failed += 1
                else:
                    # Rename in place, remembering the old name as an alias
                    db.run(
                        f"MATCH (n:{label} {{name: $old}}) "
                        "SET n.aliases = reduce(acc = [], x IN coalesce(n.aliases, []) + [$old] | "
                        "  CASE WHEN x IN acc THEN acc ELSE acc + [x] END), "
                        "n.name=$new",
                        old=node["name"], new=canonical,
                    )
                    merged += 1
    if failed:
        logger.warning("Synonym pass: %d merge(s) failed", failed)
    return merged


def resolve_normalized_duplicates(db: "GraphDB") -> int:
    """Pass 2: merge nodes with identical normalized names.

    The node with the highest MENTIONED_IN edge count becomes canonical.
    Returns count of merges performed. Failed merges are logged.
    """
    from src.graph import ENTITY_LABELS

    merged = 0
    failed = 0
    for label in ENTITY_LABELS:
        nodes = db.run(
            f"MATCH (n:{label}) "
            "OPTIONAL MATCH (n)-[r:MENTIONED_IN]->() "
            "RETURN n.name AS name, count(r) AS mention_count "
            "ORDER BY mention_count DESC"
        )

        norm_to_canonical: dict[str, str] = {}
        for node in nodes:
            norm = _normalize(node["name"])
            if norm in norm_to_canonical:
                if _merge_nodes(db, label, node["name"], norm_to_canonical[norm]):
                    merged += 1
                else:
                    failed += 1
            else:
                norm_to_canonical[norm] = node["name"]
    if failed:
        logger.warning("Normalization pass: %d merge(s) failed", failed)
    return merged


def run_resolution(db: "GraphDB") -> None:
    """Run both resolution passes and print a summary."""
    if not apoc_available(db):
        print("  WARNING: APOC not available - alias merges will fail (renames still apply).")
    print("Pass 1: synonym merge...")
    n1 = resolve_synonyms(db)
    print(f"  Merged {n1} alias nodes.")

    print("Pass 2: normalization merge...")
    n2 = resolve_normalized_duplicates(db)
    print(f"  Merged {n2} duplicate nodes.")
