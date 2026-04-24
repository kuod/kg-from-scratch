"""
Entity resolution: collapse aliases and case variants to canonical nodes.

Pass 1: APOC synonym merge using curated SYNONYM_GROUPS.
Pass 2: String normalization merge (lowercase + strip punctuation).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph import GraphDB

# Maps alias → canonical name (HGNC symbols, MONDO disease names, etc.)
SYNONYM_GROUPS: dict[str, str] = {
    # Genes / Proteins
    "p53": "TP53",
    "tumor protein p53": "TP53",
    "trp53": "TP53",
    "her2": "ERBB2",
    "her-2": "ERBB2",
    "neu": "ERBB2",
    "c-erbb2": "ERBB2",
    "kras2": "KRAS",
    "k-ras": "KRAS",
    "braf v600e": "BRAF",
    "egfr": "EGFR",
    "her1": "EGFR",
    "erbb1": "EGFR",
    "rb1": "RB1",
    "retinoblastoma": "RB1",
    "pten": "PTEN",
    "mmac1": "PTEN",
    "brca-1": "BRCA1",
    "brca-2": "BRCA2",
    "alk fusion": "ALK",
    "vegf": "VEGFA",
    "vegf-a": "VEGFA",
    "vegf165": "VEGFA",
    "pdgf": "PDGFA",
    "tgfb": "TGFB1",
    "tgf-beta": "TGFB1",
    "tgf-β": "TGFB1",
    "nf-kb": "NFKB1",
    "nfkb": "NFKB1",
    "nf-κb": "NFKB1",
    "mtor": "MTOR",
    "m-tor": "MTOR",
    "pi3k": "PIK3CA",
    "erk": "MAPK1",
    "erk2": "MAPK1",
    "jnk": "MAPK8",
    "jnk1": "MAPK8",
    "akt": "AKT1",
    "pkb": "AKT1",
    "akt serine/threonine kinase 1": "AKT1",
    # Diseases
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
    "ad": "Alzheimer's disease",
    "alzheimer": "Alzheimer's disease",
    "pd": "Parkinson's disease",
    "parkinson": "Parkinson's disease",
    "ms": "Multiple sclerosis",
    # Drugs / Chemicals
    "tamoxifen": "Tamoxifen",
    "tam": "Tamoxifen",
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
    # Pathways
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


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def resolve_synonyms(db: "GraphDB") -> int:
    """Pass 1: merge nodes whose normalized name matches a SYNONYM_GROUPS key.

    Returns count of merges performed.
    """
    from src.graph import ENTITY_LABELS

    merged = 0
    for label in ENTITY_LABELS:
        nodes = db.run(f"MATCH (n:{label}) RETURN n.name AS name")
        for node in nodes:
            name_lower = node["name"].lower().strip()
            canonical = SYNONYM_GROUPS.get(name_lower)
            if canonical and canonical.lower() != name_lower:
                # Check if canonical node exists; if not, rename in place
                canon_rows = db.run(
                    f"MATCH (c:{label} {{name: $canon}}) RETURN c.name AS name",
                    canon=canonical,
                )
                if canon_rows:
                    # Merge alias into canonical using APOC
                    try:
                        db.run(
                            f"MATCH (alias:{label} {{name: $alias}}) "
                            f"MATCH (canon:{label} {{name: $canon}}) "
                            "CALL apoc.refactor.mergeNodes([canon, alias], "
                            "  {properties: 'combine', mergeRels: true}) YIELD node "
                            "SET node.name=$canon "
                            "RETURN node",
                            alias=node["name"], canon=canonical,
                        )
                        merged += 1
                    except Exception:
                        pass
                else:
                    # Rename the alias node to canonical
                    db.run(
                        f"MATCH (n:{label} {{name: $old}}) SET n.name=$new",
                        old=node["name"], new=canonical,
                    )
                    merged += 1
    return merged


def resolve_normalized_duplicates(db: "GraphDB") -> int:
    """Pass 2: merge nodes with identical normalized names.

    The node with the highest MENTIONED_IN edge count becomes canonical.
    Returns count of merges performed.
    """
    from src.graph import ENTITY_LABELS

    merged = 0
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
                canonical = norm_to_canonical[norm]
                try:
                    db.run(
                        f"MATCH (alias:{label} {{name: $alias}}) "
                        f"MATCH (canon:{label} {{name: $canon}}) "
                        "CALL apoc.refactor.mergeNodes([canon, alias], "
                        "  {properties: 'combine', mergeRels: true}) YIELD node "
                        "SET node.name=$canon "
                        "RETURN node",
                        alias=node["name"], canon=canonical,
                    )
                    merged += 1
                except Exception:
                    pass
            else:
                norm_to_canonical[norm] = node["name"]
    return merged


def run_resolution(db: "GraphDB") -> None:
    """Run both resolution passes and print a summary."""
    print("Pass 1: synonym merge...")
    n1 = resolve_synonyms(db)
    print(f"  Merged {n1} alias nodes.")

    print("Pass 2: normalization merge...")
    n2 = resolve_normalized_duplicates(db)
    print(f"  Merged {n2} duplicate nodes.")
