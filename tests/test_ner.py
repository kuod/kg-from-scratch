"""Tests for NER module (mocked to avoid requiring scispaCy install in CI)."""
import pytest
from unittest.mock import patch, MagicMock

from src.ner import NEREntity, format_ner_hints, UMLS_LABEL_MAP


def test_ner_entity_dataclass():
    ent = NEREntity(
        text="TP53",
        label="Gene",
        raw_label="GENE_OR_GENE_PRODUCT",
        start_char=0,
        end_char=4,
    )
    assert ent.text == "TP53"
    assert ent.label == "Gene"
    assert ent.umls_cui is None


def test_format_ner_hints_empty():
    result = format_ner_hints([])
    assert "none" in result.lower()


def test_format_ner_hints_with_entities():
    entities = [
        NEREntity("TP53", "Gene", "GENE_OR_GENE_PRODUCT", 0, 4, umls_cui="C0079419", canonical_name="TP53"),
        NEREntity("Imatinib", "Drug", "SIMPLE_CHEMICAL", 10, 18),
    ]
    result = format_ner_hints(entities)
    assert "TP53" in result
    assert "Gene" in result
    assert "Imatinib" in result
    assert "Drug" in result
    assert "UMLS:C0079419" in result


def test_umls_label_map_covers_key_types():
    assert "GENE_OR_GENE_PRODUCT" in UMLS_LABEL_MAP
    assert "SIMPLE_CHEMICAL" in UMLS_LABEL_MAP
    assert "DISEASE_OR_PHENOTYPIC_FEATURE" in UMLS_LABEL_MAP
    assert UMLS_LABEL_MAP["GENE_OR_GENE_PRODUCT"] == "Gene"
    assert UMLS_LABEL_MAP["SIMPLE_CHEMICAL"] == "Drug"
    assert UMLS_LABEL_MAP["DISEASE_OR_PHENOTYPIC_FEATURE"] == "Disease"
