"""Tests for entity resolution (issue #18). Uses a fake DB; no Neo4j needed."""
import logging

from src import resolve
from src.resolve import SYNONYM_GROUPS, resolve_synonyms


class FakeDB:
    def __init__(self, nodes=None, fail_merge=False):
        self.nodes = nodes or {}  # label -> list of names
        self.fail_merge = fail_merge
        self.queries: list[tuple[str, dict]] = []

    def run(self, query, **params):
        self.queries.append((query, params))
        if "RETURN n.name AS name" in query and "MATCH (n:" in query and "OPTIONAL" not in query:
            label = query.split("(n:")[1].split(")")[0]
            return [{"name": n} for n in self.nodes.get(label, [])]
        if "RETURN c.name AS name" in query:
            label = query.split("(c:")[1].split(" ")[0]
            return [{"name": params["canon"]}] if params["canon"] in self.nodes.get(label, []) else []
        if "mergeNodes" in query and self.fail_merge:
            raise RuntimeError("no APOC")
        return []


def test_ambiguous_short_keys_removed():
    keys = {k for _, k in SYNONYM_GROUPS}
    for bad in ("ms", "ad", "pd", "tam", "neu", "pi3k", "akt", "erk", "pdgf",
                "braf v600e", "vegf165"):
        assert bad not in keys


def test_map_is_keyed_by_label():
    assert ("Disease", "nsclc") in SYNONYM_GROUPS
    assert ("Gene", "nsclc") not in SYNONYM_GROUPS
    assert ("Disease", "p53") not in SYNONYM_GROUPS


def test_gene_named_like_disease_not_renamed():
    db = FakeDB({"Gene": ["MS"], "Disease": ["NSCLC"]})
    resolve_synonyms(db)
    renames = [p for q, p in db.queries if "SET n.aliases" in q]
    assert renames == [{"old": "NSCLC", "new": "Non-small-cell lung carcinoma"}]


def test_rename_records_alias():
    db = FakeDB({"Gene": ["p53"]})
    assert resolve_synonyms(db) == 1
    q = next(q for q, _ in db.queries if "n.name=$new" in q)
    assert "n.aliases" in q and "$old" in q


def test_merge_failure_is_logged_not_swallowed(caplog):
    db = FakeDB({"Gene": ["p53", "TP53"]}, fail_merge=True)
    with caplog.at_level(logging.ERROR, logger="src.resolve"):
        merged = resolve_synonyms(db)
    assert merged == 0
    assert "Failed to merge" in caplog.text


def test_merge_query_protects_evidence_count_and_adds_alias():
    db = FakeDB({"Gene": ["p53", "TP53"]})
    assert resolve_synonyms(db) == 1
    merge_q = next(q for q, _ in db.queries if "mergeNodes" in q)
    assert "evidence_count: 'discard'" in merge_q
    assert "node.aliases" in merge_q
    recompute = next(q for q, _ in db.queries if "r.evidence_count = size(ids)" in q)
    assert "source_chunk_ids" in recompute


def test_apoc_available_false_when_missing(caplog):
    class NoApoc(FakeDB):
        def run(self, query, **p):
            raise RuntimeError("Unknown function apoc.version")
    assert resolve.apoc_available(NoApoc()) is False
