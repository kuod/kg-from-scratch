"""Query-shape tests for GraphDB using a recording stub (no Neo4j needed)."""


def test_incoming_edges_query_points_at_named_entity(recdb):
    recdb.get_entity_relationships("Gene", "TP53")
    query, params = recdb.calls[0]
    first, second = query.split("UNION")
    assert "(s:Gene {name: $name})-[r]->(t)" in first
    # incoming half: another node points AT the named entity
    assert "(src)-[r]->(n:Gene {name: $name})" in second
    assert params == {"name": "TP53"}


def test_upsert_entity_merges_aliases_instead_of_overwriting(recdb):
    recdb.upsert_entity("Gene", "TP53")  # no aliases, as for relationship endpoints
    query, params = recdb.calls[0]
    assert "n.aliases=$aliases" not in query.replace(" ", "")
    assert "coalesce(n.aliases, [])" in query
    assert params["aliases"] == []


def test_upsert_entity_still_sets_extra_props(recdb):
    recdb.upsert_entity("Gene", "TP53", aliases=["p53"], umls_cui="C0079419")
    query, params = recdb.calls[0]
    assert "n.umls_cui=$umls_cui" in query
    assert params["umls_cui"] == "C0079419"


def test_relationship_evidence_only_increments_for_new_chunk(recdb):
    recdb.upsert_relationship("Gene", "MDM2", "INHIBITS", "Gene", "TP53", "p__0000")
    query, _ = recdb.calls[0]
    assert "WHEN $chunk_id IN r.source_chunk_ids THEN r.evidence_count ELSE" in " ".join(query.split())


def test_mention_count_is_not_incremented_on_rerun(recdb):
    recdb.upsert_mention("Gene", "TP53", "p__0000")
    assert "r.count+1" not in recdb.calls[0][0]


import pytest


@pytest.mark.parametrize("bad", ["Gene) DETACH DELETE (x", "Chunk", "", "gene"])
def test_unknown_labels_are_rejected(recdb, bad):
    with pytest.raises(ValueError):
        recdb.get_entity_relationships(bad, "TP53")
    with pytest.raises(ValueError):
        recdb.find_path(bad, "a", "Gene", "b")
    with pytest.raises(ValueError):
        recdb.get_edge_confidence(bad, "a", "INHIBITS", "Gene", "b")
    with pytest.raises(ValueError):
        recdb.upsert_mention(bad, "a", "c")
    assert recdb.calls == []  # nothing reached the database


def test_unknown_relationship_type_rejected_for_edge_confidence(recdb):
    with pytest.raises(ValueError):
        recdb.get_edge_confidence("Gene", "a", "X]-() DETACH DELETE (n //", "Gene", "b")
    assert recdb.calls == []


def test_find_path_max_hops_is_bounded_int(recdb):
    recdb.find_path("Gene", "a", "Gene", "b", max_hops="3")
    assert "[*1..3]" in recdb.calls[0][0]
    with pytest.raises(ValueError):
        recdb.find_path("Gene", "a", "Gene", "b", max_hops="1] DELETE")
