"""Query-shape tests for GraphDB using a recording stub (no Neo4j needed)."""


def test_incoming_edges_query_points_at_named_entity(recdb):
    recdb.get_entity_relationships("Gene", "TP53")
    query, params = recdb.calls[0]
    first, second = query.split("UNION")
    assert "(s:Gene {name: $name})-[r]->(t)" in first
    # incoming half: another node points AT the named entity
    assert "(src)-[r]->(n:Gene {name: $name})" in second
    assert params == {"name": "TP53"}
