"""
Tool-using RAG agent for biomedical knowledge graph Q&A.

Tools:
  rank_papers              - rank papers by query similarity
  search_entity            - find entities by name/type
  get_entity_relationships - edges for an entity with CI bounds
  search_chunks            - hybrid keyword + vector search
  get_chunk_text           - full text of a chunk
  find_path                - shortest graph path between entities
  get_edge_confidence      - Wilson CI for a specific relationship

Returns: (answer, chunk_ids, confidence_dict, tool_trace)
"""
from __future__ import annotations

import json
from typing import Any

import anthropic  # type: ignore

from src.config import AGENT_MODEL
from src.graph import GraphDB, ENTITY_LABELS
from src.embed import query_similar_chunks
from src.confidence import calibrate_answer_confidence

_SYSTEM_PROMPT = """\
You are a biomedical research assistant with access to a knowledge graph built from \
peer-reviewed papers. The graph contains: Genes, Proteins, Drugs, Diseases, Pathways, \
CellTypes, Organisms, and Mechanisms — and relationships between them.

Each relationship edge carries:
  - evidence_count: number of source passages supporting it
  - confidence_lower / confidence_upper: 95% Wilson CI on evidence proportion

When answering:
1. Use the tools to find relevant evidence before answering.
2. Cite every factual claim as [paper_id § section].
3. For relationship claims, report confidence bounds if available \
   (e.g. "CI: [0.62, 0.91]").
4. If evidence is insufficient or contradictory, say so explicitly.
5. Prefer specific claims over vague generalities.
"""

_TOOLS = [
    {
        "name": "rank_papers",
        "description": "Rank all papers in the graph by semantic similarity to the query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The research question or topic"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_entity",
        "description": "Search for biomedical entities by name or alias, optionally filtered by type.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Entity name or partial name"},
                "entity_type": {
                    "type": "string",
                    "enum": ENTITY_LABELS + ["Any"],
                    "description": "Entity type to filter by, or 'Any'",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_entity_relationships",
        "description": "Get all relationships for a named entity, with confidence interval bounds.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_type": {"type": "string", "enum": ENTITY_LABELS},
                "entity_name": {"type": "string"},
            },
            "required": ["entity_type", "entity_name"],
        },
    },
    {
        "name": "search_chunks",
        "description": "Hybrid keyword + semantic search over paper chunks, with optional graph-walk expansion.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
                "expand_graph": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, also return chunks for entities found in top results",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_chunk_text",
        "description": "Retrieve the full text of a specific chunk by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
            },
            "required": ["chunk_id"],
        },
    },
    {
        "name": "find_path",
        "description": "Find the shortest path between two biomedical entities in the graph.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_type": {"type": "string", "enum": ENTITY_LABELS},
                "source_name": {"type": "string"},
                "target_type": {"type": "string", "enum": ENTITY_LABELS},
                "target_name": {"type": "string"},
                "max_hops": {"type": "integer", "default": 4},
            },
            "required": ["source_type", "source_name", "target_type", "target_name"],
        },
    },
    {
        "name": "get_edge_confidence",
        "description": "Get the Wilson CI confidence bounds for a specific relationship edge.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_type": {"type": "string", "enum": ENTITY_LABELS},
                "source_name": {"type": "string"},
                "relationship": {"type": "string"},
                "target_type": {"type": "string", "enum": ENTITY_LABELS},
                "target_name": {"type": "string"},
            },
            "required": ["source_type", "source_name", "relationship", "target_type", "target_name"],
        },
    },
]


def ask_with_confidence(
    question: str,
    db: GraphDB,
    max_turns: int = 8,
) -> tuple[str, list[str], dict[str, Any], list[dict]]:
    """Answer a biomedical question using the knowledge graph.

    Returns:
        answer: the final answer text
        chunk_ids: list of chunk IDs used as evidence
        confidence: confidence dict from calibrate_answer_confidence
        tool_trace: list of {tool, input, output} dicts for display
    """
    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": question}]
    tool_trace: list[dict] = []
    used_chunk_ids: list[str] = []

    for _ in range(max_turns):
        response = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            tools=_TOOLS,  # type: ignore
            messages=messages,
        )

        # Collect assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason != "tool_use":
            break

        # Execute tool calls
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = _execute_tool(block.name, block.input, db)
            tool_trace.append({"tool": block.name, "input": block.input, "output": result})

            # Collect chunk IDs from search results
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict) and "id" in item:
                        cid = item["id"]
                        if "__" in str(cid) and cid not in used_chunk_ids:
                            used_chunk_ids.append(cid)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            })

        messages.append({"role": "user", "content": tool_results})

    # Extract final answer text
    answer = ""
    for block in response.content:
        if hasattr(block, "text"):
            answer += block.text

    # Calibrate confidence
    source_texts = _fetch_chunk_texts(used_chunk_ids[:8], db)
    confidence = calibrate_answer_confidence(question, answer, source_texts)

    return answer, used_chunk_ids, confidence, tool_trace


def _execute_tool(name: str, inputs: dict, db: GraphDB) -> Any:
    if name == "rank_papers":
        return _rank_papers(inputs.get("query", ""), inputs.get("top_k", 5), db)

    if name == "search_entity":
        return _search_entity(inputs.get("name", ""), inputs.get("entity_type", "Any"), db)

    if name == "get_entity_relationships":
        return db.get_entity_relationships(inputs["entity_type"], inputs["entity_name"])

    if name == "search_chunks":
        return _search_chunks(
            inputs.get("query", ""),
            inputs.get("top_k", 5),
            inputs.get("expand_graph", False),
            db,
        )

    if name == "get_chunk_text":
        rows = db.run("MATCH (c:Chunk {id: $id}) RETURN c.text AS text", id=inputs["chunk_id"])
        return rows[0] if rows else {"text": "Chunk not found"}

    if name == "find_path":
        return db.find_path(
            inputs["source_type"], inputs["source_name"],
            inputs["target_type"], inputs["target_name"],
            inputs.get("max_hops", 4),
        )

    if name == "get_edge_confidence":
        rel = inputs["relationship"].upper()
        rows = db.run(
            f"MATCH (s:{inputs['source_type']} {{name: $src}})-[r:{rel}]->(t:{inputs['target_type']} {{name: $tgt}}) "
            "RETURN r.confidence_lower AS lower, r.confidence_upper AS upper, r.evidence_count AS evidence_count",
            src=inputs["source_name"], tgt=inputs["target_name"],
        )
        return rows[0] if rows else {"lower": None, "upper": None, "evidence_count": 0}

    return {"error": f"Unknown tool: {name}"}


def _rank_papers(query: str, top_k: int, db: GraphDB) -> list[dict]:
    chunks = query_similar_chunks(db, query, top_k=top_k * 3)
    seen_papers: dict[str, float] = {}
    for c in chunks:
        pid = c["paper_id"]
        score = c.get("score", 0.0)
        if pid not in seen_papers or score > seen_papers[pid]:
            seen_papers[pid] = score
    ranked = sorted(seen_papers.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"paper_id": p, "score": s} for p, s in ranked]


def _search_entity(name: str, entity_type: str, db: GraphDB) -> list[dict]:
    labels = ENTITY_LABELS if entity_type == "Any" else [entity_type]
    results = []
    for label in labels:
        rows = db.run(
            f"MATCH (n:{label}) WHERE toLower(n.name) CONTAINS toLower($name) "
            "OR ANY(a IN n.aliases WHERE toLower(a) CONTAINS toLower($name)) "
            "RETURN n.name AS name, labels(n)[0] AS label, n.aliases AS aliases LIMIT 10",
            name=name,
        )
        results.extend(rows)
    return results


def _search_chunks(query: str, top_k: int, expand_graph: bool, db: GraphDB) -> list[dict]:
    # Semantic search
    semantic = query_similar_chunks(db, query, top_k=top_k)

    # Keyword search
    keyword = db.keyword_search_chunks(query, limit=top_k)

    # Merge and deduplicate by id
    seen: set[str] = set()
    combined = []
    for chunk in semantic + keyword:
        cid = chunk.get("id", "")
        if cid not in seen:
            seen.add(cid)
            combined.append(chunk)

    if expand_graph and combined:
        # Find entities mentioned in top chunks, then pull related chunks
        for chunk in combined[:3]:
            entities = db.run(
                "MATCH (e)-[:MENTIONED_IN]->(c:Chunk {id: $id}) RETURN e.name AS name, labels(e)[0] AS label",
                id=chunk.get("id", ""),
            )
            for ent in entities[:3]:
                extra = db.run(
                    f"MATCH (e {{name: $name}})-[:MENTIONED_IN]->(c:Chunk) "
                    "RETURN c.id AS id, c.paper_id AS paper_id, c.section AS section, c.text AS text LIMIT 3",
                    name=ent["name"],
                )
                for c in extra:
                    if c["id"] not in seen:
                        seen.add(c["id"])
                        combined.append(c)

    return combined[:top_k * 2]


def _fetch_chunk_texts(chunk_ids: list[str], db: GraphDB) -> list[str]:
    texts = []
    for cid in chunk_ids:
        rows = db.run("MATCH (c:Chunk {id: $id}) RETURN c.text AS text", id=cid)
        if rows:
            texts.append(rows[0]["text"])
    return texts
