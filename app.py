"""
Streamlit chatbot for biomedical knowledge graph Q&A.

Tabs:
  1. Ask      — chatbot with confidence intervals + PyVis subgraph
  2. Explorer — entity search + relationship table
  3. Papers   — browse chunks per paper
  4. Stats    — graph statistics
"""
from __future__ import annotations

import json
import streamlit as st
from pyvis.network import Network  # type: ignore
import tempfile, os

from src.graph import GraphDB, ENTITY_LABELS
from src.agent import ask_with_confidence

st.set_page_config(page_title="BioKG Chatbot", layout="wide")


@st.cache_resource
def get_db() -> GraphDB:
    return GraphDB()


db = get_db()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("BioKG Navigator")
    st.caption("Biomedical Knowledge Graph Q&A")
    st.divider()
    st.markdown("**Graph connection:** `bolt://localhost:7687`")
    try:
        stats = db.get_graph_stats()
        st.metric("Papers", stats.get("Paper", 0))
        st.metric("Chunks", stats.get("Chunk", 0))
        total_entities = sum(stats.get(l, 0) for l in ENTITY_LABELS)
        st.metric("Entities", total_entities)
    except Exception as e:
        st.error(f"Neo4j connection failed: {e}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_ask, tab_explore, tab_papers, tab_stats = st.tabs(["Ask", "Graph Explorer", "Papers", "Stats"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — ASK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_ask:
    st.header("Ask the Knowledge Graph")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render previous turns
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            _render_confidence(turn["confidence"])
            if turn.get("tool_trace"):
                with st.expander("Tool calls", expanded=False):
                    for t in turn["tool_trace"]:
                        st.markdown(f"**{t['tool']}**")
                        st.json(t["input"])
                        st.json(t["output"])

    question = st.chat_input("Ask about genes, drugs, diseases, pathways…")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.status("Searching knowledge graph…", expanded=True) as status:
                answer, chunk_ids, confidence, tool_trace = ask_with_confidence(question, db)
                status.update(label="Done", state="complete", expanded=False)

            st.markdown(answer)
            _render_confidence(confidence)

            if chunk_ids:
                with st.expander("Evidence chunks", expanded=False):
                    for cid in chunk_ids[:6]:
                        rows = db.run("MATCH (c:Chunk {id: $id}) RETURN c.paper_id AS pid, c.section AS sec, c.text AS text", id=cid)
                        if rows:
                            st.markdown(f"**[{rows[0]['pid']} § {rows[0]['sec']}]**")
                            st.caption(rows[0]["text"][:500])

            if tool_trace:
                with st.expander("Tool calls", expanded=False):
                    for t in tool_trace:
                        st.markdown(f"**{t['tool']}**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption("Input")
                            st.json(t["input"])
                        with col2:
                            st.caption("Output")
                            st.json(t["output"])

            # Knowledge subgraph for entity mentions
            if chunk_ids:
                _render_subgraph(chunk_ids[:3], db)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "tool_trace": tool_trace,
        })


def _render_confidence(conf: dict) -> None:
    label = conf.get("label", "Medium")
    score = conf.get("score", 50)
    lo = conf.get("lower_bound", 0.3)
    hi = conf.get("upper_bound", 0.7)
    rationale = conf.get("rationale", "")
    contradictions = conf.get("contradictions", "none")

    color = {"High": "green", "Medium": "orange", "Low": "red"}.get(label, "grey")
    badge = f":{color}[**{label}** confidence]"

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown(f"{badge} — score {score}/100")
        st.markdown(f"95% CI: **[{lo:.2f}, {hi:.2f}]**")
    with col2:
        with st.expander("Confidence rationale", expanded=False):
            st.write(rationale)
            if contradictions and contradictions != "none":
                st.warning(f"Contradictions: {contradictions}")


def _render_subgraph(chunk_ids: list[str], db: GraphDB) -> None:
    """Render a PyVis knowledge subgraph for entities in the given chunks."""
    entity_rows = []
    for cid in chunk_ids:
        rows = db.run(
            "MATCH (e)-[:MENTIONED_IN]->(c:Chunk {id: $id}) RETURN e.name AS name, labels(e)[0] AS label LIMIT 8",
            id=cid,
        )
        entity_rows.extend(rows)

    if not entity_rows:
        return

    net = Network(height="350px", width="100%", bgcolor="#0e1117", font_color="white")
    net.set_options('{"physics": {"stabilization": {"iterations": 50}}}')

    added_nodes: set[str] = set()
    label_colors = {
        "Gene": "#4e9af1", "Protein": "#7ecba1", "Drug": "#f4a261",
        "Disease": "#e76f51", "Pathway": "#a8dadc", "CellType": "#9b5de5",
        "Mechanism": "#f72585", "Organism": "#b5e48c",
    }

    for ent in entity_rows[:20]:
        name = ent["name"]
        lbl = ent["label"]
        if name not in added_nodes:
            net.add_node(name, label=name, color=label_colors.get(lbl, "#888"), title=lbl)
            added_nodes.add(name)

    # Add edges between visible entities
    if len(added_nodes) > 1:
        names_list = list(added_nodes)
        for i, src in enumerate(names_list):
            for tgt in names_list[i+1:]:
                rels = db.run(
                    "MATCH (s {name: $src})-[r]->(t {name: $tgt}) RETURN type(r) AS rel LIMIT 1",
                    src=src, tgt=tgt,
                )
                if rels:
                    net.add_edge(src, tgt, label=rels[0]["rel"], color="#888888")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        html = open(f.name).read()
    os.unlink(f.name)

    with st.expander("Knowledge subgraph", expanded=False):
        st.components.v1.html(html, height=370)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — GRAPH EXPLORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_explore:
    st.header("Graph Explorer")
    col1, col2 = st.columns([2, 1])
    with col1:
        entity_name = st.text_input("Entity name", placeholder="e.g. TP53, Imatinib, KRAS")
    with col2:
        entity_type = st.selectbox("Type", ["Any"] + ENTITY_LABELS)

    if entity_name:
        labels = ENTITY_LABELS if entity_type == "Any" else [entity_type]
        matches = []
        for lbl in labels:
            rows = db.run(
                f"MATCH (n:{lbl}) WHERE toLower(n.name) CONTAINS toLower($name) "
                "RETURN n.name AS name, labels(n)[0] AS label LIMIT 5",
                name=entity_name,
            )
            matches.extend(rows)

        if not matches:
            st.info("No entities found.")
        else:
            selected = st.selectbox("Select entity", [f"{m['name']} ({m['label']})" for m in matches])
            sel_name, sel_label = selected.rsplit(" (", 1)
            sel_label = sel_label.rstrip(")")

            rels = db.get_entity_relationships(sel_label, sel_name)
            if rels:
                import pandas as pd
                df = pd.DataFrame(rels)
                df["CI"] = df.apply(
                    lambda r: f"[{r['confidence_lower']:.2f}, {r['confidence_upper']:.2f}]"
                    if r.get("confidence_lower") is not None else "—", axis=1
                )
                st.dataframe(df[["rel_type", "tgt_label", "tgt_name", "evidence_count", "CI"]], use_container_width=True)
            else:
                st.info("No relationships found for this entity.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — PAPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_papers:
    st.header("Papers")
    papers = db.run("MATCH (p:Paper) RETURN p.id AS id, p.title AS title ORDER BY p.id")
    if not papers:
        st.info("No papers ingested yet. Drop .md files in data/papers/ and run build_graph.py.")
    else:
        paper_ids = [p["id"] for p in papers]
        selected_paper = st.selectbox("Select paper", paper_ids)
        chunks = db.run(
            "MATCH (c:Chunk {paper_id: $pid}) RETURN c.id AS id, c.section AS section, c.text AS text ORDER BY c.chunk_index",
            pid=selected_paper,
        )
        for chunk in chunks:
            with st.expander(chunk["section"], expanded=False):
                st.write(chunk["text"])
                entities = db.run(
                    "MATCH (e)-[:MENTIONED_IN]->(c:Chunk {id: $id}) RETURN e.name AS name, labels(e)[0] AS label",
                    id=chunk["id"],
                )
                if entities:
                    tags = " ".join(f"`{e['name']} ({e['label']})`" for e in entities)
                    st.markdown(tags)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — STATS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_stats:
    st.header("Graph Statistics")
    try:
        stats = db.get_graph_stats()
        cols = st.columns(4)
        node_labels = ["Paper", "Chunk"] + ENTITY_LABELS
        for i, lbl in enumerate(node_labels):
            cols[i % 4].metric(lbl, stats.get(lbl, 0))

        st.subheader("Relationship types")
        rels = stats.get("relationships", {})
        if rels:
            import pandas as pd
            df = pd.DataFrame(list(rels.items()), columns=["Relationship", "Count"])
            st.bar_chart(df.set_index("Relationship"))
        else:
            st.info("No relationships in graph yet.")
    except Exception as e:
        st.error(f"Could not load stats: {e}")
