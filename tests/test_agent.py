"""Agent error-path tests with a stubbed Anthropic client and DB."""
from types import SimpleNamespace as NS

import src.agent as agent


class StubDB:
    def run(self, *a, **k):
        return []

    def keyword_search_chunks(self, kw, limit=10):
        return [{"id": "p__0000", "paper_id": "p", "section": "s", "text": "kw hit"}]


def _client(responses):
    it = iter(responses)
    return NS(messages=NS(create=lambda **kw: next(it)))


def _tool_use(name="get_chunk_text", inp=None):
    return NS(
        stop_reason="tool_use",
        content=[NS(type="tool_use", id="t1", name=name, input=inp or {})],
    )


def test_tool_exception_is_reported_not_raised(monkeypatch):
    final = NS(stop_reason="end_turn", content=[NS(type="text", text="done")])
    # get_chunk_text without chunk_id -> KeyError inside the tool
    monkeypatch.setattr(agent.anthropic, "Anthropic", lambda: _client([_tool_use(), final]))
    monkeypatch.setattr(agent, "calibrate_answer_confidence", lambda *a, **k: {})
    answer, _, _, trace = agent.ask_with_confidence("q", StubDB())
    assert answer == "done"
    assert "error" in trace[0]["output"]


def test_turn_limit_yields_explicit_message(monkeypatch):
    monkeypatch.setattr(
        agent.anthropic, "Anthropic", lambda: _client([_tool_use(inp={"chunk_id": "x"})] * 2)
    )
    monkeypatch.setattr(agent, "calibrate_answer_confidence", lambda *a, **k: {})
    answer, *_ = agent.ask_with_confidence("q", StubDB(), max_turns=2)
    assert "No final answer" in answer


def test_search_chunks_falls_back_to_keyword_without_vector_index(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no such index")

    monkeypatch.setattr(agent, "query_similar_chunks", boom)
    out = agent._search_chunks("kw", 5, False, StubDB())
    assert out and out[0]["id"] == "p__0000"
