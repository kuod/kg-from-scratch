"""Tests for LLM-output validation and the extraction cache (no network)."""
import json

import pytest

import src.extract as extract
from src.ingest import Chunk


@pytest.mark.parametrize(
    "raw",
    [
        '{"entities":[{"name":null,"type":"Gene"}]}',
        "[1]",
        '{"entities":[{"name":"x","type":"Gene","aliases":null}]}',
        "garbage {x}",
        "",
        '{"entities": "nope", "relationships": 3}',
        '{"relationships":[{"relationship":null,"source_name":"a","target_name":"b"}]}',
    ],
)
def test_parse_json_never_raises(raw):
    out = extract._parse_json(raw)
    assert set(out) == {"entities", "relationships"}


def test_parse_json_keeps_valid_items_and_strips_fences():
    raw = '```json\n{"entities":[{"name":" TP53 ","type":"Gene","aliases":["p53",1]}]}\n```'
    out = extract._parse_json(raw)
    assert out["entities"] == [{"name": "TP53", "type": "Gene", "aliases": ["p53"]}]


def test_corrupt_cache_file_is_re_extracted(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "CACHE_DIR", tmp_path)
    chunk = Chunk("p", "s", 0, "some text", 0)
    (tmp_path / "p__0000.json").write_text('{"entities": [')  # truncated
    monkeypatch.setattr(extract, "_call_llm", lambda *a: {"entities": [], "relationships": []})
    out = extract.extract_from_chunk(chunk, [])
    assert out["entities"] == []
    assert json.loads((tmp_path / "p__0000.json").read_text())["_text_key"]
    assert not list(tmp_path.glob("*.tmp"))


def _chunk():
    return Chunk("p", "s", 0, "some text", 0)


def test_cache_key_stable_and_sensitive(monkeypatch):
    base = extract._cache_key("txt")
    assert extract._cache_key("txt") == base
    assert extract._cache_key("other") != base
    monkeypatch.setattr(extract, "LLM_MODEL", "another-model")
    assert extract._cache_key("txt") != base
    monkeypatch.undo()
    monkeypatch.setattr(extract, "_SYSTEM_PROMPT", extract._SYSTEM_PROMPT + " tweak")
    assert extract._cache_key("txt") != base
    monkeypatch.undo()
    monkeypatch.setattr(extract, "PROMPT_VERSION", "999")
    assert extract._cache_key("txt") != base
    monkeypatch.undo()
    monkeypatch.setattr(extract, "LLM_FALLBACK_MODEL", "other-fallback")
    assert extract._cache_key("txt") != base


def test_model_change_causes_re_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(
        extract, "_call_llm", lambda *a: calls.append(1) or {"entities": [], "relationships": []}
    )
    extract.extract_from_chunk(_chunk(), [])
    extract.extract_from_chunk(_chunk(), [])
    assert len(calls) == 1  # second call served from cache
    monkeypatch.setattr(extract, "LLM_MODEL", "swapped")
    extract.extract_from_chunk(_chunk(), [])
    assert len(calls) == 2


def test_legacy_cache_file_is_miss_with_log(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(extract, "CACHE_DIR", tmp_path)
    (tmp_path / "p__0000.json").write_text(
        json.dumps({"entities": [{"name": "OLD"}], "relationships": [], "_text_key": "abc"})
    )
    monkeypatch.setattr(extract, "_call_llm", lambda *a: {"entities": [], "relationships": []})
    with caplog.at_level("WARNING"):
        out = extract.extract_from_chunk(_chunk(), [])
    assert out["entities"] == []
    assert "Legacy cache entry" in caplog.text


def test_cache_status_and_purge(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(extract, "_call_llm", lambda *a: {"entities": [], "relationships": []})
    extract.extract_from_chunk(_chunk(), [])  # current
    (tmp_path / "old__0000.json").write_text(json.dumps({"_text_key": "x"}))  # legacy
    (tmp_path / "bad__0000.json").write_text("{")  # corrupt
    (tmp_path / "mod__0000.json").write_text(
        json.dumps({"_cache_key": "k", "_config_key": "different"})
    )  # stale
    assert extract.cache_status() == {"current": 1, "stale": 1, "legacy": 1, "corrupt": 1}
    assert len(list(tmp_path.glob("*.json"))) == 4  # status does not delete
    extract.cache_status(purge=True)
    assert [p.name for p in tmp_path.glob("*.json")] == ["p__0000.json"]
