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
