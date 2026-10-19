"""--resume must only skip chunks that were fully processed."""
import src.build_graph as bg


class FakeDB:
    def __init__(self, processed=()):
        self.processed = set(processed)
        self.chunks = []
        self.fail_on_extract = False

    def upsert_paper(self, *a, **k): pass
    def upsert_chunk(self, chunk_id, **k): self.chunks.append(chunk_id)
    def upsert_entity(self, *a, **k): pass
    def upsert_mention(self, *a, **k): pass
    def upsert_relationship(self, *a, **k): pass
    def mark_chunk_processed(self, cid): self.processed.add(cid)
    def is_chunk_processed(self, cid): return cid in self.processed


def _setup(tmp_path, monkeypatch, extract):
    md = tmp_path / "p.md"
    md.write_text("## Z\n\nzeta text\n\n## A\n\nalpha text\n\n## B\n\nbeta text\n")
    monkeypatch.setattr(bg, "extract_entities_ner", lambda t: [])
    monkeypatch.setattr(bg, "extract_from_chunk", extract)
    return md


def test_interrupted_chunk_is_retried_on_resume(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky(chunk, ents):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("rate limited")
        return {"entities": [], "relationships": []}

    md = _setup(tmp_path, monkeypatch, flaky)
    db = FakeDB()
    try:
        bg._process_paper(md, db, resume=False)
    except RuntimeError:
        pass
    assert db.processed == {"p__0000"}  # chunk 1 upserted but NOT marked done

    db.chunks.clear()
    bg._process_paper(md, db, resume=True)
    assert db.chunks == ["p__0001", "p__0002"]  # finished chunk 0 skipped, rest redone
    assert db.processed == {"p__0000", "p__0001", "p__0002"}
