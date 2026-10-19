from src.clean_graph import clean_graph


class Recorder:
    def __init__(self):
        self.queries = []

    def run(self, q, **p):
        self.queries.append(q)
        return []


def test_trivial_name_filter_excludes_paper_and_chunk_for_both_conditions():
    db = Recorder()
    clean_graph(db)
    q = next(q for q in db.queries if "size(n.name)" in q)
    # the OR must be grouped so "AND NOT n:Paper" applies to both branches
    assert "WHERE (size(n.name) <= 1 OR n.name =~ '^[0-9.]+$') AND NOT n:Paper" in q
