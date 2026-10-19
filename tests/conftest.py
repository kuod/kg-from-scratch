import pytest

from src.graph import GraphDB


class RecordingDB(GraphDB):
    """GraphDB that records queries instead of talking to Neo4j."""

    def __init__(self, rows=None):  # no driver
        self.calls = []
        self._rows = rows if rows is not None else []

    def run(self, query, **params):
        self.calls.append((query, params))
        return self._rows


@pytest.fixture
def recdb():
    return RecordingDB()
