import pytest

from docrag.index.qdrant_store import Hit
from docrag.ingest.chunkers import Chunk


def make_chunk(cid: str, url: str = "u", start: int = 0, end: int = 10, text: str = "x") -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", source="s", url=url, title="T", headings=[], text=text, start=start, end=end)


def hits(*ids_scores):
    return [Hit(make_chunk(i), s) for i, s in ids_scores]


@pytest.fixture
def mk():
    return make_chunk
