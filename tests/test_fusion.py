import pytest

from docrag.retrieve import fusion
from tests.conftest import hits


def test_rrf_matches_hand_computation():
    dense = hits(("a", 0.9), ("b", 0.8), ("c", 0.7))
    sparse = hits(("c", 12.0), ("a", 9.0), ("d", 1.0))
    out = fusion.rrf([dense, sparse], k=60)
    scores = {h.chunk.chunk_id: h.score for h in out}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [h.chunk.chunk_id for h in out] == ["a", "c", "b", "d"]


def test_rrf_is_deterministic_on_ties_and_respects_limit():
    out = fusion.rrf([hits(("b", 1), ("a", 1)), hits(("a", 1), ("b", 1))], limit=1)
    assert [h.chunk.chunk_id for h in out] == ["a"]


def test_weighted_fusion_normalises_scales():
    dense = hits(("a", 0.9), ("b", 0.5))
    sparse = hits(("b", 30.0), ("c", 10.0))
    out = fusion.weighted(dense, sparse, dense_weight=0.5)
    scores = {h.chunk.chunk_id: h.score for h in out}
    assert scores == pytest.approx({"a": 0.5, "b": 0.5, "c": 0.0})
