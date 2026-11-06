from docrag.config import DynamicKCfg
from docrag.retrieve.dynamic_k import select
from tests.conftest import hits

H = hits(("a", 0.95), ("b", 0.9), ("c", 0.4), ("d", 0.2), ("e", 0.1))


def ids(xs):
    return [h.chunk.chunk_id for h in xs]


def test_fixed():
    assert ids(select(H, DynamicKCfg(policy="fixed"), top_k=3)) == ["a", "b", "c"]


def test_threshold_with_bounds():
    cfg = DynamicKCfg(policy="score_threshold", threshold=0.3, min_k=1, max_k=10)
    assert ids(select(H, cfg, 5)) == ["a", "b", "c"]
    cfg = DynamicKCfg(policy="score_threshold", threshold=0.99, min_k=2, max_k=10)
    assert ids(select(H, cfg, 5)) == ["a", "b"]


def test_relative_drop():
    cfg = DynamicKCfg(policy="relative_drop", relative_drop=0.5, min_k=1, max_k=10)
    assert ids(select(H, cfg, 5)) == ["a", "b"]
    cfg = DynamicKCfg(policy="relative_drop", relative_drop=0.0, min_k=1, max_k=3)
    assert ids(select(H, cfg, 5)) == ["a", "b", "c"]
