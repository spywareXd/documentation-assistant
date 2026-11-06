"""Rank fusion for hybrid search.

RRF (Cormack et al., 2009):  score(d) = sum over rankings r of  1 / (k + rank_r(d)),  rank starting at 1.
It only uses ranks, so it needs no score calibration between BM25 and cosine similarity.

Weighted: min-max normalise each list's scores to [0, 1], then w * dense + (1 - w) * sparse.
"""
from __future__ import annotations

from typing import Sequence

from docrag.index.qdrant_store import Hit


def rrf(rankings: Sequence[Sequence[Hit]], k: int = 60, limit: int | None = None) -> list[Hit]:
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            best.setdefault(cid, hit)
    order = sorted(scores, key=lambda c: (-scores[c], c))  # tie-break on id -> deterministic
    out = [Hit(best[c].chunk, scores[c]) for c in order]
    return out[:limit] if limit else out


def _minmax(hits: Sequence[Hit]) -> dict[str, float]:
    if not hits:
        return {}
    lo, hi = min(h.score for h in hits), max(h.score for h in hits)
    span = hi - lo or 1.0
    return {h.chunk.chunk_id: (h.score - lo) / span for h in hits}


def weighted(dense: Sequence[Hit], sparse: Sequence[Hit], dense_weight: float = 0.5,
             limit: int | None = None) -> list[Hit]:
    d, s = _minmax(dense), _minmax(sparse)
    chunks = {h.chunk.chunk_id: h.chunk for h in [*dense, *sparse]}
    scores = {c: dense_weight * d.get(c, 0.0) + (1 - dense_weight) * s.get(c, 0.0) for c in chunks}
    order = sorted(scores, key=lambda c: (-scores[c], c))
    out = [Hit(chunks[c], scores[c]) for c in order]
    return out[:limit] if limit else out
