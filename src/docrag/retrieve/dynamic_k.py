"""Dynamic k: decide how many chunks to send to the LLM from the scores, not a fixed number.

Needs calibrated scores (reranker sigmoid probabilities in [0, 1]); RRF or cosine scores
are not comparable across queries, so the retriever only applies adaptive policies after reranking.
- fixed:           the caller's top_k
- score_threshold: keep every chunk with score >= threshold
- relative_drop:   keep chunks while score >= ratio * best score (stops at the "elbow")
Both adaptive policies are clamped to [min_k, max_k] so an easy question gets 2 chunks
and a broad one can get up to 10.
"""
from __future__ import annotations

from docrag.config import DynamicKCfg
from docrag.index.qdrant_store import Hit


def select(hits: list[Hit], cfg: DynamicKCfg, top_k: int) -> list[Hit]:
    if cfg.policy == "fixed" or not hits:
        return hits[:top_k]
    if cfg.policy == "score_threshold":
        n = sum(1 for h in hits if h.score >= cfg.threshold)
    elif cfg.policy == "relative_drop":
        best = hits[0].score
        n = sum(1 for h in hits if h.score >= cfg.relative_drop * best)
    else:
        raise ValueError(cfg.policy)
    n = max(cfg.min_k, min(cfg.max_k, n))
    return hits[:n]
