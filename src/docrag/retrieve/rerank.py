"""Cross-encoder reranking: scores (query, chunk) pairs jointly, far more precise than
bi-encoder similarity but too slow for the whole corpus, so it only sees fused candidates."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from docrag.config import RerankCfg
from docrag.index.qdrant_store import Hit


class Reranker:
    def __init__(self, cfg: RerankCfg):
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg
        self.model = CrossEncoder(cfg.model, device=device, max_length=cfg.max_length)
        if device == "cuda":
            self.model.model.half()

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return []
        # activation_fn=None -> the model's default head activation (sigmoid), i.e. probabilities in [0, 1]
        probs = np.asarray(self.model.predict(
            [(query, h.chunk.embed_text()) for h in hits],
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
        ), dtype=np.float64)
        order = np.argsort(-probs, kind="stable")
        return [Hit(hits[i].chunk, float(probs[i])) for i in order]


@lru_cache(maxsize=2)
def _cached(cfg_json: str) -> Reranker:
    return Reranker(RerankCfg.model_validate_json(cfg_json))


def get_reranker(cfg: RerankCfg) -> Reranker:
    return _cached(cfg.model_dump_json())
