"""The configurable retriever: every retrieval experiment is a config of this class.

    query transform -> per-query candidate search (dense | sparse | tfidf | hybrid + fusion)
    -> multi-query fusion -> cross-encoder rerank -> dynamic-k cut

Returns the full ranked list (for recall@k at several k) and the selected chunks (sent to the LLM),
plus wall-clock timings per stage.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from docrag.config import Config
from docrag.index.qdrant_store import Hit, QdrantStore
from docrag.retrieve import dynamic_k, fusion


@dataclass
class RetrievalResult:
    query: str
    search_queries: list[str]
    ranked: list[Hit]  # after fusion (+ rerank), before the k cut
    selected: list[Hit]  # what the generator sees
    timings_ms: dict[str, float] = field(default_factory=dict)
    hyde_passage: str | None = None


class _Timer:
    """Accumulates wall-clock ms per stage: `with timer("dense"): ...`."""

    def __init__(self):
        self.t: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t0) * 1000


class Retriever:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rcfg = cfg.retrieval
        self.store = QdrantStore(cfg)
        self._embedder = None
        self._reranker = None
        self._tfidf = None
        self._llm = None

    # lazy components ---------------------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            from docrag.index.embedder import get_embedder

            self._embedder = get_embedder(self.cfg.embedding)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from docrag.retrieve.rerank import get_reranker

            self._reranker = get_reranker(self.rcfg.rerank)
        return self._reranker

    @property
    def tfidf(self):
        if self._tfidf is None:
            from docrag.ingest.pipeline import load_chunks
            from docrag.retrieve.tfidf import TfidfRetriever

            self._tfidf = TfidfRetriever(load_chunks(self.cfg))
        return self._tfidf

    @property
    def llm(self):
        if self._llm is None:
            from docrag.llm import get_llm

            self._llm = get_llm(self.cfg)
        return self._llm

    def warmup(self) -> None:
        """Load models before timing anything."""
        if self.rcfg.mode in ("dense", "hybrid"):
            self.embedder.embed_query("warmup")
        if self.rcfg.mode in ("sparse", "hybrid"):
            self.store.sparse.encode_query("warmup")
        if self.rcfg.mode == "tfidf":
            self.tfidf.search("warmup", 1)
        if self.rcfg.rerank.enabled:
            self.reranker.model.predict([("warmup", "warmup")])

    # search ------------------------------------------------------------------
    def _candidates(self, q: str, timer: _Timer, dense_text: str | None = None) -> list[Hit]:
        n, mode = self.rcfg.candidates, self.rcfg.mode
        if mode == "tfidf":
            with timer("tfidf"):
                return self.tfidf.search(q, n)
        dense_hits, sparse_hits = [], []
        if mode in ("dense", "hybrid"):
            with timer("embed_query"):
                vec = (self.embedder.embed_documents([dense_text])[0] if dense_text
                       else self.embedder.embed_query(q))
            if mode == "hybrid" and self.rcfg.fusion == "qdrant_rrf":
                with timer("hybrid_native"):
                    return self.store.search_hybrid_native(vec, q, limit=n, candidates=n)
            with timer("dense"):
                dense_hits = self.store.search_dense(vec, n)
        if mode in ("sparse", "hybrid"):
            with timer("sparse"):
                sparse_hits = self.store.search_sparse(q, n)
        if mode == "dense":
            return dense_hits
        if mode == "sparse":
            return sparse_hits
        with timer("fusion"):
            if self.rcfg.fusion == "weighted":
                return fusion.weighted(dense_hits, sparse_hits, self.rcfg.dense_weight, limit=n)
            return fusion.rrf([dense_hits, sparse_hits], k=self.rcfg.rrf_k, limit=n)

    def retrieve(self, query: str, history: list[dict] | None = None) -> RetrievalResult:
        timer = _Timer()
        t_all = time.perf_counter()
        transform, hyde_passage = self.rcfg.query_transform, None

        from docrag.retrieve import query as qt

        with timer("query_transform"):
            if transform == "rewrite":
                queries = [qt.rewrite(self.llm, query, history)]
            elif transform == "multi_query":
                queries = qt.multi_query(self.llm, query, self.rcfg.multi_query_n, history)
            elif transform == "hyde":
                hyde_passage = qt.hyde(self.llm, query)
                queries = [query]
            else:
                queries = [query]

        lists = [self._candidates(q, timer, dense_text=hyde_passage) for q in queries]
        if len(lists) > 1:
            with timer("fusion"):
                ranked = fusion.rrf(lists, k=self.rcfg.rrf_k, limit=self.rcfg.candidates)
        else:
            ranked = lists[0]

        if self.rcfg.rerank.enabled:
            with timer("rerank"):
                head = self.reranker.rerank(query, ranked[: self.rcfg.rerank.top_n])
            ranked = head + ranked[self.rcfg.rerank.top_n :]
            selected = dynamic_k.select(head, self.rcfg.dynamic_k, self.rcfg.top_k)
        else:
            selected = ranked[: self.rcfg.top_k]  # adaptive k needs calibrated (reranker) scores

        timings = {k: round(v, 2) for k, v in timer.t.items()}
        timings["total"] = round((time.perf_counter() - t_all) * 1000, 2)
        return RetrievalResult(query, queries, ranked, selected, timings, hyde_passage)
