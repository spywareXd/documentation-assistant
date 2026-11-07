"""Qdrant collection with two named vectors per chunk:

  dense : bge embedding (cosine)
  bm25  : sparse term-frequency vector from fastembed "Qdrant/bm25"; the collection uses the
          IDF modifier, so Qdrant computes BM25 scores at query time over the live corpus.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from docrag.config import Config
from docrag.ingest.chunkers import Chunk
from docrag.logger import log_info, log_success

DENSE, SPARSE = "dense", "bm25"


@dataclass
class Hit:
    chunk: Chunk
    score: float


class SparseEncoder:
    def __init__(self, model: str):
        from fastembed import SparseTextEmbedding

        from docrag.config import ROOT

        # keep the model out of %TEMP% (gets cleaned) and avoid HF symlinks (need admin on Windows)
        self.model = SparseTextEmbedding(model_name=model, cache_dir=str(ROOT / "data" / "models" / "fastembed"))

    def encode_docs(self, texts: list[str]) -> list[models.SparseVector]:
        return [models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
                for e in self.model.embed(texts, batch_size=256)]

    def encode_query(self, text: str) -> models.SparseVector:
        e = next(iter(self.model.query_embed(text)))
        return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


def _point_id(chunk_id: str) -> int:
    return int(chunk_id, 16)  # 16 hex chars -> fits in Qdrant's unsigned 64-bit ids


def _to_chunk(payload: dict) -> Chunk:
    return Chunk(**{k: payload[k] for k in Chunk.__dataclass_fields__ if k in payload})


class QdrantStore:
    def __init__(self, cfg: Config, collection: str | None = None):
        self.cfg = cfg
        self.name = collection or cfg.collection_name()
        self.client = QdrantClient(url=cfg.index.url or os.environ.get("QDRANT_URL", "http://localhost:6333"), timeout=60)
        self._sparse: SparseEncoder | None = None

    @property
    def sparse(self) -> SparseEncoder:
        if self._sparse is None:
            self._sparse = SparseEncoder(self.cfg.sparse.model)
        return self._sparse

    # ------------------------------------------------------------------ build
    def exists(self) -> bool:
        return self.client.collection_exists(self.name)

    def count(self) -> int:
        return self.client.count(self.name, exact=True).count if self.exists() else 0

    def build(self, chunks: list[Chunk], embedder, recreate: bool = False, batch: int = 256) -> dict:
        if self.exists() and not recreate and self.count() == len(chunks):
            log_info(f"Qdrant: {self.name} already has {len(chunks)} points, skipping build")
            return {"skipped": True}
        if self.exists():
            self.client.delete_collection(self.name)
        self.client.create_collection(
            self.name,
            vectors_config={DENSE: models.VectorParams(size=embedder.dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        self.client.create_payload_index(self.name, "source", models.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.name, "url", models.PayloadSchemaType.KEYWORD)

        t0 = time.perf_counter()
        texts = [c.embed_text() for c in chunks]
        dense = embedder.embed_documents(texts, show_progress=True)
        t_dense = time.perf_counter() - t0
        sparse = self.sparse.encode_docs(texts)
        t_sparse = time.perf_counter() - t0 - t_dense

        for i in tqdm(range(0, len(chunks), batch), desc="upserting"):
            self.client.upsert(
                self.name,
                points=[
                    models.PointStruct(
                        id=_point_id(c.chunk_id),
                        vector={DENSE: dense[j].tolist(), SPARSE: sparse[j]},
                        payload=c.to_dict(),
                    )
                    for j, c in enumerate(chunks[i : i + batch], start=i)
                ],
                wait=True,
            )
        stats = {
            "points": len(chunks),
            "dense_embed_s": round(t_dense, 2),
            "sparse_embed_s": round(t_sparse, 2),
            "total_s": round(time.perf_counter() - t0, 2),
        }
        log_success(f"Qdrant: built {self.name} {stats}")
        return stats

    # ------------------------------------------------------------------ search
    def _hits(self, points) -> list[Hit]:
        return [Hit(_to_chunk(p.payload), float(p.score)) for p in points]

    def search_dense(self, vec: np.ndarray, limit: int, flt: models.Filter | None = None) -> list[Hit]:
        res = self.client.query_points(self.name, query=vec.tolist(), using=DENSE, limit=limit,
                                       query_filter=flt, with_payload=True)
        return self._hits(res.points)

    def search_sparse(self, query: str, limit: int, flt: models.Filter | None = None) -> list[Hit]:
        res = self.client.query_points(self.name, query=self.sparse.encode_query(query), using=SPARSE,
                                       limit=limit, query_filter=flt, with_payload=True)
        return self._hits(res.points)

    def search_hybrid_native(self, vec: np.ndarray, query: str, limit: int, candidates: int) -> list[Hit]:
        """Server-side RRF (used to cross-check our own fusion implementation)."""
        res = self.client.query_points(
            self.name,
            prefetch=[
                models.Prefetch(query=vec.tolist(), using=DENSE, limit=candidates),
                models.Prefetch(query=self.sparse.encode_query(query), using=SPARSE, limit=candidates),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return self._hits(res.points)

    def all_chunks(self) -> list[Chunk]:
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(self.name, limit=1000, offset=offset, with_payload=True)
            out.extend(_to_chunk(p.payload) for p in points)
            if offset is None:
                return out
