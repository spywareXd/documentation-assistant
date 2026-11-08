"""Dense embedders.

sentence_transformers : local model on GPU (default, bge: prefix on queries only).
gemini                : gemini-embedding-001 through the API, configured exactly like the original
                        system (RETRIEVAL_DOCUMENT task type for documents *and* queries), used only
                        to reproduce the E0 baseline.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Sequence

import numpy as np
from tqdm import tqdm

from docrag.config import EmbeddingCfg


class Embedder:
    dim: int

    def embed_documents(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray: ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]


def _normalize(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-9)).astype(np.float32)


class STEmbedder(Embedder):
    def __init__(self, cfg: EmbeddingCfg):
        import torch
        from sentence_transformers import SentenceTransformer

        device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model, device=device)
        self.model.max_seq_length = cfg.max_seq_length
        if device == "cuda":
            self.model.half()  # fp16 inference on the GPU
        self.dim = self.model.get_sentence_embedding_dimension()
        self.device = device

    def embed_documents(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            list(texts), batch_size=self.cfg.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress,
        ).astype(np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed_documents([self.cfg.query_prefix + t for t in texts])


class GeminiEmbedder(Embedder):
    def __init__(self, cfg: EmbeddingCfg):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self.cfg = cfg
        self.dim = cfg.output_dimensionality or 3072
        self.client = GoogleGenerativeAIEmbeddings(
            model=cfg.model, task_type="RETRIEVAL_DOCUMENT", output_dimensionality=cfg.output_dimensionality
        )

    # Free tier: 30k embedding tokens per minute. Stay under it with small, paced batches.
    TOKENS_PER_MIN = 25_000
    BATCH_TOKENS = 6_000

    def _cache(self):
        if not hasattr(self, "_db"):
            import sqlite3

            from docrag.config import ROOT

            path = ROOT / "data" / "emb_cache" / f"{self.cfg.model}-{self.dim}.sqlite"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path)
            self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        return self._db

    @staticmethod
    def _key(text: str) -> str:
        import hashlib

        return hashlib.sha1(text.encode()).hexdigest()

    def embed_documents(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """Cached per text, so a run cut off by the free tier's daily request limit resumes next day."""
        from docrag.llm import QuotaExhausted

        db = self._cache()
        keys = [self._key(t) for t in texts]
        have = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(chunk))})"
            have.update({k: np.frombuffer(v, dtype=np.float32) for k, v in db.execute(q, chunk)})
        todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in have]
        if todo:
            print(f"gemini: {len(have)} cached, embedding {len(todo)}")

        # token-bounded batches (~4 chars per token)
        batches, cur, cur_tok = [], [], 0
        for k, t in todo:
            tok = len(t) // 4 + 1
            if cur and cur_tok + tok > self.BATCH_TOKENS:
                batches.append(cur)
                cur, cur_tok = [], 0
            cur.append((k, t))
            cur_tok += tok
        if cur:
            batches.append(cur)

        window = []  # (time, tokens) sent in the last 60 s
        for batch in tqdm(batches, disable=not show_progress, desc="gemini embed"):
            tok = sum(len(t) // 4 + 1 for _, t in batch)
            while True:
                now = time.monotonic()
                window = [(t0, n) for t0, n in window if now - t0 < 60]
                if sum(n for _, n in window) + tok <= self.TOKENS_PER_MIN:
                    break
                time.sleep(60 - (now - window[0][0]) + 0.5)
            for attempt in range(8):
                try:
                    vecs = self.client.embed_documents([t for _, t in batch])
                    window.append((time.monotonic(), tok))
                    break
                except Exception as err:
                    msg = str(err)
                    if "free_tier_requests" in msg or "per day" in msg.lower():
                        raise QuotaExhausted(f"gemini embeddings: daily request limit reached ({len(have)} cached)") from err
                    if attempt == 7:
                        raise
                    print(f"gemini retry {attempt + 1} in 35s: {msg[:100]}")
                    time.sleep(35)
            rows = [(k, np.asarray(v, dtype=np.float32).tobytes()) for (k, _), v in zip(batch, vecs)]
            db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
            db.commit()
            have.update({k: np.frombuffer(b, dtype=np.float32) for k, b in rows})
        return _normalize(np.stack([have[k] for k in keys]))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed_documents(texts)  # original system embedded queries the same way


@lru_cache(maxsize=2)
def _cached(cfg_json: str) -> Embedder:
    cfg = EmbeddingCfg.model_validate_json(cfg_json)
    return GeminiEmbedder(cfg) if cfg.provider == "gemini" else STEmbedder(cfg)


def get_embedder(cfg: EmbeddingCfg) -> Embedder:
    return _cached(cfg.model_dump_json())
