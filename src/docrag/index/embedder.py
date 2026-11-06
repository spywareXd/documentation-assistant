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

    def embed_documents(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        out, batch = [], 50
        for i in tqdm(range(0, len(texts), batch), disable=not show_progress, desc="gemini embed"):
            for attempt in range(6):
                try:
                    out.extend(self.client.embed_documents(list(texts[i : i + batch])))
                    break
                except Exception as err:  # free-tier rate limits
                    if attempt == 5:
                        raise
                    time.sleep(2 ** attempt * 5)
                    print(f"gemini retry {attempt + 1}: {str(err)[:120]}")
        return _normalize(np.array(out, dtype=np.float32))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed_documents(texts)  # original system embedded queries the same way


@lru_cache(maxsize=2)
def _cached(cfg_json: str) -> Embedder:
    cfg = EmbeddingCfg.model_validate_json(cfg_json)
    return GeminiEmbedder(cfg) if cfg.provider == "gemini" else STEmbedder(cfg)


def get_embedder(cfg: EmbeddingCfg) -> Embedder:
    return _cached(cfg.model_dump_json())
