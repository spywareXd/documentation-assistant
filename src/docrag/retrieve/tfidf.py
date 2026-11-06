"""TF-IDF baseline (eval only): the classic lexical baseline that BM25 improves on
(BM25 adds term-frequency saturation and document-length normalisation)."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from docrag.index.qdrant_store import Hit
from docrag.ingest.chunkers import Chunk


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", token_pattern=r"(?u)\b\w\w+\b")
        self.matrix = self.vec.fit_transform([c.embed_text() for c in chunks])  # rows are L2-normalised

    def search(self, query: str, limit: int) -> list[Hit]:
        q = self.vec.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        top = np.argsort(-scores, kind="stable")[:limit]
        return [Hit(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]
