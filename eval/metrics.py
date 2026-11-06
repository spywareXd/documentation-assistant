"""Retrieval metrics against chunking-independent gold evidence.

Gold evidence = (url, quote). At eval time the quote is located in the cleaned page to get a
character span. A retrieved chunk is *relevant to an evidence item* if it is from the same URL
and covers at least `min_cover` of the evidence span. This lets one golden set score every
chunking strategy.

For a question with E evidence items and a ranked list of chunks:
  hit@k    1 if any evidence item is covered in the top k
  recall@k (# evidence items covered in top k) / E
  MRR      1 / rank of the first relevant chunk (0 if none in the list)
  nDCG@k   gain 1 for a chunk that covers a *new* evidence item, ideal = min(E, k) gains up front
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from docrag.ingest.chunkers import Chunk


@dataclass
class Evidence:
    url: str
    quote: str
    start: int = -1
    end: int = -1

    @property
    def located(self) -> bool:
        return self.start >= 0


_WS = re.compile(r"\s+")


def _norm_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace and lowercase; return mapping from normalised index -> original index."""
    out, idx, prev_space = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            prev_space = True
        else:
            out.append(ch.lower())
            prev_space = False
        idx.append(i)
    return "".join(out), idx


def locate(ev: Evidence, doc_text: str) -> Evidence:
    """Find the quote in the page (whitespace/case-insensitive) and fill start/end."""
    q = _WS.sub(" ", ev.quote.strip().lower())
    if not q:
        return ev
    norm, idx = _norm_with_map(doc_text)
    pos = norm.find(q)
    if pos < 0:
        return Evidence(ev.url, ev.quote)
    return Evidence(ev.url, ev.quote, idx[pos], idx[pos + len(q) - 1] + 1)


def covers(chunk: Chunk, ev: Evidence, min_cover: float = 0.5) -> bool:
    if chunk.url != ev.url:
        return False
    if not ev.located:  # fall back to text containment
        return _WS.sub(" ", ev.quote.lower()).strip() in _WS.sub(" ", chunk.text.lower())
    overlap = min(chunk.end, ev.end) - max(chunk.start, ev.start)
    return overlap > 0 and overlap / (ev.end - ev.start) >= min_cover


def relevance_matrix(chunks: Sequence[Chunk], evidence: Sequence[Evidence]) -> list[set[int]]:
    """For each ranked chunk, the set of evidence indices it covers."""
    return [{j for j, ev in enumerate(evidence) if covers(c, ev)} for c in chunks]


def hit_at_k(rel: list[set[int]], k: int) -> float:
    return float(any(rel[:k]))


def recall_at_k(rel: list[set[int]], n_evidence: int, k: int) -> float:
    if n_evidence == 0:
        return 0.0
    covered = set().union(*rel[:k]) if rel[:k] else set()
    return len(covered) / n_evidence


def mrr(rel: list[set[int]]) -> float:
    for rank, r in enumerate(rel, start=1):
        if r:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(rel: list[set[int]], n_evidence: int, k: int) -> float:
    if n_evidence == 0:
        return 0.0
    seen: set[int] = set()
    dcg = 0.0
    for rank, r in enumerate(rel[:k], start=1):
        new = r - seen
        if new:
            dcg += 1.0 / math.log2(rank + 1)
            seen |= new
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(n_evidence, k) + 1))
    return dcg / idcg


def precision(rel: list[set[int]]) -> float:
    """Share of the given chunks that are relevant to any evidence (context precision)."""
    return sum(1 for r in rel if r) / len(rel) if rel else 0.0


def retrieval_scores(ranked: Sequence[Chunk], selected: Sequence[Chunk], evidence: Sequence[Evidence],
                     ks: Sequence[int]) -> dict[str, float]:
    rel = relevance_matrix(ranked, evidence)
    rel_sel = relevance_matrix(selected, evidence)
    n = len(evidence)
    out = {f"recall@{k}": recall_at_k(rel, n, k) for k in ks}
    out |= {f"hit@{k}": hit_at_k(rel, k) for k in ks}
    out["mrr"] = mrr(rel)
    out["ndcg@10"] = ndcg_at_k(rel, n, 10)
    out["context_recall"] = recall_at_k(rel_sel, n, len(rel_sel))  # coverage of what the LLM saw
    out["context_precision"] = precision(rel_sel)
    out["k_selected"] = float(len(selected))
    return out
