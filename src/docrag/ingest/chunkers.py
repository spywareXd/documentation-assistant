"""Three chunking strategies, all returning Chunks with exact character offsets.

- recursive:        LangChain RecursiveCharacterTextSplitter (the original approach), size/overlap tunable.
- markdown_header:  split on headings (never inside code fences), keep the heading breadcrumb,
                    merge tiny sections, recursively split sections that are too long.
- semantic:         split into blocks (paragraphs / code fences), embed them, cut where the cosine
                    distance between neighbours exceeds a percentile threshold; size-capped.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docrag.config import ChunkingCfg


@dataclass
class Document:
    doc_id: str
    source: str
    url: str
    title: str
    text: str
    commit: str = ""


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    url: str
    title: str
    headings: list[str]
    text: str
    start: int
    end: int
    context: str = ""  # contextual-retrieval prefix (filled later, optional)
    meta: dict = field(default_factory=dict)

    def embed_text(self) -> str:
        """What gets embedded and BM25-indexed: context + breadcrumb + body."""
        head = f"{self.context}\n\n" if self.context else ""
        if not self.meta.get("breadcrumb", True):
            return f"{head}{self.text}"
        crumb = " > ".join([self.title, *[h for h in self.headings if h != self.title]])
        return f"{head}{crumb}\n\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def make_chunk_id(url: str, start: int, chunk_hash: str) -> str:
    return hashlib.sha1(f"{url}|{start}|{chunk_hash}".encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- helpers
FENCE = re.compile(r"^\s*```")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _sub_split(text: str, base: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Recursive split of text[...] -> list of (start, end) offsets relative to the document."""
    if len(text) <= size:
        return [(base, base + len(text))]
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, add_start_index=True)
    spans = []
    for d in splitter.create_documents([text]):
        s = d.metadata["start_index"]
        spans.append((base + s, base + s + len(d.page_content)))
    return spans


def _sections(text: str) -> list[tuple[int, int, list[str]]]:
    """Split on markdown headings outside code fences. Returns (start, end, breadcrumb)."""
    sections, crumb = [], []
    start, pos, in_fence = 0, 0, False
    current_crumb: list[str] = []
    for line in text.splitlines(keepends=True):
        if FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            if pos > start:
                sections.append((start, pos, list(current_crumb)))
            level, title = len(m.group(1)), m.group(2).strip()
            crumb = crumb[: level - 1] + [""] * max(0, level - 1 - len(crumb)) + [title]
            current_crumb = [c for c in crumb if c]
            start = pos
        pos += len(line)
    if pos > start:
        sections.append((start, pos, list(current_crumb)))
    return sections


# --------------------------------------------------------------------------- strategies
def chunk_recursive(doc: Document, cfg: ChunkingCfg) -> list[tuple[int, int, list[str]]]:
    return [(s, e, []) for s, e in _sub_split(doc.text, 0, cfg.chunk_size, cfg.chunk_overlap)]


def chunk_markdown_header(doc: Document, cfg: ChunkingCfg) -> list[tuple[int, int, list[str]]]:
    merged: list[list] = []  # [start, end, crumb]
    for s, e, crumb in _sections(doc.text):
        # merge a small section into the previous one if they share a parent and still fit
        if merged and (e - s) < cfg.min_chunk_chars * 3 and (e - merged[-1][0]) <= cfg.chunk_size:
            merged[-1][1] = e
            continue
        if merged and (merged[-1][1] - merged[-1][0]) < cfg.min_chunk_chars * 3 and (e - merged[-1][0]) <= cfg.chunk_size:
            merged[-1][1] = e
            merged[-1][2] = crumb if len(crumb) >= len(merged[-1][2]) else merged[-1][2]
            continue
        merged.append([s, e, crumb])
    out = []
    for s, e, crumb in merged:
        for ss, ee in _sub_split(doc.text[s:e], s, cfg.chunk_size, cfg.chunk_overlap):
            out.append((ss, ee, crumb))
    return out


def _blocks(text: str) -> list[tuple[int, int]]:
    """Paragraph-level blocks; a fenced code block (with its blank lines) is one block."""
    blocks, start, pos, in_fence = [], None, 0, False
    for line in text.splitlines(keepends=True):
        if FENCE.match(line):
            in_fence = not in_fence
        blank = not line.strip() and not in_fence
        if blank:
            if start is not None:
                blocks.append((start, pos))
                start = None
        elif start is None:
            start = pos
        pos += len(line)
    if start is not None:
        blocks.append((start, pos))
    return blocks


def chunk_semantic(
    doc: Document, cfg: ChunkingCfg, embed: Callable[[Sequence[str]], np.ndarray]
) -> list[tuple[int, int, list[str]]]:
    blocks = _blocks(doc.text)
    if len(blocks) <= 1:
        return [(s, e, []) for s, e in _sub_split(doc.text, 0, cfg.chunk_size, cfg.chunk_overlap)]
    vecs = embed([doc.text[s:e] for s, e in blocks])
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
    dist = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)  # distance between neighbours
    threshold = _breakpoint_threshold(dist, cfg.semantic_breakpoint, cfg.semantic_threshold)

    groups, cur_start, cur_end = [], blocks[0][0], blocks[0][1]
    for i in range(1, len(blocks)):
        s, e = blocks[i]
        too_big = (e - cur_start) > cfg.chunk_size
        if dist[i - 1] > threshold or too_big:
            groups.append((cur_start, cur_end))
            cur_start = s
        cur_end = e
    groups.append((cur_start, cur_end))

    out = []
    for s, e in groups:  # a single huge block (e.g. long code) still gets capped
        out.extend((ss, ee, []) for ss, ee in _sub_split(doc.text[s:e], s, cfg.chunk_size, cfg.chunk_overlap))
    return out


def _breakpoint_threshold(dist: np.ndarray, kind: str, amount: float) -> float:
    if kind == "percentile":
        return float(np.percentile(dist, amount))
    if kind == "standard_deviation":
        return float(dist.mean() + (amount / 100 if amount > 10 else amount) * dist.std())
    if kind == "interquartile":
        q1, q3 = np.percentile(dist, [25, 75])
        return float(dist.mean() + 1.5 * (q3 - q1))
    if kind == "gradient":
        return float(np.percentile(np.gradient(dist), amount))
    raise ValueError(kind)


# --------------------------------------------------------------------------- entry point
def chunk_documents(
    docs: list[Document],
    cfg: ChunkingCfg,
    chunk_hash: str,
    embed: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        if cfg.strategy == "recursive":
            spans = chunk_recursive(doc, cfg)
        elif cfg.strategy == "markdown_header":
            spans = chunk_markdown_header(doc, cfg)
        elif cfg.strategy == "semantic":
            if embed is None:
                raise ValueError("semantic chunking needs an embed function")
            spans = chunk_semantic(doc, cfg, embed)
        else:
            raise ValueError(cfg.strategy)

        for s, e, crumb in spans:
            text = doc.text[s:e].strip()
            if len(text) < cfg.min_chunk_chars:
                continue
            lead = len(doc.text[s:e]) - len(doc.text[s:e].lstrip())
            start = s + lead
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.url, start, chunk_hash),
                    doc_id=doc.doc_id,
                    source=doc.source,
                    url=doc.url,
                    title=doc.title,
                    headings=crumb,
                    text=text,
                    start=start,
                    end=start + len(text),
                    meta={"breadcrumb": cfg.breadcrumb},
                )
            )
    return chunks
