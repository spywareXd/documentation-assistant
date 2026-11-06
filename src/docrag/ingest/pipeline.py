"""Ingestion: fetch -> clean -> chunk -> (contextualize) -> index.

    python -m docrag.ingest.pipeline                       # default config, build index
    python -m docrag.ingest.pipeline --config configs/experiments/e1_semantic.yaml
    python -m docrag.ingest.pipeline --no-index            # only produce chunk JSONL

Outputs (gitignored):
    data/corpus/docs.jsonl                 cleaned documents
    data/chunks/<chunk_hash>.jsonl         chunks for this chunking config
    data/chunks/<chunk_hash>.stats.json    corpus + chunk statistics
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from docrag.config import Config, load_config
from docrag.ingest.chunkers import Chunk, Document, chunk_documents
from docrag.ingest.clean import clean_page
from docrag.ingest.sources import fetch_pages
from docrag.logger import log_header, log_info, log_success, log_warning

MIN_DOC_CHARS = 200


def _jsonl_write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _jsonl_read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def docs_path(cfg: Config) -> Path:
    return cfg.path(cfg.corpus.data_dir) / "docs.jsonl"


def chunks_path(cfg: Config) -> Path:
    return cfg.path("data/chunks") / f"{cfg.chunk_hash()}.jsonl"


def build_documents(cfg: Config, refresh: bool = False) -> list[Document]:
    path = docs_path(cfg)
    if not path.exists() or refresh:
        _fetch_all(cfg, path)
    docs = [Document(**d) for d in _jsonl_read(path)]
    if cfg.corpus.sources:
        docs = [d for d in docs if d.source in cfg.corpus.sources]
    return docs


def _fetch_all(cfg: Config, path: Path) -> None:
    """docs.jsonl always holds every source; configs filter after loading."""
    log_header("FETCH + CLEAN")
    full = cfg.model_copy(update={"corpus": cfg.corpus.model_copy(update={"sources": None})})
    docs, seen = [], set()
    for page in fetch_pages(full):
        title, text = clean_page(page)
        if len(text) < MIN_DOC_CHARS:
            continue
        if page.url in seen:  # e.g. index.md and a sibling mapping to the same URL
            log_warning(f"duplicate url skipped: {page.url}")
            continue
        seen.add(page.url)
        doc_id = hashlib.sha1(page.url.encode()).hexdigest()[:12]
        docs.append(Document(doc_id, page.source, page.url, title, text, page.commit))
    _jsonl_write(path, [d.__dict__ for d in docs])
    log_success(f"{len(docs)} documents -> {path}")


def load_chunks(cfg: Config) -> list[Chunk]:
    path = chunks_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m docrag.ingest.pipeline --config ...` first")
    return [Chunk(**c) for c in _jsonl_read(path)]


def build_chunks(cfg: Config, docs: list[Document], refresh: bool = False) -> list[Chunk]:
    path = chunks_path(cfg)
    if path.exists() and not refresh:
        return load_chunks(cfg)
    log_header(f"CHUNK ({cfg.chunking.strategy}, size={cfg.chunking.chunk_size}, contextual={cfg.chunking.contextual})")
    embed = None
    if cfg.chunking.strategy == "semantic":
        from docrag.index.embedder import get_embedder

        embedder = get_embedder(cfg.embedding)
        embed = lambda texts: embedder.embed_documents(texts)  # noqa: E731
    chunks = chunk_documents(docs, cfg.chunking, cfg.chunk_hash(), embed=embed)
    if cfg.chunking.contextual:
        from docrag.ingest.contextual import add_context

        add_context(chunks, docs, cfg)
    _jsonl_write(path, [c.to_dict() for c in chunks])

    lengths = np.array([len(c.text) for c in chunks])
    stats = {
        "chunk_hash": cfg.chunk_hash(),
        "chunking": cfg.chunking.model_dump(),
        "documents": len(docs),
        "documents_per_source": dict(Counter(d.source for d in docs)),
        "corpus_chars": int(sum(len(d.text) for d in docs)),
        "chunks": len(chunks),
        "chunks_per_source": dict(Counter(c.source for c in chunks)),
        "chunk_chars": {"mean": float(lengths.mean()), "p50": float(np.median(lengths)),
                        "p95": float(np.percentile(lengths, 95)), "max": int(lengths.max())},
    }
    path.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    log_success(f"{len(chunks)} chunks -> {path}")
    log_info(json.dumps(stats["chunk_chars"]))
    return chunks


def build_index(cfg: Config, chunks: list[Chunk], recreate: bool = False) -> dict:
    from docrag.index.embedder import get_embedder
    from docrag.index.qdrant_store import QdrantStore

    log_header(f"INDEX -> {cfg.collection_name()}")
    return QdrantStore(cfg).build(chunks, get_embedder(cfg.embedding), recreate=recreate)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--refresh-docs", action="store_true", help="re-fetch and re-clean the corpus")
    ap.add_argument("--refresh-chunks", action="store_true")
    ap.add_argument("--recreate-index", action="store_true")
    ap.add_argument("--no-index", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    docs = build_documents(cfg, refresh=args.refresh_docs)
    chunks = build_chunks(cfg, docs, refresh=args.refresh_chunks or args.refresh_docs)
    if not args.no_index:
        build_index(cfg, chunks, recreate=args.recreate_index)


if __name__ == "__main__":
    main()
