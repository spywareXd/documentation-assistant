"""Contextual retrieval: an LLM writes a short "where does this chunk sit" line per chunk.

The line is prepended to the chunk for embedding and BM25 (Chunk.embed_text), so a chunk
like "Set it to False to disable it." becomes findable by what "it" refers to.
The document is sent once per chunk; responses are cached, so this runs once per corpus.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from docrag.config import Config
from docrag.ingest.chunkers import Chunk, Document
from docrag.llm import get_llm

MAX_DOC_CHARS = 24_000  # keep the prompt bounded for very long pages

PROMPT = """<document title="{title}" source="{source}">
{doc}
</document>

Here is a chunk from that document:
<chunk>
{chunk}
</chunk>

Write one or two sentences that situate this chunk within the document, to improve search retrieval.
Name the library, the feature and what the chunk is about. Answer with only those sentences."""


def _window(doc: str, chunk: Chunk) -> str:
    """Whole doc if it fits, else a window centred on the chunk."""
    if len(doc) <= MAX_DOC_CHARS:
        return doc
    mid = (chunk.start + chunk.end) // 2
    lo = max(0, min(mid - MAX_DOC_CHARS // 2, len(doc) - MAX_DOC_CHARS))
    return doc[lo : lo + MAX_DOC_CHARS]


def add_context(chunks: list[Chunk], docs: list[Document], cfg: Config, workers: int = 4) -> list[Chunk]:
    llm = get_llm(cfg)
    by_id = {d.doc_id: d for d in docs}

    def work(c: Chunk) -> None:
        d = by_id[c.doc_id]
        msg = PROMPT.format(title=d.title, source=d.source, doc=_window(d.text, c), chunk=c.text)
        c.context = llm.chat([{"role": "user", "content": msg}], role="utility", max_tokens=400).strip()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(tqdm(pool.map(work, chunks), total=len(chunks), desc="contextualizing"))
    return chunks
