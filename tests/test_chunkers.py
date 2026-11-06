import numpy as np

from docrag.config import ChunkingCfg
from docrag.ingest.chunkers import Document, _sections, chunk_documents

DOC = Document("d1", "fastapi", "https://x/a/", "Guide", """# Guide

Intro paragraph that is long enough to be kept as its own little chunk of text here.

## Install

Run the installer. It sets things up for you and prints a message when it is done ok.

```python
# not a heading
x = 1
```

## Usage

### Advanced

Use the advanced mode when you need more control over the behaviour of the tool here.
""")


def test_sections_ignore_headings_in_code_and_track_breadcrumbs():
    crumbs = [c for _, _, c in _sections(DOC.text)]
    assert ["Guide", "Install"] in crumbs
    assert ["Guide", "Usage", "Advanced"] in crumbs
    assert not any("not a heading" in " ".join(c) for c in crumbs)


def test_offsets_are_exact_and_ids_deterministic():
    for strategy in ("recursive", "markdown_header"):
        cfg = ChunkingCfg(strategy=strategy, chunk_size=200, chunk_overlap=20, min_chunk_chars=10)
        a = chunk_documents([DOC], cfg, "h")
        b = chunk_documents([DOC], cfg, "h")
        assert a and [c.chunk_id for c in a] == [c.chunk_id for c in b]
        for c in a:
            assert DOC.text[c.start:c.end] == c.text
            assert len(c.text) <= 200


def test_semantic_cuts_on_topic_change():
    doc = Document("d2", "s", "u", "T", "alpha one.\n\nalpha two.\n\nbeta one.\n\nbeta two.")

    def embed(texts):  # alpha blocks point one way, beta blocks another
        return np.array([[1.0, 0.0] if "alpha" in t else [0.0, 1.0] for t in texts])

    cfg = ChunkingCfg(strategy="semantic", chunk_size=1000, min_chunk_chars=1, semantic_threshold=50)
    texts = [c.text for c in chunk_documents([doc], cfg, "h", embed=embed)]
    assert texts == ["alpha one.\n\nalpha two.", "beta one.\n\nbeta two."]


def test_breadcrumb_toggle():
    on = chunk_documents([DOC], ChunkingCfg(strategy="markdown_header", min_chunk_chars=10), "h")[0]
    off = chunk_documents([DOC], ChunkingCfg(strategy="markdown_header", min_chunk_chars=10, breadcrumb=False), "h")[0]
    assert on.embed_text().startswith("Guide") and off.embed_text() == off.text
