from docrag.config import ROOT, deep_merge, load_config
from docrag.ingest.sources import glob_match, page_url


def test_deep_merge():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}


def test_experiment_extends_default():
    cfg = load_config(ROOT / "configs/experiments/e2_bm25.yaml")
    assert cfg.name == "e2_bm25" and cfg.retrieval.mode == "sparse" and not cfg.retrieval.rerank.enabled
    assert cfg.chunking.strategy == load_config().chunking.strategy


def test_chunk_hash_tracks_chunking_only():
    a = load_config()
    assert a.chunk_hash() == load_config(retrieval={"top_k": 9}).chunk_hash()
    assert a.chunk_hash() != load_config(chunking={"chunk_size": 512}).chunk_hash()


def test_all_experiment_configs_load():
    for f in sorted((ROOT / "configs/experiments").glob("*.yaml")):
        assert load_config(f).name == f.stem


def test_glob():
    assert glob_match("index.md", "*.md") and not glob_match("tutorial/a.md", "*.md")
    assert glob_match("tutorial/a.md", "tutorial/**/*.md") and glob_match("tutorial/x/a.md", "tutorial/**/*.md")
    assert glob_match("x/foo-js.mdx", "**/*-js.mdx") and glob_match("foo-js.mdx", "**/*-js.mdx")


def test_page_url():
    assert page_url("https://f.dev/{path}/", "tutorial/first-steps.md") == "https://f.dev/tutorial/first-steps/"
    assert page_url("https://f.dev/{path}/", "index.md") == "https://f.dev/"
    assert page_url("https://q.tech/documentation/{path}/", "manage-data/_index.md") == "https://q.tech/documentation/manage-data/"
