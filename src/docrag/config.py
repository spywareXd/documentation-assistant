"""Typed configuration loaded from YAML.

Experiment files can say `extends: <path>` and override only the keys they change:

    extends: ../default.yaml
    name: e2_bm25
    retrieval: {mode: sparse, rerank: {enabled: false}}
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


class CorpusCfg(BaseModel):
    sources_file: str = "sources.yaml"
    data_dir: str = "data/corpus"
    sources: list[str] | None = None  # restrict to these source names (None = all)


class ChunkingCfg(BaseModel):
    strategy: Literal["recursive", "markdown_header", "semantic"] = "markdown_header"
    chunk_size: int = 1000  # characters
    chunk_overlap: int = 150
    min_chunk_chars: int = 80  # drop tiny fragments (nav leftovers, lone headings)
    semantic_breakpoint: Literal["percentile", "standard_deviation", "interquartile", "gradient"] = "percentile"
    semantic_threshold: float = 90.0
    contextual: bool = False  # prepend an LLM-written context line to each chunk
    breadcrumb: bool = True  # prepend "Page title > Section" to the text that is embedded / BM25-indexed


class EmbeddingCfg(BaseModel):
    provider: Literal["sentence_transformers", "gemini"] = "sentence_transformers"  # gemini = E0 baseline only
    model: str = "BAAI/bge-base-en-v1.5"
    output_dimensionality: int | None = None  # gemini only (baseline used 1536)
    batch_size: int = 64
    device: str = "cuda"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    max_seq_length: int = 512


class SparseCfg(BaseModel):
    model: str = "Qdrant/bm25"


class IndexCfg(BaseModel):
    url: str | None = None  # default: $QDRANT_URL or http://localhost:6333
    collection_prefix: str = "docs"


class RerankCfg(BaseModel):
    enabled: bool = True
    model: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 50  # how many fused candidates go into the cross-encoder
    batch_size: int = 16
    max_length: int = 512


class DynamicKCfg(BaseModel):
    policy: Literal["fixed", "score_threshold", "relative_drop"] = "fixed"
    threshold: float = 0.3  # score_threshold: keep chunks with score >= threshold
    relative_drop: float = 0.5  # relative_drop: stop when score < ratio * top score
    min_k: int = 2
    max_k: int = 10


class RetrievalCfg(BaseModel):
    mode: Literal["dense", "sparse", "tfidf", "hybrid"] = "hybrid"
    fusion: Literal["rrf", "weighted", "qdrant_rrf"] = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.5  # only for fusion=weighted
    candidates: int = 50  # per retriever, before fusion
    top_k: int = 5
    rerank: RerankCfg = Field(default_factory=RerankCfg)
    dynamic_k: DynamicKCfg = Field(default_factory=DynamicKCfg)
    query_transform: Literal["none", "rewrite", "multi_query", "hyde"] = "none"
    multi_query_n: int = 3


class LLMCfg(BaseModel):
    provider: Literal["groq"] = "groq"
    generator: str = "openai/gpt-oss-120b"
    utility: str = "openai/gpt-oss-20b"  # rewriting, grading, contextual summaries
    judge: str = "openai/gpt-oss-120b"
    temperature: float = 0.0
    reasoning_effort: Literal["low", "medium", "high"] = "low"
    cache_dir: str = "data/llm_cache"
    requests_per_minute: int = 25
    max_retries: int = 6


class PipelineCfg(BaseModel):
    mode: Literal["single", "graph"] = "graph"
    max_rewrites: int = 2  # corrective-RAG retry budget
    grade_documents: bool = True
    groundedness_check: bool = True


class EvalCfg(BaseModel):
    golden_set: str = "eval/golden_set.jsonl"
    split: Literal["dev", "test", "all"] = "dev"
    ks: list[int] = [1, 3, 5, 10]
    retrieval_only: bool = False
    limit: int | None = None


class Config(BaseModel):
    name: str = "default"
    corpus: CorpusCfg = Field(default_factory=CorpusCfg)
    chunking: ChunkingCfg = Field(default_factory=ChunkingCfg)
    embedding: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    sparse: SparseCfg = Field(default_factory=SparseCfg)
    index: IndexCfg = Field(default_factory=IndexCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)
    pipeline: PipelineCfg = Field(default_factory=PipelineCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)

    # ---- identities used for caching / collection names -------------------
    def chunk_hash(self) -> str:
        """Changes whenever the chunk set would change."""
        return _short_hash({"corpus": self.corpus.model_dump(), "chunking": self.chunking.model_dump()})

    def collection_name(self) -> str:
        model_slug = self.embedding.model.split("/")[-1].replace(".", "-").lower()
        return f"{self.index.collection_prefix}__{self.chunk_hash()}__{model_slug}"

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p


def _short_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:10]


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_raw(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        data = deep_merge(_load_raw((path.parent / parent).resolve()), data)
    return data


def load_config(path: str | Path | None = None, **overrides) -> Config:
    """Load a YAML config (default: configs/default.yaml). Keyword overrides are deep-merged last."""
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    raw = _load_raw(path.resolve())
    if overrides:
        raw = deep_merge(raw, overrides)
    return Config(**raw)
