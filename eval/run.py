"""Run one experiment config over the golden set and save every raw output.

    python -m eval.run --config configs/experiments/e2_hybrid_rrf.yaml --retrieval-only
    python -m eval.run --config configs/experiments/e5_crag.yaml --split test

Writes
    results/<name>__<split>.json    aggregate metrics + environment (committed)
    results/raw/<name>__<split>.jsonl  per-question outputs (committed, small)
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm

from docrag.config import ROOT, Config, load_config
from docrag.ingest.pipeline import build_documents, chunks_path
from eval.metrics import Evidence, locate, relevance_matrix, retrieval_scores

RESULTS = ROOT / "results"
WARMUP_QUERIES = ["How do I define a model?", "What is a collection?", "How do I create an agent?",
                  "How do path parameters work?", "What is a checkpointer?"]


def load_golden(cfg: Config) -> list[dict]:
    path = cfg.path(cfg.eval.golden_set)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if cfg.eval.split != "all":
        rows = [r for r in rows if r["split"] == cfg.eval.split]
    if cfg.corpus.sources:
        rows = [r for r in rows if r["source"] in cfg.corpus.sources]
    rows.sort(key=lambda r: r["id"])
    return rows[: cfg.eval.limit] if cfg.eval.limit else rows


def _pcts(values: list[float]) -> dict:
    if not values:
        return {}
    a = np.array(values)
    return {"p50": round(float(np.percentile(a, 50)), 1), "p95": round(float(np.percentile(a, 95)), 1),
            "p99": round(float(np.percentile(a, 99)), 1), "mean": round(float(a.mean()), 1)}


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def _bootstrap_ci(vals: list[float], n: int = 2000, seed: int = 0) -> list[float] | None:
    if len(vals) < 5:
        return None
    rng = np.random.default_rng(seed)
    a = np.array(vals)
    means = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def environment(cfg: Config) -> dict:
    env = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "python": platform.python_version(),
           "platform": platform.platform()}
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                           cwd=ROOT).stdout.strip()
        env["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                               cwd=ROOT).stdout.strip())
    except OSError:
        pass
    try:
        import torch

        env["torch"] = torch.__version__
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    for pkg in ("qdrant_client", "sentence_transformers", "fastembed", "langgraph", "groq"):
        try:
            env[pkg] = __import__(pkg).__version__
        except (ImportError, AttributeError):
            pass
    lock = ROOT / "sources.lock.json"
    if lock.exists():
        env["corpus_lock"] = json.loads(lock.read_text())
    stats = chunks_path(cfg).with_suffix(".stats.json")
    if stats.exists():
        s = json.loads(stats.read_text())
        env["corpus"] = {k: s[k] for k in ("documents", "chunks", "documents_per_source", "chunk_chars")}
    return env


def _hit_row(h) -> dict:
    return {"chunk_id": h.chunk.chunk_id, "url": h.chunk.url, "score": round(h.score, 5)}


def run(cfg: Config, retrieval_only: bool) -> dict:
    golden = load_golden(cfg)
    docs = {d.url: d for d in build_documents(cfg)}
    ks = cfg.eval.ks
    tag = f"{cfg.name}__{cfg.eval.split}"
    print(f"{tag}: {len(golden)} questions, collection={cfg.collection_name()}, retrieval_only={retrieval_only}")

    if retrieval_only:
        from docrag.retrieve.retriever import Retriever

        retriever, pipeline = Retriever(cfg), None
    else:
        from docrag.graph.rag_graph import RAGPipeline

        pipeline = RAGPipeline(cfg)
        retriever = pipeline.retriever
    if not retriever.store.exists() and cfg.retrieval.mode != "tfidf":
        raise SystemExit(f"collection {cfg.collection_name()} missing: python -m docrag.ingest.pipeline --config <this config>")
    retriever.warmup()
    for q in WARMUP_QUERIES:  # discarded: first queries pay CUDA/kernel and connection set-up
        retriever.retrieve(q)

    llm = pipeline.llm if pipeline else None
    rows = []
    for item in tqdm(golden, desc=tag):
        evidence = [locate(Evidence(e["url"], e["quote"]), docs[e["url"]].text) if e["url"] in docs
                    else Evidence(e["url"], e["quote"]) for e in item["evidence"]]
        row = {"id": item["id"], "type": item["type"], "source": item["source"], "question": item["question"]}
        t0 = time.perf_counter()
        if retrieval_only:
            res = retriever.retrieve(item["question"])
            row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        else:
            out = pipeline.answer(item["question"])
            res = out["first_retrieval"]
            row |= {"latency_ms": out["latency_ms"], "answer": out["answer"], "answerable": out["answerable"],
                    "citations": out["citations"], "grounded": out["grounded"], "rewrites": out["rewrites"],
                    "trace": out["trace"]}
        row["search_queries"] = res.search_queries
        row["timings_ms"] = res.timings_ms
        row["retrieved"] = [_hit_row(h) for h in res.ranked[: max(ks)]]
        row["selected"] = [_hit_row(h) for h in res.selected]

        if item["type"] != "unanswerable":
            row |= retrieval_scores([h.chunk for h in res.ranked], [h.chunk for h in res.selected], evidence, ks)
        if not retrieval_only:
            if item["type"] == "unanswerable":
                row["refusal_correct"] = float(not out["answerable"])
            else:
                from eval.judge import judge_answer

                row["false_refusal"] = float(not out["answerable"])
                row |= judge_answer(llm, item["question"], item["reference_answer"], out["answer"], out["context"])
                cited = out["cited"]
                rel = relevance_matrix([h.chunk for h in cited], evidence)
                row["citation_precision"] = (sum(1 for r in rel if r) / len(rel)) if rel else None
                row["context_tokens_est"] = sum(len(h.chunk.text) for h in out["context"]) // 4
        rows.append(row)

    answerable = [r for r in rows if r["type"] != "unanswerable"]
    metric_keys = [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks] + [
        "mrr", "ndcg@10", "context_recall", "context_precision", "k_selected"]
    agg: dict = {"n_questions": len(rows), "n_answerable": len(answerable)}
    agg["retrieval"] = {k: _mean(answerable, k) for k in metric_keys}
    agg["retrieval_ci95"] = {k: _bootstrap_ci([r[k] for r in answerable]) for k in ("recall@5", "mrr", "ndcg@10")}
    agg["by_type"] = {t: {k: _mean([r for r in answerable if r["type"] == t], k) for k in ("recall@5", "mrr")}
                      for t in sorted({r["type"] for r in answerable})}
    agg["by_source"] = {s: {k: _mean([r for r in answerable if r["source"] == s], k) for k in ("recall@5", "mrr")}
                        for s in sorted({r["source"] for r in answerable})}

    stage = defaultdict(list)
    for r in rows:
        for k, v in r["timings_ms"].items():
            stage[k].append(v)
    agg["latency_ms"] = {"end_to_end": _pcts([r["latency_ms"] for r in rows]),
                         **{f"retrieval.{k}": _pcts(v) for k, v in stage.items()}}

    if not retrieval_only:
        unans = [r for r in rows if r["type"] == "unanswerable"]
        agg["generation"] = {
            "correctness": _mean(answerable, "correctness"),
            "correctness_ci95": _bootstrap_ci([r["correctness"] for r in answerable if r.get("correctness") is not None]),
            "faithfulness": _mean(answerable, "faithfulness"),
            "citation_precision": _mean(answerable, "citation_precision"),
            "false_refusal_rate": _mean(answerable, "false_refusal"),
            "unanswerable_refusal_rate": _mean(unans, "refusal_correct"),
            "groundedness_check_pass": _mean([{"g": float(r["grounded"])} for r in rows
                                              if r.get("grounded") is not None], "g"),
            "avg_rewrites": _mean(rows, "rewrites"),
            "avg_context_tokens_est": _mean(answerable, "context_tokens_est"),
        }
        agg["llm_usage"] = llm.usage_report()
        agg["llm_usage_per_query"] = {
            role: {k: round(v / len(rows), 1) for k, v in u.items() if k.endswith("tokens")}
            for role, u in agg["llm_usage"].items()
        }

    result = {"name": cfg.name, "split": cfg.eval.split, "retrieval_only": retrieval_only,
              "config": cfg.model_dump(), "collection": cfg.collection_name(), "environment": environment(cfg),
              "metrics": agg}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "raw").mkdir(exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(result, indent=2))
    with (RESULTS / "raw" / f"{tag}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(agg["retrieval"], indent=2))
    if "generation" in agg:
        print(json.dumps(agg["generation"], indent=2))
    print(f"-> results/{tag}.json")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["dev", "test", "all"], default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()
    over = {"eval": {k: v for k, v in {"split": args.split, "limit": args.limit}.items() if v is not None}}
    cfg = load_config(args.config, **over)
    from docrag.llm import QuotaExhausted

    try:
        run(cfg, retrieval_only=args.retrieval_only or cfg.eval.retrieval_only)
    except QuotaExhausted as err:
        print(f"\nSTOPPED: {err}\nFinished LLM calls are cached; re-run the same command after the quota resets.")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
