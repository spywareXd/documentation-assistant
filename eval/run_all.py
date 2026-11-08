"""Run the whole benchmark, resumably. One command, safe to re-run any time:

    python -m eval.run_all                 # everything that is not done yet
    python -m eval.run_all --phase retrieval   # only the no-LLM retrieval experiments
    python -m eval.run_all --phase llm         # only the LLM experiments (Groq quota)

- A run whose results file exists (same config) is skipped.
- Missing chunk sets / Qdrant collections are built automatically.
- When a Groq model's daily quota runs out, it stops cleanly; finished LLM calls are cached,
  so the next invocation continues where this one stopped.
- `python -m eval.report` is called at the end to regenerate RESULTS.md.

Protocol: retrieval settings are chosen on the dev split; the test split is only used to report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from docrag.config import ROOT, load_config
from docrag.llm import QuotaExhausted

EXP = ROOT / "configs" / "experiments"
RESULTS = ROOT / "results"

# (config name, split, retrieval_only)
RETRIEVAL = [
    # E1: chunking, with retrieval held at hybrid RRF + rerank
    ("e1_recursive_4000", True), ("e1_recursive_1000", True), ("e1_recursive_512", True),
    ("e1_markdown_header", True), ("e1_markdown_header_1500", True), ("e1_semantic", True),
    # E2: retrieval methods
    ("e2_tfidf", True), ("e2_bm25", True), ("e2_dense", True), ("e2_hybrid_weighted", True),
    ("e2_hybrid_rrf", True), ("e2_hybrid_qdrant_rrf", True), ("e2_hybrid_qdrant_rrf_default", True), ("e2_dense_rerank", True),
    ("e2_hybrid_rrf_rerank", True),
    # E3: dynamic k (retrieval side: context precision / chunks sent to the LLM)
    ("e3_fixed_k3", True), ("e3_fixed_k5", True), ("e3_fixed_k10", True),
    ("e3_score_threshold", True), ("e3_relative_drop", True),
    # E0: the original retrieval design (4000-char chunks, dense only; k=4 is the original bug),
    # same local embedder as the new system
    ("e0_original_k4", True), ("e0_original_k5_fixed", True),
]
# E0, faithful: the original Gemini API embeddings. The free tier allows ~1000 embedding requests/day,
# so this is cached per chunk and finishes over a couple of days of re-runs.
GEMINI = [("e0_gemini_k4", True), ("e0_gemini_k5_fixed", True)]
LLM = [  # test split only (quota); judge runs on answer-generating rows
    ("e4_none", True), ("e4_rewrite", True), ("e4_multi_query", True), ("e4_hyde", True),
    ("e0_original_k5_fixed", False),
    ("e5_single", False), ("e5_crag", False),
]


def _done(cfg, split: str, retrieval_only: bool) -> bool:
    f = RESULTS / f"{cfg.name}__{split}.json"
    if not f.exists():
        return False
    r = json.loads(f.read_text())
    return r.get("config") == cfg.model_dump() and r.get("retrieval_only") == retrieval_only


def _ensure_index(cfg) -> None:
    if cfg.retrieval.mode == "tfidf":
        from docrag.ingest.pipeline import build_chunks, build_documents

        build_chunks(cfg, build_documents(cfg))
        return
    from docrag.index.qdrant_store import QdrantStore
    from docrag.ingest.pipeline import build_chunks, build_documents, build_index

    chunks = build_chunks(cfg, build_documents(cfg))
    if QdrantStore(cfg).count() != len(chunks):
        build_index(cfg, chunks)


def run_plan(plan: list[tuple[str, str, bool]], suffix: str = "") -> bool:
    from eval.run import run

    for name, split, retrieval_only in plan:
        cfg = load_config(EXP / f"{name}.yaml", eval={"split": split})
        if not retrieval_only and suffix:
            cfg = cfg.model_copy(update={"name": cfg.name + suffix})
        if _done(cfg, split, retrieval_only):
            print(f"[skip] {cfg.name}__{split} (done)")
            continue
        print(f"\n[run]  {cfg.name}__{split}  retrieval_only={retrieval_only}")
        t0 = time.time()
        try:
            _ensure_index(cfg)
            run(cfg, retrieval_only=retrieval_only)
        except QuotaExhausted as err:
            print(f"\n[stop] daily quota reached: {err}\nRe-run `python -m eval.run_all` after it resets.")
            return False
        except Exception as err:  # one broken run must not kill the whole suite
            import traceback

            traceback.print_exc()
            print(f"[fail] {cfg.name}__{split}: {type(err).__name__}: {str(err)[:200]}")
            continue
        print(f"[ok]   {cfg.name}__{split} in {time.time() - t0:.0f}s")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "retrieval", "llm", "gemini"], default="all")
    ap.add_argument("--split", choices=["dev", "test", "both"], default="both",
                    help="for the retrieval phase")
    args = ap.parse_args()

    finished = True
    if args.phase in ("all", "retrieval"):
        splits = ["dev", "test"] if args.split == "both" else [args.split]
        plan = [(n, s, ro) for s in splits for n, ro in RETRIEVAL]
        finished = run_plan(plan)
    if finished and args.phase in ("all", "llm"):
        finished = run_plan([(n, "test", ro) for n, ro in LLM], suffix="_gen")
    if args.phase in ("all", "retrieval", "gemini"):  # separate Google quota: never blocks the rest
        splits = ["dev", "test"] if args.split == "both" else [args.split]
        run_plan([(n, s, ro) for s in splits for n, ro in GEMINI])

    from eval.report import main as report

    report()
    sys.exit(0 if finished else 3)


if __name__ == "__main__":
    main()
