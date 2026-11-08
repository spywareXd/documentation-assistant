"""Offline analyses over saved raw results (no re-retrieval needed).

    python -m eval.analysis budget  --split dev     # recall at an equal context-token budget (E1 chunking)
    python -m eval.analysis dynk    --split dev     # sweep dynamic-k thresholds on reranker scores (E3)

Both read results/raw/<run>__<split>.jsonl plus the chunk files, so they are cheap and reproducible.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from docrag.config import ROOT, Config
from docrag.ingest.chunkers import Chunk
from docrag.ingest.pipeline import build_documents, chunks_path
from eval.metrics import Evidence, covers, locate

RESULTS = ROOT / "results"


def _load(run: str, split: str) -> tuple[Config, list[dict]]:
    meta = json.loads((RESULTS / f"{run}__{split}.json").read_text())
    rows = [json.loads(l) for l in (RESULTS / "raw" / f"{run}__{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    return Config(**meta["config"]), rows


def _chunks(cfg: Config) -> dict[str, Chunk]:
    path = chunks_path(cfg)
    return {c["chunk_id"]: Chunk(**c) for c in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def _evidence(golden: dict, docs: dict) -> dict[str, list[Evidence]]:
    out = {}
    for qid, g in golden.items():
        out[qid] = [locate(Evidence(e["url"], e["quote"]), docs[e["url"]].text) for e in g["evidence"] if e["url"] in docs]
    return out


def _golden() -> dict:
    return {r["id"]: r for r in map(json.loads, (ROOT / "eval/golden_set.jsonl").read_text(encoding="utf-8").splitlines())}


def budget(split: str, runs: list[str], budgets=(500, 1000, 1500, 2500)) -> dict:
    """Evidence recall when context is filled in rank order until ~N tokens (chars/4)."""
    golden, out = _golden(), {}
    docs = {d.url: d for d in build_documents(Config())}
    ev = _evidence(golden, docs)
    for run in runs:
        cfg, rows = _load(run, split)
        chunks = _chunks(cfg)
        res = {}
        for b in budgets:
            vals = []
            for r in rows:
                evs = ev.get(r["id"]) or []
                if r["type"] == "unanswerable" or not evs:
                    continue
                used, covered = 0, set()
                for h in r["retrieved"]:
                    c = chunks[h["chunk_id"]]
                    t = len(c.text) // 4
                    if used + t > b and used > 0:
                        break
                    used += t
                    covered |= {j for j, e in enumerate(evs) if covers(c, e)}
                vals.append(len(covered) / len(evs))
            res[f"recall@{b}tok"] = round(float(np.mean(vals)), 4)
        out[run] = res
        print(f"{run:28} " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()))
    return out


def dynk(split: str, run: str = "e3_fixed_k10") -> list[dict]:
    """Sweep score_threshold and relative_drop on the reranked top-10 saved by a k=10 run."""
    golden = _golden()
    docs = {d.url: d for d in build_documents(Config())}
    ev = _evidence(golden, docs)
    cfg, rows = _load(run, split)
    chunks = _chunks(cfg)
    rows = [r for r in rows if r["type"] != "unanswerable" and ev.get(r["id"])]

    def score(select) -> dict:
        ks, rec, prec = [], [], []
        for r in rows:
            hits = r["retrieved"][:10]
            n = select([h["score"] for h in hits])
            sel = hits[:n]
            evs = ev[r["id"]]
            rel = [{j for j, e in enumerate(evs) if covers(chunks[h["chunk_id"]], e)} for h in sel]
            ks.append(n)
            rec.append(len(set().union(*rel)) / len(evs) if rel else 0.0)
            prec.append(sum(1 for x in rel if x) / n if n else 0.0)
        return {"k": round(float(np.mean(ks)), 2), "context_recall": round(float(np.mean(rec)), 4),
                "context_precision": round(float(np.mean(prec)), 4)}

    out = []
    for k in (3, 5, 10):
        out.append({"policy": f"fixed k={k}", **score(lambda s, k=k: k)})
    for mn, mx in ((2, 10), (3, 10)):
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            out.append({"policy": f"threshold {t} [{mn},{mx}]",
                        **score(lambda s, t=t: max(mn, min(mx, sum(x >= t for x in s))))})
        for d in (0.3, 0.5, 0.7, 0.9):
            out.append({"policy": f"relative {d} [{mn},{mx}]",
                        **score(lambda s, d=d: max(mn, min(mx, sum(x >= d * s[0] for x in s))))})
    for o in out:
        print(f"{o['policy']:24} k={o['k']:5.2f}  context_recall={o['context_recall']:.3f}  context_precision={o['context_precision']:.3f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["budget", "dynk"])
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    if a.what == "budget":
        res = budget(a.split, ["e1_recursive_512", "e1_recursive_1000", "e1_recursive_4000", "e1_markdown_header",
                               "e1_markdown_header_1500", "e1_semantic"])
    else:
        res = dynk(a.split)
    (RESULTS / f"analysis_{a.what}__{a.split}.json").write_text(json.dumps(res, indent=2))
