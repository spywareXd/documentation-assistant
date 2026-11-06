"""Hand-check the LLM judge on a random sample and report agreement.

    python -m eval.spotcheck results/raw/e5_crag__test.jsonl --n 20

For each sampled answer you grade correctness (1 / 0.5 / 0) yourself; the script reports exact
agreement and mean absolute difference vs the judge, saved to results/spotcheck.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from docrag.config import ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    golden = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "eval/golden_set.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    rows = [json.loads(l) for l in args.raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("correctness") is not None]
    sample = random.Random(args.seed).sample(rows, min(args.n, len(rows)))

    pairs = []
    for i, r in enumerate(sample, 1):
        print("\n" + "=" * 80 + f"\n[{i}/{len(sample)}] {r['question']}\n")
        print(f"REFERENCE: {golden[r['id']]['reference_answer']}\n\nANSWER:\n{r['answer']}\n")
        while True:
            v = input("your correctness (1 / 0.5 / 0): ").strip()
            if v in ("1", "0.5", "0"):
                pairs.append((float(v), r["correctness"]))
                break

    agree = sum(h == j for h, j in pairs) / len(pairs)
    mad = sum(abs(h - j) for h, j in pairs) / len(pairs)
    out = {"file": str(args.raw), "n": len(pairs), "exact_agreement": round(agree, 3),
           "mean_abs_diff": round(mad, 3), "pairs_human_judge": pairs}
    (ROOT / "results" / "spotcheck.json").write_text(json.dumps(out, indent=2))
    print(f"\nagreement {agree:.0%}, mean |diff| {mad:.2f} -> results/spotcheck.json")


if __name__ == "__main__":
    main()
