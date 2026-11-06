"""Human review of golden-set drafts. Nothing reaches golden_set.jsonl without passing here.

    python -m eval.review            # review pending drafts (progress is saved after every item)
    python -m eval.review --export   # write accepted items to eval/golden_set.jsonl

Keys: [a]ccept  [e]dit question/answer then accept  [r]eject  [s]kip  [q]uit
Reject anything ambiguous, trivially lexical, wrong, or not answerable from the quoted text.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from docrag.config import load_config
from docrag.ingest.pipeline import build_documents
from eval.metrics import Evidence, locate

HERE = Path(__file__).parent
DRAFT = HERE / "golden_draft.jsonl"
GOLDEN = HERE / "golden_set.jsonl"
BOLD, DIM, YEL, END = "\033[1m", "\033[2m", "\033[93m", "\033[0m"


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _save(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _show(item: dict, docs_by_url: dict) -> None:
    print("\n" + "=" * 80)
    print(f"{BOLD}[{item['type']}] {item['source']}  id={item['id']}  split={item['split']}{END}")
    print(f"\n{BOLD}Q:{END} {item['question']}\n{BOLD}A:{END} {item['reference_answer']}")
    for ev in item["evidence"]:
        doc = docs_by_url.get(ev["url"])
        print(f"\n{DIM}{ev['url']}{END}")
        if not doc:
            print("  (page not in current corpus!)")
            continue
        e = locate(Evidence(ev["url"], ev["quote"]), doc.text)
        if e.located:
            lo, hi = max(0, e.start - 400), min(len(doc.text), e.end + 400)
            print(doc.text[lo:e.start] + YEL + doc.text[e.start:e.end] + END + doc.text[e.end:hi])
        else:
            print(f"  quote not found: {ev['quote']}")


def review() -> None:
    rows = _load(DRAFT)
    docs_by_url = {d.url: d for d in build_documents(load_config())}
    pending = [r for r in rows if r["status"] == "pending"]
    print(f"{len(pending)} pending of {len(rows)}  {dict(Counter(r['status'] for r in rows))}")
    for item in pending:
        _show(item, docs_by_url)
        while True:
            key = input(f"\n{BOLD}[a]ccept [e]dit [r]eject [s]kip [q]uit > {END}").strip().lower()
            if key == "a":
                item["status"] = "accepted"
            elif key == "e":
                q = input("question (enter = keep): ").strip()
                a = input("answer   (enter = keep): ").strip()
                item["question"] = q or item["question"]
                item["reference_answer"] = a or item["reference_answer"]
                item["status"], item["edited"] = "accepted", True
            elif key == "r":
                item["status"] = "rejected"
                item["reject_reason"] = input("reason (optional): ").strip()
            elif key == "s":
                pass
            elif key == "q":
                _save(DRAFT, rows)
                return
            else:
                continue
            break
        _save(DRAFT, rows)
    print("done. Run with --export.")


def export() -> None:
    rows = [r for r in _load(DRAFT) if r["status"] == "accepted"]
    out = [{k: r[k] for k in ("id", "split", "type", "source", "question", "reference_answer", "evidence")} | {"edited": r.get("edited", False)} for r in rows]
    _save(GOLDEN, out)
    c = Counter((r["split"], r["type"]) for r in out)
    print(f"{len(out)} items -> {GOLDEN}\n{dict(sorted(c.items()))}")
    print(dict(Counter(r["source"] for r in out)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    (export if ap.parse_args().export else review)()
