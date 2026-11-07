"""Draft golden-set questions with an LLM. Every draft must then be human-reviewed (eval/review.py).

    python -m eval.build_golden --per-source 36            # ~180 drafts -> eval/golden_draft.jsonl

Mix per source: ~60% single-passage, ~30% multi-passage, ~10% unanswerable.
Guards against a biased set:
- questions are asked "as a developer would", with an instruction not to reuse the passage's
  wording (otherwise BM25 wins by construction);
- every quote must appear verbatim in the page, or the draft is discarded;
- unanswerable drafts are kept only if the current best retriever + an LLM check find no answer
  in the top 10 (the human reviewer still confirms);
- the dev/test split is a hash of the question id, fixed before anyone looks at results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from pydantic import BaseModel, Field
from tqdm import tqdm

from docrag.config import load_config
from docrag.ingest.chunkers import Document
from docrag.ingest.pipeline import build_documents
from docrag.llm import get_llm, QuotaExhausted
from eval.metrics import Evidence, locate

OUT = Path(__file__).parent / "golden_draft.jsonl"
PASSAGE_CHARS = 1800
TEST_SHARE = 0.2

STYLE = (
    "Write the question the way a developer would type it into a docs search box or ask a colleague, "
    "WITHOUT having seen this passage: describe the goal or problem in your own words, do not copy "
    "phrases from the passage (exact API names are fine if a developer would know them). "
    "No 'according to the docs' or 'in this passage'."
)


class Draft(BaseModel):
    question: str
    answer: str = Field(description="complete reference answer, 1-4 sentences, may include a short code snippet")
    quotes: list[str] = Field(description="exact verbatim sentence(s) copied from the passage(s) that prove the answer")


class Unanswerable(BaseModel):
    questions: list[str]


def split_for(qid: str) -> str:
    return "test" if int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 100 < TEST_SHARE * 100 else "dev"


def _passage(doc: Document, rng: random.Random) -> tuple[int, str]:
    """A paragraph-aligned window of the page."""
    text = doc.text
    if len(text) <= PASSAGE_CHARS:
        return 0, text
    start = rng.randrange(0, len(text) - PASSAGE_CHARS)
    para = text.rfind("\n\n", 0, start)
    start = para + 2 if para >= 0 else 0
    return start, text[start : start + PASSAGE_CHARS]


def _evidence(doc: Document, quotes: list[str]) -> list[Evidence] | None:
    out = []
    for q in quotes:
        q = q.strip().strip('"')
        if len(q) < 20:
            continue
        ev = locate(Evidence(doc.url, q), doc.text)
        if not ev.located:
            return None  # the model paraphrased instead of quoting -> reject the draft
        out.append(ev)
    return out or None


def draft_single(llm, doc: Document, rng) -> dict | None:
    _, passage = _passage(doc, rng)
    msg = (
        f"Library docs page: {doc.title} ({doc.source})\n\nPASSAGE:\n{passage}\n\n"
        "Write ONE question that this passage fully answers and that a real developer would plausibly ask. "
        f"{STYLE} Prefer questions about behaviour, configuration, errors or how-to over trivia.\n"
        'Return JSON: {"question": str, "answer": str, "quotes": [verbatim sentences from PASSAGE]}'
    )
    d = llm.structured([{"role": "user", "content": msg}], Draft, role="generator")
    ev = _evidence(doc, d.quotes)
    return ev and {"type": "single", "question": d.question, "reference_answer": d.answer, "evidence": ev}


def draft_multi(llm, docs: list[Document], rng) -> dict | None:
    """Two passages from the same source (often the same page) that must be combined."""
    a = rng.choice(docs)
    same_source = [d for d in docs if d.source == a.source and d.doc_id != a.doc_id]
    b = a if (rng.random() < 0.5 or not same_source) else rng.choice(same_source)
    (sa, pa), (sb, pb) = _passage(a, rng), _passage(b, rng)
    if a is b and abs(sa - sb) < PASSAGE_CHARS:
        return None
    msg = (
        f"PASSAGE A ({a.title}):\n{pa}\n\nPASSAGE B ({b.title}):\n{pb}\n\n"
        "Write ONE realistic developer question whose complete answer needs information from BOTH passages "
        f"(e.g. combining two features, or a setting plus its caveat). {STYLE} If no natural question needs "
        'both, return {"question": "", "answer": "", "quotes": []}.\n'
        'Return JSON: {"question": str, "answer": str, "quotes": [one verbatim sentence from A, one from B]}'
    )
    d = llm.structured([{"role": "user", "content": msg}], Draft, role="generator")
    if not d.question or len(d.quotes) < 2:
        return None
    ev_a, ev_b = _evidence(a, d.quotes[:1]), _evidence(b, d.quotes[1:2])
    if not ev_a or not ev_b:
        return None
    return {"type": "multi", "question": d.question, "reference_answer": d.answer, "evidence": ev_a + ev_b}


def draft_unanswerable(llm, source: str, titles: list[str], n: int) -> list[dict]:
    msg = (
        f"These are page titles from the {source} documentation: {', '.join(titles[:60])}.\n\n"
        f"Write {n} realistic developer questions about {source} that this documentation most likely does "
        "NOT answer: features that do not exist, integrations with unrelated tools, version-specific trivia, "
        "or operational details outside the docs' scope. They must sound plausible, not absurd.\n"
        'Return JSON: {"questions": [str, ...]}'
    )
    qs = llm.structured([{"role": "user", "content": msg}], Unanswerable, role="generator").questions
    return [{"type": "unanswerable", "question": q, "reference_answer": "Not covered by the documentation.",
             "evidence": []} for q in qs[:n]]


def confirm_unanswerable(llm, retriever, item: dict) -> bool:
    """Drop 'unanswerable' drafts the corpus actually answers."""
    hits = retriever.retrieve(item["question"]).ranked[:10]
    excerpts = "\n\n---\n\n".join(f"[{i}] {h.chunk.text[:1500]}" for i, h in enumerate(hits, 1))
    msg = (f"Question: {item['question']}\n\nExcerpts:\n{excerpts}\n\n"
           'Does any excerpt answer the question, fully or mostly? Return JSON: {"answered": bool}')

    class V(BaseModel):
        answered: bool

    return not llm.structured([{"role": "user", "content": msg}], V, role="judge").answered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--per-source", type=int, default=36, help="drafts per source before review")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-unanswerable-check", action="store_true")
    ap.add_argument("--sources", nargs="*", help="only draft these sources; appends to the existing draft file")
    ap.add_argument("--drafter", help="model that writes drafts (default: config llm.generator)")
    args = ap.parse_args()

    cfg = load_config(args.config, **({"llm": {"generator": args.drafter}} if args.drafter else {}))
    llm = get_llm(cfg)
    rng = random.Random(args.seed)
    docs = [d for d in build_documents(cfg) if len(d.text) > 800]
    by_source: dict[str, list[Document]] = {}
    for d in docs:
        by_source.setdefault(d.source, []).append(d)

    retriever = None
    if not args.skip_unanswerable_check:
        from docrag.retrieve.retriever import Retriever

        retriever = Retriever(cfg)

    # drafts are written per source, so a crash keeps finished sources; never overwrite reviewed work
    done = {json.loads(l)["source"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()} if OUT.exists() else set()
    if args.sources:
        by_source = {k: v for k, v in by_source.items() if k in args.sources}
    elif done:
        raise SystemExit(f"{OUT} already has drafts for {sorted(done)}; use --sources to add more")
    if done & set(by_source):
        raise SystemExit(f"already drafted: {sorted(done & set(by_source))}")
    total = 0
    for source, sdocs in by_source.items():
        n_single = round(args.per_source * 0.6)
        n_multi = round(args.per_source * 0.3)
        n_unans = args.per_source - n_single - n_multi
        got: list[dict] = []
        with tqdm(total=n_single + n_multi, desc=source) as bar:
            attempts = 0
            while len(got) < n_single + n_multi and attempts < 4 * (n_single + n_multi):
                attempts += 1
                want_single = sum(i["type"] == "single" for i in got) < n_single
                try:
                    it = draft_single(llm, rng.choice(sdocs), rng) if want_single else draft_multi(llm, sdocs, rng)
                except Exception as err:
                    if isinstance(err, QuotaExhausted):
                        raise
                    print(f"draft failed: {str(err)[:120]}")
                    continue
                if it:
                    got.append(it)
                    bar.update(1)
        try:
            unans = draft_unanswerable(llm, source, [d.title for d in sdocs], n_unans * 2)
        except Exception as err:
            if isinstance(err, QuotaExhausted):
                raise
            print(f"unanswerable drafting failed for {source}: {str(err)[:120]}")
            unans = []
        kept = []
        for u in unans:
            if len(kept) == n_unans:
                break
            if retriever is None:
                kept.append(u)
                continue
            try:
                if confirm_unanswerable(llm, retriever, u):
                    kept.append(u)
            except Exception as err:  # unclear -> keep it and let the human reviewer decide
                if isinstance(err, QuotaExhausted):
                    raise
                print(f"unanswerable check failed, keeping for review: {str(err)[:100]}")
                kept.append(u | {"check_failed": True})
        got.extend(kept)

        with OUT.open("a", encoding="utf-8") as f:
            for it in got:
                qid = hashlib.sha1(it["question"].encode()).hexdigest()[:10]
                row = {"id": qid, "split": split_for(qid), "status": "pending", "source": source,
                   "drafted_by": cfg.llm.generator, **it,
                       "evidence": [{"url": e.url, "quote": e.quote} for e in it["evidence"]]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        total += len(got)
        print(f"{source}: {len(got)} drafts saved")

    print(f"{total} drafts -> {OUT}\nNext: python -m eval.review")
    print(json.dumps(get_llm(cfg).usage_report(), indent=2))


if __name__ == "__main__":
    main()
