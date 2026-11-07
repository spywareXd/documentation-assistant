"""Query transformations (utility model, cached).

rewrite      : resolve follow-ups against chat history and turn the question into a
               self-contained, keyword-rich search query.
multi_query  : n diverse reformulations; each is retrieved and the lists are RRF-fused.
hyde         : write a hypothetical documentation passage and search with *its* embedding
               (Gao et al., 2022); BM25 still uses the original question.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from docrag.llm import LLM, QuotaExhausted

DOMAIN = "Python developer documentation for FastAPI, Pydantic, Qdrant, LangChain and LangGraph"


class _Queries(BaseModel):
    queries: list[str] = Field(description="search queries")


def _history_block(history: list[dict] | None) -> str:
    if not history:
        return ""
    turns = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])
    return f"Conversation so far:\n{turns}\n\n"


def rewrite(llm: LLM, question: str, history: list[dict] | None = None) -> str:
    msg = (
        f"{_history_block(history)}You turn user questions into search queries over {DOMAIN}.\n"
        "Rewrite the question below as ONE standalone search query: resolve pronouns and references "
        "to earlier turns, keep exact identifiers (class names, function names, parameters, error "
        "messages), add the library name if it is implied. Do not answer it.\n\n"
        f"Question: {question}\n\nReturn only the query."
    )
    out = llm.chat([{"role": "user", "content": msg}], role="utility", max_tokens=300).strip()
    return out.strip('"') or question


def multi_query(llm: LLM, question: str, n: int = 3, history: list[dict] | None = None) -> list[str]:
    msg = (
        f"{_history_block(history)}Write {n} different search queries for retrieving passages from {DOMAIN} "
        "that answer the question below. Vary vocabulary and angle (API name, concept, error, task). "
        f'Keep exact identifiers.\n\nQuestion: {question}\n\nReturn JSON: {{"queries": [...]}}'
    )
    try:
        qs = llm.structured([{"role": "user", "content": msg}], _Queries, role="utility").queries
    except Exception as _err:
        if isinstance(_err, QuotaExhausted):
            raise
        qs = []
    seen, out = set(), []
    for q in [question, *qs[:n]]:
        if q.strip() and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q.strip())
    return out


def hyde(llm: LLM, question: str) -> str:
    msg = (
        f"Write a short passage (under 150 words) as it would appear in {DOMAIN}, answering the question "
        "below. Include a small code example if it helps. It is fine to be approximately right.\n\n"
        f"Question: {question}"
    )
    return llm.chat([{"role": "user", "content": msg}], role="utility", max_tokens=500).strip()
