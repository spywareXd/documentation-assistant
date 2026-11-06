"""Prompts and output schemas for generation, document grading and groundedness checks."""
from __future__ import annotations

from pydantic import BaseModel, Field

from docrag.index.qdrant_store import Hit

REFUSAL = "I couldn't find this in the documentation I have indexed."

GENERATE_SYSTEM = f"""You are a documentation assistant for Python developers. You answer questions about \
FastAPI, Pydantic, Qdrant, LangChain and LangGraph using ONLY the numbered documentation excerpts provided.

Rules:
- Every factual sentence must end with citations to the excerpts that support it, like [1] or [2][4].
- Use only information in the excerpts. Do not use outside knowledge, even if you are confident.
- If the excerpts do not contain the answer, set "answerable" to false and answer exactly: "{REFUSAL}"
  You may add one sentence on what the excerpts do cover.
- Keep code identifiers exact and use fenced code blocks for code. Be concise.

Return JSON: {{"answerable": bool, "answer": "markdown answer with [n] citations", "citations": [n, ...]}}"""


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        crumb = " > ".join([c.title, *[x for x in c.headings if x != c.title]])
        blocks.append(f"[{i}] {c.source} | {crumb}\nURL: {c.url}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def generate_messages(question: str, hits: list[Hit], history: list[dict] | None = None,
                      feedback: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": GENERATE_SYSTEM}]
    for m in (history or [])[-6:]:
        msgs.append({"role": m["role"], "content": m["content"][:2000]})
    user = f"Documentation excerpts:\n\n{format_context(hits)}\n\nQuestion: {question}"
    if feedback:
        user += f"\n\nA reviewer rejected your previous draft: {feedback}\nWrite a corrected answer that only states supported facts."
    msgs.append({"role": "user", "content": user})
    return msgs


class Answer(BaseModel):
    answerable: bool = True
    answer: str
    citations: list[int] = Field(default_factory=list)


# --------------------------------------------------------------------------- grading
class Relevance(BaseModel):
    relevant: list[int] = Field(description="numbers of excerpts that help answer the question")


def grade_messages(question: str, hits: list[Hit]) -> list[dict]:
    return [{
        "role": "user",
        "content": (
            "Which of these documentation excerpts contain information that helps answer the question? "
            "An excerpt is relevant if it answers part of the question or defines something the answer needs. "
            "Be strict: same library but different topic is NOT relevant.\n\n"
            f"Question: {question}\n\nExcerpts:\n\n{format_context(hits)}\n\n"
            'Return JSON: {"relevant": [excerpt numbers]} (empty list if none).'
        ),
    }]


class Groundedness(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)


def groundedness_messages(answer: str, hits: list[Hit]) -> list[dict]:
    return [{
        "role": "user",
        "content": (
            "Check whether every factual claim in the ANSWER is supported by the EXCERPTS. "
            "Ignore style; refusals and statements that the docs do not cover something count as grounded.\n\n"
            f"EXCERPTS:\n\n{format_context(hits)}\n\nANSWER:\n{answer}\n\n"
            'Return JSON: {"grounded": bool, "unsupported_claims": ["claim", ...]}'
        ),
    }]


def crag_rewrite_messages(question: str, failed_query: str) -> list[dict]:
    return [{
        "role": "user",
        "content": (
            "A search over Python docs (FastAPI, Pydantic, Qdrant, LangChain, LangGraph) for the question below "
            f'returned nothing relevant using the query "{failed_query}". Write a different search query: '
            "use other terms, name the likely library and API, or search for the underlying concept.\n\n"
            f"Question: {question}\n\nReturn only the new query."
        ),
    }]
