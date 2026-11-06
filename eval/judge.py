"""LLM-as-judge for answer quality (temperature 0, cached, model from config.llm.judge).

Per answerable question:
  correctness   0 / 0.5 / 1   vs the human-verified reference answer
  faithfulness  supported claims / all claims, judged against the context the generator saw
Deterministic (no LLM):
  refusal_correct      unanswerable question -> model said answerable=false
  false_refusal        answerable question  -> model refused
  citation_precision   share of cited chunks that cover gold evidence

Validate the judge: `python -m eval.spotcheck results/<run>.jsonl` samples 20 verdicts for a
human to grade, and reports agreement.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from docrag.graph.prompts import format_context
from docrag.llm import LLM


class Claim(BaseModel):
    claim: str
    supported: bool


class Verdict(BaseModel):
    correctness: float = Field(description="1 = fully correct and complete, 0.5 = partially correct or incomplete, 0 = wrong or missing")
    reason: str
    claims: list[Claim] = Field(default_factory=list)


JUDGE_PROMPT = """You are grading an answer from a documentation assistant.

QUESTION:
{question}

REFERENCE ANSWER (human-verified):
{reference}

CONTEXT the assistant was given:
{context}

ASSISTANT ANSWER:
{answer}

1. correctness: compare the ASSISTANT ANSWER with the REFERENCE ANSWER. 1 if it contains the key facts \
of the reference and nothing contradicting it; 0.5 if partially correct or missing an important part; \
0 if wrong, contradicting, or a refusal. Extra correct detail is fine.
2. claims: split the ASSISTANT ANSWER into its factual claims (ignore citations markers and filler) and mark \
each supported=true only if the CONTEXT states or directly implies it.

Return JSON: {{"correctness": 0|0.5|1, "reason": "one sentence", "claims": [{{"claim": str, "supported": bool}}]}}"""


def judge_answer(llm: LLM, question: str, reference: str, answer: str, context_hits) -> dict:
    msg = JUDGE_PROMPT.format(question=question, reference=reference,
                              context=format_context(context_hits) or "(none)", answer=answer)
    v = llm.structured([{"role": "user", "content": msg}], Verdict, role="judge", max_tokens=2048)
    correctness = min(1.0, max(0.0, round(v.correctness * 2) / 2))
    n = len(v.claims)
    faithfulness = sum(c.supported for c in v.claims) / n if n else 1.0
    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "n_claims": n,
        "unsupported_claims": [c.claim for c in v.claims if not c.supported],
        "judge_reason": v.reason,
    }
