"""Answering pipelines.

single : retrieve -> generate.
graph  : corrective RAG (CRAG-style) as a LangGraph state machine

    retrieve -> grade_documents --(none relevant, budget left)--> rewrite_query -> retrieve
                      |
                      v
                  generate -> check_groundedness --(unsupported claims, 1 retry)--> generate
                                     |
                                     v
                                    END

Every node appends to `trace` (node name, timing, decision) so the UI and the eval harness
can show what happened.
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from docrag.config import Config
from docrag.graph import prompts as P
from docrag.index.qdrant_store import Hit
from docrag.llm import get_llm
from docrag.retrieve.retriever import RetrievalResult, Retriever


class RAGState(TypedDict, total=False):
    question: str
    history: list[dict]
    search_query: str
    retrieval: RetrievalResult
    retrievals: list[RetrievalResult]
    docs: list[Hit]
    rewrites: int
    regenerations: int
    answer: P.Answer
    grounded: bool | None
    feedback: str | None
    trace: list[dict]


def _log(state: RAGState, node: str, t0: float, **info) -> list[dict]:
    return [*state.get("trace", []), {"node": node, "ms": round((time.perf_counter() - t0) * 1000, 1), **info}]


class RAGPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.retriever = Retriever(cfg)
        self.llm = get_llm(cfg)
        self.graph = self._build() if cfg.pipeline.mode == "graph" else None

    # ------------------------------------------------------------------ nodes
    def retrieve(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        query = state.get("search_query") or state["question"]
        res = self.retriever.retrieve(query, history=state.get("history"))
        return {
            "retrieval": res,
            "retrievals": [*state.get("retrievals", []), res],
            "docs": res.selected,
            "trace": _log(state, "retrieve", t0, query=query, n=len(res.selected), timings_ms=res.timings_ms),
        }

    def grade_documents(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        hits = state["docs"]
        if not hits:
            return {"docs": [], "trace": _log(state, "grade_documents", t0, kept=0, of=0)}
        try:
            rel = self.llm.structured(P.grade_messages(state["question"], hits), P.Relevance, role="utility")
            keep = sorted({i for i in rel.relevant if 1 <= i <= len(hits)})
        except Exception as err:  # grading is an optimisation; on failure keep every chunk
            return {"docs": hits, "trace": _log(state, "grade_documents", t0, error=str(err)[:200])}
        return {"docs": [hits[i - 1] for i in keep],
                "trace": _log(state, "grade_documents", t0, kept=len(keep), of=len(hits))}

    def rewrite_query(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        failed = state.get("search_query") or state["question"]
        new = self.llm.chat(P.crag_rewrite_messages(state["question"], failed), role="utility", max_tokens=200)
        new = new.strip().strip('"') or state["question"]
        return {"search_query": new, "rewrites": state.get("rewrites", 0) + 1,
                "trace": _log(state, "rewrite_query", t0, new_query=new)}

    def generate(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        hits = state.get("docs") or []
        if not hits:
            ans = P.Answer(answerable=False, answer=P.REFUSAL, citations=[])
        else:
            msgs = P.generate_messages(state["question"], hits, state.get("history"), state.get("feedback"))
            try:
                ans = self.llm.structured(msgs, P.Answer, role="generator", max_tokens=2048)
            except Exception as err:
                ans = P.Answer(answerable=False, answer=f"Generation failed: {err}", citations=[])
            ans.citations = sorted({c for c in ans.citations if 1 <= c <= len(hits)})
        return {"answer": ans, "regenerations": state.get("regenerations", -1) + 1,
                "trace": _log(state, "generate", t0, answerable=ans.answerable, citations=ans.citations)}

    def check_groundedness(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        ans = state["answer"]
        if not ans.answerable or not state.get("docs"):
            return {"grounded": True, "feedback": None, "trace": _log(state, "check_groundedness", t0, skipped=True)}
        try:
            g = self.llm.structured(P.groundedness_messages(ans.answer, state["docs"]), P.Groundedness, role="utility")
        except Exception as err:
            return {"grounded": None, "trace": _log(state, "check_groundedness", t0, error=str(err)[:200])}
        feedback = None if g.grounded else "Unsupported claims: " + "; ".join(g.unsupported_claims[:5])
        return {"grounded": g.grounded, "feedback": feedback,
                "trace": _log(state, "check_groundedness", t0, grounded=g.grounded, unsupported=g.unsupported_claims[:5])}

    # ------------------------------------------------------------------ edges
    def _after_grading(self, state: RAGState) -> str:
        if state.get("docs"):
            return "generate"
        if state.get("rewrites", 0) < self.cfg.pipeline.max_rewrites:
            return "rewrite_query"
        return "generate"  # out of budget -> generate() refuses with no docs

    def _after_check(self, state: RAGState) -> str:
        if state.get("grounded") is False and state.get("regenerations", 0) < 1:
            return "generate"
        return END

    def _build(self):
        g = StateGraph(RAGState)
        g.add_node("retrieve", self.retrieve)
        g.add_node("generate", self.generate)
        g.add_edge(START, "retrieve")
        if self.cfg.pipeline.grade_documents:
            g.add_node("grade_documents", self.grade_documents)
            g.add_node("rewrite_query", self.rewrite_query)
            g.add_edge("retrieve", "grade_documents")
            g.add_conditional_edges("grade_documents", self._after_grading, ["generate", "rewrite_query"])
            g.add_edge("rewrite_query", "retrieve")
        else:
            g.add_edge("retrieve", "generate")
        if self.cfg.pipeline.groundedness_check:
            g.add_node("check_groundedness", self.check_groundedness)
            g.add_edge("generate", "check_groundedness")
            g.add_conditional_edges("check_groundedness", self._after_check, ["generate", END])
        else:
            g.add_edge("generate", END)
        return g.compile()

    # ------------------------------------------------------------------ entry
    def answer(self, question: str, history: list[dict] | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        init: RAGState = {"question": question, "history": history or [], "trace": []}
        if self.graph is None:
            state = {**init, **self.retrieve(init)}
            state = {**state, **self.generate(state)}
        else:
            state = self.graph.invoke(init, {"recursion_limit": 25})
        ans: P.Answer = state["answer"]
        docs: list[Hit] = state.get("docs") or []
        return {
            "question": question,
            "answer": ans.answer,
            "answerable": ans.answerable,
            "citations": ans.citations,
            "cited": [docs[i - 1] for i in ans.citations if i <= len(docs)],
            "context": docs,
            "retrieval": state["retrieval"],  # last retrieval (for retrieval metrics)
            "first_retrieval": state["retrievals"][0] if state.get("retrievals") else state["retrieval"],
            "grounded": state.get("grounded"),
            "rewrites": state.get("rewrites", 0),
            "trace": state.get("trace", []),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
