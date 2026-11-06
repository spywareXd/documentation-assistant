"""Streamlit UI.  Run:  streamlit run src/docrag/app.py   (optionally: -- --config configs/experiments/e5_crag.yaml)"""
from __future__ import annotations

import argparse
import re
import sys

import streamlit as st

from docrag.config import load_config
from docrag.graph.rag_graph import RAGPipeline

st.set_page_config(page_title="DocRAG: Python docs assistant", layout="wide")


@st.cache_resource(show_spinner="Loading models and index...")
def get_pipeline(config_path: str | None) -> RAGPipeline:
    cfg = load_config(config_path)
    pipe = RAGPipeline(cfg)
    pipe.retriever.warmup()
    return pipe


def _config_arg() -> str | None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args, _ = ap.parse_known_args(sys.argv[1:])
    return args.config


def _link_citations(answer: str, cited_urls: dict[int, str]) -> str:
    return re.sub(r"\[(\d+)\]", lambda m: f"[[{m.group(1)}]]({cited_urls[int(m.group(1))]})"
                  if int(m.group(1)) in cited_urls else m.group(0), answer)


def _render(result: dict) -> None:
    urls = {i: h.chunk.url for i, h in enumerate(result["context"], start=1)}
    st.markdown(_link_citations(result["answer"], urls))
    if result["citations"]:
        with st.expander(f"Sources ({len(result['citations'])})"):
            for i in result["citations"]:
                c = result["context"][i - 1].chunk
                crumb = " > ".join([c.title, *[h for h in c.headings if h != c.title]])
                st.markdown(f"**[{i}]** `{c.source}` [{crumb}]({c.url})")
    with st.expander("How this answer was produced"):
        st.caption(f"total {result['latency_ms']:.0f} ms, query rewrites: {result['rewrites']}, "
                   f"grounded: {result['grounded']}")
        for step in result["trace"]:
            info = {k: v for k, v in step.items() if k not in ("node", "ms")}
            st.markdown(f"- **{step['node']}** ({step['ms']:.0f} ms) `{info}`")
        r = result["retrieval"]
        st.markdown("**Retrieved (after rerank)**")
        st.dataframe([{"#": i, "score": round(h.score, 3), "source": h.chunk.source, "title": h.chunk.title,
                       "section": " > ".join(h.chunk.headings[1:]), "sent to LLM": h in r.selected}
                      for i, h in enumerate(r.ranked[:10], start=1)], hide_index=True, use_container_width=True)


pipe = get_pipeline(_config_arg())
cfg = pipe.cfg

with st.sidebar:
    st.header("DocRAG")
    st.caption("FastAPI · Pydantic · Qdrant · LangChain · LangGraph")
    st.markdown(
        f"- **Retrieval:** {cfg.retrieval.mode}"
        f"{' + ' + cfg.retrieval.fusion.upper() if cfg.retrieval.mode == 'hybrid' else ''}"
        f"{' + rerank' if cfg.retrieval.rerank.enabled else ''}\n"
        f"- **Chunking:** {cfg.chunking.strategy} ({cfg.chunking.chunk_size})\n"
        f"- **k:** {cfg.retrieval.dynamic_k.policy if cfg.retrieval.dynamic_k.policy != 'fixed' else cfg.retrieval.top_k}\n"
        f"- **Pipeline:** {cfg.pipeline.mode}\n- **LLM:** {cfg.llm.generator} (Groq)"
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

st.session_state.setdefault("messages", [])
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        _render(msg["result"]) if msg.get("result") else st.markdown(msg["content"])

if prompt := st.chat_input("Ask about FastAPI, Pydantic, Qdrant, LangChain or LangGraph"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving, grading and answering..."):
                result = pipe.answer(prompt, history=history)
            _render(result)
            st.session_state.messages.append({"role": "assistant", "content": result["answer"], "result": result})
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)
