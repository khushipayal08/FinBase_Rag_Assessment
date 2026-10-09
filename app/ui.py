"""Streamlit chat UI — standalone, no FastAPI needed."""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# ⚠️ set_page_config MUST be the first Streamlit command
st.set_page_config(page_title="FinBase RAG Assistant", page_icon="💰", layout="wide")

# Everything else imports AFTER set_page_config
import logging

from src.embed import ensure_index
from src.generate import generate_answer
from src.guardrails import ABSTENTION, confidence_ok, groundedness_check
from src.query_rewrite import rewrite
from src.retrieve import retrieve

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ui")


@st.cache_resource(show_spinner="Preparing knowledge base (first run only)…")
def _prepare_index() -> int:
    """Build Chroma index once, cached across reruns and sessions."""
    return ensure_index()


# Build/load index (cached)
_count = _prepare_index()
if _count == 0:
    st.error("Failed to load knowledge base. Check app logs.")
    st.stop()


# ---- Page content ----
st.title("FinBase Customer Support Assistant")
st.caption("RAG · Hybrid retrieval · Grounded answers with citations")

# Session state
if "conv_id" not in st.session_state:
    st.session_state.conv_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Session")
    st.code(f"conversation_id:\n{st.session_state.conv_id}", language=None)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.conv_id = str(uuid.uuid4())
        st.rerun()
    st.divider()
    st.markdown("**Demo questions — try these**")
    demo_qs = [
        "What is the FD interest rate for a 1-year deposit?",
        "And what is it for senior citizens?",
        "What happens if I prematurely withdraw my FD?",
        "What is the minimum SIP investment?",
        "Does FinBase offer cryptocurrency investment?",
        "What is the Section 21 Master FD Interest Rate Matrix?",
    ]
    for q in demo_qs:
        if st.button(q, key=f"demo_{q[:25]}"):
            st.session_state.pending = q


def _run_rag(question: str) -> dict:
    t0 = time.perf_counter()
    standalone = rewrite(question, st.session_state.history)
    log.info("q=%r rewritten=%r", question, standalone)

    chunks = retrieve(standalone, mode="hybrid_rerank")
    passes, conf = confidence_ok(chunks)

    if not passes:
        return {
            "answer": ABSTENTION,
            "sources": [],
            "confidence": round(conf, 4),
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }

    try:
        answer = generate_answer(standalone, chunks)
        g = groundedness_check(answer, chunks)
        if not g["passed"]:
            log.warning("groundedness failed: %s", g)
            return {
                "answer": ABSTENTION,
                "sources": [],
                "confidence": round(conf, 4),
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            }
        sources = [
            {
                "chunk_id": c.chunk_id,
                "section_number": c.metadata.get("section_number", ""),
                "section_title": c.metadata.get("section_title", ""),
                "chunk_type": c.metadata.get("chunk_type", ""),
                "source_page": c.metadata.get("source_page", 0),
                "faq_ids": c.metadata.get("faq_ids", ""),
                "text": c.text,
                "rerank_score": round(c.rerank_score, 4),
            }
            for c in chunks
        ]
        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(conf, 4),
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }
    except Exception as e:
        log.exception("generation failed")
        return {
            "answer": f"⚠️ Error: {e}",
            "sources": [],
            "confidence": round(conf, 4),
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }


# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(
                f"📎 Sources ({len(msg['sources'])}) · confidence {msg.get('confidence', 0):.2f}"
            ):
                for s in msg["sources"]:
                    st.markdown(
                        f"**Section {s['section_number']}** · {s['section_title']}  \n"
                        f"`{s['chunk_type']}` · page {s['source_page']} · "
                        f"rerank `{s['rerank_score']:.3f}`"
                    )
                    st.text_area(
                        "chunk", s["text"], height=100, disabled=True,
                        key=f"c_{s['chunk_id']}_{uuid.uuid4().hex[:4]}",
                    )

# Chat input
prompt = st.chat_input("Ask about FinBase products…")
if "pending" in st.session_state and st.session_state.pending:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            result = _run_rag(prompt)
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander(
                    f"📎 Sources ({len(result['sources'])}) · "
                    f"confidence {result['confidence']:.2f} · "
                    f"{result['latency_ms']} ms"
                ):
                    for s in result["sources"]:
                        st.markdown(
                            f"**Section {s['section_number']}** · {s['section_title']}  \n"
                            f"`{s['chunk_type']}` · page {s['source_page']} · "
                            f"rerank `{s['rerank_score']:.3f}`"
                        )
                        st.text_area(
                            "chunk", s["text"], height=100, disabled=True,
                            key=f"c_{s['chunk_id']}_{uuid.uuid4().hex[:4]}",
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "confidence": result["confidence"],
            })
            st.session_state.history.append({"role": "user", "content": prompt})
            st.session_state.history.append(
                {"role": "assistant", "content": result["answer"]}
            )