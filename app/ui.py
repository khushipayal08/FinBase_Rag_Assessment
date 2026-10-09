"""Streamlit chat UI. Run: streamlit run app/ui.py"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import uuid

import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(page_title="FinBase RAG Assistant", page_icon="💰", layout="wide")
st.title("FinBase Customer Support Assistant")
st.caption("RAG · Hybrid retrieval · Grounded answers with citations")

if "conv_id" not in st.session_state:
    st.session_state.conv_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Session")
    st.code(f"conversation_id:\n{st.session_state.conv_id}", language=None)
    if st.button("Clear conversation"):
        st.session_state.messages = []
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
                    if s.get("faq_ids"):
                        st.caption(f"FAQ IDs: {', '.join(s['faq_ids'])}")
                    st.text_area(
                        "chunk", s["text"], height=100, disabled=True,
                        key=f"c_{s['chunk_id']}_{uuid.uuid4().hex[:4]}",
                    )

prompt = st.chat_input("Ask about FinBase products…")
if "pending" in st.session_state and st.session_state.pending:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            try:
                r = requests.post(
                    f"{API}/ask",
                    json={
                        "question": prompt,
                        "conversation_id": st.session_state.conv_id,
                    },
                    timeout=60,
                ).json()
                answer = r.get("answer", "⚠️ Empty response")
                st.markdown(answer)
                if r.get("sources"):
                    with st.expander(
                        f"📎 Sources ({len(r['sources'])}) · "
                        f"confidence {r.get('confidence', 0):.2f} · "
                        f"{r.get('latency_ms', 0)} ms"
                    ):
                        for s in r["sources"]:
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
                    "content": answer,
                    "sources": r.get("sources", []),
                    "confidence": r.get("confidence", 0.0),
                })
            except Exception as e:
                st.error(f"Request failed: {e}")