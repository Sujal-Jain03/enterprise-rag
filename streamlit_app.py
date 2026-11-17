"""Streamlit chat UI for the enterprise RAG application.

Calls the FastAPI backend — run it first:
  uvicorn app.main:app --port 8000
then:
  streamlit run streamlit_app.py
"""

import os
import time

import httpx
import streamlit as st

# Inside Docker Compose the API is reachable as http://api:8000 (service name);
# locally it is http://localhost:8000. Override with RAG_API_URL.
API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
ASK_TIMEOUT = 120.0


def check_health() -> dict:
    """GET /health; returns {} when the API is unreachable."""
    try:
        r = httpx.get(f"{API_URL}/health", timeout=5.0)
        return r.json()
    except Exception:
        return {}


def ask(question: str) -> dict:
    """POST /ask; returns the response JSON (raises on HTTP failure)."""
    r = httpx.post(f"{API_URL}/ask", json={"question": question}, timeout=ASK_TIMEOUT)
    r.raise_for_status()
    return r.json()


def render_sources(sources: list) -> None:
    """One expander per cited source chunk, with the passage text."""
    for s in sources:
        label = f"{s['citation']} {s['filename']} (chunk {s['chunk_index']})"
        with st.expander(label):
            st.write(s["text"])


def main():
    st.set_page_config(page_title="Enterprise RAG", page_icon="📚")
    st.title("📚 Enterprise RAG")
    st.caption("Grounded answers from your documents — Kubernetes docs + CS papers.")

    health = check_health()
    with st.sidebar:
        st.header("System status")
        if health.get("qdrant_ok"):
            st.success("API connected · Qdrant OK")
            st.write("Collection:", ", ".join(health.get("collections", [])))
        else:
            st.error("API unreachable — start it first:\n`uvicorn app.main:app --port 8000`")
        st.divider()
        st.header("About this system")
        st.write("- **Corpus:** 3,899 chunks (63 files)")
        st.write("- **Embeddings:** BAAI/bge-base-en-v1.5 (local, 768-dim)")
        st.write("- **Retrieval:** Qdrant + FlashRank rerank, gate 0.6")
        st.write("- **Generation:** Groq openai/gpt-oss-120b")
        st.write("- **Guardrails:** regex + Llama Prompt Guard 2")
        st.write("- **Eval:** Recall@5 1.00 · Faithful 0.86 · Citations 1.00")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m["role"] == "assistant" and m.get("sources"):
                render_sources(m["sources"])
            if m["role"] == "assistant" and m.get("latency") is not None:
                st.caption(f"{m['latency']:.1f}s")

    question = st.chat_input("Ask about Kubernetes, neural networks, ...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                t0 = time.perf_counter()
                try:
                    result = ask(question)
                except Exception as e:
                    st.error(f"API request failed: {e}")
                    return
                latency = time.perf_counter() - t0
            answer = result.get("answer") or "No answer returned."
            st.write(answer)
            sources = result.get("sources") or []
            if sources:
                render_sources(sources)
            st.caption(f"{latency:.1f}s")
            if result.get("error"):
                st.caption(f"Note: {result['error']}")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "latency": latency,
            }
        )


if __name__ == "__main__":
    main()
