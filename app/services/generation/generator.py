"""Answer generation: grounded answers with citations from retrieved chunks."""
import re

import logfire
from groq import Groq

from app.config import settings
from app.services.retrieval.retriever import retrieve
from app.services.guardrails.guards import check_input, check_output



SYSTEM_PROMPT = """You are a precise technical assistant. Answer the user's question
USING ONLY the context below. Every factual claim must end with a citation in
EXACTLY this format: a number in square brackets, like [1] or [2], referring to
the numbered context chunk it came from. Use ONLY this format — never any other
citation style (no footnotes, no links, no special characters). Example: "HPA
scales pods based on CPU utilization [1]." If the context does not contain the
answer, say "I don't know from the provided documents." Do not
use outside knowledge."""


TEMPERATURE = 0.2
MAX_TOKENS = 1024


def _get_clients():
    """Primary + fallback Groq clients (fallback skipped if its key is empty)."""
    clients = []
    if settings.GROQ_API_KEY:
        clients.append(Groq(api_key=settings.GROQ_API_KEY))
    if settings.GROQ_FALLBACK_API_KEY:
        clients.append(Groq(api_key=settings.GROQ_FALLBACK_API_KEY))
    return clients


def _build_context(chunks):
    """Number the chunks for citation: [1] filename (chunk N) + text."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] Source: {c['filename']} (chunk {c['chunk_index']})\n{c['text']}")
    return "\n\n".join(parts)


def _call_llm(system, user):
    """Try each Groq key in turn; return the answer text or raise."""
    clients = _get_clients()
    if not clients:
        raise RuntimeError("No Groq API key configured in .env")
    last_error = None
    for client in clients:
        try:
            completion = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return completion.choices[0].message.content
        except Exception as e:
            last_error = e
            logfire.warning("Groq call failed, trying next key: {error}", error=str(e))
    raise RuntimeError(f"All Groq keys failed: {last_error}")

@logfire.instrument("generate")

def generate(query: str) -> dict:
    """Retrieve + generate a cited answer.

    Returns {"answer", "sources", "found", "error"}. found=False means no
    relevant context existed, so the answer is "I don't know" with no LLM call.
    """
    allowed, reason, _details = check_input(query)
    if not allowed:
        return {"answer": "I can't help with that request.", "sources": [], "found": False, "error": f"blocked: {reason}"}

    result = retrieve(query)
    if result["error"]:
        return {"answer": None, "sources": [], "found": False, "error": result["error"]}
    if not result["found"]:
        return {
            "answer": "I don't know from the provided documents.",
            "sources": [],
            "found": False,
            "error": None,
            "top_vector_score": result.get("top_vector_score"),
        }
    chunks = result["chunks"]
    context = _build_context(chunks)
    user_msg = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer with citations like [1], [2]."
    try:
        with logfire.span("Generate", model=settings.GROQ_MODEL, num_chunks=len(chunks)):
            answer = _call_llm(SYSTEM_PROMPT, user_msg)
    except Exception as e:
        logfire.error("Generation failed: {error}", error=str(e))
        return {"answer": None, "sources": [], "found": True, "error": f"generation failed: {e}"}
    answer, flags = check_output(answer, len(chunks))
    if flags.get("leaked"):
        logfire.warning("Output guard blocked response.")
        return {"answer": "I can't provide that response.", "sources": [], "found": True, "error": "blocked: output failed safety check"}
    if not re.search(r"\[\d+\]", answer):
        # One retry: the model sometimes answers with no citations at all
        # (eval run 2: 4/8 answers uncited). Ask once for the same answer
        # with citations; if the retry also fails, keep the original.
        logfire.info("Answer had no citations, retrying once with an explicit citation request.")
        retry_msg = (
            f"Context:\n{context}\n\nQuestion: {query}\n\n"
            "Your previous answer had no citations. Rewrite the same answer, "
            "adding citations like [1], [2] after each factual claim, "
            "referring to the numbered context chunks above."
        )
        try:
            with logfire.span("Generate-retry", model=settings.GROQ_MODEL, num_chunks=len(chunks)):
                retry_answer = _call_llm(SYSTEM_PROMPT, retry_msg)
            retry_answer, retry_flags = check_output(retry_answer, len(chunks))
            if not retry_flags.get("leaked"):
                answer = retry_answer
        except Exception as e:
            logfire.warning("Citation retry failed, keeping original answer: {error}", error=str(e))
    sources = [
        {"citation": f"[{i}]", "filename": c["filename"],
         "chunk_index": c["chunk_index"], "source_type": c["source_type"],
         "text": c["text"]}
        for i, c in enumerate(chunks, 1)
    ]
    return {"answer": answer, "sources": sources, "found": True, "error": None,
            "top_vector_score": result.get("top_vector_score")}
