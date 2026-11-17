"""Retrieval service: dense search + FlashRank reranking + fallbacks.

Pipeline: embed query -> fetch CANDIDATES from Qdrant -> drop anything below
MIN_SCORE (no relevant context) -> rerank the rest with FlashRank -> top_k.
"""
from pathlib import Path

import logfire
from flashrank import Ranker, RerankRequest

from app.services.retrieval import embeddings as emb
from app.services.retrieval import vector_store as vs


CANDIDATES = 20
TOP_K = 5
# Eval run 2 (16 Nov 2026): in-corpus top scores 0.735-0.857, out-of-corpus
# 0.412/0.503 — 0.6 sits between them with margin on both sides.
MIN_SCORE = 0.6
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
_CACHE_DIR = Path(__file__).resolve().parents[3] / "models" / "flashrank"

_ranker = None
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = vs.get_client()
    return _client


def _get_ranker():
    """Lazy-load FlashRank (downloads the ~22MB model on first use)."""
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name=RERANK_MODEL, cache_dir=str(_CACHE_DIR))
    return _ranker


def _to_chunk(hit, vector_rank, rerank_score=None):
    p = hit.payload
    return {
        "text": p["text"],
        "filename": p["filename"],
        "chunk_index": p["chunk_index"],
        "source_type": p["source_type"],
        "vector_score": float(hit.score),
        "vector_rank": vector_rank,
        "rerank_score": rerank_score,
    }

@logfire.instrument("retrieve")
def retrieve(query: str, top_k: int = TOP_K) -> dict:
    """Retrieve the top_k most relevant chunks for a query.

    Returns {"chunks": [...], "found": bool, "error": str | None,
             "top_vector_score": float | None}.
    found=False means no chunk passed MIN_SCORE — the caller should say
    "I don't know" instead of answering from nothing.
    """
    if not query or not query.strip():
        return {"chunks": [], "found": False, "error": "empty query", "top_vector_score": None}

    try:
        qvec = emb.embed_query(query)
    except Exception as e:
        logfire.error("Query embedding failed: {error}", error=str(e))
        return {"chunks": [], "found": False, "error": f"embedding failed: {e}", "top_vector_score": None}

    try:
        with logfire.span("Vector search", candidates=CANDIDATES):
            hits = vs.search(_get_client(), qvec, limit=CANDIDATES)
    except Exception as e:
        logfire.error("Qdrant search failed: {error}", error=str(e))
        return {"chunks": [], "found": False, "error": f"qdrant unavailable: {e}", "top_vector_score": None}

    if not hits:
        return {"chunks": [], "found": False, "error": None, "top_vector_score": None}

    top_vector_score = float(hits[0].score)
    # Relevance gate: below MIN_SCORE, nothing is worth answering from.
    # (A borderline chunk below the gate never gets reranked — accepted
    # tradeoff; Phase 8 evaluation can revisit the threshold.)
    passing = [(i, h) for i, h in enumerate(hits) if h.score >= MIN_SCORE]
    if not passing:
        logfire.info("No chunk passed MIN_SCORE.", top_score=top_vector_score)
        return {"chunks": [], "found": False, "error": None, "top_vector_score": top_vector_score}

    chunks = [_to_chunk(h, i + 1) for i, h in passing]

    try:
        ranker = _get_ranker()
        request = RerankRequest(
            query=query,
            passages=[{"id": i, "text": c["text"]} for i, c in enumerate(chunks)],
        )
        with logfire.span("Rerank", num_passages=len(chunks)):
            ranked = ranker.rerank(request)
        reranked = []
        for r in ranked:
            c = dict(chunks[r["id"]])
            c["rerank_score"] = float(r["score"])
            reranked.append(c)
        chunks = reranked
    except Exception as e:
        # Reranker down (model missing, ONNX error): degrade to vector order.
        logfire.warning("Rerank failed, using vector order: {error}", error=str(e))

    return {
        "chunks": chunks[:top_k],
        "found": True,
        "error": None,
        "top_vector_score": top_vector_score,
    }
