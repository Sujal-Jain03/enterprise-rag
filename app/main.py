"""FastAPI backend for the enterprise RAG application.

Endpoints:
- GET  /health — liveness + Qdrant connectivity check
- POST /ask    — ask a question, get a grounded, cited answer

Run: uvicorn app.main:app --port 8000
Docs: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Optional

import logfire
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.services.generation.generator import generate
from app.services.retrieval import vector_store as vs
from app.telemetry import setup_telemetry


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    citation: str
    filename: str
    chunk_index: int
    source_type: str
    text: str


class AskResponse(BaseModel):
    answer: Optional[str]
    sources: list[Source]
    found: bool
    error: Optional[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_telemetry()
    try:
        logfire.instrument_fastapi(app)
    except Exception as e:  # noqa: BLE001 - instrumentation must never kill the app
        logfire.warning("FastAPI instrumentation failed: {error}", error=str(e))
    yield


app = FastAPI(title="Enterprise RAG", lifespan=lifespan)


@app.get("/health")
def health():
    """Liveness + Qdrant connectivity. Always 200; qdrant_ok flags the dependency."""
    try:
        client = vs.get_client()
        names = [c.name for c in client.get_collections().collections]
        return {"status": "ok", "qdrant_ok": True, "collections": names}
    except Exception as e:  # noqa: BLE001 - health must report, not crash
        return {"status": "ok", "qdrant_ok": False, "error": str(e)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Ask a question; returns the grounded answer + cited sources."""
    result = generate(req.question)
    return AskResponse(
        answer=result.get("answer"),
        sources=result.get("sources") or [],
        found=result.get("found", False),
        error=result.get("error"),
    )
