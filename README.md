# Enterprise-Grade RAG Application

A production-style Retrieval-Augmented Generation system built phase by phase:
document ingestion, chunking, vector retrieval with reranking, guardrailed
generation with citations, observability, and a hand-rolled evaluation harness —
served through a FastAPI backend and a Streamlit chat UI, fully containerized
with Docker Compose.

## Architecture

```
Streamlit UI (port 8501) → FastAPI API (port 8000) → generate()
  ├─ check_input — guardrails (regex + Llama Prompt Guard 2)
  ├─ retrieve — Qdrant vector search → FlashRank rerank (MIN_SCORE gate 0.6)
  └─ Groq LLM (openai/gpt-oss-120b) — grounded answer with [n] citations
```

Observability: Logfire Cloud traces (`generate` → `check_input` → `retrieve` → `Generate`).

## Stack

| Layer | Choice |
| --- | --- |
| Embeddings | BAAI/bge-base-en-v1.5, local, 768-dim (Gemini path available via `EMBEDDING_PROVIDER=gemini`) |
| Vector store | Qdrant (local, Docker), collection `enterprise_rag` |
| Reranker | FlashRank ms-marco-MiniLM-L-12-v2 |
| Generation | Groq `openai/gpt-oss-120b`, temperature 0.2, retry-if-uncited |
| Guardrails | Regex injection patterns + Llama Prompt Guard 2; citation validation + leak checks on output |
| API / UI | FastAPI + Uvicorn / Streamlit |
| Observability | Logfire Cloud |
| Evaluation | Hand-rolled harness (Recall@5 / MRR / nDCG@5 + Groq LLM judge) |

## Corpus

63 files → 3,899 chunks (RecursiveCharacterTextSplitter, 1500 chars / 200 overlap):
6 Kubernetes docs (`DATA/true_data/`) + 57 CS papers (`DATA/noisy_data/`).
One file skipped in ingestion: a fully image-based PDF that needs OCR.

## Evaluation

10-question gold set (8 in-corpus, 2 out-of-corpus abstention checks).
Full report: `eval_report.json`.

| Metric | Score |
| --- | --- |
| Recall@5 | 1.00 |
| MRR | 0.94 |
| nDCG@5 | 0.86 |
| Faithfulness (LLM judge) | 0.86 |
| Answers question | 1.00 |
| Citations valid | 1.00 |
| Abstention (gate / answer) | 1.00 / 1.00 |

Known gap, honestly reported: one answer added an unsourced detail
(`kubectl logs`) the judge flagged as unfaithful.

## Project structure

```
.
├── app/
│   ├── main.py               # FastAPI app: GET /health, POST /ask
│   ├── config.py             # Settings from .env
│   ├── telemetry.py          # Logfire setup (cloud if token, else local-only)
│   ├── ingestions/
│   │   ├── processor.py      # parse → chunk → save JSON
│   │   ├── chunking/splitter.py
│   │   └── loaders/          # pdf, html, text, office
│   └── services/
│       ├── retrieval/        # embeddings, vector_store, retriever
│       ├── generation/       # generator (Groq + citations)
│       ├── guardrails/       # input/output safety checks
│       └── evaluation/       # metrics, judge, eval set, runner
├── DATA/                     # source documents (not in git)
├── processed_data/           # chunked JSON (generated, not in git)
├── models/                   # FlashRank model (downloaded, not in git)
├── streamlit_app.py          # chat UI (calls the API over HTTP)
├── index_data.py             # embed + upsert (resume-safe)
├── run_evaluation.py         # eval runner → eval_report.json
├── test_*.py                 # per-phase smoke tests
├── Dockerfile
├── docker-compose.yml        # qdrant + api + ui (reuses qdrant_storage volume)
├── requirements.txt          # full dev environment
└── requirements-docker.txt   # runtime-only deps for the image
```

## Quickstart

### Docker (recommended)

```bash
docker compose up --build   # first run builds the image (~4 min)
docker compose up           # later runs just start the stack
```

- Chat UI: http://localhost:8501
- API docs: http://localhost:8000/docs
- Qdrant dashboard: http://localhost:6333/dashboard

Stop with `docker compose down` (the Qdrant volume is kept, so the index survives).

### Local

```bash
uvicorn app.main:app --port 8000
streamlit run streamlit_app.py
```

## Configuration (.env)

| Variable | Required | Purpose |
| --- | --- | --- |
| GROQ_API_KEY | yes | Generation + Prompt Guard |
| GROQ_FALLBACK_API_KEY | no | Second Groq key (used if the first fails) |
| QDRANT_CLUSTER_ENDPOINT | yes | Qdrant URL (`http://localhost:6333` for local) |
| QDRANT_API_KEY | only for Qdrant Cloud | Cloud API key (ignored by local Qdrant) |
| EMBEDDING_PROVIDER | no | `local` (BGE, default) or `gemini` |
| GEMINI_API_KEY | only if `EMBEDDING_PROVIDER=gemini` | Gemini embeddings |
| LOGFIRE_TOKEN | no | Logfire Cloud write token; without it telemetry is local-only |

## Notes

- First query after startup takes ~30s (embedding + reranker models load);
  steady state is ~2–4s per query.
- Free-tier Groq rate limits inflate evaluation latency; they don't affect
  answer quality.
- The API has no auth — it is built for local development.
