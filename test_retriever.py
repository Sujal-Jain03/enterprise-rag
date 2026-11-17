"""Test the Phase 4 retriever: in-corpus (true + noisy) + out-of-corpus fallback.

Usage:
    python test_retriever.py
"""
import time

import logfire

logfire.configure(service_name="retriever-test", send_to_logfire=False)

from app.services.retrieval.retriever import retrieve

QUERIES = [
    "How does Kubernetes HPA scale pods?",   # true corpus — must find pods_autoscale
    "What is a neural network?",             # noisy corpus — must find the Neural Networks doc
    "What is the capital of France?",        # out of corpus — must hit the fallback (found=False)
]

for q in QUERIES:
    start = time.time()
    result = retrieve(q)
    elapsed = time.time() - start
    print(f"Query: {q}")
    print(f"  found={result['found']}  error={result['error']}  "
          f"top_vector_score={result['top_vector_score']}  ({elapsed:.2f}s)")
    for i, c in enumerate(result["chunks"], 1):
        rs = f"{c['rerank_score']:.4f}" if c["rerank_score"] is not None else "n/a"
        print(f"  #{i} [vec {c['vector_score']:.4f} rank {c['vector_rank']} | "
              f"rerank {rs}] {c['filename']} (chunk {c['chunk_index']}, {c['source_type']})")
        preview = c["text"][:120].replace("\n", " ")
        print(f"      {preview}...")
    print()
