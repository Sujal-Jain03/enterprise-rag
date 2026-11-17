"""Smoke-test retrieval: embed a question, print top-k chunks from Qdrant.

Usage:
    python test_search.py "How does Kubernetes HPA scale pods?"
"""
import sys
import logfire

logfire.configure(service_name="search-test", send_to_logfire=False)

from app.services.retrieval import embeddings as emb
from app.services.retrieval import vector_store as vs

query = sys.argv[1] if len(sys.argv) > 1 else "How does Kubernetes HPA scale pods?"
client = vs.get_client()
qvec = emb.embed_query(query)
hits = vs.search(client, qvec, limit=5)
print(f"Query: {query}\n")
for h in hits:
    p = h.payload
    print(f"[{h.score:.4f}] {p['filename']} (chunk {p['chunk_index']}, {p['source_type']})")
    print(f"    {p['text'][:200]}...\n")
