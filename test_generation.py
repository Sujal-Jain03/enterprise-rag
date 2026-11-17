"""Test Phase 5 generation: cited answers + the no-context fallback.

Usage:
    python test_generation.py
"""
import logfire

logfire.configure(service_name="generation-test", send_to_logfire=False)

from app.services.generation.generator import generate

QUERIES = [
    "How does Kubernetes HPA scale pods?",
    "What is a neural network?",
    "What is the capital of France?",
]

for q in QUERIES:
    result = generate(q)
    print(f"Question: {q}")
    print(f"  found={result['found']}  error={result['error']}")
    print(f"  Answer: {result['answer']}")
    for s in result["sources"]:
        print(f"    {s['citation']} {s['filename']} (chunk {s['chunk_index']}, {s['source_type']})")
    print()
