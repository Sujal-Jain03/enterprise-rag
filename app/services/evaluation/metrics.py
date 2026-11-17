"""Retrieval metrics for the RAG evaluation harness.

Same definitions as the Auric retrieval engine eval:
- Recall@k: 1.0 if a relevant chunk is in the top-k, else 0.0
- MRR: 1 / rank of the first relevant chunk (0 if none in top-k)
- nDCG@k: discounted gain, normalised by the ideal ranking (all top-k relevant)

Relevance is binary: a chunk is relevant if its filename contains the
expected_source substring.
"""

import math


def _is_relevant(filename: str, expected_source: str) -> bool:
    return expected_source.lower() in filename.lower()


def recall_at_k(filenames, expected_source, k=5):
    """1.0 if any of the top-k retrieved chunks is from the expected source."""
    top_k = filenames[:k]
    return 1.0 if any(_is_relevant(f, expected_source) for f in top_k) else 0.0


def reciprocal_rank(filenames, expected_source, k=5):
    """1 / rank of the first relevant chunk; 0.0 if none in top-k."""
    for i, f in enumerate(filenames[:k]):
        if _is_relevant(f, expected_source):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(filenames, expected_source, k=5):
    """Binary-relevance nDCG@k. Ideal ranking = every top-k chunk relevant."""
    dcg = 0.0
    for i, f in enumerate(filenames[:k]):
        if _is_relevant(f, expected_source):
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg else 0.0
