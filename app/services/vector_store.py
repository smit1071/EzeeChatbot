"""
vector_store.py — FAISS index builder and retriever.

Index type: IndexFlatIP (exact inner-product / cosine search).
  - No approximation errors — correctness > speed for typical KB sizes
  - For very large KBs (>500k chunks) swap to IndexIVFFlat for speed
  - Each bot has its own isolated index (multi-bot isolation guarantee)
"""

import faiss
import numpy as np
from typing import List, Tuple


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP from a (N, D) float32 embedding matrix.
    Vectors must already be L2-normalised (embedder.embed_texts guarantees this).
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve(
    index: faiss.IndexFlatIP,
    chunks: List[str],
    query_embedding: np.ndarray,
    top_k: int,
    score_threshold: float = 0.25,
) -> List[Tuple[str, float]]:
    """
    Query the index and return (chunk_text, score) pairs sorted by relevance.

    score_threshold filters out low-similarity chunks that could mislead the LLM.
    A score of 0.25 (cosine) is a conservative floor; chunks scoring below this
    are almost certainly irrelevant to the query.
    """
    k = min(top_k, index.ntotal)
    if k == 0:
        return []

    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if float(score) < score_threshold:
            continue
        results.append((chunks[idx], float(score)))

    return results
