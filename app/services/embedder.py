"""
embedder.py — Thin wrapper around sentence-transformers.

Uses all-MiniLM-L6-v2:
  - 384-dimensional embeddings
  - ~22 MB model, runs on CPU in <100 ms per batch
  - Excellent retrieval quality for English text
  - No Groq / OpenAI API key needed — completely free

The model is loaded once at module import and reused across all requests.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from app.core.config import settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns a float32 numpy array of shape (len(texts), 384).
    Vectors are L2-normalised so cosine similarity == dot product.
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,   # L2 norm → dot product == cosine
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Single-vector convenience wrapper. Returns shape (1, 384)."""
    return embed_texts([query])
