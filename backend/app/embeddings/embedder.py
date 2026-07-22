"""
Wraps a sentence-transformers model so the rest of the app never has
to think about model loading, batching, or normalization.

Model runs fully locally and is free (no API key) -- swapping to a
bigger model (e.g. bge-large or e5-large) for a quality bump is a
one-line change in config.py, which is a good thing to call out as a
"retrieval accuracy" tuning knob.
"""
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Returns L2-normalized embeddings, shape (len(texts), dim)."""
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    )
    return vectors.astype("float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
