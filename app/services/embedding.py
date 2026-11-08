from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.monitoring.telemetry import tracer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once and reuse it. Runs (and is traced) only the first time."""
    with tracer.start_as_current_span("embedding.load_model") as span:
        span.set_attribute("embedding.model", MODEL_NAME)
        return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed many texts at once. Returns an array of shape (len(texts), 384)."""
    with tracer.start_as_current_span("embedding.encode") as span:
        span.set_attribute("embedding.model", MODEL_NAME)
        span.set_attribute("embedding.num_texts", len(texts))
        vectors = get_model().encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        span.set_attribute("embedding.dimension", int(vectors.shape[1]))
        return vectors


def embed_query(text: str) -> np.ndarray:
    """Embed a single text (used for questions). Shape: (384,)."""
    return embed_texts([text])[0]