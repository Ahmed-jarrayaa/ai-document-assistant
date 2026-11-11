import threading

import numpy as np

from app.monitoring.telemetry import tracer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

_model = None
_model_lock = threading.Lock()


def get_model():
    """Load the model once and reuse it (thread-safe).

    The heavy `sentence_transformers` import (torch) is done lazily here,
    NOT at module import, so the web server starts and binds its port fast.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # re-check: another thread may have loaded it
                with tracer.start_as_current_span("embedding.load_model") as span:
                    span.set_attribute("embedding.model", MODEL_NAME)
                    from sentence_transformers import SentenceTransformer

                    _model = SentenceTransformer(MODEL_NAME)
    return _model


def warmup() -> None:
    """Load the model once, e.g. in a background thread at startup."""
    get_model()


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
