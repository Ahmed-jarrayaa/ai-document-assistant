import threading

import numpy as np

from app.monitoring.telemetry import tracer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

_model = None
_model_lock = threading.Lock()


def get_model():
    """Load the embedding model once and reuse it (thread-safe).

    Uses fastembed (ONNX runtime, quantized model) instead of
    sentence-transformers/torch: ~100 MB of RAM instead of ~700 MB, which
    lets the app run on small hosting instances (e.g. Render 512 MB).
    The lazy import keeps the web server starting fast.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # re-check: another thread may have loaded it
                with tracer.start_as_current_span("embedding.load_model") as span:
                    span.set_attribute("embedding.model", MODEL_NAME)
                    from fastembed import TextEmbedding

                    _model = TextEmbedding(MODEL_NAME)
    return _model


def warmup() -> None:
    """Load the model once, e.g. in a background thread at startup."""
    get_model()


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed many texts at once. Returns an array of shape (len(texts), 384)."""
    with tracer.start_as_current_span("embedding.encode") as span:
        span.set_attribute("embedding.model", MODEL_NAME)
        span.set_attribute("embedding.num_texts", len(texts))
        vectors = np.array(list(get_model().embed(texts, batch_size=32)))
        # Guarantee unit-norm vectors (cosine similarity), like before
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        span.set_attribute("embedding.dimension", int(vectors.shape[1]))
        return vectors


def embed_query(text: str) -> np.ndarray:
    """Embed a single text (used for questions). Shape: (384,)."""
    return embed_texts([text])[0]
