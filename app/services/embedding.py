import os
import threading
from functools import lru_cache

import numpy as np

from app.monitoring.telemetry import tracer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# Embeddings provider:
#   "local"  -> fastembed (ONNX in-process, needs ~200 MB RAM)
#   "remote" -> HuggingFace Inference API (needs HF_TOKEN, ~0 MB RAM on the server)
#   "auto"   -> remote if HF_TOKEN is set, else local
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "auto").lower()

_model = None
_model_lock = threading.Lock()


def _effective_provider() -> str:
    if EMBEDDINGS_PROVIDER in ("local", "remote"):
        return EMBEDDINGS_PROVIDER
    return "remote" if os.getenv("HF_TOKEN", "") else "local"


@lru_cache(maxsize=1)
def _get_hf_client():
    """Build the HuggingFace Inference client on first use (lazy import)."""
    from huggingface_hub import InferenceClient

    api_key = os.getenv("HF_TOKEN", "")
    if not api_key:
        raise RuntimeError(
            "HF_TOKEN is not set: cannot use remote embeddings. "
            "Create a free token at https://huggingface.co/settings/tokens "
            "or set EMBEDDINGS_PROVIDER=local."
        )
    return InferenceClient(api_key=api_key)


def get_model():
    """Load the LOCAL embedding model once and reuse it (thread-safe).

    The heavy import is done lazily here, NOT at module import, so the web
    server starts and binds its port fast. Only used when the provider is
    "local" (no HF_TOKEN configured).
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # re-check: another thread may have loaded it
                with tracer.start_as_current_span("embedding.load_model") as span:
                    span.set_attribute("embedding.model", MODEL_NAME)
                    from fastembed import TextEmbedding

                    # threads=1: on small instances (0.5 CPU / 512 MB), onnxruntime
                    # spawns one thread per core by default, which inflates memory.
                    _model = TextEmbedding(MODEL_NAME, threads=1)
    return _model


def warmup() -> None:
    """Load the model once, e.g. in a background thread at startup."""
    if _effective_provider() == "remote":
        # Remote: nothing to load locally; just check the API key is present.
        _get_hf_client()
        return
    get_model()


def _embed_texts_remote(texts: list[str]) -> np.ndarray:
    """Embed texts via the HuggingFace Inference API (one call per text)."""
    client = _get_hf_client()
    vectors = []
    for text in texts:
        result = client.feature_extraction(text, model=MODEL_NAME)
        array = np.asarray(result, dtype=np.float32)
        if array.ndim > 1:
            # Some models return token embeddings (tokens x dim): mean-pool
            array = array.mean(axis=0)
        if array.shape[-1] != EMBEDDING_DIM:
            raise RuntimeError(
                f"Unexpected embedding dimension {array.shape[-1]} "
                f"(expected {EMBEDDING_DIM}) from HuggingFace API."
            )
        vectors.append(array)
    return np.array(vectors)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed many texts at once. Returns an array of shape (len(texts), 384)."""
    with tracer.start_as_current_span("embedding.encode") as span:
        span.set_attribute("embedding.model", MODEL_NAME)
        span.set_attribute("embedding.num_texts", len(texts))
        span.set_attribute("embedding.provider", _effective_provider())
        if _effective_provider() == "remote":
            vectors = _embed_texts_remote(texts)
        else:
            vectors = np.array(list(get_model().embed(texts, batch_size=32)))
        # Guarantee unit-norm vectors (cosine similarity)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        span.set_attribute("embedding.dimension", int(vectors.shape[1]))
        return vectors


def embed_query(text: str) -> np.ndarray:
    """Embed a single text (used for questions). Shape: (384,)."""
    return embed_texts([text])[0]
