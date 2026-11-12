import os
import threading
import time
from functools import lru_cache

import httpx
import numpy as np

from app.monitoring.telemetry import tracer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # dimension of the LOCAL models (fastembed / HF MiniLM)

# Embeddings provider:
#   "local"  -> fastembed (ONNX in-process, ~384 dims)
#   "gemini" -> Google Gemini API (gemini-embedding-001, free tier, GEMINI_API_KEY)
#   "hf"     -> HuggingFace Inference API (paid credits since 2026, HF_TOKEN)
#   "auto"   -> gemini if GEMINI_API_KEY is set, else hf if HF_TOKEN, else local
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "auto").lower()

# --- Gemini (cloud) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_EMBEDDING_DIM = int(os.getenv("GEMINI_EMBEDDING_DIM", "768"))
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"

_model = None
_model_lock = threading.Lock()


def _effective_provider() -> str:
    if EMBEDDINGS_PROVIDER in ("local", "gemini", "hf"):
        return EMBEDDINGS_PROVIDER
    if os.getenv("GEMINI_API_KEY", ""):
        return "gemini"
    if os.getenv("HF_TOKEN", ""):
        return "hf"
    return "local"


def get_target_dim() -> int:
    """Embedding dimension of the ACTIVE provider (used by the DB schema)."""
    if _effective_provider() == "gemini":
        return GEMINI_EMBEDDING_DIM
    return EMBEDDING_DIM


@lru_cache(maxsize=1)
def _get_hf_client():
    """Build the HuggingFace Inference client on first use (lazy import)."""
    from huggingface_hub import InferenceClient

    api_key = os.getenv("HF_TOKEN", "")
    if not api_key:
        raise RuntimeError(
            "HF_TOKEN is not set: cannot use HF embeddings. "
            "Set EMBEDDINGS_PROVIDER=local or use Gemini instead."
        )
    return InferenceClient(api_key=api_key)


def get_model():
    """Load the LOCAL embedding model once and reuse it (thread-safe).

    The heavy import is done lazily here, NOT at module import, so the web
    server starts and binds its port fast. Only used when the provider is
    "local" (no cloud API key configured).
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
    """Validate the configured provider, e.g. in a background thread at startup."""
    provider = _effective_provider()
    if provider == "gemini" and not os.getenv("GEMINI_API_KEY", ""):
        print(
            "[startup] EMBEDDINGS_PROVIDER=gemini but GEMINI_API_KEY is not set: "
            "embedding calls will fail. Get a free key at "
            "https://aistudio.google.com/apikey",
            flush=True,
        )
        return
    if provider == "local":
        get_model()
    # "hf"/"gemini": nothing to load in memory; the API is called per request.


def _gemini_embed(texts: list[str], task_type: str) -> np.ndarray:
    """Embed texts with the Gemini API (batchEmbedContents, max 100 per call)."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set: cannot use Gemini embeddings. "
            "Get a free key at https://aistudio.google.com/apikey "
            "or set EMBEDDINGS_PROVIDER=local."
        )
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 100):
        batch = texts[start : start + 100]
        body = {
            "requests": [
                {
                    "model": f"models/{GEMINI_EMBEDDING_MODEL}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": GEMINI_EMBEDDING_DIM,
                }
                for text in batch
            ]
        }
        url = f"{_GEMINI_URL}/{GEMINI_EMBEDDING_MODEL}:batchEmbedContents"
        response = httpx.post(url, headers=headers, json=body, timeout=120)
        if response.status_code == 429:
            raise RuntimeError(
                "Gemini rate limit reached (100 embeddings/minute on the free "
                "tier). Wait a minute and re-upload the document."
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini embedding failed ({response.status_code}): {response.text[:300]}"
            )
        vectors.extend(e["values"] for e in response.json()["embeddings"])
        if start + 100 < len(texts):
            time.sleep(1)  # gentle pacing for the free tier
    return np.array(vectors, dtype=np.float32)


def _hf_embed(texts: list[str]) -> np.ndarray:
    """Embed texts via the HuggingFace Inference API (paid credits since 2026)."""
    client = _get_hf_client()
    vectors = []
    for text in texts:
        result = client.feature_extraction(text, model=MODEL_NAME)
        array = np.asarray(result, dtype=np.float32)
        if array.ndim > 1:
            array = array.mean(axis=0)  # token embeddings -> mean-pool
        vectors.append(array)
    return np.array(vectors, dtype=np.float32)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed many texts at once. Returns an array of shape (len(texts), dim)."""
    provider = _effective_provider()
    with tracer.start_as_current_span("embedding.encode") as span:
        span.set_attribute("embedding.provider", provider)
        span.set_attribute("embedding.num_texts", len(texts))
        if provider == "gemini":
            vectors = _gemini_embed(texts, "RETRIEVAL_DOCUMENT")
            expected = GEMINI_EMBEDDING_DIM
        elif provider == "hf":
            vectors = _hf_embed(texts)
            expected = EMBEDDING_DIM
        else:
            vectors = np.array(list(get_model().embed(texts, batch_size=32)))
            expected = EMBEDDING_DIM
        if vectors.shape[1] != expected:
            raise RuntimeError(
                f"Unexpected embedding dimension {vectors.shape[1]} "
                f"(expected {expected}) from provider '{provider}'."
            )
        span.set_attribute("embedding.dimension", int(vectors.shape[1]))
        return _normalize(vectors)


def embed_query(text: str) -> np.ndarray:
    """Embed a single question. Shape: (dim,)."""
    if _effective_provider() == "gemini":
        return _normalize(_gemini_embed([text], "RETRIEVAL_QUERY"))[0]
    return embed_texts([text])[0]
