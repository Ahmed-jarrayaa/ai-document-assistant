import os
from functools import lru_cache

import httpx
import ollama
import openai
from dotenv import load_dotenv
from app.monitoring.telemetry import tracer

load_dotenv()

# Provider switch: "ollama" (local server) or "groq" (cloud API).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# --- Ollama (local) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# --- Groq (cloud, OpenAI-compatible API) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Model of the ACTIVE provider: used by /health and tracing attributes.
LLM_MODEL = GROQ_MODEL if LLM_PROVIDER == "groq" else OLLAMA_MODEL

# Generous timeout: the first Ollama call loads the model into memory (slow on CPU)
_client = ollama.Client(host=OLLAMA_HOST, timeout=300)


@lru_cache(maxsize=1)
def _get_groq_client() -> openai.OpenAI:
    """Build the Groq client on first use (it requires a non-empty API key)."""
    return openai.OpenAI(
        api_key=os.getenv("GROQ_API_KEY", ""),
        base_url=GROQ_BASE_URL,
        timeout=300.0,
        max_retries=2,
    )

SYSTEM_PROMPT = """You are a document assistant.
Answer the question using ONLY the context passages provided by the user.

Rules:
- Do not use outside knowledge and do not invent facts.
- If the passages do not contain the answer, say clearly that the documents do not contain it.
- Answer in the same language as the question.
- Be concise and precise.
- When you use a passage, mention its page, for example (page 12)."""


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce an answer (server down, model missing...)."""


def build_context(documents: list[dict]) -> str:
    """Format retrieved chunks as labeled passages for the prompt."""
    parts = [
        f"[Source: {d['source']}, page {d['page']}]\n{d['content']}" for d in documents
    ]
    return "\n\n".join(parts)


def _generate_ollama(
    question: str, context: str
) -> tuple[str, int | None, int | None]:
    """Ask the local Ollama server to answer `question` from `context`."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context passages:\n\n{context}\n\nQuestion: {question}",
        },
    ]
    try:
        response = _client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0},
        )
    except (ConnectionError, httpx.TimeoutException) as exc:
        raise LLMError(
            f"Cannot reach Ollama at {OLLAMA_HOST}, or it took too long to answer. ({exc})"
        ) from exc
    except ollama.ResponseError as exc:
        raise LLMError(
            f"Ollama error with model '{OLLAMA_MODEL}': {exc}. "
            f"Did you run `ollama pull {OLLAMA_MODEL}`?"
        ) from exc

    # Ollama reports token counts: very useful to explain slow answers
    input_tokens = response.get("prompt_eval_count")
    output_tokens = response.get("eval_count")
    answer = response["message"]["content"].strip()
    return answer, input_tokens, output_tokens


def _generate_groq(
    question: str, context: str
) -> tuple[str, int | None, int | None]:
    """Ask the Groq API to answer `question` from `context` (OpenAI-compatible)."""
    if not GROQ_API_KEY:
        raise LLMError(
            "GROQ_API_KEY is not set. Add it to the environment "
            "(or .env) and restart the application."
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context passages:\n\n{context}\n\nQuestion: {question}",
        },
    ]
    try:
        response = _get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0,
        )
    except (openai.APIConnectionError, openai.APITimeoutError) as exc:
        raise LLMError(
            f"Cannot reach Groq at {GROQ_BASE_URL}, or it took too long to answer. ({exc})"
        ) from exc
    except openai.APIStatusError as exc:
        raise LLMError(
            f"Groq error with model '{GROQ_MODEL}': {exc}. "
            "Check the API key and the model name "
            "(https://console.groq.com/docs/models)."
        ) from exc
    except openai.OpenAIError as exc:
        raise LLMError(
            f"Groq client error: {exc}. Check GROQ_API_KEY and the model name."
        ) from exc

    # OpenAI-style usage counts
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else None
    output_tokens = usage.completion_tokens if usage else None
    answer = (response.choices[0].message.content or "").strip()
    return answer, input_tokens, output_tokens


def generate(question: str, documents: list[dict]) -> str:
    """Ask the configured provider to answer `question` from `documents`."""
    context = build_context(documents)

    with tracer.start_as_current_span("llm.generate") as span:
        span.set_attribute("gen_ai.system", LLM_PROVIDER)
        span.set_attribute("gen_ai.request.model", LLM_MODEL)
        span.set_attribute("llm.context_chars", len(context))
        span.set_attribute("llm.context_chunks", len(documents))
        if LLM_PROVIDER == "groq":
            answer, input_tokens, output_tokens = _generate_groq(question, context)
        else:
            answer, input_tokens, output_tokens = _generate_ollama(question, context)

        if input_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        if output_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

        span.set_attribute("llm.answer_chars", len(answer))
        return answer


def is_ready() -> bool:
    """True if the configured provider answers and the model is available."""
    try:
        if LLM_PROVIDER == "groq":
            if not os.getenv("GROQ_API_KEY", ""):
                return False
            _get_groq_client().models.retrieve(GROQ_MODEL)
        else:
            _client.show(OLLAMA_MODEL)
        return True
    except Exception:
        return False
