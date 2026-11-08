import os
import httpx
import ollama
from dotenv import load_dotenv
from app.monitoring.telemetry import tracer

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Generous timeout: the first call loads the model into memory (slow on CPU)
_client = ollama.Client(host=OLLAMA_HOST, timeout=300)

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


def generate(question: str, documents: list[dict]) -> str:
    """Ask the LLM to answer `question` from `documents`."""
    context = build_context(documents)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context passages:\n\n{context}\n\nQuestion: {question}",
        },
    ]

    with tracer.start_as_current_span("llm.generate") as span:
        span.set_attribute("gen_ai.system", "ollama")
        span.set_attribute("gen_ai.request.model", OLLAMA_MODEL)
        span.set_attribute("llm.context_chars", len(context))
        span.set_attribute("llm.context_chunks", len(documents))
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
        if input_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        if output_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

        answer = response["message"]["content"].strip()
        span.set_attribute("llm.answer_chars", len(answer))
        return answer
def is_ready() -> bool:
    """True if Ollama answers and the configured model is available."""
    try:
        _client.show(OLLAMA_MODEL)
        return True
    except Exception:
        return False