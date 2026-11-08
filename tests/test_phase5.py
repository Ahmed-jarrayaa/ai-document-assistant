import sys
from pathlib import Path

import ollama
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database.vector_store import count_chunks
from app.services import llm
from app.services.llm import LLMError, build_context, generate


def ollama_ready() -> bool:
    try:
        llm._client.show(llm.OLLAMA_MODEL)
        return True
    except Exception:
        return False


needs_ollama = pytest.mark.skipif(
    not ollama_ready(), reason="Ollama not running or model not pulled"
)
needs_pdf = pytest.mark.skipif(
    count_chunks("IntroML_Azencott.pdf") == 0, reason="Ingest the PDF first"
)


# ---------- no LLM needed ----------

def test_build_context_contains_source_and_page():
    docs = [{"source": "a.pdf", "page": 7, "content": "hello world"}]
    context = build_context(docs)
    assert "a.pdf" in context and "page 7" in context and "hello world" in context


def test_unreachable_ollama_raises_llm_error(monkeypatch):
    dead_client = ollama.Client(host="http://localhost:1", timeout=2)
    monkeypatch.setattr(llm, "_client", dead_client)
    with pytest.raises(LLMError):
        generate("question", [{"source": "a.pdf", "page": 1, "content": "text"}])


# ---------- need Ollama ----------

@needs_ollama
def test_answer_is_grounded_in_the_context():
    """The fact is invented, so the model can only know it from the context."""
    docs = [{
        "source": "fake.pdf", "page": 3,
        "content": "Le fichier secret du projet Aurora s'appelle Zorglub-42.",
    }]
    answer = generate("Comment s'appelle le fichier secret du projet Aurora ?", docs)
    assert "Zorglub" in answer


@needs_ollama
@needs_pdf
def test_full_graph_returns_llm_answer_with_sources():
    from app.graph.workflow import ask

    final = ask("Comment fonctionne une forêt aléatoire ?")
    assert "Placeholder" not in final["answer"]
    assert "Sources:" in final["answer"]
    assert "IntroML_Azencott.pdf" in final["answer"]