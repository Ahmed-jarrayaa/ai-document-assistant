import sys
from pathlib import Path

import ollama
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database.vector_store import count_chunks
from app.graph.nodes import evaluate_documents
from app.monitoring.telemetry import traced
from app.services import llm
from app.services.embedding import embed_texts
from app.services.llm import LLMError, generate

PDF_SOURCE = "IntroML_Azencott.pdf"


def ollama_ready() -> bool:
    try:
        llm._client.show(llm.OLLAMA_MODEL)
        return True
    except Exception:
        return False


needs_pdf = pytest.mark.skipif(count_chunks(PDF_SOURCE) == 0, reason="Ingest the PDF first")
needs_ollama = pytest.mark.skipif(not ollama_ready(), reason="Ollama not ready")


@pytest.fixture(scope="module")
def exporter():
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    mem = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(mem))
    return mem


@pytest.fixture(autouse=True)
def clear_spans(exporter):
    exporter.clear()


def test_traced_decorator_records_errors(exporter):
    @traced("test.boom")
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError):
        boom()

    span = exporter.get_finished_spans()[0]
    assert span.name == "test.boom"
    assert span.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)


def test_evaluate_node_span_has_attributes(exporter):
    docs = [{"similarity": 0.70, "content": "a"}, {"similarity": 0.20, "content": "b"}]
    evaluate_documents({"retrieved_documents": docs})

    span = exporter.get_finished_spans()[0]
    assert span.name == "node.evaluate_documents"
    assert span.attributes["evaluation.relevance_score"] == 0.70
    assert span.attributes["evaluation.kept_chunks"] == 1
    assert span.attributes["evaluation.is_relevant"] is True


def test_embedding_span_has_attributes(exporter):
    embed_texts(["a", "b"])
    span = next(s for s in exporter.get_finished_spans() if s.name == "embedding.encode")
    assert span.attributes["embedding.num_texts"] == 2
    assert span.attributes["embedding.dimension"] == 384


def test_llm_failure_is_recorded_on_span(exporter, monkeypatch):
    monkeypatch.setattr(llm, "_client", ollama.Client(host="http://localhost:1", timeout=2))
    with pytest.raises(LLMError):
        generate("question", [{"source": "a.pdf", "page": 1, "content": "text"}])

    span = next(s for s in exporter.get_finished_spans() if s.name == "llm.generate")
    assert span.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)


@needs_pdf
def test_offtopic_question_produces_one_trace_tree(exporter):
    from app.graph.workflow import ask

    ask("Meilleure recette de pizza à Naples")
    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}

    assert {
        "rag.ask", "node.analyze_question", "node.retrieve_documents",
        "embedding.encode", "pgvector.similarity_search",
        "node.evaluate_documents", "node.fallback_answer",
    } <= names
    assert "node.generate_answer" not in names and "llm.generate" not in names

    # one single trace, one root, every other span has a parent in the same trace
    assert len({s.context.trace_id for s in spans}) == 1
    ids = {s.context.span_id for s in spans}
    roots = [s for s in spans if s.parent is None]
    assert len(roots) == 1 and roots[0].name == "rag.ask"
    assert all(s.parent.span_id in ids for s in spans if s.parent is not None)


@needs_pdf
@needs_ollama
def test_relevant_question_trace_has_llm_usage(exporter):
    from app.graph.workflow import ask

    ask("Qu'est-ce que le surapprentissage ?")
    span = next(s for s in exporter.get_finished_spans() if s.name == "llm.generate")
    assert span.attributes["gen_ai.request.model"] == llm.OLLAMA_MODEL
    assert span.attributes.get("gen_ai.usage.output_tokens", 0) > 0