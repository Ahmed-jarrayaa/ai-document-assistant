import sys
from pathlib import Path

import ollama
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database.vector_store import count_chunks
from app.graph.nodes import (
    RELEVANCE_THRESHOLD,
    analyze_question,
    evaluate_documents,
    fallback_answer,
)
from app.graph.workflow import graph, route_after_evaluation
from app.services import llm

PDF_SOURCE = "IntroML_Azencott.pdf"
needs_pdf = pytest.mark.skipif(
    count_chunks(PDF_SOURCE) == 0, reason="Ingest the PDF first (Phase 2)"
)


def ollama_ready() -> bool:
    try:
        llm._client.show(llm.OLLAMA_MODEL)
        return True
    except Exception:
        return False


needs_ollama = pytest.mark.skipif(
    not ollama_ready(), reason="Ollama not running or model not pulled"
)


# ---------- unit tests: no database, no model ----------

def test_analyze_question_cleans_whitespace():
    assert analyze_question({"question": "  hello   world \n"}) == {"question": "hello world"}


def test_analyze_question_rejects_empty():
    with pytest.raises(ValueError):
        analyze_question({"question": "   "})


def test_evaluate_documents_filters_and_scores():
    docs = [
        {"similarity": 0.70, "content": "a"},
        {"similarity": 0.50, "content": "b"},
        {"similarity": 0.20, "content": "c"},
    ]
    update = evaluate_documents({"retrieved_documents": docs})
    assert update["relevance_score"] == 0.70
    assert [d["content"] for d in update["retrieved_documents"]] == ["a", "b"]


def test_evaluate_documents_with_no_documents():
    update = evaluate_documents({"retrieved_documents": []})
    assert update == {"retrieved_documents": [], "relevance_score": 0.0}


def test_router_chooses_branch_from_score():
    assert route_after_evaluation({"relevance_score": RELEVANCE_THRESHOLD + 0.1}) == "generate_answer"
    assert route_after_evaluation({"relevance_score": RELEVANCE_THRESHOLD - 0.1}) == "fallback_answer"


def test_fallback_message_says_nothing_found():
    assert "could not find" in fallback_answer({})["answer"]


# ---------- integration tests: need Postgres + the ingested PDF ----------

@needs_ollama
@needs_pdf
def test_relevant_question_goes_to_generate_answer():
    steps = [list(s.keys())[0] for s in graph.stream(
        {"question": "Comment fonctionne une forêt aléatoire ?"}, stream_mode="updates")]
    assert steps == ["analyze_question", "retrieve_documents", "evaluate_documents", "generate_answer"]


@needs_pdf
def test_offtopic_question_goes_to_fallback():
    steps = [list(s.keys())[0] for s in graph.stream(
        {"question": "Meilleure recette de pizza à Naples"}, stream_mode="updates")]
    assert steps == ["analyze_question", "retrieve_documents", "evaluate_documents", "fallback_answer"]


@needs_ollama
@needs_pdf
def test_final_state_contains_all_fields():
    final = graph.invoke({"question": "Qu'est-ce que le surapprentissage ?"})
    assert set(final) >= {"question", "retrieved_documents", "relevance_score", "answer"}
    assert final["relevance_score"] >= RELEVANCE_THRESHOLD