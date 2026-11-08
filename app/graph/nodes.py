import os

from dotenv import load_dotenv
from opentelemetry import trace

from app.database.vector_store import similarity_search
from app.graph.state import GraphState
from app.monitoring.telemetry import traced
from app.services.embedding import embed_query
from app.services.llm import generate

load_dotenv()

RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.40"))
TOP_K = int(os.getenv("TOP_K", "4"))


@traced("node.analyze_question")
def analyze_question(state: GraphState) -> dict:
    """Clean the question. Raises an error if it is empty."""
    question = " ".join(state["question"].split())
    if not question:
        raise ValueError("The question is empty.")
    return {"question": question}


@traced("node.retrieve_documents")
def retrieve_documents(state: GraphState) -> dict:
    """Embed the question and fetch the top-k closest chunks from pgvector."""
    query_vector = embed_query(state["question"])
    documents = similarity_search(query_vector, k=TOP_K)
    trace.get_current_span().set_attribute("retrieval.num_results", len(documents))
    return {"retrieved_documents": documents}


@traced("node.evaluate_documents")
def evaluate_documents(state: GraphState) -> dict:
    """Score the retrieval and keep only chunks above the threshold."""
    documents = state["retrieved_documents"]
    relevant = [d for d in documents if d["similarity"] >= RELEVANCE_THRESHOLD]
    best_score = max((d["similarity"] for d in documents), default=0.0)

    span = trace.get_current_span()
    span.set_attribute("evaluation.relevance_score", best_score)
    span.set_attribute("evaluation.threshold", RELEVANCE_THRESHOLD)
    span.set_attribute("evaluation.kept_chunks", len(relevant))
    span.set_attribute("evaluation.is_relevant", best_score >= RELEVANCE_THRESHOLD)
    return {"retrieved_documents": relevant, "relevance_score": best_score}


@traced("node.generate_answer")
def generate_answer(state: GraphState) -> dict:
    """Generate an answer with the LLM from the relevant chunks, then append sources."""
    documents = state["retrieved_documents"]
    answer = generate(state["question"], documents)

    sources = sorted({(d["source"], d["page"]) for d in documents})
    sources_text = ", ".join(f"{src} p.{page}" for src, page in sources)
    return {"answer": f"{answer}\n\nSources: {sources_text}"}


@traced("node.fallback_answer")
def fallback_answer(state: GraphState) -> dict:
    """Answer used when the documents do not contain relevant information."""
    return {
        "answer": (
            "I could not find relevant information about this question "
            "in the provided documents."
        )
    }