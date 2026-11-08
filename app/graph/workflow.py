from langgraph.graph import END, START, StateGraph
from app.monitoring.telemetry import tracer
from app.graph.nodes import (
    RELEVANCE_THRESHOLD,
    analyze_question,
    evaluate_documents,
    fallback_answer,
    generate_answer,
    retrieve_documents,
)
from app.graph.state import GraphState


def route_after_evaluation(state: GraphState) -> str:
    """Conditional edge: returns the NAME of the next node."""
    if state["relevance_score"] >= RELEVANCE_THRESHOLD:
        return "generate_answer"
    return "fallback_answer"


def build_graph():
    builder = StateGraph(GraphState)

    # Nodes
    builder.add_node("analyze_question", analyze_question)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("evaluate_documents", evaluate_documents)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("fallback_answer", fallback_answer)

    # Fixed edges
    builder.add_edge(START, "analyze_question")
    builder.add_edge("analyze_question", "retrieve_documents")
    builder.add_edge("retrieve_documents", "evaluate_documents")

    # Conditional edge: the router's return value selects the next node
    builder.add_conditional_edges(
        "evaluate_documents",
        route_after_evaluation,
        {
            "generate_answer": "generate_answer",
            "fallback_answer": "fallback_answer",
        },
    )

    builder.add_edge("generate_answer", END)
    builder.add_edge("fallback_answer", END)

    return builder.compile()


graph = build_graph()  # compiled once, reused by the whole application


def ask(question: str) -> dict:
    """Run the workflow inside one root span; every node span becomes its child."""
    with tracer.start_as_current_span("rag.ask") as span:
        span.set_attribute("rag.question", question[:200])
        final = graph.invoke({"question": question})

        score = final["relevance_score"]
        route = "generate_answer" if score >= RELEVANCE_THRESHOLD else "fallback_answer"
        span.set_attribute("rag.relevance_score", score)
        span.set_attribute("rag.route", route)
        span.set_attribute("rag.answer_chars", len(final["answer"]))

        trace_id = span.get_span_context().trace_id  # 0 when tracing is off
        return {
            **final,
            "route": route,
            "trace_id": format(trace_id, "032x") if trace_id else None,
        }