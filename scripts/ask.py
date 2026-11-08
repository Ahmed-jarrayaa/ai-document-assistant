import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from app.monitoring.telemetry import flush_telemetry, setup_telemetry, tracer

setup_telemetry()  # must run before the graph is used

from app.graph.nodes import RELEVANCE_THRESHOLD
from app.graph.workflow import graph


def main(question: str) -> None:
    print(f"\nQuestion: {question}\n")
    final_answer = ""
    score = 0.0

    # Same root span as ask(), so the streamed run is also a single trace
    with tracer.start_as_current_span("rag.ask") as span:
        span.set_attribute("rag.question", question[:200])
        for step in graph.stream({"question": question}, stream_mode="updates"):
            for node_name, update in step.items():
                keys = ", ".join(update.keys())
                extra = ""
                if "relevance_score" in update:
                    score = update["relevance_score"]
                    extra = f" | relevance_score={score:.3f}"
                if "retrieved_documents" in update:
                    extra += f" | {len(update['retrieved_documents'])} chunk(s)"
                print(f"  -> {node_name:20s} updated: [{keys}]{extra}")
                if "answer" in update:
                    final_answer = update["answer"]
        span.set_attribute("rag.relevance_score", score)
        span.set_attribute(
            "rag.route",
            "generate_answer" if score >= RELEVANCE_THRESHOLD else "fallback_answer",
        )

    print(f"\nAnswer:\n{final_answer}\n")
    flush_telemetry()  # short script: send spans before the process exits


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python scripts\\ask.py "your question"')
        sys.exit(1)
    main(sys.argv[1])