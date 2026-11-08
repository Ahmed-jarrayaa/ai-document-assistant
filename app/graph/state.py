from typing import TypedDict


class GraphState(TypedDict, total=False):
    """Data shared by all nodes.

    total=False means a key may be missing: at START only `question` exists,
    and each node adds its own fields as the workflow progresses.
    """

    question: str                 # the user's question (cleaned by analyze_question)
    retrieved_documents: list[dict]  # chunks from pgvector, with similarity scores
    relevance_score: float        # best similarity among retrieved chunks
    answer: str                   # final answer (or fallback message)