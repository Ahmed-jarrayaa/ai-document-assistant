import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database.connection import get_connection
from app.database.vector_store import delete_source, insert_chunks, similarity_search
from app.services.embedding import embed_query, embed_texts
from app.services.document_loader import clean_text
TEST_SOURCE = "__test_phase3__.pdf"

CHUNKS = [
    {"page": 1, "chunk_index": 0, "content": "A kitten is sleeping on the couch next to the cat."},
    {"page": 1, "chunk_index": 1, "content": "The tax declaration must be filed before the end of March."},
    {"page": 2, "chunk_index": 2, "content": "Python is a programming language used for data science."},
    {"page": 2, "chunk_index": 3, "content": "Neural networks learn by adjusting weights with gradient descent."},
]


@pytest.fixture(scope="module", autouse=True)
def test_data():
    delete_source(TEST_SOURCE)
    insert_chunks(TEST_SOURCE, CHUNKS, embed_texts([c["content"] for c in CHUNKS]))
    yield
    delete_source(TEST_SOURCE)


def test_top_result_is_the_right_chunk():
    results = similarity_search(embed_query("Where is the cat sleeping?"), k=2, source=TEST_SOURCE)
    assert "kitten" in results[0]["content"]


def test_results_sorted_and_limited_to_k():
    results = similarity_search(embed_query("machine learning training"), k=3, source=TEST_SOURCE)
    assert len(results) == 3
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)
    assert all(0.0 <= r["distance"] <= 2.0 for r in results)


def test_source_filter_only_returns_that_source():
    results = similarity_search(embed_query("anything"), k=10, source=TEST_SOURCE)
    assert len(results) == 4
    assert all(r["source"] == TEST_SOURCE for r in results)


def test_relevant_question_scores_higher_than_unrelated():
    relevant = similarity_search(embed_query("How do neural networks learn?"), k=1, source=TEST_SOURCE)
    unrelated = similarity_search(embed_query("Best pizza recipe in Naples"), k=1, source=TEST_SOURCE)
    assert relevant[0]["similarity"] > unrelated[0]["similarity"]


def test_hnsw_index_exists():
    with get_connection() as conn:
        row = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'document_chunks_embedding_idx'"
        ).fetchone()
    assert row is not None
    assert "hnsw" in row[0] and "vector_cosine_ops" in row[0]
def test_clean_text_fixes_ligatures():
    assert clean_text("identiﬁer une règle diﬃcile") == "identifier une règle difficile"


def test_clean_text_removes_dotted_leaders():
    cleaned = clean_text("Chapitre 1 . . . . . . . . . . 12")
    assert "." not in cleaned
    assert "Chapitre 1" in cleaned