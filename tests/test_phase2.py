import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database.vector_store import count_chunks, delete_source, insert_chunks
from app.services.document_loader import split_pages
from app.services.embedding import EMBEDDING_DIM, embed_query, embed_texts

TEST_SOURCE = "__test_phase2__.pdf"


def test_chunking_keeps_page_and_order():
    pages = [
        {"page": 1, "text": "Sentence one. " * 200},
        {"page": 2, "text": "Another page. " * 200},
    ]
    chunks = split_pages(pages)
    assert len(chunks) > 2
    assert chunks[0]["page"] == 1
    assert chunks[-1]["page"] == 2
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(len(c["content"]) <= 800 for c in chunks)


def test_embeddings_shape_and_normalization():
    vectors = embed_texts(["hello world", "bonjour le monde"])
    assert vectors.shape == (2, EMBEDDING_DIM)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_similar_texts_are_closer_than_unrelated():
    a = embed_query("The cat sits on the sofa")
    b = embed_query("A kitten is resting on the couch")
    c = embed_query("Quarterly tax declaration deadline")
    # vectors are normalized, so dot product = cosine similarity
    assert float(a @ b) > float(a @ c)


def test_insert_and_count_roundtrip():
    delete_source(TEST_SOURCE)
    chunks = [
        {"page": 1, "chunk_index": 0, "content": "first test chunk"},
        {"page": 1, "chunk_index": 1, "content": "second test chunk"},
    ]
    embeddings = embed_texts([c["content"] for c in chunks])
    assert insert_chunks(TEST_SOURCE, chunks, embeddings) == 2
    assert count_chunks(TEST_SOURCE) == 2
    assert delete_source(TEST_SOURCE) == 2
    assert count_chunks(TEST_SOURCE) == 0