import sys
from pathlib import Path

# Allow "from app...." imports when running this script directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database.connection import get_connection

EMBEDDING_DIM = 384  # must match the embedding model used in Phase 2

CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS document_chunks (
    id          SERIAL PRIMARY KEY,
    source      TEXT NOT NULL,          -- PDF file name
    page        INTEGER,                -- page number in the PDF
    chunk_index INTEGER NOT NULL,       -- position of the chunk in the document
    content     TEXT NOT NULL,          -- the chunk text
    embedding   vector({EMBEDDING_DIM}) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
"""
CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
ON document_chunks USING hnsw (embedding vector_cosine_ops);
"""

def init_db() -> None:
    # No vector registration yet: the extension may not exist
    with get_connection(register_vectors=False) as conn:
        conn.execute(CREATE_EXTENSION)
        conn.execute(CREATE_TABLE)
        conn.execute(CREATE_INDEX)
        conn.commit()
    print("Database initialized: extension 'vector' + table 'document_chunks'.")


if __name__ == "__main__":
    init_db()