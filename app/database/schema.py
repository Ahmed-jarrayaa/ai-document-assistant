"""Database schema creation and migration (idempotent, safe at every startup)."""

from app.database.connection import get_connection
from app.services.embedding import get_target_dim
from app.monitoring.telemetry import tracer

CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_META = """
CREATE TABLE IF NOT EXISTS _schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS document_chunks (
    id          SERIAL PRIMARY KEY,
    source      TEXT NOT NULL,          -- PDF file name
    page        INTEGER,                -- page number in the PDF
    chunk_index INTEGER NOT NULL,       -- position of the chunk in the document
    content     TEXT NOT NULL,          -- the chunk text
    embedding   vector({{dim}}) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
ON document_chunks USING hnsw (embedding vector_cosine_ops);
"""


@tracer.start_as_current_span("database.ensure_schema")
def ensure_schema() -> None:
    """Create (or migrate) the vector extension, table and index.

    The expected embedding dimension depends on the configured provider
    (384 for local/HF models, 768 for Gemini by default). When the dimension
    changes, the table is dropped and recreated: existing vectors from an
    incompatible model cannot be compared with new ones, so the documents
    must be re-uploaded.
    """
    dim = get_target_dim()

    # No vector registration yet: the extension may not exist
    with get_connection(register_vectors=False) as conn:
        conn.execute(CREATE_EXTENSION)
        conn.execute(CREATE_META)

        has_chunks = conn.execute(
            "SELECT to_regclass('document_chunks') IS NOT NULL;"
        ).fetchone()[0]
        row = conn.execute(
            "SELECT value FROM _schema_meta WHERE key = 'embedding_dim';"
        ).fetchone()
        # Legacy databases (before _schema_meta) were always created with 384
        stored = int(row[0]) if row else (384 if has_chunks else None)

        if stored is not None and stored != dim:
            print(
                f"[schema] embedding dimension changed {stored} -> {dim}: "
                "dropping and recreating 'document_chunks' "
                "(re-upload your documents).",
                flush=True,
            )
            conn.execute("DROP TABLE document_chunks;")
            stored = None

        if stored is None:
            conn.execute(CREATE_TABLE.format(dim=dim))

        conn.execute(CREATE_INDEX)
        conn.execute(
            "INSERT INTO _schema_meta (key, value) VALUES ('embedding_dim', %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;",
            (str(dim),),
        )
        conn.commit()

    print(f"Database schema ready: extension 'vector' + table 'document_chunks' (vector({dim})).", flush=True)
