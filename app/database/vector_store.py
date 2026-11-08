import numpy as np

from app.database.connection import get_connection
from app.monitoring.telemetry import tracer

def delete_source(source: str) -> int:
    """Remove all chunks of a document (lets us re-ingest without duplicates)."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM document_chunks WHERE source = %s", (source,))
        conn.commit()
        return cur.rowcount


def insert_chunks(source: str, chunks: list[dict], embeddings: np.ndarray) -> int:
    """Insert chunks with their embeddings. Returns the number of rows inserted."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    rows = [
        (source, c["page"], c["chunk_index"], c["content"], emb)
        for c, emb in zip(chunks, embeddings)
    ]
    with tracer.start_as_current_span("pgvector.insert_chunks") as span:
        span.set_attribute("db.system", "postgresql")
        span.set_attribute("db.operation", "INSERT")
        span.set_attribute("ingest.source", source)
        span.set_attribute("ingest.num_chunks", len(rows))
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO document_chunks (source, page, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    rows,
                )
            conn.commit()
    return len(rows)
def count_chunks(source: str | None = None) -> int:
    """Count stored chunks, optionally for a single document."""
    with get_connection() as conn:
        if source:
            row = conn.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE source = %s", (source,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()
        return row[0]
def similarity_search(
    query_embedding: np.ndarray,
    k: int = 4,
    source: str | None = None,
) -> list[dict]:
    """Return the k chunks closest to the query (cosine distance)."""
    where = "WHERE source = %s" if source else ""
    params = [query_embedding]
    if source:
        params.append(source)
    params += [query_embedding, k]

    sql = f"""
        SELECT id, source, page, chunk_index, content,
               embedding <=> %s AS distance
        FROM document_chunks
        {where}
        ORDER BY embedding <=> %s
        LIMIT %s
    """

    with tracer.start_as_current_span("pgvector.similarity_search") as span:
        span.set_attribute("db.system", "postgresql")
        span.set_attribute("db.operation", "SELECT")
        span.set_attribute("db.query.text", " ".join(sql.split()))
        span.set_attribute("retrieval.k", k)
        span.set_attribute("retrieval.source_filter", source or "")

        with get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        results = [
            {
                "id": r[0],
                "source": r[1],
                "page": r[2],
                "chunk_index": r[3],
                "content": r[4],
                "distance": float(r[5]),
                "similarity": 1.0 - float(r[5]),
            }
            for r in rows
        ]
        span.set_attribute("retrieval.num_results", len(results))
        if results:
            span.set_attribute("retrieval.top_similarity", results[0]["similarity"])
        return results
def list_sources() -> list[dict]:
    """List ingested documents with their number of chunks."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT source, COUNT(*), MAX(created_at)
            FROM document_chunks
            GROUP BY source
            ORDER BY source
            """
        ).fetchall()
    return [
        {"source": r[0], "chunks": r[1], "ingested_at": r[2].isoformat()}
        for r in rows
    ]