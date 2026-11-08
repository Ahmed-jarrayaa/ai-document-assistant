import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database.connection import get_connection


def test_connection_works():
    with get_connection() as conn:
        assert conn.execute("SELECT 1").fetchone()[0] == 1


def test_pgvector_extension_installed():
    with get_connection() as conn:
        row = conn.execute(
            "SELECT extname FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        assert row is not None


def test_table_has_384_dim_embedding():
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT format_type(atttypid, atttypmod)
            FROM pg_attribute
            WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding'
            """
        ).fetchone()
        assert row[0] == "vector(384)"


def test_cosine_distance_operator():
    """Identical vectors -> distance 0. Orthogonal vectors -> distance 1."""
    with get_connection() as conn:
        same = conn.execute(
            "SELECT '[1,0,0]'::vector <=> '[1,0,0]'::vector"
        ).fetchone()[0]
        orthogonal = conn.execute(
            "SELECT '[1,0,0]'::vector <=> '[0,1,0]'::vector"
        ).fetchone()[0]
        assert abs(same - 0.0) < 1e-6
        assert abs(orthogonal - 1.0) < 1e-6