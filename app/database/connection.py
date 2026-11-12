import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()  # reads the .env file into environment variables


def get_connection_string() -> str:
    """Build the PostgreSQL connection string from environment variables.

    If POSTGRES_URL (or DATABASE_URL) is set — e.g. the "Internal Database
    URL" of a managed Postgres like Render's — it is used directly.
    Otherwise the individual POSTGRES_* variables are assembled.
    """
    url = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
    if url:
        return url
    return (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5433')} "
        f"dbname={os.getenv('POSTGRES_DB')} "
        f"user={os.getenv('POSTGRES_USER')} "
        f"password={os.getenv('POSTGRES_PASSWORD')}"
    )


def get_connection(register_vectors: bool = True) -> psycopg.Connection:
    """Open a new connection to PostgreSQL.

    register_vectors=True teaches psycopg how to convert Python lists / numpy
    arrays to the pgvector `vector` type and back. It requires the extension
    to already exist in the database, so init_db.py calls this with False.
    """
    conn = psycopg.connect(get_connection_string())
    if register_vectors:
        register_vector(conn)
    return conn