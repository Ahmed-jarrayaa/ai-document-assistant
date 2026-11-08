import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")  # avoids Windows console encoding errors

from app.database.vector_store import similarity_search
from app.services.embedding import embed_query


def search(question: str, k: int = 4) -> None:
    t0 = time.perf_counter()
    query_vec = embed_query(question)
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = similarity_search(query_vec, k=k)
    t_search = time.perf_counter() - t0

    print(f"\nQuestion: {question}")
    print(f"(embedding: {t_embed * 1000:.0f} ms | SQL search: {t_search * 1000:.0f} ms)\n")
    for rank, r in enumerate(results, start=1):
        preview = r["content"].replace("\n", " ")[:220]
        print(f"#{rank}  similarity={r['similarity']:.3f}  page={r['page']}  chunk={r['chunk_index']}")
        print(f"    {preview}...\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python scripts\\search.py "your question" [k]')
        sys.exit(1)
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    search(sys.argv[1], k)