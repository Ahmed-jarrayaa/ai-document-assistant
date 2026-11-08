from pathlib import Path

from pypdf.errors import PdfReadError

from app.database.vector_store import delete_source, insert_chunks
from app.monitoring.telemetry import tracer
from app.services.document_loader import load_and_chunk
from app.services.embedding import embed_texts


class IngestionError(ValueError):
    """The PDF cannot be ingested (unreadable, or no extractable text)."""


def ingest_pdf(pdf_path: str | Path) -> dict:
    """PDF -> chunks -> embeddings -> pgvector. Re-ingesting replaces the old chunks."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    source = path.name

    with tracer.start_as_current_span("ingest.pdf") as span:
        span.set_attribute("ingest.source", source)

        with tracer.start_as_current_span("ingest.load_and_chunk") as load_span:
            try:
                chunks = load_and_chunk(path)
            except PdfReadError as exc:
                raise IngestionError(f"Unreadable PDF: {exc}") from exc
            load_span.set_attribute("ingest.num_chunks", len(chunks))

        if not chunks:
            raise IngestionError("No text extracted. Is the PDF scanned (images only)?")

        embeddings = embed_texts([c["content"] for c in chunks])
        replaced = delete_source(source)
        inserted = insert_chunks(source, chunks, embeddings)
        span.set_attribute("ingest.replaced_chunks", replaced)
        return {"source": source, "chunks": inserted, "replaced_chunks": replaced}