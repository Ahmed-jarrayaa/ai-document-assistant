import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, StringConstraints

from app.database.connection import get_connection
from app.database.vector_store import delete_source, list_sources
from app.graph.workflow import ask
from app.monitoring.telemetry import flush_telemetry, setup_telemetry
from app.services import embedding
from app.services import embedding
from app.services import llm
from app.services.ingestion import IngestionError, ingest_pdf
from app.services.llm import LLMError
from fastapi.responses import FileResponse
load_dotenv()
setup_telemetry()  # must run before the app is instrumented

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-up in the background: the model (torch) loads after the server is
    # listening, so the port opens immediately even on small instances.
    threading.Thread(target=embedding.warmup, name="model-warmup", daemon=True).start()
    yield
    flush_telemetry()  # send pending spans on shutdown


app = FastAPI(title="AI Document Assistant", version="1.0.0", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app, excluded_urls="health")  # one span per HTTP request
FRONTEND_FILE = Path(__file__).resolve().parent.parent / "frontend" / "index.html"



# ---------- schemas ----------

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class AskRequest(BaseModel):
    question: Question


class SourceRef(BaseModel):
    source: str
    page: int | None
    similarity: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    route: str
    relevance_score: float
    sources: list[SourceRef]
    trace_id: str | None


class IngestResponse(BaseModel):
    source: str
    chunks: int
    replaced_chunks: int


# ---------- endpoints (plain `def`: blocking code runs in a thread pool) ----------

@app.get("/health")
def health():
    database = "ok"
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        database = f"error: {type(exc).__name__}"
    llm_status = "ok" if llm.is_ready() else "unavailable"
    return {
        "status": "ok" if database == "ok" and llm_status == "ok" else "degraded",
        "database": database,
        "llm": llm_status,
        "provider": llm.LLM_PROVIDER,
        "model": llm.LLM_MODEL,
    }


@app.post("/documents", response_model=IngestResponse, status_code=201)
def upload_document(file: UploadFile = File(...)):
    # Keep only the file name and neutralize unusual characters (no path tricks)
    filename = re.sub(r"[^\w.\- ]", "_", Path(file.filename or "").name)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File is not a valid PDF.")

    DOCUMENTS_DIR.mkdir(exist_ok=True)
    path = DOCUMENTS_DIR / filename
    path.write_bytes(data)
    try:
        return ingest_pdf(path)
    except IngestionError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/documents")
def documents():
    return list_sources()


@app.delete("/documents/{source}")
def remove_document(source: str):
    removed = delete_source(source)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No document named '{source}'.")
    return {"source": source, "chunks_removed": removed}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    try:
        final = ask(request.question)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=f"Language model unavailable: {exc}")
    except psycopg.OperationalError:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    sources = [
        SourceRef(
            source=d["source"],
            page=d["page"],
            similarity=round(d["similarity"], 4),
            excerpt=d["content"].replace("\n", " ")[:200],
        )
        for d in final["retrieved_documents"]  # already filtered by evaluate_documents
    ]
    return AskResponse(
        answer=final["answer"],
        route=final["route"],
        relevance_score=round(final["relevance_score"], 4),
        sources=sources,
        trace_id=final["trace_id"],
    )
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_FILE)
