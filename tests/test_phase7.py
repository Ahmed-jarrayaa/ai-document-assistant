import os
import sys
from pathlib import Path

os.environ["TRACING_ENABLED"] = "false"  # tests must not export traces to Jaeger

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database.vector_store import count_chunks, delete_source
from app.main import DOCUMENTS_DIR, app
from app.services import llm

TEST_PDF = "__test_api__.pdf"
needs_ollama = pytest.mark.skipif(not llm.is_ready(), reason="Ollama not ready")
needs_pdf = pytest.mark.skipif(count_chunks("IntroML_Azencott.pdf") == 0, reason="Ingest the PDF first")


def make_pdf(text: str) -> bytes:
    """Build a tiny valid one-page PDF containing `text` (ASCII, no parentheses)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = "%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n"
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return out.encode("latin-1")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:  # runs the lifespan (model warm-up)
        yield test_client


@pytest.fixture
def cleanup():
    yield
    delete_source(TEST_PDF)
    (DOCUMENTS_DIR / TEST_PDF).unlink(missing_ok=True)


def test_health_reports_components(client):
    body = client.get("/health").json()
    assert {"status", "database", "ollama", "model"} <= set(body)
    assert body["database"] == "ok"


def test_ask_rejects_empty_or_missing_question(client):
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_upload_rejects_non_pdf_extension(client):
    response = client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_upload_rejects_fake_pdf(client):
    response = client.post("/documents", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")})
    assert response.status_code == 400


def test_upload_rejects_pdf_without_text(client, cleanup):
    response = client.post(
        "/documents", files={"file": (TEST_PDF, make_pdf(""), "application/pdf")}
    )
    assert response.status_code == 422


def test_upload_list_delete_roundtrip(client, cleanup):
    pdf = make_pdf("The secret file of project Aurora is called Zorglub-42.")
    upload = client.post("/documents", files={"file": (TEST_PDF, pdf, "application/pdf")})
    assert upload.status_code == 201
    assert upload.json()["source"] == TEST_PDF and upload.json()["chunks"] >= 1

    assert TEST_PDF in [d["source"] for d in client.get("/documents").json()]
    assert client.delete(f"/documents/{TEST_PDF}").status_code == 200
    assert client.delete(f"/documents/{TEST_PDF}").status_code == 404


@needs_pdf
def test_offtopic_question_returns_fallback_without_llm(client):
    body = client.post("/ask", json={"question": "Meilleure recette de pizza à Naples"}).json()
    assert body["route"] == "fallback_answer"
    assert body["sources"] == []
    assert "could not find" in body["answer"]


@needs_ollama
def test_upload_then_ask_is_grounded(client, cleanup):
    pdf = make_pdf("The secret file of project Aurora is called Zorglub-42.")
    assert client.post("/documents", files={"file": (TEST_PDF, pdf, "application/pdf")}).status_code == 201

    body = client.post("/ask", json={"question": "What is the secret file of project Aurora called?"}).json()
    assert body["route"] == "generate_answer"
    assert "Zorglub" in body["answer"]
    assert TEST_PDF in [s["source"] for s in body["sources"]]