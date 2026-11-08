from pathlib import Path
import re
import unicodedata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

CHUNK_SIZE = 800      # max characters per chunk
CHUNK_OVERLAP = 150   # characters shared between consecutive chunks


def load_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """Extract text from each page. Returns [{'page': 1, 'text': '...'}, ...]."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:  # skip empty pages (images only, blank pages...)
            pages.append({"page": page_number, "text": text})
    return pages


def split_pages(pages: list[dict]) -> list[dict]:
    """Split page texts into chunks. Returns
    [{'page': 1, 'chunk_index': 0, 'content': '...'}, ...]."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    chunk_index = 0
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            chunks.append(
                {
                    "page": page["page"],
                    "chunk_index": chunk_index,
                    "content": piece,
                }
            )
            chunk_index += 1
    return chunks


def load_and_chunk(pdf_path: str | Path) -> list[dict]:
    """Full loader pipeline: PDF file → list of chunks."""
    return split_pages(load_pdf_pages(pdf_path))
def clean_text(text: str) -> str:
    """Normalize PDF extraction artifacts."""
    text = unicodedata.normalize("NFKC", text)    # ligatures: ﬁ -> fi, ﬃ -> ffi
    text = re.sub(r"(?:\s?\.){4,}", " ", text)    # dotted leaders ". . . . ."
    text = re.sub(r"[ \t]+", " ", text)           # collapse spaces (keep newlines)
    return text.strip()