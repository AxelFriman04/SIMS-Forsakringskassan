import fitz  # PyMuPDF
from typing import List, Dict
import re
import uuid


def split_text_into_chunks(text: str, max_chars: int = 1500) -> List[str]:
    """
    Split text into chunks without cutting sentences/paragraphs abruptly.
    """
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_chars:
            current = current + "\n\n" + p if current else p
        else:
            if current:
                chunks.append(current)
            current = p

    if current:
        chunks.append(current)

    return chunks


def parse_pdf(path: str, doc_id: str = None, max_chars: int = 1500) -> List[Dict]:
    """
    Extract text from a PDF using PyMuPDF and split into chunks.
    Returns chunks in edge-compatible format.
    """
    doc_id = doc_id or str(uuid.uuid4())
    doc = fitz.open(path)
    results: List[Dict] = []

    for page_num, page in enumerate(doc, start=1):
        # Extract page text (recommended for PyMuPDF v1.22+)
        text = page.get_text("text") or ""
        if not text.strip():
            continue  # skip empty pages

        for i, chunk in enumerate(split_text_into_chunks(text, max_chars=max_chars)):
            chunk_id = f"{doc_id}::p{page_num}::c{i}"
            results.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "metadata": {
                    "page_number": page_num,
                    "source": path,
                    "doc_id": doc_id
                }
            })

    doc.close()
    return results
