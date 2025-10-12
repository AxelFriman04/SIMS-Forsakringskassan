import fitz  # PyMuPDF
from typing import List, Dict, Tuple
import re
import uuid
import os


def split_text_into_chunks(text: str, max_chars: int = 1500) -> List[str]:
    """
    Split text into coherent chunks without breaking sentences or paragraphs abruptly.
    """
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
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
    Parse a PDF file into structured text chunks.

    Returns:
        List of chunk dicts in the format:
        {
            "chunk_id": str,
            "text": str,
            "metadata": {
                "page_number": int,
                "source": str,
                "doc_id": str,
                "char_length": int,
                "num_chunks_in_page": int
            }
        }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF file not found: {path}")

    doc_id = doc_id or str(uuid.uuid4())
    doc = fitz.open(path)
    results: List[Dict] = []

    total_chars = 0
    page_stats = []  # to help with metrics later

    for page_num, page in enumerate(doc, start=1):
        # Extract text using PyMuPDF method
        try:
            text = page.get_text("text") or ""
        except Exception as e:
            print(f"[WARN] Failed to extract text from page {page_num}: {e}")
            text = ""

        text_len = len(text.strip())

        if text_len == 0:
            # Record as empty page for later metrics
            page_stats.append({
                "page_number": page_num,
                "char_length": 0,
                "num_chunks": 0,
                "status": "empty"
            })
            continue

        # Split text into chunks
        chunks = split_text_into_chunks(text, max_chars=max_chars)
        total_chars += text_len

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}::p{page_num}::c{i}"
            results.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "metadata": {
                    "page_number": page_num,
                    "source": os.path.basename(path),
                    "doc_id": doc_id,
                    "char_length": len(chunk),
                    "num_chunks_in_page": len(chunks),
                }
            })

        # Record page stats
        page_stats.append({
            "page_number": page_num,
            "char_length": text_len,
            "num_chunks": len(chunks),
            "status": "ok"
        })

    doc.close()

    # Add summary metadata for metrics
    summary_metadata = {
        "num_pages": len(page_stats),
        "num_chunks": len(results),
        "total_chars": total_chars,
        "num_empty_pages": sum(1 for p in page_stats if p["status"] == "empty"),
        "page_stats": page_stats,
    }

    # Store this summary under a special "summary" chunk for calculating M1 metrics
    results.append({
        "chunk_id": f"{doc_id}::summary",
        "text": "",
        "metadata": summary_metadata
    })

    return results
