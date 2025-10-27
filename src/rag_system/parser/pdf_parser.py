# rag_system/parser/pdf_parser.py
import fitz  # PyMuPDF
import re
import uuid
import os
from typing import List, Dict
from shared.config import settings


def split_text_into_chunks(text: str, max_chars: int = 1500, overlap_chars: int = 200) -> List[str]:
    """
    Split text into coherent chunks without breaking sentences or paragraphs abruptly.
    Respects Swedish headings and avoids cutting section titles.
    Adds an overlap between chunks to preserve context continuity.

    Args:
        text (str): The input text to split.
        max_chars (int): Maximum characters per chunk.
        overlap_chars (int): Number of characters to overlap between consecutive chunks.
    """

    max_chars = getattr(settings, "MAX_CHUNK_SIZE", max_chars)
    overlap_chars = getattr(settings, "OVERLAP_CHARS", overlap_chars)

    # Normalize spacing
    text = re.sub(r'\s+\n', '\n', text).strip()

    # Split paragraphs on double newlines or major section dividers
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    current = ""

    for p in paragraphs:
        # Detect section titles like "DEL A", "Avsnitt 2.1", etc.
        if re.match(r"^(DEL\s+[A-Z]|Avsnitt\s*\d+(\.\d+)*)", p, flags=re.IGNORECASE):
            # Force a new chunk before major section headings
            if current:
                chunks.append(current.strip())
                current = ""

        if len(current) + len(p) + 2 <= max_chars:
            current = current + "\n\n" + p if current else p
        else:
            # --- Add overlap between chunks ---
            if current:
                chunks.append(current.strip())

                # Preserve overlap portion from end of previous chunk
                overlap = current[-overlap_chars:].strip() if overlap_chars > 0 else ""
                current = (overlap + "\n\n" + p).strip()
            else:
                current = p

    if current:
        chunks.append(current.strip())

    return chunks



def parse_pdf(path: str, doc_id: str = None, max_chars: int = 1500, language: str = "sv") -> List[Dict]:
    """
    Parse a PDF file into structured text chunks.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF file not found: {path}")

    doc_id = doc_id or str(uuid.uuid4())
    doc = fitz.open(path)
    results: List[Dict] = []
    total_chars = 0
    page_stats = []
    current_section = None

    for page_num, page in enumerate(doc, start=1):
        try:
            text = page.get_text("text") or ""
        except Exception as e:
            print(f"[WARN] Failed to extract text from page {page_num}: {e}")
            text = ""

        text_len = len(text.strip())
        if text_len == 0:
            page_stats.append({
                "page_number": page_num,
                "char_length": 0,
                "num_chunks": 0,
                "status": "empty"
            })
            continue

        # Detect current section (e.g., “DEL A” or “Avsnitt 3”)
        match = re.search(r"^(DEL\s+[A-Z]|Avsnitt\s*\d+)", text, re.IGNORECASE | re.MULTILINE)
        if match:
            current_section = match.group(0).strip()

        # Split into chunks
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
                    "section": current_section,
                    "language": language  # TODO: Add user set language support
                }
            })

        page_stats.append({
            "page_number": page_num,
            "char_length": text_len,
            "num_chunks": len(chunks),
            "status": "ok"
        })

    doc.close()

    summary_metadata = {
        "num_pages": len(page_stats),
        "num_chunks": len(results),
        "total_chars": total_chars,
        "num_empty_pages": sum(1 for p in page_stats if p["status"] == "empty"),
        "page_stats": page_stats,
    }

    results.append({
        "chunk_id": f"{doc_id}::summary",
        "text": "",
        "metadata": summary_metadata
    })

    return results
