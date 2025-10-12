from rag_system.parser.pdf_parser import parse_pdf
from shared.services.embedding_service import embed_texts
from rag_system.services.vector_db_qdrant import VectorDBClient
from rag_system.graph.state import GraphState
import uuid
import numpy as np
from shared.config import settings


class IngestNode:
    def __init__(self, state: GraphState):
        self.state = state
        self.vdb = VectorDBClient(reset_collection=settings.RUN_INGEST)

    def run(self, pdf_path: str, metadata: dict = None):
        metadata = metadata or {}
        doc_id = metadata.get("doc_id") or str(uuid.uuid4())

        # --- Step 1: Parse PDF into structured chunks ---
        chunks = parse_pdf(pdf_path, doc_id=doc_id)

        # Extract the summary metadata
        summary_meta = {}
        if chunks and chunks[-1]["chunk_id"].endswith("::summary"):
            summary_meta = chunks.pop(-1)["metadata"]

        num_chunks = len(chunks)
        num_pages = summary_meta.get("num_pages", 0)
        num_empty_pages = summary_meta.get("num_empty_pages", 0)
        total_chars = summary_meta.get("total_chars", 0)

        # --- Step 2: Compute M1 Metrics ---

        # Parsing Success Rate: ratio of pages that produced text
        parsing_success_rate = (
            (num_pages - num_empty_pages) / num_pages if num_pages > 0 else 0.0
        )

        # Chunk Coverage (%): percentage of pages that yielded chunks
        pages_with_chunks = sum(1 for p in summary_meta.get("page_stats", []) if p["num_chunks"] > 0)
        chunk_coverage_pct = (
            (pages_with_chunks / num_pages * 100) if num_pages > 0 else 0.0
        )

        # Metadata Completeness: proportion of chunks with all expected metadata fields
        required_fields = {"page_number", "source", "doc_id", "char_length"}
        complete_chunks = sum(
            1 for c in chunks
            if c.get("metadata") and required_fields.issubset(set(c["metadata"].keys()))
        )
        metadata_completeness = (
            complete_chunks / num_chunks if num_chunks > 0 else 0.0
        )

        # Compute embeddings and evaluate embedding fidelity
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        # Calculate pairwise cosine similarity across random pairs to assess embedding consistency
        if len(embeddings) >= 2:
            emb_matrix = np.array(embeddings)
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            normalized = emb_matrix / (norms + 1e-8)
            sim_matrix = np.dot(normalized, normalized.T)
            # Take the mean of upper-triangular similarities as a proxy for fidelity (diversity & coherence)
            triu_indices = np.triu_indices_from(sim_matrix, k=1)
            embedding_fidelity = float(np.mean(sim_matrix[triu_indices]))
        else:
            embedding_fidelity = 1.0  # trivial single-embedding case

        # --- Step 3: Log M1 metrics ---
        m1_metrics = {
            "type": "ingest",
            "doc_id": doc_id,
            "num_chunks": num_chunks,
            "num_pages": num_pages,
            "total_chars": total_chars,
            "parsing_success_rate": parsing_success_rate,
            "chunk_coverage_pct": chunk_coverage_pct,
            "embedding_fidelity": embedding_fidelity,
            "metadata_completeness": metadata_completeness,
        }

        # --- Step 4: Upsert chunks into Qdrant ---
        self.vdb.upsert([
            (
                c["chunk_id"],
                emb,
                {
                    "text": c["text"],
                    "metadata": {**c.get("metadata", {}), **metadata}
                }
            )
            for c, emb in zip(chunks, embeddings)
        ])

        # --- Step 5: Update Graph State ---
        ingest_snapshot = {
            "doc_id": doc_id,
            "num_chunks": num_chunks,
            "num_pages": num_pages,
            "total_chars": total_chars,
        }

        self.state.last_ingest_snapshot = ingest_snapshot
        self.state.metrics_ingestion = m1_metrics
        self.state.log_metric(m1_metrics)

        return ingest_snapshot
