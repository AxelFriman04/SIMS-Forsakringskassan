from rag_system.parser.pdf_parser import parse_pdf
from shared.services.embedding_service import embed_texts
from rag_system.services.vector_db_qdrant import VectorDBClient
from rag_system.graph.state import GraphState
import uuid
from shared.config import settings


class IngestNode:
    def __init__(self, state: GraphState):
        self.state = state
        self.vdb = VectorDBClient(reset_collection=settings.RUN_INGEST)
        print("IngestNode initialized")

    def run(self, pdf_path: str, metadata: dict = None):
        print("IngestNode running!")
        metadata = metadata or {}
        doc_id = metadata.get("doc_id") or str(uuid.uuid4())

        # Parse PDF -> structured chunks
        chunks = parse_pdf(pdf_path, doc_id=doc_id)

        # Compute embeddings
        embeddings = embed_texts([c["text"] for c in chunks])

        # Upsert into Qdrant: (id, vector, payload)
        self.vdb.upsert([
            (
                c["chunk_id"],
                emb,
                {
                    "text": c["text"],  # the actual text chunk
                    "metadata": {**c["metadata"], **metadata}  # merged metadata
                }
            )
            for c, emb in zip(chunks, embeddings)
        ])

        # Log snapshot
        ingest_snapshot = {"doc_id": doc_id, "num_chunks": len(chunks)}
        self.state.last_ingest_snapshot = ingest_snapshot
        self.state.log_metric({"type": "ingest", **ingest_snapshot})

        return ingest_snapshot
