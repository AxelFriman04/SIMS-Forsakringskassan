from typing import List, Tuple, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from shared.config import settings
import uuid


class VectorDBClient:
    def __init__(self, collection_name: str = None, reset_collection: bool = False):
        self.collection_name = collection_name or settings.VECTOR_DB_COLLECTION
        self.client = QdrantClient(url=settings.VECTOR_DB_URL)

        if reset_collection:
            # Force recreate when ingesting a new snapshot
            self._recreate_collection()
        else:
            # Just ensure it exists
            self._ensure_collection()

    def _ensure_collection(self):
        # Ensure collection exists, but don't overwrite if it already does.
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self._recreate_collection()


    def _recreate_collection(self):
        """Force recreation of collection (used on ingest)."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=rest.VectorParams(
                size=settings.EMBED_DIM,
                distance=rest.Distance.COSINE
            ),
        )
    def upsert(self, items: List[Tuple[str, List[float], Dict[str, Any]]]):
        """
        Upsert items into Qdrant.
        items = [(chunk_id, vector, payload_dict)]
        Payload MUST include {"text": ..., "metadata": {...}}
        """
        points = []
        for chunk_id, vector, payload in items:
            # Validate payload
            if "text" not in payload or "metadata" not in payload:
                raise ValueError(
                    f"Payload for {chunk_id} must include 'text' and 'metadata'"
                )

            # Convert chunk_id to a valid UUID (deterministic)
            valid_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

            points.append(
                rest.PointStruct(
                    id=valid_uuid,
                    vector=vector,
                    payload={
                        "chunk_id": chunk_id,  # keep human-readable ID for citations
                        "text": payload["text"],
                        **payload["metadata"],
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, q_vector: List[float], top_k: int = None):
        """
        Search Qdrant for nearest neighbors.
        Returns list of dicts compatible with RetrievalOutput.
        """
        top_k = top_k or settings.TOP_K
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=q_vector,
            limit=top_k,
        )

        results = []
        for h in hits:
            results.append(
                {
                    "chunk_id": h.payload.get("chunk_id"),  # human-readable
                    "score": h.score,
                    "text": h.payload.get("text", ""),
                    "metadata": {
                        k: v
                        for k, v in h.payload.items()
                        if k not in {"chunk_id", "text"}
                    },
                }
            )

        return results
