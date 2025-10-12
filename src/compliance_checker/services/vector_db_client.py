# compliance_checker/services/vector_db_client.py

from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from shared.config import settings


class ComplianceVectorDB:
    """
    Lightweight, read-only vector DB client for compliance checking.
    Used to verify extracted factual claims against a curated reference database.
    """

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or settings.VECTOR_DB_COLLECTION
        self.client = QdrantClient(url=settings.VECTOR_DB_URL)

        # Ensure collection exists (read-only)
        try:
            self.client.get_collection(self.collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Compliance vector DB collection '{self.collection_name}' not found. "
                f"Ensure it exists and is populated before running compliance checks."
            ) from e

    def search_claims(self, q_vector: List[float], top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Searches for the top matching evidence chunk for a given claim embedding.
        Returns list of dicts containing citation metadata.
        """
        if not q_vector:
            return []

        query_result = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vector,
            limit=top_k,
        )

        results = []
        # New API: hits are in query_result.result
        for h in query_result.points:
            payload = h.payload or {}
            results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "score": h.score,
                    "source": payload.get("source") or payload.get("metadata", {}).get("source"),
                }
            )
        return results

    def search_claims_batch(self, q_vectors: List[List[float]], top_k: int = 1) -> List[List[Dict[str, Any]]]:
        batch_results = []

        for vec in q_vectors:
            hits = self.search_claims(vec, top_k=top_k)  # Reuse existing single-search method
            batch_results.append(hits)

        return batch_results

        """
        Batch search for multiple claim embeddings.
        Returns a list of top_k results per query vector.
        """
        """if not q_vectors:
            return [[] for _ in range(len(q_vectors))]

        query_result = self.client.query_batch_points(
            collection_name=self.collection_name,
            query=q_vectors,
            limit=top_k,
        )

        batch_results = []
        for h_list in query_result.points:  # Each h_list corresponds to one query vector
            hits = []
            for h in h_list:
                payload = h.payload or {}
                hits.append({
                    "chunk_id": payload.get("chunk_id"),
                    "score": h.score,
                    "source": payload.get("source") or payload.get("metadata", {}).get("source"),
                })
            batch_results.append(hits)

        return batch_results"""

    def get_chunk_by_id(self, chunk_id: str) -> Dict[str, Any]:
        """
        Fetch a specific chunk from Qdrant by its unique chunk_id (stored in payload).
        Returns a dict containing the chunk text and metadata.
        """
        if not chunk_id:
            return {}

        try:
            # Use Qdrant's filtering to find the specific chunk by ID
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=rest.Filter(
                    must=[rest.FieldCondition(key="chunk_id", match=rest.MatchValue(value=chunk_id))]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

            if points:
                point = points[0]
                payload = point.payload or {}
                return {
                    "chunk_id": payload.get("chunk_id"),
                    "text": payload.get("text") or payload.get("content", ""),
                    "source": payload.get("source") or payload.get("metadata", {}).get("source"),
                }

        except Exception as e:
            if getattr(settings, "DEBUG", False):
                print(f"[DEBUG] Failed to fetch chunk {chunk_id}: {e}")

        return {}
