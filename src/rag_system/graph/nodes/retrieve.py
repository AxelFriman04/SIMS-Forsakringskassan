from rag_system.graph.state import GraphState
from rag_system.services.vector_db_qdrant import VectorDBClient
from shared.services.embedding_service import embed_texts


class RetrieveNode:
    def __init__(self, state: GraphState):
        self.state = state
        self.vdb = VectorDBClient()

    def run(self, query: str, top_k: int = 5):
        """
        Embed query, retrieve top-k results from vector DB.
        """
        q_emb = embed_texts([query])[0]
        results = self.vdb.search(q_emb, top_k)

        snapshot = {"query": query, "topk": results}
        self.state.last_retrieval_snapshot = snapshot

        self.state.log_metric({
            "type": "retrieval",
            "query": query,
            "num_hits": len(results),
        })

        return snapshot
