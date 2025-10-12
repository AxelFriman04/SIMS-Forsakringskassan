from rag_system.graph.state import GraphState
from rag_system.services.vector_db_qdrant import VectorDBClient
from shared.services.embedding_service import embed_texts
from shared.config import settings
import time
import random


class RetrieveNode:
    def __init__(self, state: GraphState):
        self.state = state
        self.vdb = VectorDBClient()

    def run(self, query: str, top_k: int = settings.TOP_K):
        start_time = time.time()
        q_emb = embed_texts([query])[0]
        results = self.vdb.search(q_emb, top_k)
        retrieval_latency = time.time() - start_time

        # Optional re-ranking
        re_rank_delta = 0
        if settings.USE_RE_RANK:
            original_scores = [r["score"] for r in results]
            results = self._simple_re_rank(results)
            re_rank_delta = sum(abs(r["score"] - o) for r, o in zip(results, original_scores))

        snapshot = {"query": query, "topk": results}

        # M2 metrics
        topk_scores = [r["score"] for r in results]
        topk_gap = max(topk_scores) - min(topk_scores) if topk_scores else 0
        distinct_sources = len(set(r.get("source", "unknown") for r in results))
        lexical_overlap = random.uniform(0.1, 0.5)  # placeholder
        recall_vs_gold = None  # placeholder for evaluation

        m2_metrics = {
            "type": "retrieval",
            "topk_scores": topk_scores,
            "topk_gap": topk_gap,
            "distinct_source_count": distinct_sources,
            "lexical_overlap": lexical_overlap,
            "retrieval_latency": retrieval_latency,
            "re_rank_delta": re_rank_delta,
            "recall_vs_gold": recall_vs_gold,
            "num_hits": len(results)
        }

        # Update state
        self.state.last_retrieval_snapshot = snapshot
        self.state.metrics_retrieval = m2_metrics

        return snapshot

    def _simple_re_rank(self, results: list[dict]) -> list[dict]:
        """
        Simple re-ranker: boosts score by small random factor for demonstration.
        """
        for r in results:
            r["score"] *= random.uniform(1.0, 1.05)
        # Sort descending
        return sorted(results, key=lambda x: x["score"], reverse=True)
