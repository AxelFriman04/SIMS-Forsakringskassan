from rag_system.graph.state import GraphState
from rag_system.services.vector_db_qdrant import VectorDBClient
from shared.services.embedding_service import embed_texts
from shared.config import settings
from rag_system.model.llm import LLM
from rag_system.prompt.prompt_loader import load_rerank_prompt
import time
import json
import re


class RetrieveNode:
    def __init__(self, state: GraphState):
        self.state = state
        self.vdb = VectorDBClient()
        self.llm = LLM(model=getattr(settings, "RERANK_MODEL", "gpt-5-nano"))

        self.rerank_schema = {
            "name": "rerank",
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "passage": {"type": "integer"},
                            "relevance": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 10.0,
                            },
                        },
                        "required": ["passage", "relevance"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        }

    def run(self, query: str, top_k: int = settings.TOP_K):

        def compute_lexical_overlap(query: str, retrieved_chunks: list[dict]) -> float:
            """Compute lexical overlap between query and retrieved evidence text."""
            if not retrieved_chunks:
                return 0.0

            query_tokens = set(re.findall(r"\w+", query.lower()))
            all_text = " ".join([r.get("text", "") for r in retrieved_chunks])
            text_tokens = set(re.findall(r"\w+", all_text.lower()))

            overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            return round(overlap, 3)

        start_time = time.time()
        q_emb = embed_texts([query])[0]
        results = self.vdb.search(q_emb, top_k)
        retrieval_latency = time.time() - start_time

        # Optional re-ranking
        threshold = getattr(settings, "RERANK_THRESHOLD", 8)
        use_rerank = (
            getattr(settings, "USE_RE_RANK", False)
            and len(results) >= threshold
        )  # or force_rerank

        re_rank_delta = 0
        rerank_model_used = None

        if use_rerank:
            original_scores = [r["score"] for r in results]
            results = self._llm_re_rank(query, results)
            rerank_model_used = self.llm.model

            re_rank_delta = sum(
                abs(r["score"] - o) for r, o in zip(results, original_scores)
            ) / max(len(original_scores), 1)

        snapshot = {
            "query": query,
            "topk": results,
            "rerank_metadata": {
                "model": rerank_model_used,
            },
        }

        # M2 metrics
        topk_scores = [r["score"] for r in results]
        topk_gap = max(topk_scores) - min(topk_scores) if topk_scores else 0
        distinct_sources = len(set(r.get("source", "unknown") for r in results))
        lexical_overlap = compute_lexical_overlap(query, results)
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
        # self.state.metrics_retrieval = m2_metrics
        self.state.log_metric(m2_metrics)

        return snapshot

    # ---------- LLM Re-Ranker ----------
    def _llm_re_rank(self, query: str, results: list[dict]) -> list[dict]:
        """Use OpenAI LLM to assign semantic relevance scores to retrieved passages."""
        if not results:
            return results

        prompt = load_rerank_prompt(query, results)
        system_prompt = (
            "You are a ranking assistant. "
            "Your job is to evaluate how relevant retrieved passages are to a given query, "
            "not to answer the query. "
            "Rate relevance numerically between 0 and 10 based only on semantic alignment."
        )

        try:
            response = self.llm.complete(
                prompt,
                structured=True,
                schema=self.rerank_schema,
                system_prompt=system_prompt,
                max_tokens=16000,
            )
            parsed = json.loads(response.get("answer", "{}"))
            scores = parsed.get("scores", [])

            for s in scores:
                idx = s.get("passage") - 1
                if 0 <= idx < len(results):
                    results[idx]["score_original"] = results[idx]["score"]
                    results[idx]["score"] = float(s.get("relevance", 0.0))

            results = sorted(results, key=lambda x: x["score"], reverse=True)

        except Exception as e:
            print(f"[WARN] LLM re-ranker failed: {e}")
            return results

        return results
