# compliance_checker/graph/nodes/answer_relevance_evaluator.py
import json, numpy as np, re
from typing import Dict, Any
from shared.config import settings
from shared.services.embedding_service import embed_texts
from compliance_checker.graph.state import GraphState
from compliance_checker.model.llm import LLM
from compliance_checker.prompt.prompt_loader import (
    load_answer_relevance_prompt,
    load_answer_relevance_system_prompt
)
from shared.services.math_utils import cosine_similarity_matrix



class AnswerRelevanceEvaluatorNode:
    """Evaluates whether the RAG answer actually addresses the user query and estimates redundancy."""

    def __init__(self, state: GraphState):
        self.state = state
        self.llm = LLM(model=settings.RELEVANCE_EVAL_MODEL)
        self.alignment_schema = {
            "name": "answer_relevance",
            "type": "object",
            "properties": {
                "answers_question": {"type": "boolean"},
                "relevance_score": {"type": "number"},
                "coverage_score": {"type": "number"},
                "redundancy_ratio": {"type": "number"},
                "missing_elements": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
            },
            "required": [
                "answers_question", "relevance_score", "coverage_score",
                "redundancy_ratio", "missing_elements", "notes"
            ],
            "additionalProperties": False,
        }

    def _cheap_heuristics(self, query: str, answer: str) -> Dict[str, Any]:
        """Compute quick lexical/embedding heuristics."""
        if not answer.strip():
            return {
                "query_answer_cosine": 0.0, "keyword_coverage": 0.0,
                "redundancy_ratio": 1.0, "hallucination_risk": 1.0
            }

        q_emb = embed_texts([query])[0]
        a_emb = embed_texts([answer])[0]
        cos_sim = cosine_similarity_matrix(np.array([q_emb]), np.array([a_emb]))[0][0]

        query_terms = re.findall(r"\w+", query.lower())
        coverage = len([t for t in query_terms if t in answer.lower()]) / max(len(query_terms), 1)

        sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if s.strip()]
        sent_embs = np.array(embed_texts(sentences))

        if sent_embs.size > 0:
            sims = cosine_similarity_matrix(np.array([q_emb]), sent_embs)[0]
        else:
            sims = []
        redundancy_ratio = sum(1 for s in sims if s < 0.3) / max(len(sims), 1)

        hallucination_risk = round((1 - cos_sim) * (1 - coverage), 3)
        return {
            "query_answer_cosine": round(cos_sim, 3),
            "keyword_coverage": round(coverage, 3),
            "redundancy_ratio": round(redundancy_ratio, 3),
            "hallucination_risk": hallucination_risk
        }

    def run(self, query: str, answer_text: str) -> GraphState:
        self.state.answer = answer_text
        heuristics = self._cheap_heuristics(query, answer_text)

        prompt = load_answer_relevance_prompt(query, answer_text)
        system_prompt = load_answer_relevance_system_prompt()
        try:
            response = self.llm.complete(
                prompt, structured=True, schema=self.alignment_schema, system_prompt=system_prompt
            )
            parsed = json.loads(response.get("answer", "{}"))
        except Exception as e:
            print(f"[ERROR] LLM relevance evaluation failed: {e}")
            parsed = {
                "answers_question": False, "relevance_score": heuristics["query_answer_cosine"],
                "coverage_score": heuristics["keyword_coverage"], "redundancy_ratio": heuristics["redundancy_ratio"],
                "missing_elements": [], "notes": "LLM evaluation failed; using heuristic estimates."
            }

        self.state.metrics_relevance = {
            **heuristics,
            "answers_question": parsed.get("answers_question", False),
            "relevance_score": parsed.get("relevance_score", heuristics["query_answer_cosine"]),
            "coverage_score": parsed.get("coverage_score", heuristics["keyword_coverage"]),
            "redundancy_ratio": parsed.get("redundancy_ratio", heuristics["redundancy_ratio"]),
            "missing_elements": parsed.get("missing_elements", []),
            "notes": parsed.get("notes", ""),
            "relevance_metadata": {
                    "model": settings.RELEVANCE_EVAL_MODEL,
                }
        }

        return self.state
