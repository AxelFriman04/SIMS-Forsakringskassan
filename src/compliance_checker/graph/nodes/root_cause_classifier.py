import json
import sqlite3
from typing import Dict, Any
from shared.config import settings
from compliance_checker.graph.state import GraphState


class RootCauseClassifierNode:
    """
    Classifies the compliance verdict and identifies potential root causes
    based on the metrics from ingestion, retrieval, generation, and compliance phases.
    """

    def __init__(self, state: GraphState = None):
        self.state = state

    def _connect(self):
        conn = sqlite3.connect(settings.DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _fetch_metrics_from_db(self, answer_id: int) -> Dict[str, Any]:
        """Fetch all stage metrics for a given answer_id."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    metrics_ingestion,
                    metrics_retrieval,
                    metrics_generation,
                    metrics_compliance,
                    answer
                FROM results
                WHERE id = ?
                """,
                (answer_id,),
            )
            row = cursor.fetchone()

        if not row:
            print(f"[WARN] No record found for answer_id={answer_id}")
            return {}

        def safe_load(value):
            if not value:
                return {}
            try:
                return json.loads(value)
            except Exception:
                return {}

        return {
            "ingestion": safe_load(row["metrics_ingestion"]),
            "retrieval": safe_load(row["metrics_retrieval"]),
            "generation": safe_load(row["metrics_generation"]),
            "compliance": safe_load(row["metrics_compliance"]),
            "answer": row["answer"] or "",
        }

    def _analyze_root_causes(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose which part of the pipeline contributed most to compliance risk."""

        root_causes = []
        confidence = 1.0  # start optimistic, decrease with uncertainty

        ingestion = metrics.get("ingestion", {})
        retrieval = metrics.get("retrieval", {})
        generation = metrics.get("generation", {})
        compliance = metrics.get("compliance", {})

        ingestion_fidelity = ingestion.get("embedding_fidelity", 1.0)
        parsing_success = ingestion.get("parsing_success_rate", 1.0)
        retrieval_topk_gap = retrieval.get("topk_gap", 0.0)
        retrieval_sources = retrieval.get("distinct_source_count", 1)
        retrieval_overlap = retrieval.get("lexical_overlap", 0.0)
        hallucinations = generation.get("preliminary_hallucinations_warnings", [])
        entailment_ratio = compliance.get("verification", {}).get("entailment_ratio", 0.0)
        contradiction_ratio = compliance.get("verification", {}).get("contradiction_ratio", 0.0)
        avg_confidence = compliance.get("verification", {}).get("avg_confidence", 0.0)
        compliance_score = compliance.get("compliance_score", None)

        # --- Heuristic issue identification ---
        if parsing_success < 0.9:
            root_causes.append("Poor parsing success; document ingestion incomplete.")
            confidence -= 0.1

        if ingestion_fidelity < 0.7:
            root_causes.append("Low embedding fidelity; retrieval relevance may be reduced.")
            confidence -= 0.15

        if retrieval_sources < 2 or retrieval_overlap < 0.15:
            root_causes.append("Retrieval narrow or unbalanced; low source diversity.")
            confidence -= 0.15

        if retrieval_topk_gap > 0.05:
            root_causes.append("Retrieval instability (high top-k gap).")
            confidence -= 0.05

        if hallucinations:
            root_causes.append("Potential hallucinations detected in generation step.")
            confidence -= 0.1

        if entailment_ratio < 0.5:
            root_causes.append("Low entailment ratio between answer and sources.")
            confidence -= 0.2

        if contradiction_ratio > 0.3:
            root_causes.append("Contradictory evidence in verified claims.")
            confidence -= 0.2

        if avg_confidence < 0.6:
            root_causes.append("Low model confidence in factual verification.")
            confidence -= 0.1

        # --- Classification logic ---
        if compliance_score is not None:
            score = compliance_score
        else:
            # fallback heuristic
            score = (entailment_ratio + avg_confidence) / 2.0

        if confidence < 0.4:
            verdict = "Inconclusive"
        elif score >= 0.85:
            verdict = "Fully Compliant"
        elif score >= 0.6:
            verdict = "Partially Compliant"
        elif score >= 0.3:
            verdict = "Non-Compliant"
        else:
            verdict = "Severely Non-Compliant"

        if not root_causes:
            root_causes.append("No significant issues detected.")

        return {
            "verdict": verdict,
            "root_causes": root_causes,
            "confidence": round(max(0.0, confidence), 2),
            "compliance_score": round(score, 3),
        }

    def run(self, answer_id: int = None) -> GraphState:
        """
        Executes the root cause classification step.
        """
        if answer_id:
            metrics = self._fetch_metrics_from_db(answer_id)
        elif self.state:
            metrics = {
                "ingestion": self.state.metrics_ingestion,
                "retrieval": self.state.metrics_retrieval,
                "generation": self.state.metrics_generation,
                "compliance": {
                    "claim_extraction": self.state.metrics_claim_extraction,
                    "verification": self.state.metrics_verification,
                },
                "answer": self.state.answer,
            }
        else:
            raise ValueError("No state or answer_id provided to RootCauseClassifierNode.")

        result = self._analyze_root_causes(metrics)

        if self.state:
            self.state.verdict = result["verdict"]
            self.state.root_cause = result
            self.state.metrics_root_cause = result

        # --- NEW ---
        self.state.metrics_pipeline = {
            "ingestion": metrics.get("ingestion", {}),
            "retrieval": metrics.get("retrieval", {}),
            "generation": metrics.get("generation", {}),
            "compliance": metrics.get("compliance", {}),
        }

        # Also update the GraphState answer from DB if we fetched it
        if "answer" in metrics and metrics["answer"]:
            self.state.answer = metrics["answer"]

        if getattr(settings, "DEBUG", False):
            print("\n=== Root Cause Classifier ===")
            print(f"Verdict: {result['verdict']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Compliance Score: {result['compliance_score']}")
            print("Root Causes:")
            for cause in result["root_causes"]:
                print(f"  - {cause}")

        return self.state or result
