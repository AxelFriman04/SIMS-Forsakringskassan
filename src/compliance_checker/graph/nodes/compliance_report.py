# compliance_checker/graph/nodes/compliance_report.py

import json
from typing import Dict, Any
from compliance_checker.graph.state import GraphState
from shared.config import settings
from compliance_checker.services.results_db_client import ResultsDBClient


class ComplianceReportNode:
    """
    Generates a structured, multi-layer compliance report from the full pipeline state.
    Persists it into the results database under 'compliance_report'.
    """

    def __init__(self, state: GraphState):
        if not state:
            raise ValueError("ComplianceReportNode requires a GraphState object.")
        self.state = state
        self.db = ResultsDBClient()

    # ---------- Internal helper ----------
    def _extract_model_provenance(self) -> Dict[str, Any]:
        """
        Dynamically extract which models were used at each stage.
        Looks for model info in the metrics and metadata fields
        rather than relying on static config.
        """
        metrics = getattr(self.state, "metrics_pipeline", {}) or {}
        style_metrics = getattr(self.state, "metrics_style", {}) or {}
        relevance_metrics = getattr(self.state, "metrics_relevance", {}) or {}

        # --- RAG models ---
        ingestion = metrics.get("ingestion", {}) or {}
        retrieval = metrics.get("retrieval", {}) or {}
        generation = metrics.get("generation", {}) or {}

        # --- Compliance models ---
        compliance = metrics.get("compliance", {}) or {}
        claim_extraction = (compliance.get("claim_extraction") or {})
        verification = (compliance.get("verification") or {})

        # Extract model IDs dynamically from stored metadata
        return {
            "rag_models": {
                "embedding_model": (
                    ingestion.get("metadata", {}).get("model")
                    or settings.EMBEDDING_MODEL
                ),
                "generation_model": (
                    generation.get("metadata", {}).get("model")
                    or generation.get("generator_metadata", {}).get("model")
                    or settings.LLM_MODEL
                ),
                "rerank_model": (
                    retrieval.get("rerank_metadata", {}).get("model")
                    or settings.RERANK_MODEL
                ),
            },
            "compliance_models": {
                "claim_extraction_model": (
                    claim_extraction.get("claim_extract_metadata", {}).get("model")
                    or claim_extraction.get("metadata", {}).get("model")
                    or settings.CLAIM_EXTRACT_MODEL
                ),
                "entailment_model": (
                    verification.get("entailment_metadata", {}).get("model")
                    or verification.get("metadata", {}).get("model")
                    or settings.ENTAILMENT_LLM_MODEL
                ),
                "style_eval_model": (
                    style_metrics.get("style_metadata", {}).get("model")
                    or settings.STYLE_EVAL_MODEL
                ),
                "relevance_model": (
                    relevance_metrics.get("relevance_metadata", {}).get("model")
                    or settings.RELEVANCE_LLM_MODEL
                )
            },
        }

    # ---------- Builder ----------
    def _generate_report(self) -> Dict[str, Any]:
        """Builds a structured compliance report using pipeline state."""

        root_cause = getattr(self.state, "metrics_root_cause", {}) or {}
        metrics_pipeline = getattr(self.state, "metrics_pipeline", {}) or {}
        metrics_style = getattr(self.state, "metrics_style", {}) or {}
        metrics_heuristics = getattr(self.state, "metrics_heuristics", {}) or {}
        metrics_relevance = getattr(self.state, "metrics_relevance", {}) or {}
        claims = getattr(self.state, "claims", [])
        verified_claims = getattr(self.state, "verified_claims", [])

        model_info = self._extract_model_provenance()

        report = {
            "rag_answer": getattr(self.state, "answer", ""),
            "verdict": getattr(self.state, "verdict", "Inconclusive"),
            "compliance_score": root_cause.get("overall_score"),
            "confidence": root_cause.get("confidence", 0.0),
            "root_causes": root_cause.get("issues", []),
            "recommendations": root_cause.get("recommendations", []),

            "stage_scores": root_cause.get("stage_scores", {}),
            "summary": root_cause.get("verdict", "Summary unavailable"),

            "metrics": {
                "pipeline": metrics_pipeline,
                "style": metrics_style,
                "heuristics": metrics_heuristics,
                "relevance": metrics_relevance,
                "root_cause": root_cause,
            },

            "claims": claims,
            "verified_claims": verified_claims,

            "models": model_info,
        }

        return report

    # ---------- Execution ----------
    def run(self) -> Dict[str, Any]:
        """Generate and persist the compliance report."""
        report = self._generate_report()

        answer_id = getattr(self.state, "result_id", None)
        if answer_id:
            try:
                self.db.update_compliance_report(answer_id, report)
                if getattr(settings, "DEBUG", False):
                    print(f"[DB] ✅ Compliance report stored successfully for id={answer_id}")
            except Exception as e:
                print(f"[DB ERROR] Failed to update compliance report: {e}")
        else:
            if getattr(settings, "DEBUG", False):
                print("[WARN] No result_id found in state — report not stored in DB.")

        if getattr(settings, "DEBUG", False):
            print("\n=== 📘 COMPLIANCE REPORT ===")
            print(json.dumps(report, indent=2, ensure_ascii=False))

        return report
