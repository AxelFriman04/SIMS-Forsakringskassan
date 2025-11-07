# compliance_checker/graph/compile_graph.py

import sqlite3
from shared.config import settings
from shared.services.result_store import ResultsDBClient
from compliance_checker.graph.state import GraphState
from compliance_checker.graph.nodes.style_evaluator import StyleEvaluatorNode
from compliance_checker.graph.nodes.answer_relevance_evaluator import AnswerRelevanceEvaluatorNode
from compliance_checker.graph.nodes.claim_extractor import ClaimExtractorNode
from compliance_checker.graph.nodes.evidence_checker import EvidenceCheckerNode
from compliance_checker.graph.nodes.root_cause_classifier import RootCauseClassifierNode
from compliance_checker.graph.nodes.compliance_report import ComplianceReportNode


def run_compliance_pipeline(answer_text: str, query_text: str = "", dry_run: bool = False) -> GraphState:
    """
    Run the full compliance pipeline on a given RAG-generated answer.
    Steps:
        1. Style Evaluation
        2. Query–Answer Relevance Evaluation
        3. Claim Extraction
        4. Evidence Verification
    """
    state = GraphState()

    # --- Step 1: Style Evaluation ---
    print("\n=== Step 1: Style Evaluation ===")
    style_node = StyleEvaluatorNode(state)
    state = style_node.run(answer_text)

    # --- Step 2: Query–Answer Relevance Evaluation ---
    if query_text:
        print("\n=== Step 2: Relevance Evaluation ===")
        relevance_node = AnswerRelevanceEvaluatorNode(state)
        state = relevance_node.run(query_text, answer_text)
    else:
        print("\n[WARN] No query provided — skipping relevance evaluation.")

    # --- Step 3: Claim Extraction ---
    print("\n=== Step 3: Claim Extraction ===")
    extractor = ClaimExtractorNode(state)
    state = extractor.run(answer_text, dry_run=dry_run)

    # --- Step 4: Evidence Verification ---
    print("\n=== Step 4: Evidence Verification ===")
    verifier = EvidenceCheckerNode()
    state = verifier.run(state)

    return state


if __name__ == "__main__":
    # --- Load latest RAG answer ---
    conn = sqlite3.connect(settings.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, query, answer FROM results ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row:
        print("No sample answer found in DB.")
        exit()

    answer_id, sample_query, sample_answer = row
    db_client = ResultsDBClient()

    try:
        print("\n=== Running Compliance Pipeline ===")
        compliance_state = run_compliance_pipeline(sample_answer, query_text=sample_query, dry_run=False)

        # --- Persist node metrics and extracted data ---
        db_client.update_compliance_results(answer_id, compliance_state)

        # --- Step 5: Root Cause Classification ---
        print("\n=== Step 5: Root Cause Classifier ===")
        root_cause_node = RootCauseClassifierNode(compliance_state)
        verdict_state = root_cause_node.run(answer_id)

        # --- Step 6: Build and Store Compliance Report ---
        print("\n=== Step 6: Building Compliance Report ===")
        compliance_node = ComplianceReportNode(verdict_state)
        report = compliance_node.run()

        db_client.update_compliance_report(answer_id, report)
        print(f"\n✅ Compliance report stored successfully for record ID {answer_id}")

    except Exception as e:
        print(f"[ERROR] Compliance pipeline failed: {e}")
