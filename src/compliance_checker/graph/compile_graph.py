# compliance_checker/graph/compile_graph.py
import sqlite3
from shared.config import settings
from shared.services.result_store import ResultsDBClient
from compliance_checker.graph.state import GraphState
from compliance_checker.graph.nodes.style_evaluator import StyleEvaluatorNode
from compliance_checker.graph.nodes.claim_extractor import ClaimExtractorNode
from compliance_checker.graph.nodes.evidence_checker import EvidenceCheckerNode
from compliance_checker.graph.nodes.root_cause_classifier import RootCauseClassifierNode
from compliance_checker.graph.nodes.compliance_report import ComplianceReportNode


def run_compliance_pipeline(answer_text: str, dry_run: bool = False) -> GraphState:
    """Run full compliance pipeline."""
    state = GraphState()

    # --- #1 Style Evaluation ---
    print("\n=== Step 1: Style Evaluation ===")
    style_node = StyleEvaluatorNode(state)
    state = style_node.run(answer_text)

    # --- #2 Claim Extraction ---
    extractor = ClaimExtractorNode(state)
    state = extractor.run(answer_text, dry_run=dry_run)

    # --- #3 Evidence Verification ---
    verifier = EvidenceCheckerNode()
    state = verifier.run(state)

    return state


if __name__ == "__main__":
    # --- Load latest RAG answer ---
    conn = sqlite3.connect(settings.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, answer FROM results ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row:
        print("No sample answer found in DB.")
        exit()

    answer_id, sample_answer = row
    db_client = ResultsDBClient()

    try:
        print("\n=== Running Compliance Pipeline ===")
        compliance_state = run_compliance_pipeline(sample_answer, dry_run=False)

        # Store metrics + claims
        db_client.update_compliance_results(answer_id, compliance_state)

        # Root cause + report generation
        print("\n=== Running Root Cause Classifier ===")
        root_cause_node = RootCauseClassifierNode(compliance_state)
        verdict_state = root_cause_node.run(answer_id)

        print("\n=== Building Compliance Report ===")
        compliance_node = ComplianceReportNode(verdict_state)
        report = compliance_node.run()

        # Store report in DB
        db_client.update_compliance_report(answer_id, report)
        print(f"\n✅ Compliance report stored for record ID {answer_id}")

    except Exception as e:
        print(f"[ERROR] Compliance pipeline failed: {e}")

"""
def run_compliance_pipeline(answer_text: str, dry_run: bool = False):
    """ """
    Runs the full compliance pipeline:
      1. Claim Extraction
      2. Evidence Verification (Entailment Checking)
    """ """

    if getattr(settings, "DEBUG", False):
        print("=== Running Compliance Pipeline ===")
        print("Step 1: Claim Extraction\n")

    # Initialize shared pipeline state
    state = GraphState()

    # ---- 1. Run Claim Extraction ----
    extractor = ClaimExtractorNode(state)
    state = extractor.run(answer_text, dry_run=dry_run)

    if getattr(settings, "DEBUG", False):
        claims = state.claims
        print(f"\nExtracted {len(claims)} claims: ")
        for i, claim in enumerate(claims, 1):
            print(f"  {i}. {claim['text']}")
            citation = claim.get("citation")
            if citation:
                print(
                    f"  ↳ Citation ID: {citation.get('id')}, score: {citation.get('score')}, source: {citation.get('source')}")
            else:
                print("  ↳ No citation found.")
        print("\nProceeding to Evidence Verification...\n")

    # ---- 2. Run Evidence Checker ----
    verifier = EvidenceCheckerNode()
    state = verifier.run(state)

    if getattr(settings, "DEBUG", False):
        verified_claims = state.verified_claims
        print(f"\nVerified {len(verified_claims)} claims: ")
        for i, claim in enumerate(verified_claims, 1):
            print(
                f"  {i}. {claim['text']} — label: {claim.get('entailment', claim.get('label', 'N/A'))} (conf: {claim.get('confidence', 0.0):.2f})")

        print("\nMetrics:")
        print("  Extraction:", state.metrics_claim_extraction)
        print("  Verification:", state.metrics_verification)

    verified_claims = state.verified_claims
    print(f"\nVerified {len(verified_claims)} claims: ")
    for i, claim in enumerate(verified_claims, 1):
        print(
            f"  {i}. {claim['text']} — label: {claim.get('entailment', claim.get('label', 'N/A'))} (conf: {claim.get('confidence', 0.0):.2f})")

    print("\nMetrics:")
    print("  Extraction:", state.metrics_claim_extraction)
    print("  Verification:", state.metrics_verification)

    return state


if __name__ == "__main__":
    # --- Load latest answer from DB ---
    conn = sqlite3.connect(settings.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, answer FROM results ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row:
        print("No sample answer found in DB.")
        exit()

    answer_id, sample_answer = row

    try:
        # state = run_compliance_pipeline(sample_answer, dry_run=False)
        db_client = ResultsDBClient()
        # db_client.update_compliance_results(answer_id, state)

        # --- Run Root Cause Classification ---
        print("\n=== Running Root Cause Classifier ===")
        state = GraphState()
        root_cause_node = RootCauseClassifierNode(state)
        verdict = root_cause_node.run(answer_id)

        print("\n--- Root Cause Verdict ---")
        print(json.dumps(asdict(verdict), indent=2))

        # --- Compliance Report ---
        compliance_node = ComplianceReportNode(verdict)
        report = compliance_node.run()

    except Exception as e:
        print(f"[ERROR] Compliance pipeline failed: {e}")"""
