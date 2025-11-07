import json
import sqlite3
from typing import Any, Dict
from shared.config import settings


class ResultsDBClient:
    """
    Handles storing compliance metrics and claim verification results
    into the same SQLite database as the RAG results,
    using the same record ID.
    """
    # TODO: This file is maybe unused
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.DB_PATH

    def _connect(self):
        """Connect to SQLite DB with dict-style row access."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_compliance_columns(self, cursor):
        """Check if all compliance-related columns exist, and create them if missing."""
        cursor.execute("PRAGMA table_info(results)")
        columns = [col["name"] for col in cursor.fetchall()]

        expected_cols = [
            "metrics_style",
            "metrics_relevance",
            "metrics_heuristics",
            "metrics_compliance",
            "metadata_compliance",
        ]

        for col in expected_cols:
            if col not in columns:
                print(f"[DB] Adding missing column: {col}")
                cursor.execute(f"ALTER TABLE results ADD COLUMN {col} TEXT")

    # ---------- UPDATE FUNCTION ----------
    def update_compliance_results(self, answer_id: int, state) -> None:
        """Insert or update all compliance-related results for the given RAG result entry."""
        with self._connect() as conn:
            cursor = conn.cursor()

            # Ensure all new columns exist
            self._ensure_compliance_columns(cursor)

            # Verify record exists
            cursor.execute("SELECT * FROM results WHERE id = ?", (answer_id,))
            row = cursor.fetchone()
            if not row:
                print(f"[WARN] No record found for id={answer_id}")
                return

            # Compute compliance score if missing
            if not getattr(state, "compliance_score", None) and state.metrics_verification:
                vr = state.metrics_verification
                entailment = vr.get("entailment_ratio", 0.0)
                confidence = vr.get("avg_confidence", 0.0)
                state.compliance_score = round(entailment * confidence, 3)

            # --- Prepare all metrics ---
            compliance_metrics = {
                "claim_extraction": getattr(state, "metrics_claim_extraction", {}),
                "verification": getattr(state, "metrics_verification", {}),
                "num_claims": len(getattr(state, "claims", [])),
                "num_verified_claims": len(getattr(state, "verified_claims", [])),
                "compliance_score": getattr(state, "compliance_score", 0.0),
            }

            compliance_metadata = {
                "claims": getattr(state, "claims", []),
                "verified_claims": getattr(state, "verified_claims", []),
            }

            # --- NEW METRICS FIELDS ---
            metrics_style = getattr(state, "metrics_style", {})
            metrics_relevance = getattr(state, "metrics_relevance", {})
            metrics_heuristics = getattr(state, "metrics_heuristics", {})

            # --- Build SQL update ---
            cursor.execute(
                """
                UPDATE results
                SET
                    metrics_style = ?,
                    metrics_relevance = ?,
                    metrics_heuristics = ?,
                    metrics_compliance = ?,
                    metadata_compliance = ?
                WHERE id = ?
                """,
                (
                    json.dumps(metrics_style, ensure_ascii=False),
                    json.dumps(metrics_relevance, ensure_ascii=False),
                    json.dumps(metrics_heuristics, ensure_ascii=False),
                    json.dumps(compliance_metrics, ensure_ascii=False),
                    json.dumps(compliance_metadata, ensure_ascii=False),
                    answer_id,
                ),
            )

            conn.commit()
            print(f"[DB] ✅ Compliance results stored successfully for id={answer_id}")
