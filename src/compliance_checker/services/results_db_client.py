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

    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.DB_PATH

    def _connect(self):
        """Connect to SQLite DB with dict-style row access."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_compliance_columns(self, cursor):
        """Check if compliance columns exist, and create them if missing."""
        cursor.execute("PRAGMA table_info(results)")
        columns = [col["name"] for col in cursor.fetchall()]

        missing = []
        if "metrics_compliance" not in columns:
            missing.append("metrics_compliance")
        if "metadata_compliance" not in columns:
            missing.append("metadata_compliance")

        for col in missing:
            print(f"[DB] Adding missing column: {col}")
            cursor.execute(f"ALTER TABLE results ADD COLUMN {col} TEXT")

        if not missing:
            print("[DB] Compliance columns already exist.")

    def update_compliance_results(self, answer_id: int, state) -> None:
        """Insert or update compliance results for the given RAG result entry."""
        with self._connect() as conn:
            cursor = conn.cursor()

            # Ensure compliance columns exist
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

            # Prepare compliance metrics
            compliance_metrics = {
                "claim_extraction": state.metrics_claim_extraction,
                "verification": state.metrics_verification,
                "num_claims": len(state.claims),
                "num_verified_claims": len(state.verified_claims),
                "compliance_score": state.compliance_score,
            }

            compliance_metadata = {
                "claims": state.claims,
                "verified_claims": state.verified_claims,
            }

            # Store results
            cursor.execute(
                """
                UPDATE results
                SET metrics_compliance = ?,
                    metadata_compliance = ?
                WHERE id = ?
                """,
                (
                    json.dumps(compliance_metrics, ensure_ascii=False),
                    json.dumps(compliance_metadata, ensure_ascii=False),
                    answer_id,
                ),
            )

            conn.commit()
            print(f"[DB] ✅ Compliance results stored successfully for id={answer_id}")
