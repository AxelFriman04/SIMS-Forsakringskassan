# shared/services/results_db_client.py
import sqlite3
import json
import datetime
from typing import Dict, Any
from shared.config import settings


class ResultsDBClient:
    """Unified database client for all RAG + Compliance results."""

    def __init__(self, db_path: str = settings.DB_PATH):
        self.db_path = db_path
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                answer TEXT,
                ingest_snapshot TEXT,
                retrieval_snapshot TEXT,
                generator_snapshot TEXT,
                compliance_report TEXT,
                metrics_ingestion TEXT,
                metrics_retrieval TEXT,
                metrics_generation TEXT,
                metrics_style TEXT,
                metrics_heuristics TEXT,
                metrics_compliance TEXT,
                metadata_compliance TEXT,
                timestamp TEXT
            )
            """)
            conn.commit()

    def insert_rag_result(self, query: str, answer: str,
                          ingest_snapshot: dict = None,
                          retrieval_snapshot: dict = None,
                          generator_snapshot: dict = None,
                          metrics_ingestion: dict = None,
                          metrics_retrieval: dict = None,
                          metrics_generation: dict = None) -> int:
        """Insert RAG pipeline result and return the row ID."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO results (
                query, answer, ingest_snapshot,
                retrieval_snapshot, generator_snapshot,
                metrics_ingestion, metrics_retrieval, metrics_generation,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query,
                answer,
                json.dumps(ingest_snapshot or {}),
                json.dumps(retrieval_snapshot or {}),
                json.dumps(generator_snapshot or {}),
                json.dumps(metrics_ingestion or {}),
                json.dumps(metrics_retrieval or {}),
                json.dumps(metrics_generation or {}),
                datetime.datetime.utcnow().isoformat()
            ))
            conn.commit()
            return cursor.lastrowid

    def update_compliance_results(self, answer_id: int, state) -> None:
        """Attach compliance metrics and metadata for given RAG result."""
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

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE results
                SET metrics_compliance = ?,
                    metadata_compliance = ?
                WHERE id = ?
            """, (
                json.dumps(compliance_metrics, ensure_ascii=False),
                json.dumps(compliance_metadata, ensure_ascii=False),
                answer_id,
            ))
            conn.commit()

    def update_compliance_report(self, answer_id: int, report_json: dict):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE results
                SET compliance_report = ?
                WHERE id = ?
            """, (json.dumps(report_json, ensure_ascii=False), answer_id))
            conn.commit()

    def fetch_all(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM results")
            rows = cursor.fetchall()
        return rows

    # TODO: Probably remove this function

    """
    def close(self):
        self.conn.close()"""
