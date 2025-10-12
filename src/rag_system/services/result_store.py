# src/graph/result_store.py
import sqlite3
import json
import datetime
from shared.config import settings


class ResultStore:
    def __init__(self, db_path: str = settings.DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            answer TEXT,
            retrieval_snapshot TEXT,
            generator_snapshot TEXT,
            metrics_ingestion TEXT,
            metrics_retrieval TEXT,
            metrics_generation TEXT,
            model_id TEXT,
            timestamp TEXT
        )
        """)
        self.conn.commit()

    def insert_result(self, query: str, answer: str,
                      retrieval_snapshot: dict,
                      generator_snapshot: dict,
                      metrics_ingestion: dict = None,
                      metrics_retrieval: dict = None,
                      metrics_generation: dict = None,
                      model_id: str = None):
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO results (
            query, answer,
            retrieval_snapshot, generator_snapshot,
            metrics_ingestion, metrics_retrieval, metrics_generation,
            model_id, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            query,
            answer,
            json.dumps(retrieval_snapshot or {}),
            json.dumps(generator_snapshot or {}),
            json.dumps(metrics_ingestion or {}),
            json.dumps(metrics_retrieval or {}),
            json.dumps(metrics_generation or {}),
            model_id,
            datetime.datetime.utcnow().isoformat()
        ))
        self.conn.commit()

    def fetch_all(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM results")
        rows = cursor.fetchall()
        return rows

    def close(self):
        self.conn.close()
