import sqlite3
import json
from shared.config import settings

import re


def regex_func():

    answer = """[CITE: c4f348d2-cf03-4adf-8edd-1cc358354212::p1::c0]"""
    declared_citations = re.findall(r"\[CITE:\s*([^\]]+)\]", answer)
    print(declared_citations)

def print_db():
    conn = sqlite3.connect(settings.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM results")
    rows = cursor.fetchall()

    for row in rows:
        # Assuming table structure: id, query, answer, retrieval_snapshot, generator_snapshot,
        # metrics_ingestion, metrics_retrieval, metrics_generation, model_id, timestamp
        row_dict = {
            "id": row[0],
            "query": row[1],
            "answer": row[2],
            "retrieval_snapshot": json.loads(row[3] or "{}"),
            "generator_snapshot": json.loads(row[4] or "{}"),
            "metrics_ingestion": json.loads(row[5] or "{}"),
            "metrics_retrieval": json.loads(row[6] or "{}"),
            "metrics_generation": json.loads(row[7] or "{}"),
            "model_id": row[8],
            "timestamp": row[9],
        }
        print(json.dumps(row_dict, indent=2))

    conn.close()


def print_latest_result():
    """Print the latest row inserted in the 'results' table."""
    conn = sqlite3.connect(settings.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM results ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    if row:
        row_dict = {
            "id": row[0],
            "query": row[1],
            "answer": row[2],
            "retrieval_snapshot": json.loads(row[3] or "{}"),
            "generator_snapshot": json.loads(row[4] or "{}"),
            "metrics_ingestion": json.loads(row[5] or "{}"),
            "metrics_retrieval": json.loads(row[6] or "{}"),
            "metrics_generation": json.loads(row[7] or "{}"),
            "model_id": row[8],
            "timestamp": row[9],
        }
        print(json.dumps(row_dict, indent=2))
    else:
        print("No results found.")
    conn.close()


def clear_results():
    """Delete all entries from the 'results' table."""
    conn = sqlite3.connect(settings.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM results")
    conn.commit()
    conn.close()
    print(f"All rows cleared from {settings.DB_PATH}")


if __name__ == "__main__":
    # print_db()

    print_latest_result()

    # clear_results()

    #regex_func()
