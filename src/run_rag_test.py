from rag_system.graph.compile_graph import build_graph
from shared.config import settings


def main():
    g = build_graph()

    if settings.RUN_INGEST:
        print("=== Ingesting test PDF ===")
        ingest_snapshot = g["ingest"].run(settings.PDF_PATH)
        print("Ingest snapshot:", ingest_snapshot)

    if settings.RUN_RETRIEVE:
        print("\n=== Retrieving for a query ===")
        query = "What is this document about?"
        retrieval_snapshot = g["retrieve"].run(query, top_k=settings.TOP_K)
        print("Retrieval snapshot:", retrieval_snapshot)
    else:
        retrieval_snapshot = None

    if settings.RUN_GENERATE and retrieval_snapshot:
        print("\n=== Generating answer ===")
        generation_output = g["generate"].run(query, retrieval_snapshot)
        print("Generation output:", generation_output)

    print("\n=== Metrics log ===")
    for metric in g["state"].metrics:
        print(metric)


def validate_rag_system():
    g = build_graph()

    print("=== Ingesting test PDF ===")
    ingest_snapshot = g["ingest"].run(settings.PDF_PATH)
    print("Ingest snapshot:", ingest_snapshot)

    # Basic check: num_chunks > 0
    num_chunks = ingest_snapshot.get("num_chunks", 0)
    assert num_chunks > 0, "No chunks were ingested!"

    print("\n=== Retrieving for a query ===")
    query = "What is this document about?"
    retrieval_snapshot = g["retrieve"].run(query, top_k=settings.TOP_K)
    print("Retrieval snapshot:", retrieval_snapshot)

    hits = retrieval_snapshot.get("topk", [])
    assert len(hits) > 0, "No chunks retrieved!"
    assert len(hits) <= settings.TOP_K, "Retrieved more than top_k chunks!"

    # Check metadata on each retrieved chunk
    for h in hits:
        assert "chunk_id" in h, "Missing chunk_id in retrieval!"
        assert "text" in h, "Missing text in retrieval!"
        assert "metadata" in h, "Missing metadata in retrieval!"

    print("\n=== Testing chunk coverage ===")
    chunk_ids_ingested = set(f"{ingest_snapshot['doc_id']}::p{i}::c0" for i in range(1, num_chunks+1))
    chunk_ids_retrieved = set(h["chunk_id"] for h in hits)
    print("Ingested chunk IDs:", chunk_ids_ingested)
    print("Retrieved chunk IDs:", chunk_ids_retrieved)

    assert chunk_ids_retrieved.issubset(chunk_ids_ingested), "Retrieved chunks not in ingested chunks!"

    print("\n✅ All checks passed. RAG system structure is working in dummy mode!")


if __name__ == "__main__":
    main()
    #validate_rag_system()
