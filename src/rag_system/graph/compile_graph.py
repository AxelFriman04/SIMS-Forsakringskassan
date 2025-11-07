from rag_system.graph.nodes.generate import GenerateNode
from rag_system.graph.nodes.retrieve import RetrieveNode
from rag_system.graph.nodes.ingest import IngestNode
from rag_system.graph.state import GraphState
from shared.services.result_store import ResultsDBClient
from shared.config import settings


def build_graph(result_store: ResultsDBClient):
    """
    Build nodes and state.
    """
    state = GraphState()
    ingest = IngestNode(state) if settings.RUN_INGEST else None
    retrieve = RetrieveNode(state)
    generate = GenerateNode(state)
    return state, ingest, retrieve, generate


def run_pipeline(pdf_path: str, query: str, result_store: ResultsDBClient):
    state, ingest, retrieve, generate = build_graph(result_store)

    # 1. Ingest
    if ingest:
        ingest.run(pdf_path)

    if settings.RUN_GENERATE:

        # 2. Retrieve
        retrieval_snapshot = retrieve.run(query)

        # 3. Generate
        generate.run(query, retrieval_snapshot)

        # 4. Insert full result into ResultStore  TODO: Check new structure if changes are needed
        state.result_id = result_store.insert_rag_result(
            query=state.query,
            answer=state.answer,
            ingest_snapshot=state.last_ingest_snapshot,
            retrieval_snapshot=state.last_retrieval_snapshot,
            generator_snapshot=state.last_generation_snapshot,
            metrics_ingestion=state.metrics_ingestion,
            metrics_retrieval=state.metrics_retrieval,
            metrics_generation=state.metrics_generation,
        )

    return state


if __name__ == "__main__":
    rs = ResultsDBClient()
    query = "När kan det bli aktuellt att bedöma oskäligt?"
    rag_state = run_pipeline(settings.PDF_PATH, query, rs)
    print("Pipeline finished. Answer:", rag_state.answer)
