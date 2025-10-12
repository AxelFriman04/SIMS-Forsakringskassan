from rag_system.graph.nodes.generate import GenerateNode
from rag_system.graph.nodes.retrieve import RetrieveNode
from rag_system.graph.nodes.ingest import IngestNode
from rag_system.graph.state import GraphState
from rag_system.services.result_store import ResultStore
from shared.config import settings


def build_graph(result_store: ResultStore):
    """
    Build nodes and state.
    """
    state = GraphState()
    ingest = IngestNode(state) if settings.RUN_INGEST else None
    retrieve = RetrieveNode(state)
    generate = GenerateNode(state)
    return state, ingest, retrieve, generate


def run_pipeline(pdf_path: str, query: str, result_store: ResultStore):
    state, ingest, retrieve, generate = build_graph(result_store)

    # 1. Ingest
    if ingest:
        ingest.run(pdf_path)

    # 2. Retrieve
    retrieval_snapshot = retrieve.run(query)

    # 3. Generate
    generate.run(query, retrieval_snapshot)

    # 4. Insert full result into ResultStore
    result_store.insert_result(
        query=state.query,
        answer=state.answer,
        retrieval_snapshot=state.last_retrieval_snapshot,
        generator_snapshot=state.last_generation_snapshot,
        metrics_ingestion=state.metrics_ingestion,
        metrics_retrieval=state.metrics_retrieval,
        metrics_generation=state.metrics_generation,
        model_id=generate.llm.model
    )

    return state


if __name__ == "__main__":
    rs = ResultStore()
    query = "Summarize the key characteristics, habitat, and culinary uses of Boletus edulis, citing the relevant chunks."
    state = run_pipeline(settings.PDF_PATH, query, rs)
    print("Pipeline finished. Answer:", state.answer)
