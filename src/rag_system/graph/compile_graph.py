from rag_system.graph.nodes.generate import GenerateNode
from rag_system.graph.nodes.retrieve import RetrieveNode
from rag_system.graph.nodes.ingest import IngestNode
from rag_system.graph.state import GraphState
from shared.config import settings

def build_graph():
    """
    Build and return a simple graph instance with nodes and state.
    """
    state = GraphState()
    ingest = IngestNode(state) if settings.RUN_INGEST else None
    retrieve = RetrieveNode(state)
    generate = GenerateNode(state)

    # Pseudo orchestration map
    return {
        "state": state,
        "ingest": ingest,
        "retrieve": retrieve,
        "generate": generate
    }


if __name__ == "__main__":
    g = build_graph()
    print("Graph built:", list(g.keys()))
