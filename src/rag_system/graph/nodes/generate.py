from rag_system.model.llm import LLM
from rag_system.graph.state import GraphState


class GenerateNode:
    def __init__(self, state: GraphState):
        self.state = state
        self.llm = LLM()

    def run(self, query: str, retrieval_snapshot):
        """
        Compose a prompt with retrieved evidence and ask the LLM to generate an answer.
        """
        context = "\n\n".join([r["text"] for r in retrieval_snapshot["topk"]])
        prompt = (
            f"Use the following evidence to answer the query.\n\n"
            f"EVIDENCE:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"Answer clearly and cite sources by their chunk_id where relevant."
        )

        gen = self.llm.complete(prompt)

        self.state.log_metric({
            "type": "generation",
            "chars": len(gen["answer"]),
            "query": query,
        })

        return gen
