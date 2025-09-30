from langchain_core.documents import Document
from typing_extensions import TypedDict, List
from langgraph.graph import START, StateGraph

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State, vector_store):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State, prompt, llm):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def build_graph(State, retrieve_fn, generate_fn):
    graph_builder = StateGraph(State).add_sequence([
        ("retrieve", retrieve_fn),
        ("generate", generate_fn)
    ])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()

