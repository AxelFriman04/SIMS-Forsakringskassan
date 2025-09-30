from loader import load_and_split
from embeddings import VectorStoreManager
from rag import State, retrieve, generate, build_graph
from llm import init_llm, load_prompt

# Code inspired by LangChain tutorial on RAG https://python.langchain.com/docs/tutorials/rag

'''
To run code in Visual Studio (2022)
* Find path to Python Environment
    1. View -> Other Windows -> Python Environments
    2. Get PATH
* Install packages
    1. Open terminal (View -> Terminal)
    2. Run "PATH -m pip install beautifulsoup4 langchain-community"
        * PATH -m pip show beautifulsoup4 (Check if it worked)
    3. Run "PATH -m pip install sentence-transformers"
        * "PATH -m pip show sentence-transformers"
    4. Run "PATH -m pip install langgraph"
'''


# Load and split documents
all_splits = load_and_split("https://en.wikipedia.org/wiki/Boletus_edulis")

# Setup vector store
vector_manager = VectorStoreManager()
vector_manager.add_documents(all_splits)

# Load LLM and prompt
llm = init_llm()
prompt = load_prompt()

# Build graph
graph = build_graph(State, lambda s: retrieve(s, vector_manager), lambda s: generate(s, prompt, llm))

# Invoke
result = graph.invoke({"question": "Can you eat a porcini mushroom?"})
print(f"Context: {result['context']}\n\nAnswer: {result['answer']}")
