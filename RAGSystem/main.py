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
prompt = load_prompt("""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:""")

# Build graph
graph = build_graph(State, lambda s: retrieve(s, vector_manager), lambda s: generate(s, prompt, llm))

# Invoke
result = graph.invoke({"question": "What is the serial number of the laptop I used last week?"})
print(f"Context: {result['context']}\n\nAnswer: {result['answer']}")
