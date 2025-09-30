from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from config import EMBEDDING_MODEL

class VectorStoreManager:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = InMemoryVectorStore(self.embeddings)

    def add_documents(self, documents):
        return self.vector_store.add_documents(documents=documents)

    def similarity_search(self, query):
        return self.vector_store.similarity_search(query)
