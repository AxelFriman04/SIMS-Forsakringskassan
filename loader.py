from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import RETRIEVAL_URL, CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split(url=RETRIEVAL_URL, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    loader = WebBaseLoader(web_paths=(url,))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    return splitter.split_documents(docs)
