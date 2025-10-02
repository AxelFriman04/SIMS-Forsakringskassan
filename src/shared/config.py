import os

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class Settings(BaseSettings):
    VECTOR_DB_URL: str = os.getenv("VECTOR_DB_URL")  # Qdrant default
    VECTOR_DB_COLLECTION: str = os.getenv("VECTOR_DB_COLLECTION")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    EMBED_DIM: int = 1536  # for the text-embedding-3-small model
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-5-nano"
    TOP_K: int = 5

    USE_DUMMY_LLM: bool = True
    USE_DUMMY_EMBEDDINGS: bool = True

    # === Run control flags ===
    RUN_INGEST: bool = False         # Only set to True when you want to parse + embed + upsert PDF
    RUN_RETRIEVE: bool = True
    RUN_GENERATE: bool = True

    # === Paths ===
    PDF_PATH: str = "shared/test.pdf"

    class Config:
        env_file = ".env"

settings = Settings()
