import os

from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path
from typing import ClassVar

# Load environment variables from .env
load_dotenv()


class Settings(BaseSettings):

    """
        RAG configs
    """

    VECTOR_DB_URL: str = os.getenv("VECTOR_DB_URL")
    VECTOR_DB_COLLECTION: str = os.getenv("VECTOR_DB_COLLECTION")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    EMBED_DIM: int = 1536  # for the text-embedding-3-small model
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-5-nano"
    TOP_K: int = 5

    USE_DUMMY_LLM: bool = False
    USE_DUMMY_EMBEDDINGS: bool = False

    # === Run control flags ===
    RUN_INGEST: bool = True         # Only set to True when you want to parse + embed + upsert PDF
    RUN_RETRIEVE: bool = True
    RUN_GENERATE: bool = True
    USE_RE_RANK: bool = True

    """
        Compliance checker configs
    """
    ENTAILMENT_LLM_MODEL: str = "gpt-5-nano"

    """
        Base settings
    """

    # Base directory (your project root)
    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent  # adjust if needed

    # === Paths ===
    PDF_PATH: str = str(BASE_DIR / "Boletus_edulis.pdf")
    DB_PATH: str = str(BASE_DIR / "data/rag_results.db")

    # === Debug outputs ===
    DEBUG: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
