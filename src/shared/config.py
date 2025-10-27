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
    EMBED_DIM: int = 3072  # 1536 for the text-embedding-3-small model, 3072 for the large
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    LLM_MODEL: str = "gpt-5-nano"
    TOP_K: int = 5
    RERANK_MODEL: str = "gpt-5-nano"
    RERANK_THRESHOLD: int = 3
    PDF_LANG_IS_SWE: bool = True  # TODO: Add user set language support (Used only in prompt so far)
    USE_LOGPROBS: bool = True

    # === Ingestion settings ===
    MAX_CHUNK_SIZE: int = 1800
    OVERLAP_CHARS: int = 250

    USE_DUMMY_LLM: bool = False
    USE_DUMMY_EMBEDDINGS: bool = False

    # === Run control flags ===
    RUN_INGEST: bool = False         # Only set to True when you want to parse + embed + upsert PDF
    # RUN_RETRIEVE: bool = False
    RUN_GENERATE: bool = True
    USE_RE_RANK: bool = True

    """
        Compliance checker configs
    """
    ENTAILMENT_LLM_MODEL: str = "gpt-5-nano"
    STYLE_EVAL_MODEL: str = "gpt-5-nano"

    """
        Base settings
    """

    # Base directory (your project root)
    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent  # adjust if needed

    # === Paths ===
    PDF_PATH: str = str(BASE_DIR / "sjukpenning-rehabilitering-rehabiliteringsersattning-vagledning-2025-1.pdf")
    DB_PATH: str = str(BASE_DIR / "data/rag_results.db")

    # === Debug outputs ===
    DEBUG: bool = False

    class Config:
        env_file = ".env"


settings = Settings()
