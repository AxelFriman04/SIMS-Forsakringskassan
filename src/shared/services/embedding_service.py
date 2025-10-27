from typing import List
from openai import OpenAI
from shared.config import settings
import hashlib
import math

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def _deterministic_vector(s: str, dim: int) -> List[float]:
    """Creates a reproducible fake embedding vector for testing."""
    h = hashlib.md5(s.encode()).digest()
    v = [b / 255.0 for b in h]
    repeated = (v * (dim // len(v) + 1))[:dim]
    return repeated


def embed_texts(texts: List[str], batch_size: int = 1000) -> List[List[float]]:
    """Embed a list of texts using OpenAI’s embedding API with optional batching."""
    if not texts:
        return []

    # --- Dummy mode for offline testing ---
    if getattr(settings, "USE_DUMMY_EMBEDDINGS", False):
        return [_deterministic_vector(t, settings.EMBED_DIM) for t in texts]

    # Remove empty strings safely
    valid_texts = [t for t in texts if t.strip()]
    if not valid_texts:
        return [[] for _ in texts]

    # --- Batch processing to avoid API size limits ---
    all_embeddings = []
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i + batch_size]
        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=batch
        )
        all_embeddings.extend([d.embedding for d in response.data])

    # --- Auto-adjust EMBED_DIM dynamically (if not already correct) ---
    actual_dim = len(all_embeddings[0]) if all_embeddings else 0
    if actual_dim != getattr(settings, "EMBED_DIM", actual_dim):
        print(f"[INFO] Auto-updating settings.EMBED_DIM to {actual_dim}")
        settings.EMBED_DIM = actual_dim

    return all_embeddings
