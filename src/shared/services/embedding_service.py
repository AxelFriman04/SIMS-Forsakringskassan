from typing import List
from openai import OpenAI
from shared.config import settings
import hashlib

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    if settings.USE_DUMMY_EMBEDDINGS:
        # Deterministic fake embeddings
        def deterministic_vector(s: str, dim=settings.EMBED_DIM):
            h = hashlib.md5(s.encode()).digest()  # 16 bytes
            v = [b / 255.0 for b in h]
            repeated = (v * (dim // len(v) + 1))[:dim]
            return repeated

        return [deterministic_vector(t) for t in texts]

    # Remove empty strings
    valid_texts = [t for t in texts if t.strip()]
    if not valid_texts:
        return [[] for _ in texts]

    response = client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=valid_texts
    )

    embeddings = [d.embedding for d in response.data]

    # Sanity check
    for i, emb in enumerate(embeddings):
        if len(emb) != settings.EMBED_DIM:
            raise ValueError(
                f"Embedding length mismatch for input {i}: expected {settings.EMBED_DIM}, got {len(emb)}"
            )

    return embeddings
