from typing import List
from openai import OpenAI
from shared.config import settings
import hashlib


client = OpenAI(api_key=settings.OPENAI_API_KEY)


def embed_texts(texts: List[str]) -> List[List[float]]:
    if settings.USE_DUMMY_EMBEDDINGS:
        # Fast, deterministic fake embeddings matching EMBED_DIM
        def deterministic_vector(s: str, dim=settings.EMBED_DIM):
            h = hashlib.md5(s.encode()).digest()  # 16 bytes
            v = [b / 255.0 for b in h]
            # Repeat the vector until reaching required dim
            repeated = (v * (dim // len(v) + 1))[:dim]
            return repeated

        return [deterministic_vector(t) for t in texts]

    """
    Get embeddings from OpenAI for a list of texts.
    """
    if not texts:
        return []

    response = client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=texts
    )

    return [d.embedding for d in response.data]
