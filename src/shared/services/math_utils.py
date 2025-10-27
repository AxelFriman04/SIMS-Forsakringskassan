import numpy as np


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between two embedding matrices.
    A: shape (n, d)
    B: shape (m, d)
    Returns matrix (n, m) of cosine similarities.
    """
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return np.dot(A_norm, B_norm.T)
