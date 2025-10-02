# Define typed message shapes between nodes (helpful for testing and type-checking)
from typing import TypedDict, List, Dict, Any


class IngestOutput(TypedDict):
    doc_id: str
    chunks: List[Dict[str, Any]]  # {chunk_id, text, metadata}


class RetrievalOutput(TypedDict):
    query: str
    topk: List[Dict[str, Any]]  # {chunk_id, score, text, metadata}


class GenerationOutput(TypedDict):
    answer: str
    declared_citations: List[str]
    generator_metadata: Dict[str, Any]
