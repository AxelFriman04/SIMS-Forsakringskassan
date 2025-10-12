from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class GraphState:
    """
    Central runtime state for the RAG graph.
    Holds metrics and snapshots for ingestion, retrieval, and generation.
    """
    last_ingest_snapshot: Dict[str, Any] = field(default_factory=dict)
    last_retrieval_snapshot: Dict[str, Any] = field(default_factory=dict)
    last_generation_snapshot: Dict[str, Any] = field(default_factory=dict)
    metrics_ingestion: Dict[str, Any] = field(default_factory=dict)
    metrics_retrieval: Dict[str, Any] = field(default_factory=dict)
    metrics_generation: Dict[str, Any] = field(default_factory=dict)
    query: str = ""
    answer: str = ""

    def log_metric(self, metric: Dict[str, Any]):
        """
        Add metrics into the appropriate field.
        """
        metric_type = metric.get("type")
        if metric_type == "ingest":
            self.metrics_ingestion.update(metric)
        elif metric_type == "retrieval":
            self.metrics_retrieval.update(metric)
        elif metric_type == "generation":
            self.metrics_generation.update(metric)
