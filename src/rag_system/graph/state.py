from dataclasses import dataclass, field
from typing import Any, Dict, List
import time

@dataclass
class GraphState:
    """
    Central runtime state for the RAG graph.
    Holds metrics and small snapshots for ingestion & retrieval.
    """
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    last_ingest_snapshot: Dict[str, Any] = field(default_factory=dict)
    last_retrieval_snapshot: Dict[str, Any] = field(default_factory=dict)

    def log_metric(self, metric: Dict[str, Any]):
        """
        Add a metric to the central metrics log with a timestamp.
        """
        metric_with_ts = {**metric, "ts": time.time()}
        self.metrics.append(metric_with_ts)
