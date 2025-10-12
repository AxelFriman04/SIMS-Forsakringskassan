# compliance_checker/graph/state.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class GraphState:
    """
    Holds state throughout the compliance checking pipeline.
    Each node (Claim Extraction, Verification, Scoring, etc.)
    will update or extend this state as the pipeline runs.
    """

    # The original LLM-generated answer to be checked
    answer: str = ""

    # Extracted factual claims, each with optional linked citations
    claims: List[Dict[str, Any]] = field(default_factory=list)

    # For later steps (optional for now)
    verified_claims: List[Dict[str, Any]] = field(default_factory=list)
    compliance_score: Optional[float] = None

    # Metrics for internal logging and debugging
    metrics_claim_extraction: Dict[str, Any] = field(default_factory=dict)
    metrics_verification: Dict[str, Any] = field(default_factory=dict)

    # Root Cause Classifier
    root_cause: Dict[str, Any] = field(default_factory=dict)
    verdict: str = ""
    metrics_root_cause: Dict[str, Any] = field(default_factory=dict)
    metrics_pipeline: Dict[str, Any] = field(default_factory=dict)

    def log_metric(self, metric: Dict[str, Any], stage: str = "claim_extraction"):
        """Adds metrics to the correct stage dictionary safely."""
        stage_attr = f"metrics_{stage}"
        if not hasattr(self, stage_attr):
            # Create stage dictionary if it doesn't exist
            setattr(self, stage_attr, {})
        stage_dict = getattr(self, stage_attr)
        stage_dict.update(metric)
