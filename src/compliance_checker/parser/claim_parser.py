# compliance_checker/parser/claim_parser.py

import json
from typing import List, Dict


def parse_claims(raw_output: str) -> List[Dict[str, str]]:
    """
    Parses raw LLM output into structured claims.
    Attempts JSON parsing first, falls back to line-splitting heuristics.
    """
    if not raw_output.strip():
        return []

    # Try to parse as JSON
    try:
        data = json.loads(raw_output)
        claims = [{"text": c["text"].strip()} for c in data if isinstance(c, dict) and "text" in c and c["text"].strip()]
        return claims
    except json.JSONDecodeError:
        pass

    # Fallback: split by newlines or dashes
    lines = [ln.strip("-• ").strip() for ln in raw_output.split("\n") if ln.strip()]
    return [{"text": line} for line in lines if line]
