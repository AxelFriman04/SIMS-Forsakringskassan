from __future__ import annotations
from typing import List, Dict, Any
import re
import numpy as np
import json

from compliance_checker.graph.state import GraphState
from compliance_checker.model.llm import LLM
from compliance_checker.prompt.prompt_loader import load_claim_extraction_prompt, load_claim_extractor_system_prompt
from shared.services.embedding_service import embed_texts
from shared.services.math_utils import cosine_similarity_matrix


class ClaimExtractorNode:
    """
    Extracts atomic factual claims from the RAG-generated answer,
    aligns them with relevant segments, and attaches citations.
    """

    def __init__(self, state: GraphState):
        self.state = state
        self.llm = LLM()

        # Structured schema for extraction
        self.claim_schema = {
            "name": "claim_extractor",
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "A single verifiable factual claim"}
                        },
                        "required": ["text"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["claims"],
            "additionalProperties": False
        }

    def _split_answer_into_segments(self, answer_text: str) -> List[Dict[str, Any]]:
        """
        Splits the RAG answer into parts based on [CITE: ...] markers.
        Each segment includes the text span and its associated cite IDs.
        """
        pattern = re.compile(r"(.*?)(?:\[CITE:\s*([^\]]+)\])+", re.DOTALL)
        segments = []
        for match in re.finditer(pattern, answer_text):
            text_part = match.group(1).strip()
            cite_ids = [c.strip() for c in match.captures(2)] if hasattr(match, "captures") else [match.group(2)]
            if text_part:
                segments.append({"segment_text": text_part, "citations": cite_ids})
        return segments or [{"segment_text": answer_text.strip(), "citations": []}]

    def run(self, answer_text: str, dry_run: bool = False) -> GraphState:
        self.state.answer = answer_text

        if dry_run:
            dummy_claims = [
                {"text": "Karljohanssvamp bildar mykorrhiza med träd.", "citation": {"id": "chunk1"}},
                {"text": "Den växer i både löv- och barrskog.", "citation": {"id": "chunk2"}}
            ]
            self.state.claims = dummy_claims
            self.state.log_metric({"claim_count": len(dummy_claims), "source": "dry_run"})
            return self.state

        # --- Step 1: Prepare prompt
        prompt = load_claim_extraction_prompt(answer_text)
        system_prompt = load_claim_extractor_system_prompt()

        # --- Step 2: Extract claims using structured output
        try:
            response = self.llm.complete(prompt, structured=True, schema=self.claim_schema, system_prompt=system_prompt)
            parsed = json.loads(response.get("answer", "{}"))
            claims = parsed.get("claims", [])
        except Exception as e:
            print(f"[ERROR] Claim extraction failed: {e}")
            claims = []

        if not claims:
            self.state.claims = []
            self.state.log_metric({"claim_count": 0, "warning": "No claims parsed"})
            return self.state

        # --- Step 3: Split answer into segments
        segments = self._split_answer_into_segments(answer_text)

        # --- Step 4: Embed claims and segments
        claim_texts = [c["text"] for c in claims]
        segment_texts = [s["segment_text"] for s in segments]
        claim_embeds = np.array(embed_texts(claim_texts))
        segment_embeds = np.array(embed_texts(segment_texts))

        # --- Step 5: Compute cosine similarity
        sim_matrix = cosine_similarity_matrix(claim_embeds, segment_embeds)

        # --- Step 6: Assign each claim to the most similar segment
        combined_claims = []
        for i, claim in enumerate(claims):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_score = float(sim_matrix[i][best_idx])
            matched_segment = segments[best_idx]

            combined_claims.append({
                "text": claim["text"],
                "match_score": round(best_score, 3),
                "matched_segment": matched_segment["segment_text"],
                "citations": matched_segment["citations"]
            })

        # --- Step 7: Update state
        self.state.claims = combined_claims
        self.state.log_metric({
            "claim_count": len(combined_claims),
            "avg_match_score": float(np.mean([c["match_score"] for c in combined_claims]))
        })

        return self.state