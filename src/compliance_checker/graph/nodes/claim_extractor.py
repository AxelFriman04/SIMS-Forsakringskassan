from __future__ import annotations
from typing import List, Dict, Any
from compliance_checker.graph.state import GraphState
from compliance_checker.model.llm import LLM
from compliance_checker.prompt.prompt_loader import load_claim_extraction_prompt
from compliance_checker.parser.claim_parser import parse_claims
from shared.services.embedding_service import embed_texts
from compliance_checker.services.vector_db_client import ComplianceVectorDB


class ClaimExtractorNode:
    """
    Extracts atomic factual claims from the RAG-generated answer,
    embeds them, retrieves matching evidence, and attaches citations.
    """

    def __init__(self, state: GraphState):
        self.state = state
        self.llm = LLM()
        self.vdb = ComplianceVectorDB()

    def run(self, answer_text: str, top_k: int = 1, dry_run: bool = False) -> GraphState:
        self.state.answer = answer_text

        # --- Step 1 Load the LLM prompt
        prompt = load_claim_extraction_prompt(answer_text)

        if dry_run:
            fake_claims = [
                {"text": "Karljohanssvamp forms mycorrhiza with trees.", "citation": {"id": "dummy_cite_1"}},
                {"text": "It grows in deciduous and coniferous forests.", "citation": {"id": "dummy_cite_2"}}
            ]
            self.state.claims = fake_claims
            self.state.log_metric({"claim_count": len(fake_claims), "source": "dry_run"})
            # return {"claims": fake_claims, "metadata": {"source": "dry_run"}}
            return self.state

        # --- Step 2 Ask LLM to extract claims
        response = self.llm.complete(prompt, max_tokens=8000)
        raw_output = response.get("answer", "").strip()

        # --- Step 3 Parse atomic claims
        claims = parse_claims(raw_output)
        if not claims:
            self.state.claims = []
            self.state.log_metric({"claim_count": 0, "warning": "No claims parsed"})
            # return {"claims": [], "metadata": {"error": "No claims found"}}
            return self.state

        # --- Step 4 Batch embed all claims
        claim_texts = [c["text"] for c in claims]
        embeddings = embed_texts(claim_texts)

        # --- Step 5 Batch query DB for all embeddings
        batch_results = self.vdb.search_claims_batch(embeddings, top_k=top_k)

        # --- Step 6 Combine claims with their best citation
        # TODO: Potentially incorporate spaCy to make a more robust
        #  citation match, embedd rag sentences and claims and do cosine similarity between them
        combined_claims = []
        for claim_text, hits in zip(claim_texts, batch_results):
            best_hit = hits[0] if hits else None
            citation = {
                "id": best_hit.get("chunk_id"),
                "score": best_hit.get("score"),
                "source": best_hit.get("source")
            } if best_hit else None

            combined_claims.append({
                "text": claim_text,
                "citation": citation
            })

        # --- Step 7 Update state
        self.state.claims = combined_claims
        self.state.log_metric({"claim_count": len(combined_claims)})
        # TODO: Change to store metrics in the given metrics variable in state.

        """return {
            "claims": combined_claims,
            "metadata": {
                "num_claims": len(combined_claims),
                "num_citations_found": sum(1 for c in combined_claims if c["citation"]),
            },
        }"""
        return self.state
