import math
import json
from typing import List, Dict, Any, Literal
from pydantic import BaseModel
from shared.config import settings
from compliance_checker.prompt.prompt_loader import (
    load_entailment_prompt,
    load_entailment_batch_prompt_v2,
)
from compliance_checker.model.llm import LLM
from compliance_checker.services.vector_db_client import ComplianceVectorDB
from compliance_checker.graph.state import GraphState


class ClaimEntailment(BaseModel):
    # label: str # ->  later if we want to have more support for other classifications
    label: Literal["entailment", "contradiction", "neutral"]
    confidence: float


class EvidenceCheckerNode:
    def __init__(self):
        self.llm = LLM(model=settings.ENTAILMENT_LLM_MODEL)
        self.vdb = ComplianceVectorDB()

        print(f"[DEBUG] EvidenceChecker using model: {settings.ENTAILMENT_LLM_MODEL}")

        self.entailment_schema = {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ["entailment", "contradiction", "neutral"]},
                "confidence": {"type": "number"},
            },
            "required": ["label", "confidence"],
            "additionalProperties": False,
        }

        self.batch_schema = {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": self.entailment_schema
                }
            },
            "required": ["claims"],
            "additionalProperties": False
        }

    # ---------- DB Helpers ----------
    def fetch_provenance_texts(self, claims: List[Dict[str, Any]]) -> Dict[str, str]:
        unique_ids = {c["citation"]["id"] for c in claims if c.get("citation")}
        provenance_texts = {}

        for cid in unique_ids:
            try:
                chunk = self.vdb.get_chunk_by_id(cid)  # TODO: Finish the database lookup and retrieval (Fixed I think)
                if chunk:
                    provenance_texts[cid] = chunk.get("text")
            except Exception:
                provenance_texts[cid] = ""
        return provenance_texts

    # ---------- Single Pair ----------
    def check_evidence(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        prompt = load_entailment_prompt(premise, hypothesis)
        # print(f"\n[DEBUG] Single claim prompt for hypothesis:\n{hypothesis}\nPremise:\n{premise}\n")
        try:
            response = self.llm.complete(
                prompt,
                structured=True,
                schema=self.entailment_schema,
            )
            # Parse the JSON string returned in 'answer'
            answer_str = response.get("answer", "{}")
            answer_obj = json.loads(answer_str)
            result = answer_obj.get("claims", [])
            # result = response.get("answer", {})
            return {
                "label": result.get("label", "neutral"),
                "confidence": float(result.get("confidence", 0.0)),
            }
        except Exception as e:
            return {"label": "neutral", "confidence": 0.0}

    # ---------- Batch ----------
    def check_evidence_batch(self, claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        provenance_texts = self.fetch_provenance_texts(claims)
        # print(f"[DEBUG] Fetched provenance texts for {len(provenance_texts)} chunks: \n{provenance_texts}\n")
        if not provenance_texts:
            print("If not provenance_texts: True")
            return {
                "claims": [
                    {**c, "entailment": "neutral", "confidence": 0.0} for c in claims
                ],
                "metrics": {"entailment_ratio": 0.0, "contradiction_ratio": 0.0, "avg_confidence": 0.0},
            }

        # --- Build evidence map and pairs ---
        # TODO: Works for up to 26 claims, okay for now but implement a more robust system down the line
        evidence_map = {cid: f"Evidence {chr(65+i)}" for i, cid in enumerate(provenance_texts.keys())}
        evidence_texts = {evidence_map[cid]: text for cid, text in provenance_texts.items()}

        pairs = []
        for claim in claims:
            citation = claim.get("citation")
            evidence_ref = evidence_map.get(citation["id"]) if citation else None
            pairs.append({"evidence_ref": evidence_ref, "hypothesis": claim["text"]})

        # --- Step 1: Try full batch
        try:
            prompt = load_entailment_batch_prompt_v2(evidence_texts, pairs)
            # print(f"\n[DEBUG] Batch prompt for {len(pairs)} claims:\n{prompt}\n")
            response = self.llm.complete(prompt, structured=True, schema=self.batch_schema)
            print("Full batch response:\n\n")
            print(response)
            # Parse the JSON string returned in 'answer'
            answer_str = response.get("answer", "{}")
            answer_obj = json.loads(answer_str)
            results = answer_obj.get("claims", [])
        except Exception as e:
            print(f"[ERROR] Full batch failed: {e}")
            results = []

        # --- Step 2: Try smaller batches if full batch failed
        if not results or len(results) != len(claims):
            if getattr(settings, "DEBUG", False):
                print(
                    f"[WARN] Full batch failed or incomplete ({len(results)}/{len(claims)}). Retrying in smaller "
                    f"batches...")

            chunk_size = max(3, math.ceil(len(claims) / 3))
            sub_results = []

            for i in range(0, len(claims), chunk_size):
                sub_claims = claims[i: i + chunk_size]

                # Map sub-claims to evidence
                sub_evidence = {}
                sub_pairs = []
                for c in sub_claims:
                    citation = c.get("citation", {})
                    cid = citation.get("id")
                    if cid and cid in evidence_map:
                        sub_evidence[evidence_map[cid]] = provenance_texts.get(cid, "")
                    sub_pairs.append({
                        "evidence_ref": evidence_map.get(cid),
                        "hypothesis": c["text"]
                    })

                sub_prompt = load_entailment_batch_prompt_v2(sub_evidence, sub_pairs)
                # print("Prompt for sub_batches: ", sub_prompt)
                try:
                    sub_response = self.llm.complete(sub_prompt, structured=True, schema=self.batch_schema)
                    print("Sub batch response:\n\n")
                    print(sub_response)
                    # batch_out = sub_response.get("answer", [])
                    # batch_out = sub_response.get("answer", {}).get("claims", [])

                    # Parse the JSON string returned in 'answer'
                    answer_str = sub_response.get("answer", "{}")
                    answer_obj = json.loads(answer_str)
                    batch_out = answer_obj.get("answer", [])
                except Exception as e:
                    print(f"[ERROR] Sub-batch {i // chunk_size + 1} failed: {e}")
                    batch_out = []

                if not batch_out or len(batch_out) != len(sub_claims):
                    print("If not batch_out: True")
                    batch_out = [{"label": "neutral", "confidence": 0.0} for _ in sub_claims]
                    if getattr(settings, "DEBUG", False):
                        print(f"[WARN] Sub-batch {i // chunk_size + 1} failed — using neutral defaults.")

                sub_results.extend(batch_out)

            results = sub_results

        # --- Step 3: Fallback to per-claim mode if still broken
        if not results or len(results) != len(claims):
            print("[WARN] All batch attempts failed — switching to per-claim fallback.")
            results = []
            for idx, claim in enumerate(claims):
                citation = claim.get("citation")
                premise = provenance_texts.get(citation["id"]) if citation and citation.get(
                    "id") in provenance_texts else ""
                res = self.check_evidence(premise, claim["text"])
                print(res)
                results.append(res)

            if getattr(settings, "DEBUG", False):
                print(f"[INFO] Completed per-claim fallback for {len(results)} claims.")

        # --- Step 4: Attach results to claims
        for claim, result in zip(claims, results):
            claim["entailment"] = result.get("label", "neutral")
            claim["confidence"] = result.get("confidence", 0.0)

        # --- Step 5: Compute metrics
        entailments = sum(1 for c in claims if c["entailment"] == "entailment")
        contradictions = sum(1 for c in claims if c["entailment"] == "contradiction")
        avg_conf = sum(c["confidence"] for c in claims) / len(claims) if claims else 0.0

        metrics = {
            "entailment_ratio": entailments / len(claims) if claims else 0.0,
            "contradiction_ratio": contradictions / len(claims) if claims else 0.0,
            "avg_confidence": avg_conf,
        }

        # --- Step 6: Optional debug output
        if getattr(settings, "DEBUG", False):
            print("\n=== 🧠 Evidence Checking Summary ===")
            for i, claim in enumerate(claims, 1):
                citation = claim.get("citation")
                evidence_id = citation.get("id") if citation else None
                evidence_text = provenance_texts.get(evidence_id, "[no evidence]")
                entailment = claim.get("entailment", "neutral")
                conf = claim.get("confidence", 0.0)

                print(f"\nClaim {i}: {claim['text']}")
                print(f" → Evidence source: {evidence_id or 'N/A'}")
                # print(f" → Evidence snippet: {evidence_text[:180].strip()}...")
                print(f" → Entailment: {entailment} (conf={conf: .2f})")

            print("\n--- Metrics ---")
            for k, v in metrics.items():
                print(f"{k}: {v: .3f}")
            print("===============================\n")

        return {"claims": claims, "metrics": metrics}

    # ---------- Graph Entry ----------
    def run(self, state: GraphState) -> GraphState:
        claims = state.claims
        if not claims:
            state.log_metric({
                "entailment_ratio": 0.0,
                "contradiction_ratio": 0.0,
                "avg_confidence": 0.0
            }, stage="verification")
            return state

        result = self.check_evidence_batch(claims)
        state.verified_claims = result["claims"]
        state.log_metric(result["metrics"], stage="verification")
        return state
