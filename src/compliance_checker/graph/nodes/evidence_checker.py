import math
import json
import re
from typing import List, Dict, Any, Literal
from pydantic import BaseModel
from shared.config import settings
from compliance_checker.prompt.prompt_loader import (
    load_entailment_prompt,
    load_entailment_batch_prompt,
    load_evidence_checker_system_prompt,
)
from compliance_checker.model.llm import LLM
from compliance_checker.services.vector_db_client import ComplianceVectorDB
from compliance_checker.graph.state import GraphState


class ClaimEntailment(BaseModel):
    label: Literal["entailment", "contradiction", "neutral"]
    confidence: float


class EvidenceCheckerNode:
    """
    Verifies factual claims against retrieved evidence chunks using LLM-based entailment checking.
    Supports multiple evidence citations per claim.
    Includes multi-stage fallback: batch → sub-batch → per-claim.
    """

    def __init__(self):
        self.llm = LLM(model=settings.ENTAILMENT_LLM_MODEL)
        self.vdb = ComplianceVectorDB()

        # Structured output schemas
        self.entailment_schema = {
            "name": "entailment_evaluation",
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ["entailment", "contradiction", "neutral"]},
                "confidence": {"type": "number"},
            },
            "required": ["label", "confidence"],
            "additionalProperties": False,
        }

        self.batch_schema = {
            "name": "entailment_batch",
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": self.entailment_schema,
                }
            },
            "required": ["claims"],
            "additionalProperties": False,
        }

    # --- Heuristic Accuracy check ---
    # TODO: Maybe move to evaluate RAG answer instead of extracted claims
    def heuristic_accuracy_score(self, claim_text: str, evidence_text: str) -> Dict[str, Any]:
        """Compute fast lexical & structural heuristics for factual reliability."""
        if not evidence_text.strip():
            return {"coverage": 0.0, "lexical_overlap": 0.0, "entity_match": 0.0,
                    "numeric_match": 0.0, "heuristic_confidence": 0.0}

        # --- Entity extraction ---
        words = re.findall(r"\b[A-ZÅÄÖ][a-zåäö]+\b", claim_text)
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', claim_text) if s.strip()]
        first_words = [s.split()[0] for s in sentences if s.split()]
        entities_claim = [w for w in words if w not in first_words]

        # --- Normalize text ---
        claim_tokens = re.findall(r"\w+", claim_text.lower())
        evidence_tokens = re.findall(r"\w+", evidence_text.lower())

        # --- Lexical overlap ---
        overlap = len(set(claim_tokens) & set(evidence_tokens)) / max(len(set(claim_tokens)), 1)

        # --- Entity match ---
        entity_hits = sum(1 for e in entities_claim if e.lower() in evidence_text.lower())
        entity_match = entity_hits / max(len(entities_claim), 1) if entities_claim else 1.0

        # --- Numerical match ---
        nums_claim = re.findall(r"\d+(?:[\.,]\d+)?", claim_text)
        num_hits = sum(1 for n in nums_claim if n in evidence_text)
        numeric_match = num_hits / max(len(nums_claim), 1) if nums_claim else 1.0

        # --- Coverage heuristic ---
        coverage = 1.0 if len(evidence_text.split()) > 0 else 0.0

        # --- Aggregate confidence ---
        heuristic_conf = round(0.25 * (coverage + overlap + entity_match + numeric_match), 3)

        return {
            "coverage": round(coverage, 3),
            "lexical_overlap": round(overlap, 3),
            "entity_match": round(entity_match, 3),
            "numeric_match": round(numeric_match, 3),
            "heuristic_confidence": heuristic_conf,
        }

    # ---------- DB Helper ----------
    def fetch_provenance_texts(self, claims: List[Dict[str, Any]]) -> Dict[str, str]:
        """Fetch all unique cited chunks from DB."""
        all_ids = {cid for c in claims for cid in c.get("citations", []) if cid}
        provenance_texts = {}
        for cid in all_ids:
            try:
                chunk = self.vdb.get_chunk_by_id(cid)
                if chunk and "text" in chunk:
                    provenance_texts[cid] = chunk["text"]
            except Exception as e:
                print(f"[WARN] Could not fetch chunk {cid}: {e}")
                provenance_texts[cid] = ""
        return provenance_texts

    # ---------- Core batch checker ----------
    def check_evidence_batch(self, claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs multi-evidence entailment checking for all claims in batch."""
        provenance_texts = self.fetch_provenance_texts(claims)

        # --- Compute heuristics ---
        for claim in claims:
            cited_texts = [provenance_texts.get(cid, "") for cid in claim.get("citations", [])]
            combined_text = "\n".join(cited_texts)
            claim["heuristics"] = self.heuristic_accuracy_score(claim["text"], combined_text)

        # --- If provenance completely failed, skip entailment ---
        if not provenance_texts:
            print("[WARN] No provenance texts found — defaulting to neutral results.")
            for claim in claims:
                claim["entailment"] = "neutral"
                claim["confidence"] = 0.0

            metrics = {
                "entailment_ratio": 0.0,
                "contradiction_ratio": 0.0,
                "avg_confidence": 0.0,
            }

            return {"claims": claims, "metrics": metrics}

        # --- Build evidence map: cid -> Evidence A/B/C...
        evidence_map = {cid: f"Evidence {chr(65 + i)}" for i, cid in enumerate(provenance_texts.keys())}
        evidence_texts = {label: provenance_texts[cid] for cid, label in evidence_map.items()}

        # --- Build claim–evidence pairs
        pairs = []
        for claim in claims:
            cited_ids = claim.get("citations", [])
            evidence_refs = [evidence_map[cid] for cid in cited_ids if cid in evidence_map]
            pairs.append({
                "hypothesis": claim["text"],
                "evidence_refs": evidence_refs,
            })

        # --- Step 1: Try full batch
        prompt = load_entailment_batch_prompt(evidence_texts, pairs)
        results = self._try_batch(prompt, self.batch_schema)

        # --- Step 2: Try smaller sub-batches if full batch failed
        if not results or len(results) != len(claims):
            if getattr(settings, "DEBUG", False):
                print(
                    f"[WARN] Full batch failed or incomplete ({len(results)}/{len(claims)}). "
                    "Retrying in smaller batches..."
                )

            chunk_size = max(3, math.ceil(len(claims) / 3))
            sub_results = []

            for i in range(0, len(claims), chunk_size):
                sub_claims = claims[i:i + chunk_size]

                # Build sub-evidence mapping and pairs
                sub_evidence = {}
                sub_pairs = []
                for c in sub_claims:
                    cited_ids = c.get("citations", [])
                    for cid in cited_ids:
                        if cid in evidence_map:
                            sub_evidence[evidence_map[cid]] = provenance_texts.get(cid, "")
                    sub_pairs.append({
                        "hypothesis": c["text"],
                        "evidence_refs": [evidence_map[cid] for cid in cited_ids if cid in evidence_map],
                    })

                sub_prompt = load_entailment_batch_prompt(sub_evidence, sub_pairs)
                sub_batch_out = self._try_batch(sub_prompt, self.batch_schema)

                if not sub_batch_out or len(sub_batch_out) != len(sub_claims):
                    sub_batch_out = [{"label": "neutral", "confidence": 0.0} for _ in sub_claims]
                    if getattr(settings, "DEBUG", False):
                        print(f"[WARN] Sub-batch {i // chunk_size + 1} failed — using neutral defaults.")

                sub_results.extend(sub_batch_out)

            results = sub_results

        # --- Step 3: Fallback to per-claim mode if still broken
        if not results or len(results) != len(claims):
            print("[WARN] All batch attempts failed — switching to per-claim fallback.")
            results = [self._check_single_claim(c, provenance_texts, evidence_map) for c in claims]

        # --- Step 4: Attach results to claims
        for claim, result in zip(claims, results):
            claim["entailment"] = result.get("label", "neutral")
            claim["confidence"] = float(result.get("confidence", 0.0))

        # --- Step 5: Compute metrics
        entailments = sum(1 for c in claims if c["entailment"] == "entailment")
        contradictions = sum(1 for c in claims if c["entailment"] == "contradiction")
        avg_conf = sum(c.get("confidence", 0.0) for c in claims) / len(claims) if claims else 0.0

        # Safely compute heuristic average (if any claim has it)
        heur_list = [c.get("heuristics", {}) for c in claims if c.get("heuristics")]
        avg_heur_conf = (
            sum(h.get("heuristic_confidence", 0.0) for h in heur_list) / len(heur_list)
            if heur_list else 0.0
        )

        metrics = {
            "entailment_ratio": entailments / len(claims) if claims else 0.0,
            "contradiction_ratio": contradictions / len(claims) if claims else 0.0,
            "avg_confidence": avg_conf,
            "heuristic_avg_confidence": round(avg_heur_conf, 3),
        }

        # --- Step 6: Debug
        if getattr(settings, "DEBUG", False):
            print("\n=== 🧩 Evidence Verification Summary ===")
            for i, claim in enumerate(claims, 1):
                ev_refs = ", ".join(claim.get("citations", []))
                print(f"{i}. {claim['text']} → {claim['entailment']} ({claim['confidence']:.2f}) [cites: {ev_refs}]")
            print("Metrics:", metrics)
            print("========================================")

        return {"claims": claims, "metrics": metrics}

    # ---------- Helpers ----------
    def _try_batch(self, prompt: str, schema: dict) -> List[Dict[str, Any]]:
        """Helper to run a structured LLM batch and safely parse JSON."""
        system_prompt = load_evidence_checker_system_prompt()
        try:
            response = self.llm.complete(prompt, structured=True, schema=schema, system_prompt=system_prompt)
            parsed = json.loads(response.get("answer", "{}"))
            return parsed.get("claims", [])
        except Exception as e:
            print(f"[ERROR] Batch attempt failed: {e}")
            return []

    def _check_single_claim(self, claim: Dict[str, Any], provenance_texts: Dict[str, str], evidence_map: Dict[str, str]) -> Dict[str, Any]:
        """Fallback: Check one claim using concatenated evidence text."""
        cited_ids = claim.get("citations", [])
        if not cited_ids:
            return {"label": "neutral", "confidence": 0.0}

        combined_evidence = "\n".join([provenance_texts.get(cid, "") for cid in cited_ids])
        if not combined_evidence.strip():
            return {"label": "neutral", "confidence": 0.0}

        prompt = load_entailment_prompt(combined_evidence, claim["text"])
        system_prompt = load_evidence_checker_system_prompt()

        try:
            response = self.llm.complete(prompt, structured=True, schema=self.entailment_schema, system_prompt=system_prompt)
            parsed = json.loads(response.get("answer", "{}"))
            return {
                "label": parsed.get("label", "neutral"),
                "confidence": float(parsed.get("confidence", 0.0)),
            }
        except Exception as e:
            print(f"[ERROR] Single claim entailment failed: {e}")
            return {"label": "neutral", "confidence": 0.0}

    # ---------- Graph Entry ----------
    def run(self, state: GraphState) -> GraphState:
        claims = state.claims
        if not claims:
            state.log_metric({
                "entailment_ratio": 0.0,
                "contradiction_ratio": 0.0,
                "avg_confidence": 0.0
            }, stage="verification")
            # log an empty heuristics block for consistency
            state.log_metric({}, stage="heuristics")
            return state

        result = self.check_evidence_batch(claims)
        state.verified_claims = result["claims"]
        # Add entailment model metadata to the metrics dict
        result_metrics = {
            **result["metrics"],
            "entailment_metadata": {
                "model": settings.ENTAILMENT_LLM_MODEL
            }
        }

        # Log metrics (with metadata)
        state.log_metric(result_metrics, stage="verification")
        # --- NEW: Aggregate + log heuristics across verified claims ---
        heuristics_list = [c.get("heuristics", {}) for c in state.verified_claims if c.get("heuristics")]
        if heuristics_list:
            avg_heur_conf = sum(h.get("heuristic_confidence", 0.0) for h in heuristics_list) / len(heuristics_list)
            avg_lexical = sum(h.get("lexical_overlap", 0.0) for h in heuristics_list) / len(heuristics_list)
            avg_entity = sum(h.get("entity_match", 0.0) for h in heuristics_list) / len(heuristics_list)
            avg_numeric = sum(h.get("numeric_match", 0.0) for h in heuristics_list) / len(heuristics_list)
            avg_coverage = sum(h.get("coverage", 0.0) for h in heuristics_list) / len(heuristics_list)

            state.log_metric({
                "avg_heuristic_confidence": round(avg_heur_conf, 3),
                "avg_lexical_overlap": round(avg_lexical, 3),
                "avg_entity_match": round(avg_entity, 3),
                "avg_numeric_match": round(avg_numeric, 3),
                "avg_coverage": round(avg_coverage, 3),
            }, stage="heuristics")
        else:
            # Keep shape stable even if no heuristics were computed
            state.log_metric({}, stage="heuristics")

        return state
