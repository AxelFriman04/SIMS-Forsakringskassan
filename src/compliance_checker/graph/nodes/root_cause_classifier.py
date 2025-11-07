# compliance_checker/graph/nodes/root_cause_classifier.py
import json
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from shared.config import settings
from compliance_checker.graph.state import GraphState


class RootCauseClassifierNode:
    """
    Scores each pipeline stage and synthesizes a compliance verdict
    with prioritized root causes and actionable recommendations.
    """

    def __init__(self, state: GraphState | None = None):
        self.state = state

    # ---------- DB ----------
    def _connect(self):
        conn = sqlite3.connect(settings.DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _safe_json(self, v: Optional[str]) -> Dict[str, Any]:
        if not v:
            return {}
        try:
            return json.loads(v)
        except Exception:
            return {}

    def _fetch_metrics_from_db(self, answer_id: int) -> Dict[str, Any]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                  metrics_ingestion,
                  metrics_retrieval,
                  metrics_generation,
                  metrics_style,
                  metrics_relevance,
                  metrics_compliance,
                  compliance_report,
                  answer
                FROM results
                WHERE id = ?
                """,
                (answer_id,),
            )
            row = cur.fetchone()

        if not row:
            print(f"[WARN] No record found for answer_id={answer_id}")
            return {}

        return {
            "ingestion": self._safe_json(row["metrics_ingestion"]),
            "retrieval": self._safe_json(row["metrics_retrieval"]),
            "generation": self._safe_json(row["metrics_generation"]),
            "style": self._safe_json(row["metrics_style"]),
            "relevance": self._safe_json(row["metrics_relevance"]),
            "compliance": self._safe_json(row["metrics_compliance"]),
            "answer": row["answer"] or "",
        }

    # ---------- Utilities ----------
    # TODO: Move to math_utils
    @staticmethod
    def _nz(val: Optional[float], default: float = 0.0) -> float:
        try:
            return float(val)
        except Exception:
            return default

    @staticmethod
    def _bounded(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _score_completion_coverage(self, metrics: Dict[str, Any]) -> tuple[float, list[str]]:
        """
        Evaluates whether the answer sufficiently covers the query and evidence base.
        Uses entailment ratio, retrieval coverage, and extraction completeness.
        """
        issues = []
        compl = metrics.get("compliance", {}) or {}
        ver = compl.get("verification", {}) or {}
        extr = compl.get("claim_extraction", {}) or {}
        retrieval = metrics.get("retrieval", {}) or {}
        ingestion = metrics.get("ingestion", {}) or {}

        entailment = self._nz(ver.get("entailment_ratio"), 0.0)
        claim_count = int(extr.get("claim_count") or 0)
        retrieval_sources = int(retrieval.get("distinct_source_count") or 0)
        chunk_cov = self._nz(ingestion.get("chunk_coverage_pct"), 0.0) / 100.0

        score = self._bounded(
            0.5 * entailment +
            0.2 * min(claim_count / 10.0, 1.0) +
            0.15 * min(retrieval_sources / 3.0, 1.0) +
            0.15 * chunk_cov
        )

        if entailment < 0.5:
            issues.append("Low entailment ratio reduces answer coverage.")
        if claim_count < 3:
            issues.append("Few extracted claims — incomplete factual coverage.")
        if retrieval_sources < 2:
            issues.append("Limited retrieval diversity — potential gaps in evidence.")
        if chunk_cov < 0.8:
            issues.append("Low document chunk coverage — missing context during reasoning.")

        return score, issues

    def _score_consistency(self, metrics: Dict[str, Any], stage_scores: Dict[str, float]) -> tuple[float, list[str]]:
        """
        Checks overall logical and factual consistency:
        - Contradiction rate
        - Variance across stage scores
        - Stability of entailment confidence
        """
        issues = []
        ver = (metrics.get("compliance", {}) or {}).get("verification", {}) or {}

        contra = self._nz(ver.get("contradiction_ratio"), 0.0)
        conf = self._nz(ver.get("avg_confidence"), 0.0)
        heur = self._nz(ver.get("heuristic_avg_confidence"), 0.0)

        # Stage consistency variance
        scores = list(stage_scores.values())
        mean_s = self._mean(scores)
        var = self._mean([(s - mean_s) ** 2 for s in scores])
        stage_consistency = self._bounded(1.0 - min(var * 2.5, 1.0))

        score = self._bounded(0.5 * (1.0 - contra) + 0.25 * conf + 0.25 * stage_consistency)

        if contra > 0.2:
            issues.append("Detected internal contradictions among verified claims.")
        if stage_consistency < 0.7:
            issues.append("Inconsistent performance across pipeline stages.")
        if conf < 0.6:
            issues.append("Low entailment confidence indicates unstable reasoning.")

        return score, issues

    # ---------- Stage scoring ----------
    # TODO: Use real values instead of placeholder
    def _score_ingestion(self, m: Dict[str, Any]) -> Tuple[float, List[str]]:
        issues = []
        psr = self._nz(m.get("parsing_success_rate"), 1.0)          # [0..1]
        meta = self._nz(m.get("metadata_completeness"), 1.0)        # [0..1]
        emb = self._nz(m.get("embedding_fidelity"), 0.7)            # heuristic ~[0.3..0.95]
        cov = self._nz(m.get("chunk_coverage_pct"), 0.0) / 100.0    # [%] -> [0..1]

        # Weighted score
        score = self._bounded(0.35*psr + 0.25*meta + 0.25*emb + 0.15*cov)

        if psr < 0.9:  issues.append("Parsing success below 90% — possible extraction gaps.")
        if meta < 0.9: issues.append("Metadata completeness below 90% — harder to trace provenance.")
        if emb < 0.65: issues.append("Low embedding fidelity — retrieval relevance may degrade.")
        if cov < 0.8:  issues.append("Chunk coverage below 80% — some pages may be underrepresented.")

        return score, issues

    def _score_retrieval(self, m: Dict[str, Any]) -> Tuple[float, List[str]]:
        issues = []
        topk = m.get("topk_scores") or []
        max_score = max(topk) if topk else 0.0
        mean_topk = self._mean(topk)
        gap = self._nz(m.get("topk_gap"), 0.0)
        distinct = int(m.get("distinct_source_count") or 0)
        overlap = self._nz(m.get("lexical_overlap"), 0.0)
        rerank = self._nz(m.get("re_rank_delta"), 0.0)
        num_hits = int(m.get("num_hits") or 0)

        # ---- Core scoring components ----
        focus = gap / (gap + 0.25)  # S-curve-ish; 0.25 tunes knee
        comp = [
            self._bounded(max_score),               # peak match strength
            self._bounded(mean_topk),               # general retrieval quality
            self._bounded(overlap),                 # lexical / semantic alignment
            self._bounded(focus),                   # reward larger gap
            self._bounded(min(distinct, 3) / 3.0),  # source diversity
            self._bounded(min(num_hits, 5) / 5.0),  # penalize zero results
        ]
        base_score = self._bounded(self._mean(comp))

        # ---- Confidence boost/penalty based on max_score ----
        # Range: [−0.2 .. +0.1]
        if max_score >= 0.7:
            boost_factor = 1.10  # strong top hit
        elif max_score < 0.4:
            boost_factor = 0.80  # weak top hit
        else:
            # interpolate linearly between 0.4–0.7 → 0.8–1.1
            boost_factor = 0.8 + (max_score - 0.4) * (0.3 / 0.3)
        score = self._bounded(base_score * boost_factor)

        # ---- Diagnostics ----
        if distinct < 2: issues.append("Low source diversity in retrieval.")
        if overlap < 0.20:
            issues.append(
                "Low lexical overlap between query and retrieved evidence — retriever may not fully understand query intent.")
        if overlap > 0.85:
            issues.append(
                "High lexical overlap — retrieval may be overly keyword-based rather than semantically diverse.")
        if max_score < 0.5:
            issues.append(
                "Low maximum retrieval score (< 0.5) — retrieved evidence may not be semantically relevant; "
                "the source material may lack sufficient information for this query."
            )

        if gap < 0.15:
            issues.append("Low top-k score gap — retriever may lack focus; important data might be missing.")
        if num_hits == 0 or not topk:
            return 0.0, ["No retrieval hits — answer may be unsupported."]
        if rerank > 0.1:
            issues.append("Large re-ranking adjustment — consider relying more on re-ranker or tuning base similarity.")

        return score, issues

    def _score_generation(self, m: Dict[str, Any]) -> Tuple[float, List[str]]:
        issues = []
        toks = (m.get("token_logprob_stats") or {})
        avg_lp = toks.get("avg")
        min_lp = toks.get("min")
        hall = m.get("preliminary_hallucinations_warnings") or []
        ans_len = int(m.get("answer_length") or 0)

        # Heuristic: penalize if explicit hallucination flags
        base = 0.85
        if hall: base -= 0.15
        # very negative min logprob might indicate brittle spans (if available)
        if isinstance(min_lp, (int, float)) and min_lp is not None and min_lp < -2.5:
            base -= 0.05

        score = self._bounded(base)

        if hall:
            issues.append("Potential hallucinations flagged during generation.")
        if isinstance(avg_lp, (int, float)) and avg_lp is not None and avg_lp < -1.5:
            issues.append("Low average token confidence in generation.")

        return score, issues

    def _score_style(self, m: Dict[str, Any]) -> Tuple[float, List[str]]:
        issues = []
        if not m:
            return 0.5, ["No style evaluation captured."]

        clarity = self._nz(m.get("clarity_score"), 0.0)
        reg = self._nz(m.get("register_consistency"), 0.0)
        score = self._bounded(0.6 * clarity + 0.4 * reg)

        if clarity < 0.7:
            issues.append("Clarity below target.")
        if reg < 0.8:
            issues.append("Register consistency below target.")

        # Tone is noted but not scored directly here
        return score, issues

    def _score_relevance(self, m: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Evaluates how well the generated answer aligns with the query intent
        and avoids redundant or hallucinated content.
        """
        issues = []
        rel_score = self._nz(m.get("relevance_score"), 0.0)
        coverage = self._nz(m.get("coverage_score"), 0.0)
        redundancy = self._nz(m.get("redundancy_ratio"), 0.0)
        hallucination = self._nz(m.get("hallucination_risk"), 0.0)

        # Combine scores (reward relevance/coverage, penalize redundancy/hallucination)
        score = self._bounded(0.5 * rel_score + 0.3 * coverage - 0.1 * redundancy - 0.1 * hallucination)

        if rel_score < 0.5:
            issues.append("Low semantic relevance between query and answer.")
        if coverage < 0.6:
            issues.append("Incomplete coverage — answer may not fully address query intent.")
        if redundancy > 0.4:
            issues.append("Answer contains redundant or off-topic information.")
        if hallucination > 0.3:
            issues.append("Potential hallucination risk detected.")

        return score, issues

    def _score_claim_extraction(self, m: Dict[str, Any]) -> Tuple[float, List[str]]:
        issues = []
        cc = int(m.get("claim_count") or 0)
        avg_match = self._nz(m.get("avg_match_score"), 0.0)  # cosine sim claim<->segment

        # Score: encourage reasonable claim count, high match
        base = 0.75 * self._bounded(avg_match) + 0.25 * self._bounded(min(cc, 10) / 10.0)
        score = self._bounded(base)

        if cc == 0:
            return 0.0, ["No claims extracted — downstream verification impossible."]
        if avg_match < 0.6:
            issues.append("Low alignment between extracted claims and answer segments.")

        return score, issues

    def _score_verification(self, m: Dict[str, Any]) -> Tuple[float, List[str]]:
        issues = []
        ent = self._nz(m.get("entailment_ratio"), 0.0)
        contra = self._nz(m.get("contradiction_ratio"), 0.0)
        conf = self._nz(m.get("avg_confidence"), 0.0)
        heur = self._nz(m.get("heuristic_avg_confidence"), 0.0)

        # Score: entailment and confidence dominate; contradiction penalizes; heuristics smooth
        score = self._bounded(0.45*ent + 0.35*conf + 0.20*heur - 0.20*contra)

        if ent < 0.5:
            issues.append("Low entailment ratio.")
        if contra > 0.2:
            issues.append("High contradiction rate across claims.")
        if conf < 0.6:
            issues.append("Low entailment confidence.")
        if heur < 0.6:
            issues.append("Low heuristic support (lexical/entity/numeric).")

        return score, issues

    # ---------- Verdict & aggregation ----------
    def _verdict_from_score(self, score: float, confidence_hint: float) -> Tuple[str, float]:
        if score >= 0.85:
            return "Fully Compliant", self._bounded(0.7 + 0.3 * confidence_hint)
        if score >= 0.70:
            return "Largely Compliant", self._bounded(0.6 + 0.3 * confidence_hint)
        if score >= 0.55:
            return "Partially Compliant", self._bounded(0.5 + 0.3 * confidence_hint)
        if score >= 0.35:
            return "Non-Compliant", self._bounded(0.4 + 0.3 * confidence_hint)
        return "Severely Non-Compliant", self._bounded(0.35 + 0.3 * confidence_hint)

    # TODO: Improve this to more specific recommendations rather then general
    def _recommendations(self, stage_issues: Dict[str, List[str]]) -> List[str]:
        recs = []
        if stage_issues.get("ingestion"):
            recs.append("Re-parse document with improved extractor; verify page coverage and metadata.")
        if stage_issues.get("retrieval"):
            recs.append("Increase source diversity, tune similarity thresholds, and consider re-ranking.")
            for msg in stage_issues.get("retrieval", []):
                if "low maximum retrieval score" in msg.lower():
                    recs.append(
                        "Knowledge base or retriever may lack relevant information — "
                        "consider enriching indexed data or reformulating the query for better coverage."
                    )
                    break

        if stage_issues.get("generation"):
            recs.append("Constrain generation to retrieved context; enable stricter citation policies.")
        if stage_issues.get("style"):
            recs.append("Adjust prompt/tone to meet clarity and register requirements.")
        if stage_issues.get("extraction"):
            recs.append("Refine claim extraction prompt/schema; increase match alignment.")
        if stage_issues.get("verification"):
            recs.append("Review contradictory claims and strengthen evidence mapping.")
        if stage_issues.get("completion"):
            recs.append("Review retrieval breadth and ensure all relevant evidence is covered.")
        if stage_issues.get("consistency"):
            recs.append("Check for contradictory or unstable model reasoning across claims.")
        if stage_issues.get("relevance"):
            recs.append("Refine prompt or retrieval strategy to improve query–answer relevance and focus.")

        return recs

    # ---------- Run ----------
    def run(self, answer_id: int | None = None) -> GraphState | Dict[str, Any]:
        # Load metrics
        if answer_id is not None:
            m = self._fetch_metrics_from_db(answer_id)
            # prefer state style if available (DB doesn’t store style yet)
            style_metrics = (self.state.metrics_style if self.state and getattr(self.state, "metrics_style", None) else {})
        elif self.state:
            m = {
                "ingestion": getattr(self.state, "metrics_ingestion", {}),
                "retrieval": getattr(self.state, "metrics_retrieval", {}),
                "generation": getattr(self.state, "metrics_generation", {}),
                "compliance": {
                    "claim_extraction": getattr(self.state, "metrics_claim_extraction", {}),
                    "verification": getattr(self.state, "metrics_verification", {}),
                },
                "style": getattr(self.state, "metrics_style", {}),
                "answer": getattr(self.state, "answer", ""),
            }
            style_metrics = m.get("style", {})
        else:
            raise ValueError("No state or answer_id provided to RootCauseClassifierNode.")

        ingestion_score, ingest_issues = self._score_ingestion(m.get("ingestion", {}))
        retrieval_score, retr_issues = self._score_retrieval(m.get("retrieval", {}))
        generation_score, gen_issues = self._score_generation(m.get("generation", {}))
        style_score, style_issues = self._score_style(style_metrics)
        relevance_score, rel_issues = self._score_relevance(m.get("relevance", {}))
        extract_score, extr_issues = self._score_claim_extraction((m.get("compliance", {}) or {}).get("claim_extraction", {}))
        verify_score, ver_issues = self._score_verification((m.get("compliance", {}) or {}).get("verification", {}))

        stage_scores = {
            "ingestion": ingestion_score,
            "retrieval": retrieval_score,
            "generation": generation_score,
            "style": style_score,
            "relevance": relevance_score,
            "extraction": extract_score,
            "verification": verify_score,
        }

        completion_score, completion_issues = self._score_completion_coverage(m)
        consistency_score, consistency_issues = self._score_consistency(m, stage_scores)

        stage_scores.update({
            "completion": completion_score,
            "consistency": consistency_score,
        })

        # Overall score: emphasize verification + retrieval + generation
        overall = self._bounded(
            0.28 * verify_score +
            0.18 * retrieval_score +
            0.15 * generation_score +
            0.12 * relevance_score +
            0.10 * ingestion_score +
            0.07 * extract_score +
            0.05 * style_score +
            0.03 * completion_score +
            0.02 * consistency_score
        )

        # Confidence: coherence of stage scores (lower variance -> higher confidence)
        scores = list(stage_scores.values())
        if scores:
            mean_s = self._mean(scores)
            var = self._mean([(s - mean_s) ** 2 for s in scores])
            confidence_hint = self._bounded(1.0 - min(var * 3.0, 1.0))
        else:
            confidence_hint = 0.5

        verdict, confidence = self._verdict_from_score(overall, confidence_hint)

        stage_issues = {
            "ingestion": ingest_issues,
            "retrieval": retr_issues,
            "generation": gen_issues,
            "style": style_issues,
            "relevance": rel_issues,
            "extraction": extr_issues,
            "verification": ver_issues,
            "completion": completion_issues,
            "consistency": consistency_issues,
        }
        # Flatten prioritized issues (most critical first)
        priority = [
            "verification",
            "relevance",
            "retrieval",
            "generation",
            "ingestion",
            "extraction",
            "style",
            "completion",
            "consistency",
        ]

        flat_issues: List[str] = []
        for st in priority:
            for msg in stage_issues.get(st, []):
                flat_issues.append(f"[{st.upper()}] {msg}")

        result = {
            "verdict": verdict,
            "overall_score": round(overall, 3),
            "confidence": round(confidence, 2),
            "stage_scores": {k: round(v, 3) for k, v in stage_scores.items()},
            "issues": flat_issues[:12],  # avoid spam
            "recommendations": self._recommendations(stage_issues),
        }

        # Push into state if provided
        if self.state:
            self.state.verdict = verdict
            self.state.root_cause = result
            self.state.metrics_root_cause = result
            # useful for the report node
            self.state.metrics_pipeline = {
                "ingestion": m.get("ingestion", {}),
                "retrieval": m.get("retrieval", {}),
                "generation": m.get("generation", {}),
                "style": style_metrics,
                "relevance": m.get("relevance", {}),
                "compliance": m.get("compliance", {}),
            }
            if m.get("answer"):
                self.state.answer = m["answer"]

            return self.state

        return result
