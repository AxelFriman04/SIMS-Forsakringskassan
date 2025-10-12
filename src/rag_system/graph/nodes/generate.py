# src/nodes/generate.py

import re
import statistics
from rag_system.model.llm import LLM
from rag_system.graph.state import GraphState
from rag_system.prompt.prompt_loader import compose_generation_prompt
import tiktoken


class GenerateNode:
    def __init__(self, state: GraphState):
        self.state = state
        self.llm = LLM()

    def count_tokens(self, text: str, model: str) -> int:
        """
        Count the number of tokens in a string for a given OpenAI model.
        """
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))

    def compute_max_tokens(
        self,
        retrieval_snapshot: dict,
        base_tokens: int = 256,
        token_per_500_chars: int = 50,
        token_per_chunk_citation: int = 5,
        max_limit: int = 128000
    ) -> int:
        """
        Dynamically compute max_tokens based on the total text length
        of retrieved chunks and number of citations.

        Parameters:
            retrieval_snapshot: dict returned by RetrieveNode.run()
            base_tokens: minimum number of tokens for the answer
            token_per_500_chars: extra tokens per ~500 chars retrieved
            token_per_chunk_citation: extra tokens per chunk for citations
            max_limit: maximum allowed tokens

        Returns:
            int: max_tokens to use for generation
        """
        topk = retrieval_snapshot.get("topk", [])
        total_chars = sum(len(r.get("text", "")) for r in topk)
        num_chunks = len(topk)

        # Extra tokens for content
        extra_tokens_text = (total_chars // 500) * token_per_500_chars

        # Extra tokens for citations
        extra_tokens_citations = num_chunks * token_per_chunk_citation

        tokens = base_tokens + extra_tokens_text + extra_tokens_citations
        return min(tokens, max_limit)

    def run(self, query: str, retrieval_snapshot: dict, dry_run: bool = False):
        """
        Compose a prompt with retrieved evidence, ask the LLM to generate an answer,
        compute M3 metrics, and persist results.
        """

        self.state.query = query

        # 1. Build prompt
        prompt = compose_generation_prompt(query, retrieval_snapshot)

        # 2. Count tokens
        total_prompt_tokens = self.count_tokens(prompt, model=self.llm.model)
        max_tokens_estimate = self.compute_max_tokens(retrieval_snapshot)
        print(f"[DRY RUN] Total prompt tokens: {total_prompt_tokens}")
        print(f"[DRY RUN] Estimated max_tokens for generation: {max_tokens_estimate}")

        if dry_run:
            # Return a dummy response without calling the API
            gen = {
                "answer": "",
                "declared_citations": [],
                "logprobs": [],
                "generator_metadata": {
                    "tokens": 0,
                    "model": self.llm.model
                }
            }
        else:
            # 3. Call LLM normally
            gen = self.llm.complete(prompt, max_tokens=16000)

        # 2. Call LLM
        # max_tokens = self.compute_max_tokens(retrieval_snapshot)
        # gen = self.llm.complete(prompt, max_tokens=max_tokens)
        # gen = self.llm.complete(prompt)
        answer = gen.get("answer", "")
        self.state.answer = answer
        self.state.last_generation_snapshot = gen

        # 3. Compute M3 metrics
        m3_metrics = self._compute_m3_metrics(answer, gen, retrieval_snapshot)
        self.state.metrics_generation = m3_metrics


        return gen

    def _compute_m3_metrics(self, answer: str, gen: dict, retrieval_snapshot: dict) -> dict:
        """
        Compute prototype M3 metrics:
        - token_logprob_stats: average/min/count from logprobs
        - generator_declared_citations: citations found in answer
        - answer_length: char length
        - claim_count: heuristic based on sentence punctuation
        - preliminary_hallucinations_warnings: checks missing/invalid citations
        """

        # Token logprobs
        logprobs = gen.get("logprobs", [])
        avg_logprob = statistics.mean(logprobs) if logprobs else None
        min_logprob = min(logprobs) if logprobs else None

        # citations extraction
        # declared_citations = re.findall(r"\[CITE:\s*(\w+)\]", answer)
        declared_citations = re.findall(r"\[CITE:\s*([^\]]+)\]", answer)

        # claim count heuristic = number of sentences
        claim_count = answer.count(".") + answer.count("!") + answer.count("?")

        # hallucination warnings
        valid_chunk_ids = {str(r["chunk_id"]) for r in retrieval_snapshot.get("topk", [])}
        hallucination_warnings = []
        if not declared_citations:
            hallucination_warnings.append("No citations provided.")
        if any(cid not in valid_chunk_ids for cid in declared_citations):
            hallucination_warnings.append("Citations reference non-retrieved chunks.")

        return {
            "token_logprob_stats": {
                "avg": avg_logprob,
                "min": min_logprob,
                "count": len(logprobs)
            },
            "generator_declared_citations": declared_citations,
            "answer_length": len(answer),
            "claim_count": claim_count,
            "preliminary_hallucinations_warnings": hallucination_warnings
        }
