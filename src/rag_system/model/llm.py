from __future__ import annotations

from typing import Dict, List
from openai import OpenAI
from shared.config import settings
import random


class LLM:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.client = OpenAI(api_key=api_key or settings.OPENAI_API_KEY)
        self.model = model or settings.LLM_MODEL

    def complete(self, prompt: str, max_tokens: int = 512) -> Dict:
        """
        Call OpenAI chat/completions endpoint.
        Returns standardized dict with:
          - answer: str
          - declared_citations: list[str]
          - logprobs: list[float] (token-level logprobs, if available)
          - generator_metadata: dict with tokens, model id, etc.
        """

        print("Max tokens in LLM set to: ", max_tokens)

        if settings.USE_DUMMY_LLM:
            # Fake response for testing pipeline flow
            fake_answer = f"[DUMMY] Answer of length {len(prompt)}"
            fake_logprobs = [random.uniform(-2.0, -0.1) for _ in range(len(fake_answer) // 5)]
            return {
                "answer": fake_answer,
                "declared_citations": [f"chunk::{random.randint(0, 5)}"],
                "logprobs": fake_logprobs,
                "generator_metadata": {
                    "tokens": len(fake_answer) // 4,
                    "model": "dummy"
                }
            }

        # ---- Real OpenAI call ----
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers with citations by chunk id when available."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens
            # logprobs=True  # request logprobs if supported
        )

        print("LLM raw response:", response)

        choice = response.choices[0].message.content
        usage = response.usage

        # Extract token-level logprobs if the API provides them
        token_logprobs: List[float] = []
        if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
            for t in response.choices[0].logprobs.content:
                if "logprob" in t:  # OpenAI logprob object
                    token_logprobs.append(t["logprob"])

        return {
            "answer": choice,
            "declared_citations": [],  # kept for compatibility, extraction happens downstream
            "logprobs": token_logprobs,
            "generator_metadata": {
                "tokens": usage.total_tokens if usage else None,
                "model": self.model,
            }
        }