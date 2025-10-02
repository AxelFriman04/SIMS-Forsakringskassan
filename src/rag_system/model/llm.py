from __future__ import annotations

from typing import Dict
from openai import OpenAI
from shared.config import settings
import random


class LLM:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.client = OpenAI(api_key=api_key or settings.OPENAI_API_KEY)
        self.model = model or settings.LLM_MODEL

    def complete(self, prompt: str, max_tokens: int = 512) -> Dict:

        if settings.USE_DUMMY_LLM:
            # Fake response (good enough for testing pipeline flow)
            return {
                "answer": f"[DUMMY] Answer of length {len(prompt)}",
                "declared_citations": [f"chunk::{random.randint(0, 5)}"],
                "generator_metadata": {"tokens": len(prompt) // 4, "model": "dummy"}
            }

        """
        Calls OpenAI completion endpoint.
        Returns standardized dict for downstream nodes.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers with citations by chunk id when available."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )

        choice = response.choices[0].message.content

        return {
            "answer": choice,
            "declared_citations": [],  # TODO: later add citation extraction logic
            "generator_metadata": {
                "tokens": response.usage.total_tokens if response.usage else None,
                "model": self.model,
            }
        }
