from __future__ import annotations

from typing import Dict, List, Any, Optional
from openai import OpenAI
from shared.config import settings
import random


class LLM:
    def __init__(self, api_key: str = settings.OPENAI_API_KEY, model: str | None = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model or settings.LLM_MODEL

    def complete(
        self,
        prompt: str,
        structured: bool = False,
        schema: Optional[dict] = None,
        max_tokens: int = 8000,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call OpenAI chat/completions endpoint.
        Returns standardized dict with:
          - answer: str
          - declared_citations: list[str]
          - logprobs: list[float] (token-level logprobs, if available)
          - generator_metadata: dict with tokens, model id, etc.
        """

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

        # ---- OpenAI call ----
        system_message = (
            {"role": "system", "content": system_prompt}
            if system_prompt
            else {"role": "system", "content": "You are a helpful assistant that answers using provided evidence and citations by chunk id."}
        )
        response_kwargs = dict(
            model=self.model,
            messages=[
                system_message,
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens,
            # logprobs=True # Use if supported
        )

        # ---- Structured output (JSON schema enforcement) ----
        if structured:
            if schema:
                response_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.get("name", "structured_output"),
                        "strict": True,
                        "schema": schema,
                    },
                }
            else:
                response_kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**response_kwargs)

        # --- Parse result ---
        if structured and hasattr(response.choices[0].message, "parsed"):
            content = response.choices[0].message.parsed
        else:
            content = response.choices[0].message.content or ""

        usage = getattr(response, "usage", None)
        token_logprobs: List[float] = []
        if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
            for t in response.choices[0].logprobs.content:
                if "logprob" in t:
                    token_logprobs.append(t["logprob"])

        return {
            "answer": content,
            "logprobs": token_logprobs,
            "metadata": {
                "tokens": usage.total_tokens if usage else None,
                "model": self.model,
            },
        }