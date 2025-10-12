# compliance_checker/model/llm.py

from __future__ import annotations
from typing import Dict, List, Optional, Any
from openai import OpenAI
from shared.config import settings
import random


class LLM:
    """
    Wrapper for OpenAI chat models used in the compliance checker.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.client = OpenAI(api_key=api_key or settings.OPENAI_API_KEY)
        self.model = model or settings.LLM_MODEL

    def complete(
        self,
        prompt: str,
        max_tokens: int = 16000,
        system_prompt: Optional[str] = None,
        structured: bool = False,
        schema: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Calls the OpenAI chat completion API.
        Returns standardized dict with:
          - answer: str or dict (if structured=True)
          - logprobs: list[float]
          - generator_metadata: dict (tokens, model, etc.)
        """

        # Dummy fallback for offline testing
        if settings.USE_DUMMY_LLM:
            fake_answer = f"[DUMMY LLM] Extracted {len(prompt) % 3 + 1} claims."
            fake_logprobs = [random.uniform(-1.5, -0.1) for _ in range(len(fake_answer) // 6)]
            return {
                "answer": fake_answer,
                "logprobs": fake_logprobs,
                "generator_metadata": {
                    "tokens": len(fake_answer) // 4,
                    "model": "dummy-llm",
                },
            }

        system_message = (
            {"role": "system", "content": system_prompt}
            if system_prompt
            else {"role": "system", "content": "You are a factual compliance analysis assistant."}
        )

        if getattr(settings, "DEBUG", False):
            print(f"[LLM] Using model: {self.model} | structured={structured} | max_tokens={max_tokens}")

        # Prepare arguments
        response_kwargs = dict(
            model=self.model,
            messages=[system_message, {"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
        )

        # Structured output handling
        if structured:
            if schema:
                response_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "verification",  # <-- must be inside json_schema
                        "strict": True,
                        "schema": schema  # <-- your schema dict
                    }
                    # "strict": True,
                }
            else:
                response_kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**response_kwargs)

        print(response)

        if getattr(settings, "DEBUG", False):
            print("[LLM] Raw response:", response)

        # Parse structured or unstructured output
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
            "generator_metadata": {
                "tokens": usage.total_tokens if usage else None,
                "model": self.model,
            },
        }