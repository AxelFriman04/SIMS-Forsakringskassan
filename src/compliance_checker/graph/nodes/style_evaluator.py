from typing import Dict, Any
from compliance_checker.model.llm import LLM
from compliance_checker.prompt.prompt_loader import load_style_evaluation_prompt, load_style_evaluator_system_prompt
from compliance_checker.graph.state import GraphState
from shared.config import settings
import json


class StyleEvaluatorNode:
    """
    Evaluates tone, formality, and clarity of the RAG-generated answer.
    """

    def __init__(self, state: GraphState):
        self.state = state
        self.llm = LLM(model=settings.STYLE_EVAL_MODEL)

        # Schema for structured output
        self.style_schema = {
            "name": "style_evaluation",
            "type": "object",
            "properties": {
                "tone": {
                    "type": "string",
                    "enum": ["formal", "neutral", "informal", "speculative", "unclear"]
                },
                "clarity_score": {"type": "number"},  # 0–1 scale
                "register_consistency": {"type": "number"},  # 0–1, consistent tone across sentences
                "notes": {"type": "string"}
            },
            "required": ["tone", "clarity_score", "register_consistency", "notes"],
            "additionalProperties": False,
        }

    def run(self, answer_text: str) -> GraphState:
        """
        Evaluates the style and tone of the RAG answer and stores metrics in state.
        """
        #  answer_text = self.state.answer or answer_text
        if not answer_text.strip():
            self.state.log_metric(
                {"tone": "neutral", "clarity_score": 0.0, "register_consistency": 0.0},
                stage="style"
            )
            return self.state

        prompt = load_style_evaluation_prompt(answer_text)
        system_prompt = load_style_evaluator_system_prompt()

        try:
            response = self.llm.complete(prompt, structured=True, schema=self.style_schema, system_prompt=system_prompt)
            parsed = json.loads(response.get("answer", "{}"))

            self.state.metrics_style = {
                "tone": parsed.get("tone", "neutral"),
                "clarity_score": parsed.get("clarity_score", 0.0),
                "register_consistency": parsed.get("register_consistency", 0.0),
                "notes": parsed.get("notes", ""),
                "style_metadata": {
                    "model": settings.STYLE_EVAL_MODEL,
                }
            }

            if getattr(settings, "DEBUG", False):
                print("\n=== 🗣️ Style Evaluation ===")
                for k, v in self.state.metrics_style.items():
                    print(f"{k}: {v}")
                print("============================\n")

        except Exception as e:
            print(f"[ERROR] Style evaluation failed: {e}")
            self.state.metrics_style = {
                "tone": "neutral",
                "clarity_score": 0.0,
                "register_consistency": 0.0,
                "notes": "Evaluation failed"
            }

        return self.state
