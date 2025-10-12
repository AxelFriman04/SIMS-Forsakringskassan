# compliance_checker/prompt/prompt_loader.py

def load_claim_extraction_prompt(answer_text: str) -> str:
    """
    Compose the system + user prompt for claim extraction.
    The LLM should output a JSON array of atomic factual claims.
    """
    return f"""
You are a factual claim extraction assistant.

Your task is to read the following answer and extract all **independent, verifiable factual claims**.

- Each claim must represent one distinct fact that can be checked independently.
- Do NOT merge multiple claims into one sentence.
- Output a **JSON array** where each element is an object with one field:
  "text": "<claim>"

Example output:
[
  {{ "text": "Water boils at 100°C at sea level." }},
  {{ "text": "Ice melts at 0°C." }}
]

Answer to analyze:
\"\"\"{answer_text}\"\"\"
"""


def load_entailment_prompt_old(premise: str, hypothesis: str) -> str:
    """
    Compose the system + user prompt for the Evidence Checker node.
    The LLM should classify the relationship between the provided evidence (premise)
    and the claim (hypothesis) as ENTAILMENT, CONTRADICTION, or NEUTRAL.
    """
    return f"""
You are a factual verification expert.

Your task is to analyze whether the given **evidence** (premise)
supports, contradicts, or is unrelated to the **claim** (hypothesis).

Respond strictly in JSON format with the following fields:
- "label": one of ["entailment", "contradiction", "neutral"]
- "confidence": a float between 0 and 1 indicating your confidence in this classification.

Examples:

Evidence: "Water freezes at 0°C under normal conditions."
Claim: "Water becomes solid at 0°C."
Output:
{{"label": "entailment", "confidence": 0.97}}

Evidence: "The Earth revolves around the Sun."
Claim: "The Sun revolves around the Earth."
Output:
{{"label": "contradiction", "confidence": 1.0}}

Evidence: "The Moon has craters."
Claim: "The Moon is made of cheese."
Output:
{{"label": "neutral", "confidence": 0.99}}

Now analyze:

Evidence: \"\"\"{premise}\"\"\"
Claim: \"\"\"{hypothesis}\"\"\"
"""


def load_entailment_prompt(premise: str, hypothesis: str) -> str:
    """
    Compose the system + user prompt for the Evidence Checker node
    using structured output.
    """
    return f"""
You are a factual verification expert.

Your task:
Analyze whether the given **evidence** (premise) supports, contradicts,
or is unrelated to the **claim** (hypothesis).

Return an object exactly following this schema:
- label: one of ["entailment", "contradiction", "neutral"]
- confidence: a float between 0 and 1

Examples:

Evidence: "Water freezes at 0°C under normal conditions."
Claim: "Water becomes solid at 0°C."
Output:
{{"label": "entailment", "confidence": 0.97}}

Evidence: "The Earth revolves around the Sun."
Claim: "The Sun revolves around the Earth."
Output:
{{"label": "contradiction", "confidence": 1.0}}

Evidence: "The Moon has craters."
Claim: "The Moon is made of cheese."
Output:
{{"label": "neutral", "confidence": 0.99}}

Now analyze the following evidence and claim, and return the object strictly following the schema:

Evidence: \"\"\"{premise}\"\"\"
Claim: \"\"\"{hypothesis}\"\"\"
"""

def load_entailment_batch_prompt(pairs: list[dict]) -> str:
    """
    Compose a prompt for verifying multiple (evidence, claim) pairs at once.
    Each pair should have keys: 'premise' and 'hypothesis'.
    """
    formatted_pairs = "\n".join(
        [f"{i+1}. Evidence: \"\"\"{p['premise']}\"\"\"\n   Claim: \"\"\"{p['hypothesis']}\"\"\"" for i, p in enumerate(pairs)]
    )

    return f"""
You are a factual verification expert.

You will receive several (evidence, claim) pairs. 
For each pair, decide whether the **claim** is ENTAILED by, CONTRADICTED by, or NEUTRAL toward the **evidence**.

Respond strictly in **JSON array** form, where each element corresponds to the same index in order.

Each element must be an object:
{{
  "label": one of ["entailment", "contradiction", "neutral"],
  "confidence": a float between 0 and 1
}}

Example output:
[
  {{"label": "entailment", "confidence": 0.97}},
  {{"label": "neutral", "confidence": 0.88}}
]

Now analyze these pairs:

{formatted_pairs}
"""


def load_entailment_batch_prompt_v2(evidence_texts: dict, pairs: list[dict]) -> str:
    formatted_evidence = "\n".join(
        [f"{ref}: \"\"\"{text}\"\"\"" for ref, text in evidence_texts.items()]
    )

    formatted_pairs = "\n".join(
        [f"{i+1}. Claim: \"\"\"{p['hypothesis']}\"\"\" (uses {p['evidence_ref'] or 'no evidence'})"
         for i, p in enumerate(pairs)]
    )

    return f"""
You are a factual verification expert.

Below are several pieces of EVIDENCE labeled A, B, C, etc.,
followed by a list of CLAIMS. Each claim specifies which evidence it relates to.

Your task:
For each claim, classify the relationship between the evidence and the claim as:
- "entailment" (the evidence supports the claim)
- "contradiction" (the evidence disproves the claim)
- "neutral" (the evidence is unrelated)

Return an array of objects exactly following this schema:
- label: one of ["entailment", "contradiction", "neutral"]
- confidence: a float between 0 and 1

Examples:

Evidence: "Water freezes at 0°C under normal conditions."
Claim: "Water becomes solid at 0°C."
Output:
{{"label": "entailment", "confidence": 0.97}}

Evidence: "The Earth revolves around the Sun."
Claim: "The Sun revolves around the Earth."
Output:
{{"label": "contradiction", "confidence": 1.0}}

Evidence: "The Moon has craters."
Claim: "The Moon is made of cheese."
Output:
{{"label": "neutral", "confidence": 0.99}}

### EVIDENCE TEXTS ###
{formatted_evidence}

### CLAIMS ###
{formatted_pairs}

Return your output as an array of objects following the schema above.
"""


def load_entailment_batch_prompt_v2_old(evidence_texts: dict, pairs: list[dict]) -> str:
    """
    Compose a prompt that references evidence symbolically to avoid duplication.
    evidence_texts: { "Evidence A": "...", "Evidence B": "..." }
    pairs: [{ "evidence_ref": "Evidence A", "hypothesis": "..." }, ...]
    """
    # --- Format the evidence section ---
    formatted_evidence = "\n".join(
        [f"{ref}: \"\"\"{text}\"\"\"" for ref, text in evidence_texts.items()]
    )

    # --- Format the claim section ---
    formatted_pairs = "\n".join(
        [f"{i+1}. Claim: \"\"\"{p['hypothesis']}\"\"\" (uses {p['evidence_ref'] or 'no evidence'})"
         for i, p in enumerate(pairs)]
    )

    return f"""
You are a factual verification expert.

Below are several pieces of EVIDENCE labeled A, B, C, etc.,
followed by a list of CLAIMS. Each claim specifies which evidence it relates to.

Your task:
For each claim, classify the relationship between the evidence and the claim as:
- "entailment" (the evidence supports the claim)
- "contradiction" (the evidence disproves the claim)
- "neutral" (the evidence is unrelated)

Respond strictly in **JSON array** form, where each element corresponds
to the claim in order and has the fields:
{{
  "label": one of ["entailment", "contradiction", "neutral"],
  "confidence": a float between 0 and 1
}}

Examples:

Evidence: "Water freezes at 0°C under normal conditions."
Claim: "Water becomes solid at 0°C."
Output:
{{"label": "entailment", 
  "confidence": 0.97
}}

Evidence: "The Earth revolves around the Sun."
Claim: "The Sun revolves around the Earth."
Output:
{{"label": "contradiction", 
  "confidence": 1.0
}}

Evidence: "The Moon has craters."
Claim: "The Moon is made of cheese."
Output:
{{"label": "neutral", 
  "confidence": 0.99
}}

### EVIDENCE TEXTS ###
{formatted_evidence}

### CLAIMS ###
{formatted_pairs}

Now analyze all claims and output a JSON array.
"""

