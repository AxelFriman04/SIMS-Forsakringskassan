# compliance_checker/prompt/prompt_loader.py
from shared.config import settings


def load_style_evaluation_prompt(answer_text: str) -> str:
    """
    Composes the LLM prompt for evaluating tone, clarity, and register consistency.
    """

    if getattr(settings, "PDF_LANG_IS_SWE", False):
        language_instruction = (
            "Analysera texten på svenska. Bedöm om den har en formell, neutral och tydlig myndighetston. "
            "Om texten är otydlig, informell eller innehåller spekulationer – markera det."
        )
    else:
        language_instruction = (
            "Analyze the text in English. Assess whether it has a formal, neutral, and clear professional tone. "
            "If it is unclear, informal, or speculative, flag accordingly."
        )

    return f"""
You are an expert linguistic evaluator.

Your task is to analyze the following RAG-generated answer and assess:

1. **Tone** — choose one: ["formal", "neutral", "informal", "speculative", "unclear"]
2. **Clarity Score** — a number between 0 and 1 (1 = very clear and precise)
3. **Register Consistency** — a number between 0 and 1 (1 = tone is consistent and professional throughout)
4. **Notes** — short comments explaining the reasoning.

{language_instruction}

Analyze the text below and produce structured output following the schema exactly.

Answer:
\"\"\"{answer_text}\"\"\"
"""


def load_style_evaluator_system_prompt() -> str:
    """System prompt for the StyleEvaluatorNode."""
    from shared.config import settings

    if getattr(settings, "PDF_LANG_IS_SWE", False):
        return (
            "Du är en språklig utvärderingsassistent specialiserad på myndighetstexter. "
            "Ditt uppdrag är att analysera **ton**, **språklig klarhet** och **registerkonsekvens** "
            "i svar som genererats av ett språkmodellssystem. "
            "Bedömningen ska göras strikt objektivt – du ska inte omformulera eller förbättra texten, "
            "endast analysera dess språkliga egenskaper.\n\n"
            "Ditt mål är att identifiera om texten är formell, neutral och tydlig, "
            "eller om den innehåller spekulativa, oklara eller informella drag."
        )
    else:
        return (
            "You are a linguistic evaluation assistant specialized in official and professional writing. "
            "Your role is to assess the **tone**, **clarity**, and **register consistency** "
            "of a model-generated answer. "
            "You must not rewrite or edit the text – only evaluate it objectively.\n\n"
            "Your goal is to determine whether the text maintains a formal, neutral, and clear style, "
            "or whether it contains informal, speculative, or unclear elements."
        )


def load_claim_extraction_prompt(answer_text: str) -> str:
    """
    Compose user prompt for claim extraction.
    Adapts to Swedish or English language depending on config.
    Works with structured JSON output (schema-enforced).
    """

    # Language control
    if getattr(settings, "PDF_LANG_IS_SWE", False):
        language_instruction = (
            "Extrahera påståenden på svenska. "
            "Behåll originalformuleringens ton och betydelse. "
            "Hoppa över fraser som enbart anger att information saknas eller att något inte kan bedömas."
        )
        tone_instruction = (
            "Skriv neutralt och sakligt i stil med Försäkringskassans vägledningar."
        )
    else:
        language_instruction = (
            "Extract the claims in English, preserving the factual tone and meaning. "
            "Skip phrases that only express missing or unavailable information."
        )
        tone_instruction = (
            "Maintain a neutral and factual style, suitable for an official report."
        )

    return f"""
You are an expert factual analysis assistant that extracts **independent, verifiable factual claims**
from a RAG-generated answer.

The text may include citation markers like `[CITE: <id>]`. 
These indicate which source supported a part of the text — they should **not** be included in the claim text.

Your task:
- Identify each **distinct, checkable factual statement**.
- Each claim must represent **only one** factual proposition.
- Exclude vague or speculative sentences and those saying information is unavailable.
- Maintain a clear, factual, and neutral tone.

The output will be validated against a **structured JSON schema**, not parsed from text.
You do **not** need to format JSON manually — just produce the structured data that fits the schema.

{language_instruction}
{tone_instruction}

Example structured output (for illustration only):
[
  {{ "text": "Water boils at 100°C at sea level." }},
  {{ "text": "Ice melts at 0°C." }}
]

Analyze the following answer and extract all factual claims:

\"\"\"{answer_text}\"\"\"
"""


def load_claim_extractor_system_prompt() -> str:
    """System prompt for the ClaimExtractorNode."""
    from shared.config import settings

    if getattr(settings, "PDF_LANG_IS_SWE", False):
        return (
            "Du är en juridisk och språklig analysassistent med expertis inom svenska myndighetstexter. "
            "Ditt uppdrag är att identifiera **självständiga, verifierbara påståenden** i en given text. "
            "Varje påstående ska kunna kontrolleras mot källmaterial, "
            "och får inte innehålla flera fakta i samma mening. "
            "Du ska inte tolka, förklara eller lägga till information – "
            "endast extrahera de rena faktapåståendena så som de står uttryckta."
        )
    else:
        return (
            "You are a legal and linguistic analysis assistant specializing in factual decomposition. "
            "Your job is to identify **independent, verifiable factual statements** within a given text. "
            "Each claim must be self-contained, checkable, and atomic. "
            "Do not interpret, summarize, or add information — extract only the factual statements as written."
        )


def load_entailment_prompt(premise: str, hypothesis: str) -> str:
    """
    Compose the prompt for single claim verification.
    Adapts to Swedish or English.
    Structured output enforced via JSON schema.
    """

    if getattr(settings, "PDF_LANG_IS_SWE", False):
        language_instruction = (
            "Analysera sambandet mellan bevistexten (evidens) och påståendet. "
            "Klassificera om beviset **stödjer**, **motsäger** eller **inte är relaterat till** påståendet."
        )
    else:
        language_instruction = (
            "Analyze the relationship between the evidence and the claim. "
            "Classify whether the evidence **supports**, **contradicts**, or is **unrelated** to the claim."
        )

    return f"""
You are a precise factual verification expert.

Your goal is to determine how well the provided **evidence** supports the **claim**.

Follow this classification scheme:
- "entailment" → The evidence directly supports or confirms the claim.
- "contradiction" → The evidence directly contradicts the claim.
- "neutral" → The evidence is unrelated or insufficient.

Return a structured object with:
- label: one of ["entailment", "contradiction", "neutral"]
- confidence: a float between 0 and 1 indicating certainty.

Examples:
Evidence: "Water freezes at 0°C under normal conditions."
Claim: "Water becomes solid at 0°C."
→ {{ "label": "entailment", "confidence": 0.97 }}

Evidence: "The Earth revolves around the Sun."
Claim: "The Sun revolves around the Earth."
→ {{ "label": "contradiction", "confidence": 1.0 }}

Evidence: "The Moon has craters."
Claim: "The Moon is made of cheese."
→ {{ "label": "neutral", "confidence": 0.99 }}

{language_instruction}

Now analyze the following pair:

Evidence:
\"\"\"{premise}\"\"\"

Claim:
\"\"\"{hypothesis}\"\"\"
"""


def load_entailment_batch_prompt(evidence_texts: dict, pairs: list[dict]) -> str:
    """
    Build the entailment batch prompt for multiple claims.
    Each claim may reference one or more evidence segments.
    Adapts to Swedish or English and uses structured JSON output.
    """

    # Format evidence texts
    formatted_evidence = "\n".join(
        [f"{ref}: \"\"\"{text.strip()}\"\"\"" for ref, text in evidence_texts.items()]
    )

    # Format claims with multi-evidence references
    formatted_pairs = "\n".join([
        f"{i+1}. Claim: \"\"\"{p['hypothesis']}\"\"\" "
        f"(uses {', '.join(p['evidence_refs']) if p['evidence_refs'] else 'no evidence'})"
        for i, p in enumerate(pairs)
    ])

    # Language-specific instructions
    if getattr(settings, "PDF_LANG_IS_SWE", False):
        language_instruction = (
            "För varje påstående, avgör om bevisen **stödjer**, **motsäger** "
            "eller **inte är relaterade till** påståendet."
        )
    else:
        language_instruction = (
            "For each claim, determine whether the evidence **supports**, **contradicts**, "
            "or is **unrelated** to the claim."
        )

    return f"""
You are a domain expert in factual consistency checking.

Below are multiple EVIDENCE sections (A, B, C, ...) and several CLAIMS.
Each claim specifies which evidence references it relies on.

{language_instruction}

Output a **structured array** where each element is an object:
- label: one of ["entailment", "contradiction", "neutral"]
- confidence: a float between 0 and 1

Interpretation:
- "entailment" → Evidence clearly supports the claim.
- "contradiction" → Evidence clearly conflicts with the claim.
- "neutral" → Evidence is unrelated or insufficient to decide.

Examples:
Evidence: "Water freezes at 0°C under normal conditions."
Claim: "Water becomes solid at 0°C."
→ {{ "label": "entailment", "confidence": 0.97 }}

Evidence: "The Earth revolves around the Sun."
Claim: "The Sun revolves around the Earth."
→ {{ "label": "contradiction", "confidence": 1.0 }}

Evidence: "The Moon has craters."
Claim: "The Moon is made of cheese."
→ {{ "label": "neutral", "confidence": 0.99 }}

### EVIDENCE TEXTS ###
{formatted_evidence}

### CLAIMS ###
{formatted_pairs}

Return your output as an array of objects that strictly follow the schema.
"""


def load_evidence_checker_system_prompt() -> str:
    """System prompt for EvidenceCheckerNode (single and batch)."""
    from shared.config import settings

    if getattr(settings, "PDF_LANG_IS_SWE", False):
        return (
            "Du är en expert på faktagranskning och textuell semantisk analys. "
            "Ditt uppdrag är att avgöra om ett bevisavsnitt **stödjer**, **motsäger** "
            "eller **inte är relaterat till** ett givet påstående. "
            "Du ska inte skapa nya slutsatser – endast analysera sambandet mellan "
            "påståendet och den tillhandahållna evidenstexten.\n\n"
            "Fokusera på semantisk mening och faktuell överensstämmelse, "
            "inte på ordmatchning eller ytlig likhet."
        )
    else:
        return (
            "You are a factual verification specialist trained in textual entailment. "
            "Your task is to determine whether the given evidence **supports**, **contradicts**, "
            "or is **unrelated to** a provided claim. "
            "Do not generate explanations or new information — only assess the relationship.\n\n"
            "Focus on semantic meaning and factual consistency, not word overlap or tone."
        )
