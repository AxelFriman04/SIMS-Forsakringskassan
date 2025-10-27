from shared.config import settings


def load_generation_prompt(query: str, retrieval_snapshot: dict) -> str:
    """
    Build the generation prompt using retrieved evidence.
    The model will answer in Swedish if PDF_LANG_IS_SWE is True.
    """

    topk = retrieval_snapshot.get("topk", [])
    context = "\n\n".join([f"[{r['chunk_id']}] {r['text']}" for r in topk])

    # Language control
    language_instruction = (
        "Svara på svenska i en formell och myndighetslik ton. "
        "Använd tydliga och sakliga formuleringar anpassade för Försäkringskassan."
        if getattr(settings, "PDF_LANG_IS_SWE", False)
        else "Answer in English using a clear, factual, and professional tone."
    )

    return (
        f"Du är en expert på att tolka Försäkringskassans vägledningar och regelverk. "
        f"Använd ENDAST informationen i avsnitten nedan för att besvara frågan. "
        f"Om svaret inte finns i underlaget, skriv 'Information saknas i tillgängligt material.'\n\n"
        f"{language_instruction}\n\n"
        "Var noga med att:\n"
        "- Skriva ett fullständigt men koncentrerat svar baserat på fakta i texten.\n"
        "- Inkludera citatmarkörer [CITE: <chunk_id>] efter varje påstående som stöds av en eller flera källor. "
          "Om flera källor styrker samma information, ange alla relevanta citat.\n"
        "- Undvik att lägga till egna antaganden eller spekulationer.\n\n"
        f"EVIDENS:\n{context}\n\n"
        f"FRÅGA: {query}\n\n"
        "SVAR:"
    )


def load_generator_system_prompt() -> str:
    """
    System prompt for the RAG Generator node.
    Enforces grounded, citation-aware, domain-consistent output.
    """

    from shared.config import settings  # avoid circular import

    if getattr(settings, "PDF_LANG_IS_SWE", False):
        # --- Swedish Mode ---
        return (
            "Du är en expert på Försäkringskassans vägledningar och juridiska texter. "
            "Skriv på **svenska** i en formell, myndighetslik ton. "
            "Använd endast informationen från de angivna källavsnitten. "
            "Varje faktapåstående ska åtföljas av citatmarkören [CITE: <chunk_id>] "
            "direkt efter meningen eller satsen det baseras på. "
            "Om flera källor stöder samma information, ange alla relevanta citat. "
            "Om svaret inte finns i underlaget, skriv exakt: 'Information saknas i tillgängligt material.' "
            "Undvik spekulationer, åsikter och ogrundade slutsatser."
        )
    else:
        # --- English Mode ---
        return (
            "You are an expert assistant specialized in legal and policy interpretation. "
            "Write in **English** using a clear, factual, and formal tone. "
            "Use only the information present in the provided evidence sections. "
            "After every factual statement, include the citation marker [CITE: <chunk_id>] "
            "referring to the supporting text. "
            "If multiple evidence sources support the same statement, include all relevant citations. "
            "If the information is not found in the evidence, respond exactly with "
            "'Information not available in provided evidence.' "
            "Avoid opinions, speculation, or generalizations."
        )


def load_rerank_prompt(query: str, results: list[dict]) -> str:
    """
    Build a structured reranker prompt.
    Dynamically switches between Swedish and English based on config.
    """

    formatted_passages = "\n\n".join(
        [f"Passage {i+1}:\n{r['text']}" for i, r in enumerate(results)]
    )

    if getattr(settings, "PDF_LANG_IS_SWE", False):
        language_instruction = (
            "Du är en expert på att bedöma hur relevanta hämtade textavsnitt är för en given fråga.\n\n"
            "Uppgift:\n"
            "- Bedöm hur väl varje textavsnitt svarar på frågan utifrån **semantisk relevans**.\n"
            "- Tilldela varje passage ett numeriskt **relevansbetyg** mellan **0.0** (inte relevant) "
            "och **10.0** (mycket relevant).\n"
            "- Fokusera på betydelse och sammanhang, inte ordmatchning.\n"
            "- Du ska inte skriva ett svar eller sammanfatta texten – bara sätta relevansbetyg.\n\n"
            "Systemet kommer att validera utdata mot ett strukturerat JSON-schema.\n"
            "Du behöver inte formatera JSON manuellt – bara ge data som passar schemat."
        )
    else:
        language_instruction = (
            "You are an expert in information retrieval and ranking.\n\n"
            "Task:\n"
            "- Evaluate how relevant each retrieved passage is to the given query based on **semantic meaning**.\n"
            "- Assign a numerical **relevance score** between **0.0** (irrelevant) and **10.0** (highly relevant).\n"
            "- Focus only on semantic alignment, not keyword overlap.\n"
            "- Do NOT answer or paraphrase the query – just rate relevance.\n\n"
            "The system will validate your output against a structured JSON schema.\n"
            "You do not need to format JSON manually – just output data that fits the schema."
        )

    return f"""
{language_instruction}

Schema:
{{
  "scores": [
    {{"passage": <integer>, "relevance": <float>}}
  ]
}}

Exempel / Example structured output:
{{
  "scores": [
    {{"passage": 1, "relevance": 9.5}},
    {{"passage": 2, "relevance": 7.8}},
    {{"passage": 3, "relevance": 4.2}}
  ]
}}

Fråga / Query:
\"\"\"{query}\"\"\"

Hämtade textavsnitt / Retrieved Passages:
{formatted_passages}
"""

# shared/prompt/prompt_loader.py

def load_reranker_system_prompt() -> str:
    from shared.config import settings

    if getattr(settings, "PDF_LANG_IS_SWE", False):
        return (
            "Du är en professionell assistent som arbetar med informationssökning och rangordning. "
            "Ditt uppdrag är att bedöma **hur relevant varje textavsnitt är** i förhållande till frågan som ställts. "
            "Du ska inte besvara frågan, bara bedöma relevans.\n\n"
            "Anvisningar:\n"
            "- Ge varje passage ett numeriskt relevansbetyg mellan **0.0** (inte alls relevant) "
            "och **10.0** (mycket relevant och direkt svarande på frågan).\n"
            "- Bedöm utifrån **semantisk mening**, inte bara ordmatchning.\n"
            "- Ta hänsyn till både kontext, betydelse och ämnesöverensstämmelse.\n"
            "- Om flera passager är lika relevanta, ge dem liknande betyg.\n"
            "- Skala betygen så att den mest relevanta passagen får det högsta värdet.\n"
            "- Utdata ska endast innehålla strukturerade numeriska bedömningar utan förklaringar eller kommentarer.\n\n"
            "Ditt mål är att ge konsekventa och jämförbara betyg för varje passage baserat på "
            "hur väl den stödjer eller besvarar frågan i sak."
        )
    else:
        return (
            "You are an expert assistant specializing in information retrieval and ranking. "
            "Your task is to evaluate **how relevant each retrieved passage is** to a given query. "
            "You must not answer the query — only rate relevance.\n\n"
            "Guidelines:\n"
            "- Assign each passage a numerical relevance score between **0.0** (irrelevant) "
            "and **10.0** (highly relevant and directly answering the query).\n"
            "- Judge based on **semantic meaning**, not surface word overlap.\n"
            "- Consider both topic and contextual fit.\n"
            "- Passages of similar relevance should receive similar scores.\n"
            "- Normalize the scale so the most relevant passage gets the highest value.\n"
            "- Output only structured numerical judgments — no commentary or explanations.\n\n"
            "Your goal is to produce consistent, comparable scores that reflect true semantic relevance."
        )





