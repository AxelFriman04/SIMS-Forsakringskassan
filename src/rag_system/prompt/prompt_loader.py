def compose_generation_prompt(query: str, retrieval_snapshot: dict) -> str:
    """
    Build the generation prompt using retrieved evidence.
    retrieval_snapshot["topk"] should contain dicts with 'chunk_id' and 'text'.
    """
    topk = retrieval_snapshot.get("topk", [])
    print("TopK in prompt is: ", len(topk))
    context = "\n\n".join([f"[{r['chunk_id']}] {r['text']}" for r in topk])

    return (
        "You are a helpful assistant. Use ONLY the EVIDENCE below to answer the QUESTION. "
        "Cite each piece you use as [CITE: <chunk_id>]. "
        "If the evidence does not contain the answer, respond with 'Information not available in provided evidence.'\n\n"
        f"EVIDENCE:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer in 3–5 concise sentences and include citations."
    )


"""    return (
        "You are a helpful assistant. Use the EVIDENCE below to answer the QUESTION. "
        "Cite chunks as [CITE: <chunk_id>] whenever you use them.\n\n"
        f"EVIDENCE:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer in 3–5 sentences and include citations."
    )"""

"""        context = "\n\n".join([r["text"] for r in retrieval_snapshot["topk"]])
        prompt = (
            f"Use the following evidence to answer the query.\n\n"
            f"EVIDENCE:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"Answer clearly and cite sources by their chunk_id where relevant."
        )"""