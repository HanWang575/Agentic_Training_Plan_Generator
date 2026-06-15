"""Prompt builders for the evidence-grounded answer generator."""

import json


def build_answer_prompt(
    question: str,
    evidence_context: str,
    safety_mode: str,
    safety_flags: list[dict],
    memory_context: str | None = None,
) -> str:
    """Build the live answer-generation prompt.

    Parameters:
        question: normalized user question.
        evidence_context: retrieved evidence text already selected for the LLM.
        safety_mode: `normal` or `caution`.
        safety_flags: list of safety flag dictionaries.
    Returns:
        str: grounded prompt for the answer model.
    """

    memory_section = ""
    if memory_context:
        memory_section = (
            "\n\nUser memory context (personalization only):\n"
            f"{memory_context}\n\n"
            "Memory rules:\n"
            "- Use memory only to personalize framing, constraints, and follow-up suggestions.\n"
            "- Do not treat memory as evidence for scientific claims.\n"
            "- If memory conflicts with evidence or raises a safety concern, say so cautiously.\n"
        )

    return (
        "You are the Evidence-Based Q&A Assistant for a personal trainer AI.\n"
        "Your job is to answer from the evidence context only. Stay faithful to the evidence, "
        "but do not sound mechanical or overly terse.\n\n"
        "Core rules:\n"
        "1. Use only the evidence context. Do not add outside facts.\n"
        "2. If the evidence is incomplete, say exactly what is missing.\n"
        "3. Do not invent citations, source titles, page numbers, or recommendations.\n"
        "4. Write like a thoughtful coach-educator: clear, concrete, and natural.\n"
        "5. Prefer explanation over slogans. When the evidence supports it, explain what a term means, "
        "why it matters, how it is used in practice, and give a small example.\n"
        "6. Do not pad the answer with generic advice that is not grounded in the evidence.\n\n"
        "Style guidance:\n"
        "- Avoid one-line, dry answers unless the evidence is genuinely minimal.\n"
        "- In 'Short answer', give a direct answer in 2-4 sentences when evidence allows.\n"
        "- In 'Practical guidance', interpret the evidence in plain language.\n"
        "- If the question asks for explanation, definition, comparison, or examples, provide them when supported.\n"
        "- If evidence conflicts or is narrow, mention that uncertainty briefly and specifically.\n\n"
        f"Safety mode: {safety_mode}\n"
        f"Safety flags: {json.dumps(safety_flags, ensure_ascii=False)}\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence context:\n{evidence_context}\n\n"
        f"{memory_section}"
        "Output requirements:\n"
        "- Use exactly these section headers.\n"
        "- Keep all claims traceable to the evidence context.\n"
        "- In 'Evidence used', list only citations that support the answer.\n\n"
        "Use this exact structure:\n"
        "Short answer:\n"
        "Write a direct, evidence-grounded answer. If the evidence is strong enough, explain it in a few sentences "
        "instead of giving only a brief fragment.\n\n"
        "Practical guidance:\n"
        "Explain how to apply or interpret the evidence. Include examples or distinctions when the evidence supports them. "
        "If evidence is insufficient, clearly say what is missing.\n\n"
        "Evidence used:\n"
        "- [source_id / chunk_id] title, page/section\n\n"
        "Safety note:\n"
        "Add a brief safety note only when relevant. If no special safety issue is present, say that no special safety "
        "concern was identified from the evidence."
    )


def build_query_planner_prompt(
    question: str,
    topic_hint: str | None,
) -> str:
    """Build the live query-planner prompt.

    Parameters:
        question: normalized user question.
        topic_hint: optional user-provided topic hint.
    Returns:
        str: JSON-only planning prompt for retrieval preparation.
    """

    topic_hint_text = topic_hint or ""
    return (
        "You are the QueryPlanner for an evidence-based personal trainer AI.\n"
        "Your job is to improve retrieval quality before search happens.\n\n"
        "You must help with four things:\n"
        "1. Clean the user's wording without changing intent.\n"
        "2. Clarify shorthand, abbreviations, or implicit meaning when the likely meaning is obvious.\n"
        "3. Expand the query into retrieval-friendly terminology, including both the user's language and helpful "
        "English fitness or research terms when useful.\n"
        "4. Keep the output concise, specific, and faithful to the original question.\n\n"
        "Rules:\n"
        "- Do not answer the user's question.\n"
        "- Do not add facts that change the meaning of the question.\n"
        "- If the question is vague, make it more retrieval-friendly but keep uncertainty visible.\n"
        "- If the user writes in Chinese, you may include Chinese and English query terms.\n"
        "- Prefer terms that are likely to appear in training, nutrition, sports science, or medical-fitness texts.\n"
        "- Return valid JSON only. No markdown. No explanation.\n\n"
        f"Question:\n{question}\n\n"
        f"Topic hint:\n{topic_hint_text}\n\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "clean_question": "short cleaned version of the user question",\n'
        '  "clarified_question": "self-contained retrieval-friendly version of the question",\n'
        '  "search_query": "dense retrieval query string for semantic and keyword search",\n'
        '  "query_terms": ["term 1", "term 2", "term 3"],\n'
        '  "notes": ["short note about abbreviation expansion or ambiguity if useful"]\n'
        "}\n\n"
        "Requirements for fields:\n"
        "- clean_question: keep close to the user's wording.\n"
        "- clarified_question: make the topic explicit enough for retrieval.\n"
        "- search_query: 1 compact retrieval string, not a paragraph.\n"
        "- query_terms: 4 to 10 focused terms or short phrases.\n"
        "- notes: optional short notes; use an empty list if not needed."
    )
