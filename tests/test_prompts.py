"""Prompt quality checks for LLM generation nodes."""

from app.qa.prompts import build_answer_prompt, build_query_planner_prompt


def test_answer_prompt_requests_explanatory_grounded_style():
    """The live prompt should ask for richer answers without losing evidence discipline."""

    prompt = build_answer_prompt(
        question="What does VO2max mean?",
        evidence_context="[src_a / chk_a] Example Source, page 1\nVO2max means maximal oxygen uptake.",
        safety_mode="normal",
        safety_flags=[],
        max_tokens=1500,
    )

    assert "Use only the evidence context" in prompt
    assert "do not sound mechanical or overly terse" in prompt
    assert "Write like a thoughtful coach-educator" in prompt
    assert "Avoid one-line, dry answers" in prompt
    assert "you have at most about 1500 output tokens" in prompt
    assert "Avoid large tables" in prompt
    assert "starting a table that cannot fit" in prompt
    assert "Do not invent citations" in prompt
    assert "Use Arabic-number citations" in prompt
    assert "write 'text [2].', not 'text.[2]'" in prompt
    assert "Do not repeatedly say phrases like" in prompt
    assert "Evidence used:\n1. [source_id / chunk_id] title, page/section" in prompt


def test_query_planner_prompt_requests_json_and_query_expansion():
    """The planner prompt should ask for structured, retrieval-friendly output."""

    prompt = build_query_planner_prompt(
        question="VO2max means what",
        topic_hint="running",
    )

    assert "Return valid JSON only" in prompt
    assert '"clean_question"' in prompt
    assert '"clarified_question"' in prompt
    assert '"search_query"' in prompt
    assert '"query_terms"' in prompt
    assert "include Chinese and English query terms" in prompt
