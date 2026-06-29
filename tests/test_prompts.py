"""Prompt quality checks for LLM generation nodes."""

from app.qa.prompts import build_answer_prompt, build_query_planner_prompt
from app.program.prompts import build_program_evidence_query_planner_prompt, build_program_sop_step_prompt


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


def _program_sop_prompt(step: str) -> str:
    return build_program_sop_step_prompt(
        task_type="program_generation",
        sop_step=step,
        user_message="Create a strength plan.",
        collected_inputs={"goal": "strength", "experience": "intermediate"},
        memory_context="profile: test",
        weekly_summaries=[],
        evidence=[],
        previous_decisions={},
        raw_plan={"sessions": []},
    )


def test_program_sop_prompt_includes_only_current_step_rules():
    prompt = _program_sop_prompt("periodization")

    assert "Use `decision.periodization`" in prompt
    assert "deload_week_numbers" in prompt
    assert "programming_intensity_and_volume" not in prompt
    assert "weekly_estimated_progression_kg" not in prompt
    assert "session_exercise_order" not in prompt


def test_program_intensity_prompt_keeps_intensity_schema():
    prompt = _program_sop_prompt("programming_intensity_and_volume")

    assert "Use `decision.programming_intensity_and_volume`" in prompt
    assert "weekly_estimated_progression_kg" in prompt
    assert "deload_session_intensity_reduction_percent" in prompt
    assert "session_exercise_order" in prompt


def test_program_evidence_query_planner_prompt_uses_current_sop_rules():
    prompt = build_program_evidence_query_planner_prompt(
        sop_step="programming_intensity_and_volume",
        user_message="Create a strength plan.",
        collected_inputs={"goal": "strength", "experience": "intermediate"},
        relevant_previous_decisions={"periodization": {"weeks": 12, "deload_week_numbers": [4, 8, 12]}},
        attempt=1,
    )

    assert "EvidenceQueryPlanner" in prompt
    assert "weekly_estimated_progression_kg" in prompt
    assert "deload_session_intensity_reduction_percent" in prompt
    assert '"queries"' in prompt
    assert '"purpose"' in prompt
