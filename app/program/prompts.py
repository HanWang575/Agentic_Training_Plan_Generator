"""Prompts for structured program generation."""

import json
from typing import Any


def build_program_task_router_prompt(
    user_turn: str,
    pending_action: str | None,
    active_task_type: str | None,
    active_sop: str | None,
    current_sop_step: str | None,
    latest_version_id: str | None,
    approval_status: str | None,
) -> str:
    """Build the JSON-only prompt for Program Generator task routing."""

    return f"""You are the TaskAndSOPRouter for a strength-program generator.
Classify the user's current turn into exactly one route.

Allowed routes:
- new_program: design a separate new program rather than changing an existing one.
- explain_plan_rationale: explain the current generated plan or its calculations without changing it.
- clarification_answer: answer requested information for the currently pending planning task.

Rules:
- Use meaning, not keyword matching.
- Use clarification_answer only when the turn actually supplies or answers requested information for the pending task.
- Questions asking why/how without requesting a new program are rationale.
- Treat planning requests as new_program unless they are a clarification answer or a why/how rationale question.
- If no plan version exists, do not claim an existing plan can be explained; choose new_program unless this is a clarification answer.
- Return JSON only.

Current user turn:
{user_turn}

Workflow state:
{json.dumps({
    "pending_action": pending_action,
    "active_task_type": active_task_type,
    "active_sop": active_sop,
    "current_sop_step": current_sop_step,
    "latest_version_id": latest_version_id,
    "approval_status": approval_status,
}, ensure_ascii=False)}

Return exactly:
{{
  "route": "new_program | explain_plan_rationale | clarification_answer",
  "rationale": "short classification reason",
  "confidence": 0.0
}}"""


def build_program_rationale_prompt(
    question: str,
    sop_context: dict[str, Any],
    recorded_rationale: str,
    numeric_facts: dict[str, Any] | None,
) -> str:
    """Build the grounded prompt for a question-specific plan rationale."""

    return f"""You are the RationaleAnswerBuilder for a strength-program generator.
Answer the user's exact question about the current generated plan.

Rules:
- Ground the answer only in the supplied SOP/context record, recorded rationale, and numeric facts.
- Tailor the explanation to what the user actually asked; do not merely repeat the broad rationale.
- If numeric facts are supplied, reconstruct the calculation step by step from those facts.
- If `matched_prescriptions` are supplied, answer from those matched prescriptions first; do not switch to an unrelated loaded exercise merely because it has a fuller calculation chain.
- For bodyweight prescriptions with `load_kg=null`, explain the saved reps, RPE, rest, notes, and recorded rationale; do not claim they belong to another weighted exercise.
- Inspect the complete calculation chain and named progression/percentage/rounding facts before claiming that arithmetic is missing.
- Clearly distinguish the value used by the saved plan from a newer value reported by the user.
- Do not invent calculations, evidence, user facts, or design decisions.
- Do not modify the plan or promise that a plan change has occurred.
- If the recorded context cannot answer part of the question, say what is missing.
- Return JSON only.

User question:
{question}

SOP progress and context:
{json.dumps(sop_context, ensure_ascii=False, default=str)}

Recorded plan rationale:
{recorded_rationale}

Deterministically selected numeric facts:
{json.dumps(numeric_facts or {}, ensure_ascii=False, default=str)}

Return exactly:
{{"answer": "question-specific rationale"}}"""


def build_program_lift_max_extraction_prompt(user_message: str) -> str:
    """Build the JSON-only prompt for current lift-max extraction."""

    return f"""You are the LiftMaxExtractor node for a strength-program generator.
Extract only current, user-reported maximum strength values from the current user message.

Rules:
- Extract current 1RM / PR / max values only.
- Treat "SBD 110kg, 115kg, 150kg" as squat=110 kg, bench_press=115 kg, deadlift=150 kg.
- Do not extract future goals or targets such as "bench 120 kg by the end of the year".
- Do not extract bodyweight, height, age, volume, reps, working sets, or recent performance sets.
- Do not convert a recent set like "Overhead Press: 45 kg x 8 reps" into a 1RM.
- Do not extract weighted pull-up values into bench, squat, deadlift, or overhead press.
- Use only this user message. Do not infer from memory or previous turns.
- Return JSON only.

Current user message:
{user_message}

Allowed keys:
- squat_1rm_kg
- bench_press_1rm_kg
- deadlift_1rm_kg
- overhead_press_1rm_kg

Return exactly:
{{
  "lift_maxes": [
    {{
      "key": "bench_press_1rm_kg",
      "value_kg": 115,
      "source_text": "My bench PR is 115 kg",
      "confidence": 0.95
    }}
  ],
  "ignored": [
    {{"source_text": "bench 120kg by the end of the year", "reason": "future target, not current max"}}
  ]
}}"""


def build_program_input_extraction_prompt(
    user_message: str,
    recent_status_text: str,
    profile_text: str,
) -> str:
    """Build the JSON-only prompt for Program Generator input extraction."""

    return f"""You are the ProgramInputExtractor node for a strength-program generator.
Extract only concrete planning inputs supported by the schema below.

Rules:
- Extract each field from the highest-priority source that answers it.
- Priority order is: current user message, then recent status text, then profile text.
- If the current user message answers a field, always use that value even when recent status or profile disagree.
- Use recent status only when the current user message does not answer the field.
- Use profile text only when neither the current user message nor recent status answers the field.
- Do not invent values.
- For screening, return "clear" only when the selected source confirms no injuries, pain, medical limitations, relevant illness, medication concern, or clearance issue.
- For screening, return "limited" when the selected source reports pain, injury, illness, medication changes, medical limitations, or uncertainty requiring conservative handling.
- If screening is not stated or cannot be inferred, omit it.
- Convert days_per_week and session_duration_minutes to integers when stated.
- Keep equipment as a short comma-separated string.
- Return JSON only.

Current user message:
{user_message}

Recent status text:
{recent_status_text}

Profile text:
{profile_text}

Allowed output keys:
- goal: strength | hypertrophy | endurance | general fitness
- days_per_week: integer 1-7
- session_duration_minutes: integer 20-180
- experience: beginner | intermediate | advanced
- equipment: string
- screening: clear | limited

Return exactly:
{{
  "inputs": {{
    "goal": "strength",
    "days_per_week": 3,
    "session_duration_minutes": 60,
    "experience": "intermediate",
    "equipment": "full gym, barbell",
    "screening": "clear"
  }},
  "ignored": [
    {{"text": "unsupported fact", "reason": "not a required planning input"}}
  ]
}}"""


def build_program_sop_step_prompt(
    task_type: str,
    sop_step: str,
    user_message: str,
    collected_inputs: dict[str, Any],
    memory_context: str,
    weekly_summaries: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    previous_decisions: dict[str, Any],
    raw_plan: dict[str, Any],
) -> str:
    """Build the JSON-only prompt for one SOP decision in stepwise plan generation."""

    return f"""You are the SOPStepExecutor for an evidence-based strength-program generator.
Make only the decision for the current SOP step. Do not write the whole plan here.

Rules:
- Use the supplied MemoryContext, current-step evidence, previous SOP decisions, and existing raw_plan.
- Preserve safety: do not diagnose or provide medical clearance.
- State conservative assumptions when user facts are missing.
- Do not calculate load_kg or weekly/session volume in this step; calculator steps do that later.
- Put the current step's structured decision inside the `decision` object.
- In the current step's structured decision always includes a 'rationale' field.
- For consultation_and_screening, use `decision.screening`.
- For needs_analysis, use `decision.needs_analysis`.
- For periodization, use `decision.periodization` with fields including but not limited to weeks, macrocycle, mesocycles, microcycles, phases, target_timeline, deload_week_numbers (1-based program week numbers indicating deload weeks), and rationale.
- For planning_model, use `decision.planning_model` with fields including model and rationale.
- For programming_exercises_and_order, use `decision.programming_exercises_and_order` with fields including but not limited to `program_exercise_list`, `session_exercise_order` and rationale. `program_exercise_list` is the unique exercise catalog, so each exercise appears once there. `session_exercise_order` is the arranged session plan, so the same exercise may appear in multiple sessions when the role differs.
- For programming_intensity_and_volume, use `decision.programming_intensity_and_volume`. Use `previous_decisions.periodization` to make week-by-week intensity and volume choices.
  Generate one compact section for each arranged exercise slot in `session_exercise_order`, not for each unique exercise in `program_exercise_list`. If the same lift appears in two sessions, create two separate sections.
  {{"session_index": "1-based training-day index",
  "exercise_order": "1-based order within that session",
  "lift": "lift_name",
  "slot_role": "short role such as volume, intensity, technique, accessory, or deload",
  "weekly_primary_exposure": "short description",
  "sets_per_week": ["one integer per program week, range [1,20], deload behaviour should not be considered when setting this integer, for deload weeks the value should be the same as previous week"],
  "reps_per_set": ["one integer per program week, range [1,15], deload behaviour should not be considered when setting this integer, for deload weeks the value should be the same as previous week"],
  "weekly_estimated_progression_kg": ["one number per program week; expected increase in this exercise slot's estimated 1RM for that week, not cumulative; week 1 should usually be 0; deload weeks should usually be 0, choose in [0, 0.125, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]"],
  "deload_session_intensity_reduction_percent": ["one number per program week; must be 0 only for non-deload weeks, non-zero only for weeks in `deload_week_numbers`, range [0, 100]"],
  "deload_session_reps_reduction_percent": ["one number per program week; must be 0 for non-deload weeks, non-zero only for weeks in `deload_week_numbers`, range [0, 100]"],
  "target_rpe": ["one number per program week, range [1, 10]"],
  "rationale": "explain the rationale behind briefly."}}.
  The deterministic calculator derives intensity_percent and load_kg from `reps_per_set`, `target_rpe`, and cumulative `weekly_estimated_progression_kg`; do not output `week1_intensity_percent_1rm`, `base_intensity_percent_1rm`, or `weekly_progression_percent`.
  Each list length must equal `decision.periodization.weeks`. Do not express deload/taper by setting `sets_per_week` to 0 or by pre-reducing `reps_per_set`; deload/taper is handled only by `deload_session_intensity_reduction_percent`, `deload_session_reps_reduction_percent`, lower `target_rpe`, and `weekly_estimated_progression_kg: 0`.
  Use `sets_per_week: 0` only when the exercise slot is intentionally absent for a non-deload reason.
  For a repeated lift, `sets_per_week` is the weekly set count assigned to that specific session slot; the lift's total weekly set count is the sum of its slots.
- Return JSON only.

Task type:
{task_type}

Current SOP step:
{sop_step}

User request:
{user_message}

Collected inputs:
{json.dumps(collected_inputs, ensure_ascii=False)}

Memory context:
{memory_context}

Latest completed weekly summaries:
{json.dumps(weekly_summaries, ensure_ascii=False)}

Current-step approved evidence:
{json.dumps(evidence, ensure_ascii=False)}

Previous SOP decisions:
{json.dumps(previous_decisions, ensure_ascii=False, default=str)}

Existing raw_plan:
{json.dumps(raw_plan, ensure_ascii=False, default=str)}

Return exactly:
{{
  "step": "{sop_step}",
  "decision": {{"structured_fields_for_current_step": "values"}},
  "rationale": "short reason grounded in the supplied context",
  "assumptions": [],
  "warnings": []
}}"""


def build_program_plan_update_prompt(
    sop_step: str,
    decision: dict[str, Any],
    raw_plan: dict[str, Any],
    collected_inputs: dict[str, Any],
    previous_decisions: dict[str, Any],
) -> str:
    """Build the JSON-only prompt for updating the structured raw_plan."""

    return f"""You are the PlanUpdater for a strength-program generator.
Update the supplied raw_plan according to the current SOP decision.

Rules:
- Return the complete updated raw_plan JSON object, not a patch.
- Preserve the raw_plan schema and keep templates.session, templates.exercise, and templates.set.
- Fill, expand, compact, or update sessions/exercises/sets only as needed by the current SOP decision.
- Keep one week of session templates in raw_plan.sessions; deterministic code expands them across weeks.
- Every exercise must include load_type: "weighted" or "bodyweight".
- All weighted exercises must include none-zero intensity_percent, sets, reps, rpe, and rest_seconds proposals, but do not invent load_kg.
- Bodyweight exercises may keep load_kg null.
- Do not add calculated session dates, generated IDs, calculation_trace_ids, or database fields.
- Do not include screening, needs_analysis, periodization, planning_model, changed_elements, or unchanged_elements in raw_plan; those belong in sop_progress.decisions.
- Return JSON only, with no markdown.

Current SOP step:
{sop_step}

Current SOP decision:
{json.dumps(decision, ensure_ascii=False, default=str)}

Collected inputs:
{json.dumps(collected_inputs, ensure_ascii=False)}

Previous SOP decisions:
{json.dumps(previous_decisions, ensure_ascii=False, default=str)}

Current raw_plan:
{json.dumps(raw_plan, ensure_ascii=False, default=str)}

Return the complete updated raw_plan object."""
