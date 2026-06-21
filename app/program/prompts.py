"""Prompts for structured program design and adaptation."""

import json
from typing import Any


def build_program_plan_prompt(
    task_type: str,
    user_message: str,
    collected_inputs: dict[str, Any],
    memory_context: str,
    weekly_summaries: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    active_plan: dict[str, Any] | None,
) -> str:
    """Build a JSON-only prompt; deterministic code validates all numbers later."""

    active_plan_payload = active_plan or {}
    return f"""You are the program-design reasoning component of an evidence-based personal trainer.
Task: {task_type}

Follow the applicable SOP in order. For a new program cover screening, needs analysis, periodization,
planning model, programming decisions, exercise selection, frequency, exercise order, load/reps,
volume, rest periods, and final review. For adaptation protect the goal, classify the trigger,
change the smallest coherent portion of the future plan, and avoid stacking missed work.

For a missed or impossible session, first separate a schedule conflict from recovery, illness, pain,
or fatigue. Protect the training goal rather than the exact date. Reschedule conflicts within the
current or nearby microcycle; for recovery limitations reduce fatigue and leave recovery before the
next hard session. Redistribute key work coherently, adjust frequency then volume then intensity then
exercise selection/order then rest, avoid failure-based work when fatigue is high, and state what
future monitoring should trigger another revision.

Rules:
- Do not diagnose or provide medical clearance.
- Use only supplied user facts and evidence. State conservative assumptions.
- Produce a four-week plan unless the user explicitly requires another duration.
- Return session templates for one week; deterministic code expands dates and weeks.
- Numeric suggestions are proposals and will be recalculated and traced by tools.
- Return JSON only, with no markdown.

User message:
{user_message}

Collected inputs:
{json.dumps(collected_inputs, ensure_ascii=False)}

Memory context:
{memory_context}

Latest completed weekly summaries:
{json.dumps(weekly_summaries, ensure_ascii=False)}

Approved evidence:
{json.dumps(evidence, ensure_ascii=False)}

Current active plan for adaptation:
{json.dumps(active_plan_payload, ensure_ascii=False)}

Return this shape:
{{
  "title": "short plan title",
  "goal": "primary goal",
  "sport": "sport or general_fitness",
  "rationale": "concise rationale linked to supplied facts and evidence",
  "screening": {{"status": "clear|limited|medical_clearance_required", "notes": []}},
  "needs_analysis": {{"goal": "", "experience": "", "equipment": "", "constraints": []}},
  "periodization": {{"weeks": 4, "model": "parallel|sequential|emphasis", "phases": []}},
  "assumptions": [],
  "warnings": [],
  "sessions": [
    {{
      "name": "session name",
      "goal": "session goal",
      "duration_minutes": 60,
      "exercises": [
        {{"name": "exercise", "mode": "resistance|aerobic", "sets": 3, "reps": 8,
          "intensity_percent": 70, "rpe": 7, "rest_seconds": 120,
          "duration_seconds": null, "distance_meters": null, "pace_seconds_per_km": null, "notes": ""}}
      ]
    }}
  ],
  "changed_elements": [],
  "unchanged_elements": []
}}"""
