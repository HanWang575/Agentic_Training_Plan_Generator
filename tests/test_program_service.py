"""Program Generator workflow, persistence, safety, and calculator tests."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import re

import pytest
from pydantic import ValidationError
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.core.llm_provider import OpenAIProvider, ProgramTaskRouteResult, StructuredOutputError
from app.knowledge.schemas import RetrievedChunk
from app.memory.service import MemoryService
from app.memory.schemas import PlanProgressPatchRequest
from app.program.calculators import CalculatorService
from app.program.nodes import ProgramContextPatchValidator, ReasoningActionRouter, TaskAndSOPRouter
from app.program.schemas import ProgramContextPatch, ProgramReviewRequest, ProgramTurnRequest
from app.program.service import (
    NEW_PROGRAM_SOP,
    ProgramService,
    _extract_exercise_load_references,
    _extract_lift_maxes,
    _intensity_volume_policy_from_sop_progress,
)
from app.storage.database import create_app_engine, init_db
from app.storage.models import (
    CalculationTrace,
    ConversationMessage,
    Episode,
    ProgramContextRecord,
    ProgramTrace,
    ProgramVersion,
    SetPrescription,
    TrainingPlanProgress,
    TrainingSession,
)


class FakeProgramProvider:
    """Return a compact but complete program proposal without network calls."""

    def __init__(self):
        self.lift_max_requests = []
        self.input_requests = []
        self.sop_requests = []

    def extract_program_lift_maxes(self, request):
        self.lift_max_requests.append(request.user_message)
        return _extract_lift_maxes(request.user_message)

    def extract_program_inputs(self, request):
        self.input_requests.append(request.user_message)
        result = {}
        result.update(self._extract_inputs_from_text(request.profile_text))
        result.update(self._extract_inputs_from_text(request.recent_status_text))
        result.update(self._extract_inputs_from_text(request.user_message))
        return result

    def _extract_inputs_from_text(self, text):
        text = text.lower()
        result = {}
        if "strength" in text or "powerlifting" in text:
            result["goal"] = "strength"
        elif "hypertrophy" in text or "muscle" in text:
            result["goal"] = "hypertrophy"
        elif "endurance" in text:
            result["goal"] = "endurance"
        for level in ("beginner", "intermediate", "advanced"):
            if level in text:
                result["experience"] = level
                break
        if "commercial gym" in text:
            result["equipment"] = "commercial gym"
        elif "full gym" in text or "barbell" in text:
            result["equipment"] = "full gym, barbell"
        elif "bodyweight" in text:
            result["equipment"] = "bodyweight"
        if "injury-free" in text or "no injuries" in text or "nothing hurts" in text:
            result["screening"] = "clear"
        elif "pain" in text or "injury" in text or "medical" in text:
            result["screening"] = "limited"
        day_match = re.search(r"\b([1-7])\s*(?:days?|times?)\s*(?:per|a)?\s*week", text)
        if not day_match:
            day_match = re.search(r"days(?:_per_week| per week)?\s*[:=]\s*([1-7])", text)
        if day_match:
            result["days_per_week"] = int(day_match.group(1))
        else:
            for word, value in {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
            }.items():
                if re.search(rf"\b{word}\s+(?:days?|times?)\s+(?:per|a)\s+week", text):
                    result["days_per_week"] = value
                    break
        duration_match = re.search(r"\b(\d{2,3})\s*(?:minutes?|mins?)", text)
        if not duration_match:
            duration_match = re.search(r"(?:duration|session_duration_minutes)\s*[:=]\s*(\d{2,3})", text)
        if duration_match:
            result["session_duration_minutes"] = int(duration_match.group(1))
        return result

    def generate_program_task_route(self, request):
        message = request.user_turn.lower()
        if any(term in message for term in ("why", "rationale", "how did", "calculated", "calculation")):
            route = "explain_plan_rationale"
        elif request.pending_action == "waiting_for_user":
            route = "clarification_answer"
        else:
            route = "new_program"
        return ProgramTaskRouteResult(route=route, rationale="test route", confidence=1.0)

    def generate_program_rationale(self, request):
        answer = request.recorded_rationale or "No recorded rationale."
        facts = request.numeric_facts or {}
        prescription = facts.get("selected_prescription", {})
        if prescription:
            answer += f" Selected {prescription.get('exercise')} at {prescription.get('intensity_percent'):g}%."
        progression = facts.get("progression", {})
        if progression:
            inputs = progression.get("inputs", {})
            result = progression.get("result", {})
            answer += (
                f" {inputs.get('current_value'):g} * (1 + {inputs.get('percentage'):g} / 100) "
                f"= {result.get('value'):g}%."
            )
        matched = facts.get("matched_prescriptions", [])
        if matched:
            answer += " Matched prescriptions: " + ", ".join(
                f"{item.get('exercise')} {item.get('reps')} reps RPE {item.get('rpe')}"
                for item in matched
            )
        comparison = facts.get("one_rm_comparison", {})
        if comparison:
            answer += f" The saved plan used a {comparison.get('saved_plan_one_rm_kg'):g} kg 1RM."
            reported = comparison.get("reported_in_current_question_kg")
            if reported is not None:
                answer += f" The current question reports {reported:g} kg."
        return answer

    def _base_plan(self, raw_plan, inputs, task_type="program_generation"):
        plan = dict(raw_plan or {})
        plan.update(
            {
                "schema": "raw_training_plan_v1",
                "title": "Four Week Strength Foundation",
                "goal": inputs.get("goal", "strength"),
                "sport": "strength",
                "rationale": "The plan matches the user's schedule, experience, equipment, and approved evidence.",
                "assumptions": [
                    ["Bench press 1RM is 115 kg and bench loading is anchored to that value."],
                    ["No current pain, illness, or medical limitation was reported."],
                ],
                "warnings": [],
                "sessions": [
                    {
                        "name": "Squat and Bench",
                        "goal": "strength",
                        "duration_minutes": inputs.get("session_duration_minutes", 60),
                        "exercises": [
                            {"name": "Back Squat", "load_type": "weighted", "sets": 3, "reps": 5, "intensity_percent": 75, "rpe": 7,
                             "rest_seconds": 180},
                            {"name": "Bench Press", "load_type": "weighted", "sets": 3, "reps": 6, "intensity_percent": 72, "rpe": 7,
                             "rest_seconds": 150},
                        ],
                    },
                    {
                        "name": "Deadlift and Press",
                        "goal": "strength",
                        "duration_minutes": inputs.get("session_duration_minutes", 60),
                        "exercises": [
                            {"name": "Deadlift", "load_type": "weighted", "sets": 3, "reps": 4, "intensity_percent": 78, "rpe": 7.5,
                             "rest_seconds": 180},
                            {"name": "Overhead Press", "load_type": "weighted", "sets": 3, "reps": 6, "intensity_percent": 70, "rpe": 7,
                             "rest_seconds": 120},
                        ],
                    },
                    {
                        "name": "Full Body Volume",
                        "goal": "hypertrophy",
                        "duration_minutes": inputs.get("session_duration_minutes", 60),
                        "exercises": [
                            {"name": "Front Squat", "load_type": "weighted", "sets": 3, "reps": 8, "intensity_percent": 65, "rpe": 7,
                             "rest_seconds": 120},
                        ],
                    },
                ],
            }
        )
        plan.setdefault(
            "templates",
            {
                "session": {"name": "", "goal": "", "duration_minutes": inputs.get("session_duration_minutes", 60), "exercises": []},
                "exercise": {"name": "", "mode": "resistance", "load_type": "weighted", "sets": 0, "reps": 0},
                "set": {"set_number": 1, "reps": 0, "load_kg": None},
            },
        )
        return plan

    def generate_program_sop_step_decision(self, request):
        self.sop_requests.append(request)
        decisions = {
            "consultation_and_screening": {"screening": {"status": "clear", "notes": []}},
            "needs_analysis": {
                "needs_analysis": {
                    "goal": request.collected_inputs.get("goal"),
                    "experience": request.collected_inputs.get("experience"),
                    "equipment": request.collected_inputs.get("equipment"),
                    "constraints": [],
                }
            },
            "periodization": {"periodization": {"weeks": 4, "model": "parallel", "phases": ["foundation"]}},
            "planning_model": {
                "planning_model": {
                    "model": "parallel",
                    "rationale": "Train all main strength qualities together.",
                }
            },
        }
        return {
            "step": request.sop_step,
            "decision": decisions.get(request.sop_step, {"summary": f"decision for {request.sop_step}"}),
            "rationale": "test decision",
            "assumptions": [],
            "warnings": [],
        }

    def update_program_plan(self, request):
        plan = dict(request.raw_plan)
        inputs = request.collected_inputs
        if request.sop_step in {
            "programming_exercises_and_order",
            "programming_intensity_and_volume",
            "rest_periods",
        }:
            plan = self._base_plan(plan, inputs)
        elif request.sop_step == "consultation_and_screening":
            plan["title"] = "Four Week Strength Foundation"
            plan["goal"] = inputs.get("goal", "strength")
            plan["rationale"] = "The plan matches the user's schedule, experience, equipment, and approved evidence."
        return plan

class FakeKnowledgeService:
    def search(self, query, topic_hint=None, keyword_top_k=None, semantic_top_k=None):
        chunk = RetrievedChunk(
            source_id="src_strength",
            chunk_id="chk_programming",
            title="Approved strength programming source",
            text="Match frequency, intensity, volume, and rest to the trainee's goal and status.",
            page_number=10,
            section_heading="Program design",
            topic_tags=["strength"],
            token_count=20,
            score=0.9,
            rank=1,
            retrieval_method="keyword",
        )
        return [chunk], [], None


def _make_service(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        sqlite_path=tmp_path / "app.sqlite",
        upload_dir=tmp_path / "uploads",
        chroma_path=tmp_path / "chroma",
        openai_api_key=None,
    )
    engine = create_app_engine(settings)
    init_db(engine)
    session = Session(engine)
    provider = FakeProgramProvider()
    memory = MemoryService(session=session, settings=settings, llm_provider=provider)
    memory.create_user("usr_program", timezone_name="Asia/Shanghai")
    memory.update_profile(
        "usr_program",
        {
            "primary_training_goal": "strength",
            "available_training_days": 3,
            "typical_session_duration_minutes": 60,
            "experience_level": "intermediate",
            "available_equipment": "full gym, barbell",
            "medical_constraints_user_reported": "no medical issues",
            "squat_1rm_kg": 140,
            "bench_press_1rm_kg": 100,
            "deadlift_1rm_kg": 180,
            "overhead_press_1rm_kg": 65,
        },
    )
    service = ProgramService(
        session=session,
        settings=settings,
        memory_service=memory,
        knowledge_service=FakeKnowledgeService(),
        llm_provider=provider,
    )
    return session, memory, service


REQUIRED_INPUT_REPLY = (
    "goal: strength; days_per_week: 3; duration: 60; experience: intermediate; "
    "equipment: full gym and barbell; screening: no injuries"
)


def _start_new_program(
    service: ProgramService,
    conversation_id: str,
    user_id: str = "usr_program",
    message: str = "Create a strength plan. My bench PR is 115 kg.",
    reply: str = REQUIRED_INPUT_REPLY,
):
    first = service.process_turn(
        conversation_id,
        ProgramTurnRequest(user_id=user_id, message=message),
    )
    if first.status != "waiting_for_user":
        return first
    assert "Before I design this safely" in first.response
    return service.process_turn(
        conversation_id,
        ProgramTurnRequest(user_id=user_id, message=reply),
    )


def test_calculators_are_deterministic_and_safe(tmp_path: Path):
    settings = Settings(_env_file=None, sqlite_path=tmp_path / "app.sqlite")
    calculators = CalculatorService(settings)

    one_rm = calculators.run("estimate_repetition_max", {"weight": 100, "reps": 5})
    load = calculators.run("calculate_percentage_load", {"one_rm": one_rm.value, "percentage": 75})
    rounded = calculators.run("round_training_load", {"load": load.value, "increment": 2.5})
    reps_rpe_intensity = calculators.run("calculate_reps_rpe_intensity", {"reps": 5, "target_rpe": 8})

    assert one_rm.value == pytest.approx(116.67, abs=0.01)
    assert reps_rpe_intensity.value == pytest.approx(81.08, abs=0.01)
    assert rounded.value % 2.5 == 0
    with pytest.raises(ValueError):
        calculators.run("calculate_expression", {"expression": "__import__('os').system('echo no')"})
    with pytest.raises(ValidationError):
        ProgramTurnRequest(user_id="", message="Make a plan")


def test_lift_max_extraction_does_not_cross_exercise_boundaries():
    request = (
        "I am 29 yo male. Height 175 cm, BW 80kgs. I have been trained in gym for almost 3 years. "
        "SBD 110kg, 115kg, 150kg. For deadlift my PR is 155kg but I have been stopped training "
        "deadlift for months. I just finished 1RM test for bench press.I train 3-4 days per week. "
        "I can do about 25 BW pullups and a 55kg weighted pullup. Write a strength program for me. "
        "My primary goal is to achieve 400 total in Big 3, bench 120kg, and be able to do a single "
        "arm pull up by the end of the year."
    )
    clarification = "Weighted Pull-Up 1RM: 55kg; Overhead Press: 45 kg x 8 reps"

    assert _extract_lift_maxes(request) == {
        "squat_1rm_kg": 110.0,
        "bench_press_1rm_kg": 115.0,
        "deadlift_1rm_kg": 155.0,
    }
    assert _extract_lift_maxes(clarification) == {}


def test_program_service_uses_llm_input_and_lift_extractors(tmp_path: Path):
    _, _, service = _make_service(tmp_path)

    inputs = service._extract_inputs("My bench PR is 116 kg.", "")
    natural_inputs = service._extract_inputs(
        "I want strength, can train four days a week for 75 minutes, have intermediate experience, "
        "use a commercial gym, and I'm injury-free.",
        "",
    )

    assert inputs["bench_press_1rm_kg"] == 116.0
    assert natural_inputs["goal"] == "strength"
    assert natural_inputs["session_duration_minutes"] == 75
    assert natural_inputs["experience"] == "intermediate"
    assert natural_inputs["equipment"] == "commercial gym"
    assert natural_inputs["screening"] == "clear"
    assert service.llm_provider.lift_max_requests == [
        "My bench PR is 116 kg.",
        (
            "I want strength, can train four days a week for 75 minutes, have intermediate experience, "
            "use a commercial gym, and I'm injury-free."
        ),
    ]
    assert service.llm_provider.input_requests == [
        "My bench PR is 116 kg.",
        (
            "I want strength, can train four days a week for 75 minutes, have intermediate experience, "
            "use a commercial gym, and I'm injury-free."
        ),
    ]


def test_input_extraction_priority_and_required_prompt_gate(tmp_path: Path):
    _, _, service = _make_service(tmp_path)

    from_three_sources = service._extract_inputs(
        "I want hypertrophy, train 4 days per week for 75 minutes, have advanced experience, "
        "use a commercial gym, and no injuries.",
        profile_text=(
            "primary_training_goal: strength; available_training_days: 3; "
            "typical_session_duration_minutes: 60; experience_level: intermediate; "
            "available_equipment: full gym, barbell; medical_constraints_user_reported: medical issue"
        ),
        recent_status_text=(
            "Recently training for endurance, two days per week, 45 minutes, beginner, "
            "bodyweight only, knee pain."
        ),
    )
    recent_over_profile = service._extract_inputs(
        "",
        profile_text="strength, three days per week, 60 minutes, intermediate, full gym, no medical issues",
        recent_status_text="hypertrophy, four days per week, 75 minutes, advanced, commercial gym, knee pain",
    )
    first = service.process_turn(
        "conv_required_gate",
        ProgramTurnRequest(user_id="usr_program", message="Create a new strength plan."),
    )

    assert from_three_sources["goal"] == "hypertrophy"
    assert from_three_sources["days_per_week"] == 4
    assert from_three_sources["session_duration_minutes"] == 75
    assert from_three_sources["experience"] == "advanced"
    assert from_three_sources["equipment"] == "commercial gym"
    assert from_three_sources["screening"] == "clear"
    assert recent_over_profile["goal"] == "hypertrophy"
    assert recent_over_profile["days_per_week"] == 4
    assert recent_over_profile["session_duration_minutes"] == 75
    assert recent_over_profile["experience"] == "advanced"
    assert recent_over_profile["equipment"] == "commercial gym"
    assert recent_over_profile["screening"] == "limited"
    assert first.status == "waiting_for_user"
    assert "Before I design this safely" in first.response
    assert "primary training goal" not in first.response
    assert "available training days per week" in first.response
    assert service.get_context(first.context_id).sections["new_program_working_state"]["required_inputs_prompted"] is True


def test_new_program_is_saved_with_traced_numeric_prescriptions(tmp_path: Path):
    session, memory, service = _make_service(tmp_path)
    response = _start_new_program(
        service,
        "conv_program",
        message=(
            "Create a strength plan. days_per_week: 3; duration: 60; experience: intermediate; "
            "equipment: full gym and barbell; screening: no injuries. My bench PR is 115 kg."
        ),
    )

    assert response.status == "waiting_for_review"
    assert response.draft_program_id
    assert response.draft_version_id
    version = service.get_version(response.draft_program_id, response.draft_version_id)
    assert len(version.sessions) == 12
    assert all(item.exercises for item in version.sessions)
    assert all(item.duration_calculation_trace_id for item in version.sessions)
    assert all(set_item.calculation_trace_ids for item in version.sessions for ex in item.exercises for set_item in ex.sets)
    assert all(set_item.load_kg is not None for item in version.sessions for ex in item.exercises for set_item in ex.sets)
    assert all(ex.volume is not None and ex.calculation_trace_ids for item in version.sessions for ex in item.exercises)
    assert version.periodization["weekly_volume_load_kg"]
    assert version.screening == {"status": "clear", "notes": []}
    assert version.needs_analysis["experience"] == "intermediate"
    assert version.periodization["weeks"] == 4
    assert version.periodization["phases"] == ["foundation"]
    assert version.planning_model == "parallel"
    assert version.source_references["evidence"] == [
        {"source_id": "src_strength", "chunk_id": "chk_programming"}
    ]
    assert version.assumptions == [
        "Bench press 1RM is 115 kg and bench loading is anchored to that value.",
        "No current pain, illness, or medical limitation was reported.",
    ]
    assert "calculation_trace_ids" not in version.source_references
    assert session.exec(select(CalculationTrace)).all()
    percentage_traces = [
        item
        for item in session.exec(select(CalculationTrace)).all()
        if item.tool_name == "calculate_percentage_load"
    ]
    assert any(json.loads(item.inputs_json).get("one_rm") == 115 for item in percentage_traces)
    effective_one_rm_traces = [
        item
        for item in session.exec(select(CalculationTrace)).all()
        if item.tool_name == "calculate_expression"
        and json.loads(item.inputs_json).get("unit") == "kg"
        and json.loads(item.result_json).get("value") == 115
    ]
    assert any("current_user_turn_1rm" in json.loads(item.source_references_json) for item in effective_one_rm_traces)
    step_traces = [item for item in session.exec(select(ProgramTrace)).all() if item.action == "write_step_output"]
    assert len(step_traces) == len(NEW_PROGRAM_SOP)
    context = service.get_context(response.context_id)
    assert service.list_context_ids() == [response.context_id]
    stored_memory_context = context.sections["memory_context"]
    assert "rendered_context" not in stored_memory_context
    assert "debug" not in stored_memory_context
    assert stored_memory_context["sections"]["profile"]
    assert memory.get_memory_context(
        "usr_program",
        query="Create a strength plan",
        task_type="program_generator",
        include_debug=True,
    ).rendered_context
    assert context.sections["sop_progress"]["completed_steps"] == NEW_PROGRAM_SOP
    assert set(context.sections["sop_progress"]["evidence_sufficient_by_step"]) == (
        set(NEW_PROGRAM_SOP) - {"load_calculation", "volume_calculation"}
    )
    assert context.sections["retrieved_evidence"]["current_step"] == "rest_periods"
    assert len(context.sections["retrieved_evidence"]["items"]) == 1
    assert "sop_steps" not in context.sections["retrieved_evidence"]["items"][0]
    assert set(context.sections["sop_progress"]["evidence_refs_by_step"]) == (
        set(NEW_PROGRAM_SOP) - {"load_calculation", "volume_calculation"}
    )
    assert context.sections["sop_progress"]["evidence_refs_by_step"]["rest_periods"] == [
        {"source_id": "src_strength", "chunk_id": "chk_programming"}
    ]
    working = context.sections["new_program_working_state"]
    assert "calculated_plan" not in working
    assert working["raw_plan"]["title"] == "Four Week Strength Foundation"
    assert not {
        "screening",
        "needs_analysis",
        "periodization",
        "planning_model",
        "changed_elements",
        "unchanged_elements",
    } & set(working["raw_plan"])
    assert working["calculation_artifacts"]["schema"] == "calculated_plan_delta_v1"
    assert len(working["calculation_artifacts"]["calculated_sessions"]) == len(version.sessions)
    assert working["calculation_artifacts"]["calculated_periodization"]["weekly_volume_load_kg"]
    assert "step_outputs" not in working
    assert "calculation_results" not in context.sections
    assert "source_and_trace_references" not in context.sections
    assert service.llm_provider.sop_requests
    assert all("active_plan_progress" not in item.memory_context for item in service.llm_provider.sop_requests)
    latest_calculation_trace = max(
        (
            item
            for item in session.exec(select(ProgramTrace)).all()
            if item.context_id == response.context_id and item.action == "run_calculation"
        ),
        key=lambda item: item.created_at,
    )
    latest_calculation_ids = {
        item.calculation_trace_id
        for item in session.exec(
            select(CalculationTrace).where(CalculationTrace.trace_id == latest_calculation_trace.trace_id)
        ).all()
    }
    assert latest_calculation_ids
    assert len(latest_calculation_ids) < 250
    assert all(
        counters.get("rag_queries", 0) <= 5 and counters.get("user_questions", 0) <= 5
        for counters in context.sections["sop_progress"]["counters"].values()
    )
    session.close()


def test_materialization_uses_sop_periodization_and_intensity_volume_policy(tmp_path: Path):
    session, _, service = _make_service(tmp_path)
    original_decider = service.llm_provider.generate_program_sop_step_decision

    def policy_decider(request):
        if request.sop_step == "periodization":
            return {
                "step": request.sop_step,
                "decision": {
                    "periodization": {
                        "weeks": 5,
                        "deload_week_numbers": [3],
                        "phases": ["build", "deload", "build"],
                    }
                },
                "rationale": "test periodization",
                "assumptions": [],
                "warnings": [],
            }
        if request.sop_step == "programming_intensity_and_volume":
            return {
                "step": request.sop_step,
                "decision": {
                    "programming_intensity_and_volume": {
                        "bench_policy": {
                            "lift": "Bench Press",
                            "reps_per_set": [6, 6, 6, 6, 6],
                            "target_rpe": [7, 7, 5, 7, 7],
                            "weekly_estimated_progression_kg": [0, 2, 0, 2, 2],
                            "deload_session_intensity_reduction_percent": [0, 0, 20, 0, 0],
                            "deload_session_reps_reduction_percent": [0, 0, 50, 0, 0],
                        }
                    }
                },
                "rationale": "test intensity policy",
                "assumptions": [],
                "warnings": [],
            }
        return original_decider(request)

    service.llm_provider.generate_program_sop_step_decision = policy_decider
    response = _start_new_program(
        service,
        "conv_policy_materialization",
        message=(
            "Create a strength plan. days_per_week: 3; duration: 60; experience: intermediate; "
            "equipment: full gym and barbell; screening: no injuries. My bench PR is 115 kg."
        ),
    )
    version = service.get_version(response.draft_program_id, response.draft_version_id)
    context = service.get_context(response.context_id)
    calculated_periodization = context.sections["new_program_working_state"]["calculation_artifacts"]["calculated_periodization"]

    def bench_set(week: int):
        planned_session = next(item for item in version.sessions if item.week_number == week and item.day_number == 1)
        bench = next(item for item in planned_session.exercises if item.name == "Bench Press")
        return bench.sets[0]

    assert len(version.sessions) == 15
    assert version.periodization["weeks"] == 5
    assert calculated_periodization["deload_weeks"] == [3]
    assert calculated_periodization["week_policies"]["3"]["deload"] is True
    assert calculated_periodization["exercise_policy_count"] == 1
    assert bench_set(1).intensity_percent == pytest.approx(76.92)
    assert bench_set(1).reps == 6
    assert bench_set(1).load_kg == pytest.approx(88.5)
    assert bench_set(2).intensity_percent == pytest.approx(76.92)
    assert bench_set(2).load_kg == pytest.approx(90.0)
    assert bench_set(3).intensity_percent == pytest.approx(63.16)
    assert bench_set(3).reps == 3
    assert bench_set(4).intensity_percent == pytest.approx(76.92)
    assert bench_set(4).load_kg == pytest.approx(91.5)
    progression_traces = [
        item
        for item in session.exec(select(CalculationTrace)).all()
        if item.tool_name == "calculate_progression_increment"
    ]
    assert any(json.loads(item.inputs_json).get("percentage") == -20 for item in progression_traces)
    assert any(item.tool_name == "calculate_reps_rpe_intensity" for item in session.exec(select(CalculationTrace)).all())
    session.close()


def test_session_specific_intensity_policy_distinguishes_repeated_lift_slots(tmp_path: Path):
    session, _, service = _make_service(tmp_path)
    first = _start_new_program(
        service,
        "conv_repeated_lift_slots",
        message=(
            "Create a strength plan. days_per_week: 2; duration: 60; experience: intermediate; "
            "equipment: full gym and barbell; screening: no injuries. My bench PR is 115 kg."
        ),
    )
    context = service.get_context(first.context_id)
    record = service._get_context(first.context_id)
    raw_plan = dict(context.sections["new_program_working_state"]["raw_plan"])
    raw_plan["sessions"] = [
        {
            "name": "Upper Volume",
            "goal": "strength",
            "duration_minutes": 60,
            "exercises": [
                {"name": "Bench Press", "load_type": "weighted", "sets": 3, "reps": 6, "intensity_percent": 70, "rpe": 7}
            ],
        },
        {
            "name": "Upper Intensity",
            "goal": "strength",
            "duration_minutes": 60,
            "exercises": [
                {"name": "Bench Press", "load_type": "weighted", "sets": 1, "reps": 3, "intensity_percent": 85, "rpe": 8}
            ],
        },
    ]
    sop_progress = {
        "decisions": {
            "programming_intensity_and_volume": {
                "decision": {
                    "programming_intensity_and_volume": {
                        "slots": [
                            {
                                "session_index": 1,
                                "exercise_order": 1,
                                "lift": "Bench Press",
                                "sets_per_week": [4, 4],
                                "reps_per_set": [6, 6],
                                "weekly_estimated_progression_kg": [0, 2],
                                "target_rpe": [7, 7.5],
                            },
                            {
                                "session_index": 2,
                                "exercise_order": 1,
                                "lift": "Bench Press",
                                "sets_per_week": [1, 1],
                                "reps_per_set": [3, 2],
                                "weekly_estimated_progression_kg": [0, 1],
                                "target_rpe": [8, 8.5],
                            },
                        ]
                    }
                }
            }
        }
    }
    trace_id = service._trace(record, "run_calculation", {"title": "repeated lift slots"})
    artifacts = service._materialize_plan(
        raw_plan,
        context.sections["new_program_working_state"]["collected_inputs"],
        context.sections["memory_context"],
        trace_id,
        periodization={"weeks": 2},
        intensity_volume_policy=_intensity_volume_policy_from_sop_progress(sop_progress),
    )

    def bench_slot(week: int, day: int):
        planned_session = next(
            item for item in artifacts["calculated_sessions"] if item["week_number"] == week and item["day_number"] == day
        )
        return planned_session["exercises"][0]["sets"][0]

    assert artifacts["calculated_periodization"]["session_exercise_policy_count"] == 2
    assert bench_slot(1, 1)["intensity_percent"] == pytest.approx(76.92)
    assert bench_slot(1, 1)["reps"] == 6
    assert bench_slot(1, 1)["rpe"] == pytest.approx(7)
    assert bench_slot(1, 2)["intensity_percent"] == pytest.approx(85.71)
    assert bench_slot(1, 2)["reps"] == 3
    assert bench_slot(1, 2)["rpe"] == pytest.approx(8)
    assert bench_slot(2, 1)["intensity_percent"] == pytest.approx(77.92)
    assert bench_slot(2, 2)["intensity_percent"] == pytest.approx(89.55)
    assert bench_slot(2, 2)["reps"] == 2
    session.close()


def test_deload_reps_reduction_applies_to_week_indexed_reps_without_zero_sets(tmp_path: Path):
    session, _, service = _make_service(tmp_path)
    first = _start_new_program(
        service,
        "conv_deload_reps_policy",
        message=(
            "Create a strength plan. days_per_week: 1; duration: 60; experience: intermediate; "
            "equipment: full gym and barbell; screening: no injuries. My bench PR is 115 kg."
        ),
    )
    context = service.get_context(first.context_id)
    record = service._get_context(first.context_id)
    raw_plan = dict(context.sections["new_program_working_state"]["raw_plan"])
    raw_plan["sessions"] = [
        {
            "name": "Bench",
            "goal": "strength",
            "duration_minutes": 60,
            "exercises": [
                {"name": "Bench Press", "load_type": "weighted", "sets": 3, "reps": 6, "intensity_percent": 70, "rpe": 7}
            ],
        }
    ]
    sop_progress = {
        "decisions": {
            "programming_intensity_and_volume": {
                "decision": {
                    "programming_intensity_and_volume": [
                        {
                            "session_index": 1,
                            "exercise_order": 1,
                            "lift": "Bench Press",
                            "sets_per_week": [3, 3, 0, 3],
                            "reps_per_set": [6, 6, 6, 6],
                            "weekly_estimated_progression_kg": [0, 2, 0, 2],
                            "deload_session_intensity_reduction_percent": [0, 0, 20, 0],
                            "deload_session_reps_reduction_percent": [0, 0, 50, 0],
                            "target_rpe": [7, 7.5, 5, 8],
                        }
                    ]
                }
            }
        }
    }
    trace_id = service._trace(record, "run_calculation", {"title": "deload reps policy"})
    artifacts = service._materialize_plan(
        raw_plan,
        context.sections["new_program_working_state"]["collected_inputs"],
        context.sections["memory_context"],
        trace_id,
        periodization={"weeks": 4, "deload_week_numbers": [3]},
        intensity_volume_policy=_intensity_volume_policy_from_sop_progress(sop_progress),
    )

    week_3 = next(item for item in artifacts["calculated_sessions"] if item["week_number"] == 3)
    bench = week_3["exercises"][0]
    assert len(bench["sets"]) == 3
    assert bench["sets"][0]["reps"] == 3
    assert bench["sets"][0]["intensity_percent"] == pytest.approx(63.16)
    week_4 = next(item for item in artifacts["calculated_sessions"] if item["week_number"] == 4)
    assert week_4["exercises"][0]["sets"][0]["intensity_percent"] == pytest.approx(78.95)
    assert week_4["exercises"][0]["sets"][0]["load_kg"] == pytest.approx(94.0)
    session.close()


def test_user_schedule_survives_clarification_and_raw_split_controls_materialization(tmp_path: Path):
    session, _, service = _make_service(tmp_path)
    first = service.process_turn(
        "conv_four_day_sticky",
        ProgramTurnRequest(
            user_id="usr_program",
            message=(
                "Create a strength plan. I train 4 days per week for 60 minutes, have intermediate "
                "experience, use a full gym, and have no injuries. My bench PR is 115 kg."
            ),
        ),
    )
    assert first.status == "waiting_for_review"
    context = service.get_context(first.context_id)
    assert context.sections["new_program_working_state"]["collected_inputs"]["days_per_week"] == 4

    working = context.sections["new_program_working_state"]
    working["raw_plan"]["sessions"].append(
        {
            "name": "Fourth Day",
            "goal": "strength",
            "duration_minutes": 60,
            "exercises": [
                {"name": "Bench Press", "load_type": "weighted", "sets": 2, "reps": 5, "intensity_percent": 70, "rpe": 7}
            ],
        }
    )
    working["collected_inputs"]["days_per_week"] = 3
    trace_id = service._trace(service._get_context(first.context_id), "run_calculation", {"title": "raw split test"})
    artifacts = service._materialize_plan(
        working["raw_plan"],
        working["collected_inputs"],
        context.sections["memory_context"],
        trace_id,
        periodization={"weeks": 2},
    )
    assert len(working["raw_plan"]["sessions"]) == 4
    assert artifacts["calculated_periodization"]["sessions_per_week"] == 4
    assert len(artifacts["calculated_sessions"]) == 8
    session.close()


def test_weighted_exercise_load_clarification_and_bodyweight_exception(tmp_path: Path):
    session, _, service = _make_service(tmp_path)
    original_updater = service.llm_provider.update_program_plan

    def plan_with_pullups(request):
        plan = original_updater(request)
        if request.sop_step == "programming_intensity_and_volume":
            plan["sessions"][0]["exercises"].extend(
                [
                    {
                        "name": "Weighted Pull-Up",
                        "mode": "resistance",
                        "load_type": "weighted",
                        "sets": 3,
                        "reps": 5,
                        "intensity_percent": 75,
                        "rpe": 8,
                        "rest_seconds": 180,
                    },
                    {
                        "name": "Pull-Up",
                        "mode": "resistance",
                        "load_type": "bodyweight",
                        "sets": 3,
                        "reps": 8,
                        "intensity_percent": 70,
                        "rpe": 7,
                        "rest_seconds": 120,
                    },
                ]
            )
        return plan

    service.llm_provider.update_program_plan = plan_with_pullups
    waiting = service.process_turn(
        "conv_weighted_clarification",
        ProgramTurnRequest(
            user_id="usr_program",
            message=(
                "Create a strength plan. days_per_week: 3; duration: 60; experience: intermediate; "
                "equipment: full gym; screening: no injuries"
            ),
        ),
    )
    assert waiting.status == "waiting_for_user"
    assert "Weighted Pull-Up" in waiting.response
    assert "load_references" in waiting.response

    draft = service.process_turn(
        "conv_weighted_clarification",
        ProgramTurnRequest(
            user_id="usr_program",
            message="Submitted structured load references.",
            load_references=[
                {
                    "exercise_name": "Weighted Pull-Up",
                    "type": "recent_set",
                    "weight_kg": 40,
                    "reps": 5,
                }
            ],
        ),
    )
    assert draft.status == "waiting_for_review"
    version = service.get_version(draft.draft_program_id, draft.draft_version_id)
    weighted_sets = [
        prescribed_set
        for planned_session in version.sessions
        for exercise in planned_session.exercises
        if exercise.name == "Weighted Pull-Up"
        for prescribed_set in exercise.sets
    ]
    bodyweight_sets = [
        prescribed_set
        for planned_session in version.sessions
        for exercise in planned_session.exercises
        if exercise.name == "Pull-Up"
        for prescribed_set in exercise.sets
    ]
    assert weighted_sets and all(item.load_kg is not None for item in weighted_sets)
    assert bodyweight_sets and all(item.load_kg is None for item in bodyweight_sets)
    estimate_traces = [
        item
        for item in session.exec(select(CalculationTrace)).all()
        if item.tool_name == "estimate_repetition_max"
    ]
    assert any(json.loads(item.inputs_json) == {"weight": 40.0, "reps": 5, "method": "epley", "unit": "kg"} for item in estimate_traces)
    session.close()


def test_load_reference_parser_handles_press_and_structured_json():
    message = (
        "Weighted Plank 1RM: 20kg, Weighted Pull-up 1RM: 55kg, "
        "Chest-Supported Row: 40kg x 12 reps, Overhead Press: 45kg x 7 reps, "
        "Weighted Pull-up Iso Hold 1RM: 55kg, Lat Pulldown: 40kg x 10 reps, "
        "Face Pull: 40kg x 12 reps"
    )
    names = [
        "Weighted Plank",
        "Weighted Pull-up",
        "Chest-Supported Row",
        "Overhead Press",
        "Weighted Pull-up Iso Hold",
        "Lat Pulldown",
        "Face Pull",
    ]
    parsed = _extract_exercise_load_references(message, names)
    assert parsed["overhead_press"] == {
        "exercise_name": "Overhead Press",
        "weight_kg": 45.0,
        "reps": 7,
        "source": "current_user_turn_recent_performance",
    }
    assert parsed["chest_supported_row"]["reps"] == 12

    structured = _extract_exercise_load_references(
        json.dumps(
            {
                "load_references": [
                    {"exercise_name": "Overhead Press", "type": "recent_set", "weight_kg": 45, "reps": 7},
                    {"exercise_name": "Weighted Pull-up", "type": "1rm", "weight_kg": 55, "reps": 1},
                ]
            }
        ),
        ["Overhead Press", "Weighted Pull-up"],
    )
    assert structured["overhead_press"]["reps"] == 7
    assert structured["weighted_pull_up"]["source"] == "current_user_turn_1rm"


def test_malformed_program_json_returns_clean_error_response(tmp_path: Path):
    session, _, service = _make_service(tmp_path)

    def broken_updater(_request):
        raise StructuredOutputError("PlanUpdater returned malformed JSON at line 704, column 10.")

    service.llm_provider.update_program_plan = broken_updater

    response = service.process_turn(
        "conv_bad_json",
        ProgramTurnRequest(
            user_id="usr_program",
            message=(
                "Create a strength plan. days_per_week: 3; duration: 60; experience: intermediate; "
                "equipment: full gym; screening: no injuries"
            ),
        ),
    )

    assert response.status == "open"
    assert response.draft_program_id is None
    assert "malformed JSON" in response.response
    trace = session.get(ProgramTrace, response.trace_id)
    assert trace.action == "structured_output_error"
    assert "PlanUpdater returned malformed JSON" in trace.errors_json
    session.close()


def test_clarification_resumes_same_program_context(tmp_path: Path):
    session, memory, service = _make_service(tmp_path)
    memory.create_user("usr_empty")
    first = service.process_turn(
        "conv_clarify",
        ProgramTurnRequest(user_id="usr_empty", message="Please make me a plan."),
    )
    assert first.status == "waiting_for_user"
    second = service.process_turn(
        "conv_clarify",
        ProgramTurnRequest(
            user_id="usr_empty",
            message=(
                "goal: strength; days_per_week: 3; duration: 60; experience: intermediate; "
                "equipment: full gym; screening: no injuries"
            ),
        ),
    )
    assert second.context_id == first.context_id
    assert second.status == "waiting_for_user"
    assert "Back Squat" in second.response
    third = service.process_turn(
        "conv_clarify",
        ProgramTurnRequest(
            user_id="usr_empty",
            message=(
                "Back Squat 1RM: 140 kg; Bench Press 1RM: 100 kg; Deadlift 1RM: 180 kg; "
                "OHP 1RM: 65 kg; Front Squat 1RM: 110 kg"
            ),
        ),
    )
    assert third.context_id == first.context_id
    assert third.status == "waiting_for_review"
    assert len(session.exec(select(ProgramContextRecord)).all()) == 1
    rationale = service.process_turn(
        "conv_clarify",
        ProgramTurnRequest(user_id="usr_empty", message="Why was this plan designed this way?"),
    )
    assert rationale.route == "explain_plan_rationale"

    next_plan = service.process_turn(
        "conv_clarify",
        ProgramTurnRequest(user_id="usr_empty", message="Create a completely new plan."),
    )
    assert next_plan.status == "waiting_for_user"
    assert next_plan.draft_program_id is None
    assert next_plan.draft_version_id is None
    rationale_while_waiting = service.process_turn(
        "conv_clarify",
        ProgramTurnRequest(user_id="usr_empty", message="How did you get the bench weights in the saved plan?"),
    )
    assert rationale_while_waiting.route == "explain_plan_rationale"
    assert "Before I design this safely" not in rationale_while_waiting.response
    session.close()


def test_context_budget_continue_resumes_the_original_turn(tmp_path: Path):
    session, _, service = _make_service(tmp_path)
    service.settings.program_max_context_tokens = 1
    original_message = (
        "Create a strength plan. days_per_week: 3; duration: 60; experience: intermediate; "
        "equipment: full gym; screening: no injuries"
    )

    warning = service.process_turn(
        "conv_budget_resume",
        ProgramTurnRequest(user_id="usr_program", message=original_message),
    )
    assert warning.pending_action == "waiting_for_context_budget_decision"
    saved = service.get_context(warning.context_id)
    assert saved.sections["sop_progress"]["context_budget_pending_turn"]["content"] == original_message

    resumed = service.process_turn(
        "conv_budget_resume",
        ProgramTurnRequest(user_id="usr_program", message="continue"),
    )
    assert resumed.route == "new_program"
    assert resumed.status == "waiting_for_review"
    resumed_context = service.get_context(resumed.context_id)
    assert resumed_context.sections["new_program_working_state"]["task_request"] == original_message
    assert resumed_context.sections["sop_progress"]["context_budget_override"] is True
    session.close()


def test_approval_activates_plan_and_compacts_active_plan_context(tmp_path: Path):
    session, memory, service = _make_service(tmp_path)
    draft = _start_new_program(service, "conv_activate")
    review = service.review_program(
        draft.draft_program_id,
        draft.draft_version_id,
        ProgramReviewRequest(user_id="usr_program", decision="approve"),
    )
    assert review.activation_status == "activated"
    active = session.exec(select(TrainingPlanProgress).where(TrainingPlanProgress.status == "active")).one()
    assert active.version_id == draft.draft_version_id
    assert "Four Week Strength Foundation" in active.plan_snapshot_json
    first_session_id = memory.get_active_plan("usr_program").planned_sessions[0]["session_id"]
    assert "calculation_trace_ids" not in memory.get_active_plan("usr_program").plan_snapshot["source_references"]
    compact_context = memory.get_memory_context(
        "usr_program",
        query="Show my current plan",
        task_type="program_generator",
    )
    compact_plan_text = compact_context.sections["active_plan_progress"]
    assert '"format":"compact_active_plan_v1"' in compact_plan_text
    assert "calculation_trace_ids" not in compact_plan_text
    assert len(compact_plan_text) // 4 <= 5000
    memory.patch_plan_progress(
        draft.draft_program_id,
        PlanProgressPatchRequest(
            user_id="usr_program",
            session_id=first_session_id,
            session_status="completed",
        ),
    )
    updated_active = memory.get_active_plan("usr_program")
    assert updated_active.plan_progress["sessions"][first_session_id]["status"] == "completed"
    session.close()


def test_review_messages_activation_failure_and_retry_are_explicit(tmp_path: Path, monkeypatch):
    session, memory, service = _make_service(tmp_path)
    draft = _start_new_program(service, "conv_activation_failure")
    original_activate = memory.activate_plan

    def fail_activation(_request):
        raise RuntimeError("temporary activation failure")

    monkeypatch.setattr(memory, "activate_plan", fail_activation)
    failed = service.review_program(
        draft.draft_program_id,
        draft.draft_version_id,
        ProgramReviewRequest(user_id="usr_program", decision="approve"),
    )
    assert failed.activation_status == "activation_failed"
    assert session.get(ProgramVersion, draft.draft_version_id).status == "approved"

    monkeypatch.setattr(memory, "activate_plan", original_activate)
    retried = service.review_program(
        draft.draft_program_id,
        draft.draft_version_id,
        ProgramReviewRequest(user_id="usr_program", decision="approve"),
    )
    assert retried.activation_status == "activated"
    messages = session.exec(
        select(ConversationMessage).where(ConversationMessage.conversation_id == "conv_activation_failure")
    ).all()
    assert any(item.role == "assistant" and "activation failed" in item.content for item in messages)
    assert any(item.role == "assistant" and "approved and activated" in item.content.lower() for item in messages)
    session.close()


def test_patch_validation_and_episode_context_reference(tmp_path: Path):
    session, memory, service = _make_service(tmp_path)
    validator = ProgramContextPatchValidator({"sop_progress"})
    router = ReasoningActionRouter()
    task_router = TaskAndSOPRouter(FakeProgramProvider())
    assert router(
        {
            "missing_inputs": ["goal"],
            "user_question_count": 5,
            "max_user_questions": 5,
            "evidence_sufficient": True,
            "step_complete": False,
        }
    )["next_action"] == "write_step_output"
    assert router(
        {
            "missing_inputs": [],
            "evidence_sufficient": False,
            "rag_query_count": 5,
            "max_rag_queries": 5,
            "step_complete": False,
        }
    )["next_action"] == "write_step_output"
    with pytest.raises(ValueError, match="Stale"):
        validator.validate(
            ProgramContextPatch(expected_context_revision=1, target_section="sop_progress", value={}),
            current_revision=2,
        )
    with pytest.raises(ValueError, match="Unknown"):
        validator.validate(
            ProgramContextPatch(expected_context_revision=2, target_section="memory_context", value={}),
            current_revision=2,
        )

    draft = _start_new_program(service, "conv_close_program")
    memory.close_conversation("conv_close_program", program_context_id=draft.context_id)
    episode = session.exec(select(Episode).where(Episode.conversation_id == "conv_close_program")).one()
    assert episode.program_context_id == draft.context_id
    assert session.get(ProgramContextRecord, draft.context_id) is not None
    session.close()


def test_new_approval_deactivates_the_older_active_program(tmp_path: Path):
    session, _, service = _make_service(tmp_path)
    message = (
        "Create a strength plan. days_per_week: 3; duration: 60; experience: intermediate; "
        "equipment: full gym; screening: no injuries"
    )
    first = _start_new_program(service, "conv_first", message=message)
    service.review_program(
        first.draft_program_id,
        first.draft_version_id,
        ProgramReviewRequest(user_id="usr_program", decision="approve"),
    )
    second = _start_new_program(service, "conv_second", message=message)
    review = service.review_program(
        second.draft_program_id,
        second.draft_version_id,
        ProgramReviewRequest(user_id="usr_program", decision="approve"),
    )

    assert review.deactivated_plan_ids == [first.draft_program_id]
    rows = session.exec(select(TrainingPlanProgress)).all()
    assert {item.plan_id: item.status for item in rows} == {
        first.draft_program_id: "superseded",
        second.draft_program_id: "active",
    }
    session.close()


def test_rationale_is_read_only_and_red_flags_block_planning(tmp_path: Path):
    session, _, service = _make_service(tmp_path)
    draft = _start_new_program(service, "conv_rationale", message="Create a strength plan.")
    before = service.get_context(draft.context_id)
    rationale = service.process_turn(
        "conv_rationale",
        ProgramTurnRequest(user_id="usr_program", message="Why was this plan designed this way?"),
    )
    after = service.get_context(draft.context_id)
    assert "schedule" in rationale.response
    assert before.latest_version_id == after.latest_version_id
    for section in (
        "sop_progress",
        "new_program_working_state",
        "draft_artifacts",
    ):
        assert before.sections[section] == after.sections[section]

    numeric_rationale = service.process_turn(
        "conv_rationale",
        ProgramTurnRequest(
            user_id="usr_program",
            message="My bench PR is 115 kg. How did you get the bench weight numbers?",
        ),
    )
    assert "used a 100 kg 1RM" in numeric_rationale.response
    assert "reports 115 kg" in numeric_rationale.response

    intensity_followup = service.process_turn(
        "conv_rationale",
        ProgramTurnRequest(user_id="usr_program", message="I want the exact calculation behind 75.24%"),
    )
    assert "Selected Bench Press at 76.92%" in intensity_followup.response
    assert "Bench Press 6 reps RPE 7.0" in intensity_followup.response

    original_updater = service.llm_provider.update_program_plan

    def plan_with_bodyweight_pullups(request):
        plan = original_updater(request)
        if request.sop_step == "programming_intensity_and_volume":
            plan["sessions"][0]["exercises"].extend(
                [
                    {
                        "name": "Pull-Up",
                        "load_type": "bodyweight",
                        "sets": 3,
                        "reps": 6,
                        "intensity_percent": 70,
                        "rpe": 7,
                        "rest_seconds": 120,
                        "notes": "Bodyweight pull-up volume.",
                    },
                    {
                        "name": "Paused Pull-Up",
                        "load_type": "bodyweight",
                        "sets": 4,
                        "reps": 4,
                        "intensity_percent": 70,
                        "rpe": 7,
                        "rest_seconds": 75,
                        "notes": "Paused bodyweight pull-up control work.",
                    },
                ]
            )
        return plan

    service.llm_provider.update_program_plan = plan_with_bodyweight_pullups
    pullup_draft = _start_new_program(service, "conv_pullup_rationale")
    pullup_rationale = service.process_turn(
        "conv_pullup_rationale",
        ProgramTurnRequest(
            user_id="usr_program",
            message=(
                "I can do more than 25 self-weight pull ups in a row. In your plan there are sets "
                "with 6 reps of self-weight pull-ups with rpe 7 and 4 reps of paused self-weight "
                "pull-ups with rpe 7, why?"
            ),
        ),
    )
    facts = service._numeric_rationale_facts(
        service._get_context(pullup_draft.context_id),
        "I can do more than 25 self-weight pull ups in a row. Why 6 reps pull-ups and 4 reps paused pull-ups at rpe 7?",
    )
    matched = facts["matched_prescriptions"]
    assert any(item["exercise"] == "Pull-Up" and item["reps"] == 6 and item["load_kg"] is None for item in matched)
    assert any(item["exercise"] == "Paused Pull-Up" and item["reps"] == 4 and item["load_kg"] is None for item in matched)
    assert facts["reported_bodyweight_performance"][0]["reps"] == 25
    assert "Overhead Press" not in pullup_rationale.response
    assert "Pull-Up 6 reps RPE 7.0" in pullup_rationale.response
    assert "Paused Pull-Up 4 reps RPE 7.0" in pullup_rationale.response

    blocked = service.process_turn(
        "conv_safety",
        ProgramTurnRequest(
            user_id="usr_program",
            message=(
                "Create a strength plan despite severe chest pain. days_per_week: 3; duration: 60; "
                "experience: intermediate; equipment: gym; screening: medical issue"
            ),
        ),
    )
    assert blocked.status == "waiting_for_safety_clearance"
    assert "medical" in blocked.response.lower()
    session.close()


def test_weekly_summary_notes_are_stable_for_completed_weeks(tmp_path: Path):
    session, _, service = _make_service(tmp_path)
    now = datetime.now(timezone.utc)
    session.add(
        TrainingSession(
            session_id="ts_week",
            user_id="usr_program",
            source_import_id="import_test",
            session_date=now - timedelta(days=8),
            sport_type="strength",
            duration_seconds=3600,
        )
    )
    session.commit()

    first = service.refresh_weekly_summaries("usr_program", reference_date=now.date())
    second = service.refresh_weekly_summaries("usr_program", reference_date=now.date())

    assert len(first.notes) == 4
    assert [item.summary_id for item in first.notes] == [item.summary_id for item in second.notes]
    session.close()


def test_program_routes_and_missing_llm_configuration_are_explicit(tmp_path: Path):
    from app.api.main import create_app

    paths = create_app().openapi()["paths"]
    assert "/api/programs/conversations/{conversation_id}/turns" in paths
    assert "/api/programs/{program_id}/versions/{version_id}/review" in paths
    assert "/api/programs/calculators/{tool_name}" in paths

    session, memory, service = _make_service(tmp_path)
    service.llm_provider = OpenAIProvider(service.settings)
    with pytest.raises(ConfigurationError):
        service.process_turn(
            "conv_no_key",
            ProgramTurnRequest(
                user_id="usr_program",
                message=(
                    "Create a strength plan. days_per_week: 3; duration: 60; experience: intermediate; "
                    "equipment: full gym; screening: no injuries"
                ),
            ),
        )
    session.close()
