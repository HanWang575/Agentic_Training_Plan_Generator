"""Program Generator workflow, persistence, safety, and calculator tests."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

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
from app.program.service import NEW_PROGRAM_SOP, ProgramService, _extract_lift_maxes
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

    def extract_program_lift_maxes(self, request):
        self.lift_max_requests.append(request.user_message)
        return _extract_lift_maxes(request.user_message)

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


def test_calculators_are_deterministic_and_safe(tmp_path: Path):
    settings = Settings(_env_file=None, sqlite_path=tmp_path / "app.sqlite")
    calculators = CalculatorService(settings)

    one_rm = calculators.run("estimate_repetition_max", {"weight": 100, "reps": 5})
    load = calculators.run("calculate_percentage_load", {"one_rm": one_rm.value, "percentage": 75})
    rounded = calculators.run("round_training_load", {"load": load.value, "increment": 2.5})

    assert one_rm.value == pytest.approx(116.67, abs=0.01)
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


def test_program_service_uses_llm_lift_max_extractor(tmp_path: Path):
    _, _, service = _make_service(tmp_path)

    inputs = service._extract_inputs("My bench PR is 116 kg.", "")

    assert inputs["bench_press_1rm_kg"] == 116.0
    assert service.llm_provider.lift_max_requests == ["My bench PR is 116 kg."]


def test_new_program_is_saved_with_traced_numeric_prescriptions(tmp_path: Path):
    session, memory, service = _make_service(tmp_path)
    response = service.process_turn(
        "conv_program",
        ProgramTurnRequest(
            user_id="usr_program",
            message=(
                "Create a strength plan. days_per_week: 3; duration: 60; experience: intermediate; "
                "equipment: full gym and barbell; screening: no injuries. My bench PR is 115 kg."
            ),
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
    assert any("current_user_turn_1rm" in json.loads(item.source_references_json) for item in percentage_traces)
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

    draft = service.process_turn(
        "conv_weighted_clarification",
        ProgramTurnRequest(user_id="usr_program", message="Weighted Pull-Up: 40 kg x 5 reps"),
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
    draft = service.process_turn(
        "conv_activate",
        ProgramTurnRequest(
            user_id="usr_program",
            message=(
                "Create a strength plan. days_per_week: 3; duration: 60; experience: intermediate; "
                "equipment: full gym; screening: no injuries"
            ),
        ),
    )
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
    draft = service.process_turn(
        "conv_activation_failure",
        ProgramTurnRequest(
            user_id="usr_program",
            message=(
                "Create a strength plan. days_per_week: 3; duration: 60; experience: intermediate; "
                "equipment: full gym; screening: no injuries"
            ),
        ),
    )
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

    draft = service.process_turn(
        "conv_close_program",
        ProgramTurnRequest(
            user_id="usr_program",
            message=(
                "Create a strength plan. days_per_week: 3; duration: 60; experience: intermediate; "
                "equipment: full gym; screening: no injuries"
            ),
        ),
    )
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
    first = service.process_turn("conv_first", ProgramTurnRequest(user_id="usr_program", message=message))
    service.review_program(
        first.draft_program_id,
        first.draft_version_id,
        ProgramReviewRequest(user_id="usr_program", decision="approve"),
    )
    second = service.process_turn("conv_second", ProgramTurnRequest(user_id="usr_program", message=message))
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
    draft = service.process_turn(
        "conv_rationale",
        ProgramTurnRequest(
            user_id="usr_program",
            message=(
                "Create a strength plan. days_per_week: 3; duration: 60; experience: intermediate; "
                "equipment: full gym; screening: no injuries"
            ),
        ),
    )
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
    assert "Selected Bench Press at 75.24%" in intensity_followup.response
    assert "72 * (1 + 4.5 / 100) = 75.24%" in intensity_followup.response

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
    pullup_draft = service.process_turn(
        "conv_pullup_rationale",
        ProgramTurnRequest(
            user_id="usr_program",
            message=(
                "Create a strength plan. days_per_week: 3; duration: 60; experience: intermediate; "
                "equipment: full gym; screening: no injuries"
            ),
        ),
    )
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
