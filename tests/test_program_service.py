"""Program Generator workflow, persistence, safety, and calculator tests."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.core.llm_provider import OpenAIProvider
from app.knowledge.schemas import RetrievedChunk
from app.memory.service import MemoryService
from app.memory.schemas import PlanProgressPatchRequest
from app.program.calculators import CalculatorService
from app.program.nodes import ProgramContextPatchValidator, ReasoningActionRouter
from app.program.schemas import ProgramContextPatch, ProgramReviewRequest, ProgramTurnRequest
from app.program.service import ADAPTATION_SOP, NEW_PROGRAM_SOP, ProgramService
from app.storage.database import create_app_engine, init_db
from app.storage.models import (
    CalculationTrace,
    ConversationMessage,
    Episode,
    ProgramAdaptation,
    ProgramContextRecord,
    ProgramTrace,
    ProgramVersion,
    SetPrescription,
    TrainingPlanProgress,
    TrainingSession,
)


class FakeProgramProvider:
    """Return a compact but complete program proposal without network calls."""

    def generate_program_plan(self, request):
        return {
            "title": "Four Week Strength Foundation",
            "goal": request.collected_inputs.get("goal", "strength"),
            "sport": "strength",
            "rationale": "The plan matches the user's schedule, experience, equipment, and approved evidence.",
            "screening": {"status": "clear", "notes": []},
            "needs_analysis": {
                "goal": request.collected_inputs.get("goal"),
                "experience": request.collected_inputs.get("experience"),
                "equipment": request.collected_inputs.get("equipment"),
                "constraints": [],
            },
            "periodization": {"weeks": 4, "model": "parallel", "phases": ["foundation"]},
            "assumptions": [],
            "warnings": [],
            "sessions": [
                {
                    "name": "Squat and Bench",
                    "goal": "strength",
                    "duration_minutes": request.collected_inputs.get("session_duration_minutes", 60),
                    "exercises": [
                        {"name": "Back Squat", "sets": 3, "reps": 5, "intensity_percent": 75, "rpe": 7,
                         "rest_seconds": 180},
                        {"name": "Bench Press", "sets": 3, "reps": 6, "intensity_percent": 72, "rpe": 7,
                         "rest_seconds": 150},
                    ],
                },
                {
                    "name": "Deadlift and Press",
                    "goal": "strength",
                    "duration_minutes": request.collected_inputs.get("session_duration_minutes", 60),
                    "exercises": [
                        {"name": "Deadlift", "sets": 3, "reps": 4, "intensity_percent": 78, "rpe": 7.5,
                         "rest_seconds": 180},
                        {"name": "Overhead Press", "sets": 3, "reps": 6, "intensity_percent": 70, "rpe": 7,
                         "rest_seconds": 120},
                    ],
                },
                {
                    "name": "Full Body Volume",
                    "goal": "hypertrophy",
                    "duration_minutes": request.collected_inputs.get("session_duration_minutes", 60),
                    "exercises": [
                        {"name": "Front Squat", "sets": 3, "reps": 8, "intensity_percent": 65, "rpe": 7,
                         "rest_seconds": 120},
                    ],
                },
            ],
            "changed_elements": ["future session distribution"] if request.task_type == "program_adaptation" else [],
            "unchanged_elements": ["primary strength goal"] if request.task_type == "program_adaptation" else [],
        }


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


def test_new_program_is_saved_with_traced_numeric_prescriptions(tmp_path: Path):
    session, _, service = _make_service(tmp_path)
    response = service.process_turn(
        "conv_program",
        ProgramTurnRequest(
            user_id="usr_program",
            message=(
                "Create a strength plan. days_per_week: 3; duration: 60; experience: intermediate; "
                "equipment: full gym and barbell; screening: no injuries."
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
    assert all(ex.volume is not None and ex.calculation_trace_ids for item in version.sessions for ex in item.exercises)
    assert version.periodization["weekly_volume_load_kg"]
    assert version.source_references["evidence"] == [
        {"source_id": "src_strength", "chunk_id": "chk_programming"}
    ]
    assert session.exec(select(CalculationTrace)).all()
    step_traces = [item for item in session.exec(select(ProgramTrace)).all() if item.action == "write_step_output"]
    assert len(step_traces) == 12
    context = service.get_context(response.context_id)
    assert context.sections["sop_progress"]["completed_steps"] == NEW_PROGRAM_SOP
    assert set(context.sections["sop_progress"]["evidence_sufficient_by_step"]) == set(NEW_PROGRAM_SOP)
    assert context.sections["retrieved_evidence"][0]["sop_steps"] == NEW_PROGRAM_SOP
    assert all(
        counters.get("rag_queries", 0) <= 5 and counters.get("user_questions", 0) <= 5
        for counters in context.sections["sop_progress"]["counters"].values()
    )
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
    assert second.status == "waiting_for_review"
    assert len(session.exec(select(ProgramContextRecord)).all()) == 1
    rationale = service.process_turn(
        "conv_clarify",
        ProgramTurnRequest(user_id="usr_empty", message="Why was this plan designed this way?"),
    )
    assert rationale.route == "explain_plan_rationale"
    session.close()


def test_approval_activates_plan_and_adaptation_creates_new_version(tmp_path: Path):
    session, memory, service = _make_service(tmp_path)
    draft = service.process_turn(
        "conv_adapt",
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
    memory.patch_plan_progress(
        draft.draft_program_id,
        PlanProgressPatchRequest(
            user_id="usr_program",
            session_id=first_session_id,
            session_status="completed",
        ),
    )

    revision = service.process_turn(
        "conv_adapt",
        ProgramTurnRequest(user_id="usr_program", message="I missed tomorrow's session because of a meeting; adapt my plan."),
    )
    assert revision.draft_program_id == draft.draft_program_id
    revised = service.get_version(revision.draft_program_id, revision.draft_version_id)
    assert revised.version_number == 2
    adaptation = session.exec(select(ProgramAdaptation)).one()
    assert adaptation.from_version_id == draft.draft_version_id
    assert adaptation.to_version_id == revision.draft_version_id
    assert adaptation.approval_status == "pending"
    revision_context = service.get_context(revision.context_id)
    assert revision_context.sections["sop_progress"]["completed_steps"] == ADAPTATION_SOP
    trigger = revision_context.sections["revision_adaptation_working_state"]["trigger_classification"]
    assert trigger["category"] == "schedule_conflict"
    assert trigger["adjustment_priority"] == [
        "frequency", "volume", "intensity", "exercise_selection_or_order", "rest"
    ]
    service.review_program(
        revision.draft_program_id,
        revision.draft_version_id,
        ProgramReviewRequest(user_id="usr_program", decision="approve"),
    )
    revised_active = memory.get_active_plan("usr_program")
    revised_first_id = revised_active.planned_sessions[0]["session_id"]
    assert revised_active.plan_progress["sessions"][revised_first_id]["status"] == "completed"
    assert session.get(ProgramAdaptation, adaptation.adaptation_id).approval_status == "approved"
    session.close()


def test_new_conversation_adaptation_inherits_active_program_context(tmp_path: Path):
    session, _, service = _make_service(tmp_path)
    message = (
        "Create a strength plan. days_per_week: 3; duration: 60; experience: intermediate; "
        "equipment: full gym; screening: no injuries"
    )
    original = service.process_turn("conv_original", ProgramTurnRequest(user_id="usr_program", message=message))
    service.review_program(
        original.draft_program_id,
        original.draft_version_id,
        ProgramReviewRequest(user_id="usr_program", decision="approve"),
    )

    revision = service.process_turn(
        "conv_new_revision",
        ProgramTurnRequest(user_id="usr_program", message="I missed a session because of a work meeting; adapt it."),
    )

    assert revision.draft_program_id == original.draft_program_id
    context = service.get_context(revision.context_id)
    original_context = service.get_context(original.context_id)
    assert context.parent_context_id == original_context.context_id
    assert service.get_version(revision.draft_program_id, revision.draft_version_id).version_number == 2
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
        "revision_adaptation_working_state",
        "draft_artifacts",
        "calculation_results",
    ):
        assert before.sections[section] == after.sections[section]

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
