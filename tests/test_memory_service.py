"""User Profile and Memory service behavior."""

from datetime import timedelta
from pathlib import Path

from sqlmodel import Session, select

from app.core.config import Settings
from app.memory.schemas import MemoryItemApproveRequest, PlanProgressPatchRequest, TrainingLogImportRequest
from app.memory.service import MemoryService
from app.storage.database import create_app_engine, init_db
from app.storage.models import (
    Conversation,
    ConversationMessage,
    Episode,
    MemoryItem,
    MemoryTrace,
    ProfileField,
    RecentStatus,
    TrainingLogImport,
    TrainingPlanProgress,
    TrainingSession,
    UserAccount,
    utc_now,
)


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
    return session, MemoryService(session=session, settings=settings)


def test_manual_profile_and_recent_status_appear_in_context(tmp_path: Path):
    """Manual approved memory should be retrievable as compact context."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test", display_name="Alex", timezone_name="Asia/Shanghai")
    service.update_profile(
        "usr_test",
        {
            "primary_training_goal": "build strength",
            "bench_press_1rm_kg": 100,
        },
    )
    observed = utc_now() - timedelta(hours=1)
    status = service.update_recent_status(
        "usr_test",
        {"sleep_duration_hours": 5.5},
        observed_at=observed,
    ).statuses[0]

    context = service.get_memory_context(
        "usr_test",
        query="How should I train today?",
        include_debug=True,
    )

    assert "primary_training_goal: build strength" in context.sections["profile"]
    assert "sleep_duration_hours: 5.5" in context.sections["recent_status"]
    assert status.valid_until == observed + timedelta(days=1)
    assert status.status_id in context.memory_ids
    assert context.debug["memory_trace_id"].startswith("mtr_")
    session.close()


def test_list_users_returns_existing_user_ids(tmp_path: Path):
    """The frontend can list local user IDs for selection/debugging."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_b", display_name="B")
    service.create_user("usr_a", display_name="A")

    users = service.list_users()

    assert [user.user_id for user in users] == ["usr_a", "usr_b"]
    session.close()


def test_delete_user_removes_owned_memory_rows(tmp_path: Path):
    """Deleting a user should clean up the memory block's user-owned rows."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    service.update_profile("usr_test", {"primary_training_goal": "build muscle"})
    service.update_recent_status("usr_test", {"sleep_duration_hours": 7})
    conversation = service.create_conversation("usr_test", title="Delete me")
    service.append_message(conversation.conversation_id, role="user", content="I slept 7 hours today.")
    service.close_conversation(conversation.conversation_id)
    service.get_memory_context("usr_test", conversation_id=conversation.conversation_id, query="How am I doing?")
    service.patch_plan_progress(
        "plan_delete",
        PlanProgressPatchRequest(user_id="usr_test", event_type="plan_created"),
    )
    service.import_xunji(
        TrainingLogImportRequest(
            user_id="usr_test",
            external_id="delete_batch",
            payload={"sessions": [{"id": "sess_delete", "date": utc_now().date().isoformat()}]},
        )
    )

    response = service.delete_user("usr_test")

    assert response.deleted_counts["user_accounts"] == 1
    for model in (
        UserAccount,
        Conversation,
        ConversationMessage,
        Episode,
        MemoryItem,
        ProfileField,
        RecentStatus,
        MemoryTrace,
        TrainingPlanProgress,
        TrainingLogImport,
        TrainingSession,
    ):
        assert session.exec(select(model)).all() == []
    session.close()


def test_list_and_delete_conversation_by_id(tmp_path: Path):
    """The debug UI can list conversations and delete one by conversation_id."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    conversation = service.create_conversation("usr_test", title="Conversation to remove")
    service.append_message(conversation.conversation_id, role="user", content="I slept 6 hours today.")
    service.close_conversation(conversation.conversation_id)
    service.get_memory_context("usr_test", conversation_id=conversation.conversation_id, query="How should I train?")

    listed = service.list_conversations("usr_test")
    response = service.delete_conversation(conversation.conversation_id)

    assert [item.conversation_id for item in listed] == [conversation.conversation_id]
    assert response.deleted_counts["conversations"] == 1
    assert service.list_conversations("usr_test") == []
    assert session.exec(select(ConversationMessage)).all() == []
    assert session.exec(select(Episode)).all() == []
    assert session.exec(select(MemoryItem)).all() == []
    assert session.exec(select(MemoryTrace)).all() == []
    session.close()


def test_get_conversation_detail_returns_all_messages(tmp_path: Path):
    """The debug UI can inspect the full saved conversation content."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    conversation = service.create_conversation("usr_test", title="Inspect me")
    service.append_message(conversation.conversation_id, role="user", content="I slept 7 hours.")
    service.append_message(conversation.conversation_id, role="assistant", content="That sleep supports training.")

    detail = service.get_conversation_detail(conversation.conversation_id)

    assert detail.conversation.conversation_id == conversation.conversation_id
    assert detail.conversation.title == "Inspect me"
    assert [message.role for message in detail.messages] == ["user", "assistant"]
    assert [message.content for message in detail.messages] == [
        "I slept 7 hours.",
        "That sleep supports training.",
    ]
    session.close()


def test_delete_profile_and_recent_status_entries_by_id(tmp_path: Path):
    """Profile and recent-status debug deletes should target one stored row."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    profile = service.update_profile("usr_test", {"primary_training_goal": "build strength"}).fields[0]
    status = service.update_recent_status("usr_test", {"sleep_duration_hours": 8}).statuses[0]

    profile_delete = service.delete_profile_field(profile.profile_field_id)
    status_delete = service.delete_recent_status(status.status_id)

    assert profile_delete.deleted_counts == {"profile_fields": 1}
    assert status_delete.deleted_counts == {"recent_status": 1}
    assert service.list_profile("usr_test").fields == []
    assert service.list_recent_status("usr_test") == []
    session.close()


def test_context_uses_latest_profile_and_recent_status_value_per_key(tmp_path: Path):
    """Prompt context should not contain multiple active values for one memory key."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    service.update_profile("usr_test", {"primary_training_goal": "build strength"})
    service.update_profile("usr_test", {"primary_training_goal": "run a marathon"})

    first_observed = utc_now() - timedelta(hours=2)
    second_observed = utc_now() - timedelta(minutes=10)
    first_status = service.update_recent_status(
        "usr_test",
        {"sleep_duration_hours": 5.5},
        observed_at=first_observed,
    ).statuses[0]
    second_status = service.update_recent_status(
        "usr_test",
        {"sleep_duration_hours": 5.0},
        observed_at=second_observed,
    ).statuses[0]

    context = service.get_memory_context("usr_test", query="How should I train today?")
    active_statuses = service.list_recent_status("usr_test")
    audit_statuses = service.list_recent_status("usr_test", include_expired=True)

    assert "primary_training_goal: run a marathon" in context.sections["profile"]
    assert "primary_training_goal: build strength" not in context.sections["profile"]
    assert "sleep_duration_hours: 5.0" in context.sections["recent_status"]
    assert "sleep_duration_hours: 5.5" not in context.sections["recent_status"]
    assert second_status.status_id in context.memory_ids
    assert first_status.status_id not in context.memory_ids
    assert [item.status_id for item in active_statuses] == [second_status.status_id]
    assert {item.status_id for item in audit_statuses} == {first_status.status_id, second_status.status_id}
    session.close()


def test_close_conversation_creates_episode_and_reviewable_candidates(tmp_path: Path):
    """Closing should create one episode and candidate memories, then approval writes profile/status."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    conversation = service.create_conversation("usr_test", title="Check-in")
    service.append_message(
        conversation.conversation_id,
        role="user",
        content=(
            "My goal is build muscle. My bench press 1RM is 100 kg. "
            "I slept 5 hours and I have knee pain today."
        ),
    )

    closed = service.close_conversation(conversation.conversation_id)
    closed_again = service.close_conversation(conversation.conversation_id)
    keys = {item.memory_key for item in closed.candidate_memory_items}

    assert closed.episode is not None
    assert closed.episode.episode_id == closed_again.episode.episode_id
    assert {"primary_training_goal", "bench_press_1rm_kg", "sleep_duration_hours", "joint_pain"} <= keys

    goal_item = next(item for item in closed.candidate_memory_items if item.memory_key == "primary_training_goal")
    status_item = next(item for item in closed.candidate_memory_items if item.memory_key == "joint_pain")
    approved_goal = service.approve_memory_item(goal_item.memory_id, MemoryItemApproveRequest())
    approved_status = service.approve_memory_item(status_item.memory_id, MemoryItemApproveRequest())

    profile = service.list_profile("usr_test")
    context = service.get_memory_context("usr_test", query="Can I train with this knee pain?")

    assert approved_goal.profile_field.field_key == "primary_training_goal"
    assert approved_status.recent_status.status_key == "joint_pain"
    assert any(field.field_key == "primary_training_goal" for field in profile.fields)
    assert "joint_pain" in context.sections["recent_status"]
    session.close()


def test_key_proposal_allows_explicit_extension_mode(tmp_path: Path):
    """Unknown keys are rejected unless extension mode creates and approves a proposal."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")

    response = service.update_profile(
        "usr_test",
        {"favorite_barbell": "Texas power bar"},
        extension_mode=True,
    )
    proposal = response.key_proposals[0]
    service.approve_key_proposal(proposal.proposal_id)
    patched = service.update_profile("usr_test", {"favorite_barbell": "Texas power bar"})

    assert proposal.proposed_key == "favorite_barbell"
    assert any(field.field_key == "favorite_barbell" for field in patched.fields)
    assert "primary_training_goal" in service.list_profile_keys("usr_test")
    assert "favorite_barbell" in service.list_profile_keys("usr_test")
    assert "sleep_duration_hours" in service.list_recent_status_keys("usr_test")
    session.close()


def test_plan_context_router_and_progress_events(tmp_path: Path):
    """Plan context is conditional and progress changes append events."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    plan = service.patch_plan_progress(
        "plan_alpha",
        PlanProgressPatchRequest(
            user_id="usr_test",
            planned_sessions=[{"session_id": "s1", "title": "Lower body"}],
            session_id="s1",
            session_status="pending",
            event_type="plan_created",
        ),
    )
    full_context = service.get_memory_context("usr_test", query="Can you adapt today's workout?")
    summary_context = service.get_memory_context("usr_test", query="I feel sore, any training advice?")
    none_context = service.get_memory_context("usr_test", query="What does creatine do?")

    assert plan.progress_events[0]["event_type"] == "plan_created"
    assert full_context.plan_context_level == "full"
    assert "planned_sessions" in full_context.sections["active_plan_progress"]
    assert summary_context.plan_context_level == "summary"
    assert "Session status counts" in summary_context.sections["active_plan_progress"]
    assert none_context.plan_context_level == "none"
    assert none_context.sections["active_plan_progress"] == ""
    session.close()


def test_xunji_import_is_idempotent_and_summarized(tmp_path: Path):
    """Training log imports should skip duplicate external IDs and summarize sessions."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    recent_date = utc_now().date().isoformat()
    request = TrainingLogImportRequest(
        user_id="usr_test",
        external_id="xunji_batch_1",
        payload={
            "sessions": [
                {
                    "id": "sess_1",
                    "date": recent_date,
                    "sport_type": "strength",
                    "exercises": [{"name": "Squat", "sets": 3}],
                    "rpe": 7,
                    "notes": "Felt solid.",
                }
            ]
        },
    )

    imported = service.import_xunji(request)
    duplicate = service.import_xunji(request)
    summary = service.get_training_log_summary("usr_test", days=30)

    assert imported.imported is True
    assert imported.sessions_imported == 1
    assert duplicate.skipped_duplicate is True
    assert summary.session_count == 1
    assert summary.sport_counts == {"strength": 1}
    session.close()
