"""User Profile and Memory service behavior."""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from sqlmodel import Session, select

from app.core.config import Settings
from app.core.llm_provider import (
    MemoryEpisodeSummaryResult,
    MemoryExtractionResult,
    MemoryKeyProposalResult,
    MemoryPlanRouteResult,
)
from app.memory.nodes import estimate_tokens
from app.memory.schemas import (
    MemoryItemApproveRequest,
    PlanProgressPatchRequest,
    TrainingLogImportRequest,
    TrainingLogXunjiSyncRequest,
)
from app.memory.service import MemoryService, ROLLING_SUMMARY_TARGET_RATIO, ROLLING_SUMMARY_TOKEN_BUDGET
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


class FakeMemoryLLMProvider:
    """Fake memory LLM provider for graph-node tests."""

    def generate_memory_plan_route(self, request):
        return MemoryPlanRouteResult(
            plan_context_level="summary",
            rationale="LLM judged this as mildly training related.",
            confidence=0.9,
        )

    def generate_memory_rolling_summary(self, request):
        return "LLM rolling summary: user reported accumulating fatigue and schedule limits."

    def generate_memory_episode_summary(self, request):
        return MemoryEpisodeSummaryResult(
            summary="LLM episode summary: user wants strength and has limited time today.",
            topics=["strength", "schedule"],
            importance=0.72,
        )

    def generate_memory_candidates(self, request):
        return MemoryExtractionResult(
            items=[
                {
                    "category": "profile",
                    "target_type": "ProfileField",
                    "memory_key": "primary_training_goal",
                    "text": "User wants to build strength.",
                    "value": "build strength",
                    "confidence": 0.88,
                    "safety_relevant": False,
                },
                {
                    "category": "recent_status",
                    "target_type": "RecentStatus",
                    "memory_key": "available_training_time_today",
                    "text": "User has 35 minutes to train today.",
                    "value": 35,
                    "confidence": 0.84,
                    "safety_relevant": False,
                },
            ]
        )

    def review_memory_key_proposal(self, request):
        return MemoryKeyProposalResult(
            duplicate_decision="duplicate",
            nearest_existing_keys=["exercise_preferences"],
            canonical_key="exercise_preferences",
            reason="The proposed key duplicates an existing exercise preference key.",
        )

    def generate_completed_plan_episode(self, request):
        return MemoryEpisodeSummaryResult(
            summary="LLM completed-plan summary: plan_alpha was completed with useful adherence feedback.",
            topics=["training_plan", "plan_completion", "adherence"],
            importance=0.8,
        )

    def classify_training_sport_type(self, request):
        title = str(request.session_payload.get("title") or "").lower()
        if "run" in title or "跑步" in title:
            return "running"
        return "strength"


def test_manual_profile_and_recent_status_appear_in_context(tmp_path: Path):
    """Manual approved memory should be retrievable as memory context."""

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


def test_recent_best_lifts_profile_updates_append_or_replace_entries(tmp_path: Path):
    session, service = _make_service(tmp_path)
    service.create_user("usr_test")

    service.update_profile(
        "usr_test",
        {
            "recent_best_lifts_json": [
                {"exercise": "Bench Press", "weight_kg": 100, "reps": 1, "date": "2026-06-01"}
            ]
        },
    )
    service.update_profile(
        "usr_test",
        {
            "recent_best_lifts_json": {
                "exercise": "Deadlift",
                "weight_kg": 180,
                "reps": 1,
                "date": "2026-06-02",
            }
        },
    )
    service.update_profile(
        "usr_test",
        {
            "recent_best_lifts_json": {
                "exercise": "Bench Press",
                "weight_kg": 105,
                "reps": 1,
                "date": "2026-06-10",
            }
        },
    )

    field = next(
        item
        for item in service.list_profile("usr_test").fields
        if item.field_key == "recent_best_lifts_json"
    )
    assert field.value == [
        {"exercise": "Bench Press", "weight_kg": 105, "reps": 1, "date": "2026-06-10"},
        {"exercise": "Deadlift", "weight_kg": 180, "reps": 1, "date": "2026-06-02"},
    ]
    session.close()


def test_recent_best_lifts_memory_approval_merges_with_profile(tmp_path: Path):
    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    service.update_profile(
        "usr_test",
        {
            "recent_best_lifts_json": [
                {"exercise_name": "Back Squat", "weight_kg": 140, "reps": 1}
            ]
        },
    )
    item = MemoryItem(
        memory_id="mem_best_lift",
        user_id="usr_test",
        category="profile",
        target_type="ProfileField",
        memory_key="recent_best_lifts_json",
        text="Back squat recent best is now 145 kg for 1 rep.",
        value_json=json.dumps({"exercise_name": "Back Squat", "weight_kg": 145, "reps": 1}),
        confidence=0.9,
        status="candidate",
    )
    session.add(item)
    session.commit()

    approved = service.approve_memory_item("mem_best_lift", MemoryItemApproveRequest())

    assert approved.profile_field.value == [
        {"exercise_name": "Back Squat", "weight_kg": 145, "reps": 1}
    ]
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


def test_recent_status_expiry_reminders_include_review_fields_and_suggestion(tmp_path: Path):
    """Near-expiry statuses should ask the user to expire or renew with a suggested date."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    now = utc_now()
    observed = now - timedelta(days=6, hours=23)
    status = service.update_recent_status(
        "usr_test",
        {"joint_pain": "mild right knee pain"},
        observed_at=observed,
    ).statuses[0]

    response = service.list_expiring_recent_status("usr_test", within_hours=2, now=now)
    reminder = response.reminders[0]

    assert response.statuses[0].status_id == status.status_id
    assert response.statuses[0].expiration_reminder_status == "due"
    assert reminder.status_id == status.status_id
    assert reminder.status_key == "joint_pain"
    assert reminder.value == "mild right knee pain"
    assert reminder.source == "manual"
    assert reminder.current_valid_until == status.valid_until
    assert reminder.suggested_valid_until == now + timedelta(days=7)
    assert "Expire it if it is no longer true" in reminder.prompt
    session.close()


def test_recent_status_expire_and_renew_actions_keep_audit_storage(tmp_path: Path):
    """The user can expire or renew a reminder without deleting the status row."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    pain = service.update_recent_status("usr_test", {"joint_pain": "resolved knee pain"}).statuses[0]
    sleep = service.update_recent_status("usr_test", {"sleep_duration_hours": 6}).statuses[0]

    expired = service.expire_recent_status(pain.status_id)
    renewed_until = utc_now() + timedelta(days=2)
    renewed = service.update_recent_status_valid_until(sleep.status_id, renewed_until)
    context = service.get_memory_context("usr_test", query="How should I train today?")
    audit_statuses = service.list_recent_status("usr_test", include_expired=True)

    assert expired.review_status == "expired"
    assert expired.expiration_reminder_status == "expired_by_user"
    assert expired.expired_at is not None
    assert renewed.review_status == "approved"
    assert renewed.expiration_reminder_status == "renewed"
    assert renewed.expired_at is None
    assert "joint_pain" not in context.sections["recent_status"]
    assert sleep.status_id in context.memory_ids
    assert {item.status_id for item in audit_statuses} == {pain.status_id, sleep.status_id}
    session.close()


def test_recent_status_hides_after_expiry_without_user_response(tmp_path: Path):
    """Expired statuses should leave normal context while remaining available for audit."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    observed = utc_now() - timedelta(days=1, hours=1)
    status = service.update_recent_status(
        "usr_test",
        {"sleep_duration_hours": 5},
        observed_at=observed,
    ).statuses[0]

    context = service.get_memory_context("usr_test", query="How should I train today?")
    active_statuses = service.list_recent_status("usr_test")
    audit_statuses = service.list_recent_status("usr_test", include_expired=True)
    audit_status = audit_statuses[0]

    assert "sleep_duration_hours" not in context.sections["recent_status"]
    assert active_statuses == []
    assert audit_status.status_id == status.status_id
    assert audit_status.review_status == "expired"
    assert audit_status.expired_at == status.valid_until
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


def test_inactive_conversation_auto_closes_and_creates_episode(tmp_path: Path):
    """Open conversations older than the inactivity threshold should close lazily."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    conversation = service.create_conversation("usr_test", title="Idle check-in")
    service.append_message(
        conversation.conversation_id,
        role="user",
        content="My goal is build muscle. I slept 5 hours.",
    )
    row = session.get(Conversation, conversation.conversation_id)
    row.updated_at = utc_now() - timedelta(hours=1, minutes=5)
    session.add(row)
    session.commit()

    listed = service.list_conversations("usr_test")
    episode = session.exec(
        select(Episode).where(Episode.conversation_id == conversation.conversation_id)
    ).first()
    candidates = service.list_memory_items("usr_test", status="candidate")

    assert listed[0].status == "closed"
    assert listed[0].closure_reason == "inactive_timeout"
    assert episode is not None
    assert {"primary_training_goal", "sleep_duration_hours"} <= {
        item.memory_key for item in candidates
    }
    session.close()


def test_reopened_conversation_updates_episode_and_extracts_new_messages_only(tmp_path: Path):
    """Reusing a closed conversation should reopen it, then refresh the existing episode on close."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    conversation = service.create_conversation("usr_test", title="Reusable chat", conversation_id="conv_reuse")
    service.append_message(
        conversation.conversation_id,
        role="user",
        content="My goal is build muscle.",
    )
    first_close = service.close_conversation(conversation.conversation_id)
    first_episode_id = first_close.episode.episode_id
    first_candidate_ids = {item.memory_id for item in first_close.candidate_memory_items}

    reopened = service.create_conversation("usr_test", conversation_id="conv_reuse")
    service.append_message(
        conversation.conversation_id,
        role="user",
        content="My bench press 1RM is 100 kg.",
    )
    open_detail = service.get_conversation_detail(conversation.conversation_id)
    second_close = service.close_conversation(conversation.conversation_id)
    final_detail = service.get_conversation_detail(conversation.conversation_id)
    all_candidates = service.list_memory_items("usr_test", status="candidate")

    assert reopened.status == "open"
    assert open_detail.conversation.status == "open"
    assert second_close.episode.episode_id == first_episode_id
    assert len(second_close.episode.source_message_ids) == 2
    assert "bench press" in second_close.episode.summary.lower()
    assert {"primary_training_goal", "bench_press_1rm_kg"} <= {
        item.memory_key for item in second_close.candidate_memory_items
    }
    assert first_candidate_ids <= {item.memory_id for item in second_close.candidate_memory_items}
    assert len([item for item in all_candidates if item.memory_key == "primary_training_goal"]) == 1
    assert len([item for item in all_candidates if item.memory_key == "bench_press_1rm_kg"]) == 1
    assert final_detail.conversation.status == "closed"
    session.close()


def test_llm_memory_nodes_create_summary_episode_and_candidates(tmp_path: Path):
    """LLM-backed memory nodes should replace deterministic summaries/extraction when available."""

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
    service = MemoryService(session=session, settings=settings, llm_provider=FakeMemoryLLMProvider())
    service.create_user("usr_test")
    conversation = service.create_conversation("usr_test", title="LLM close")
    long_context = " ".join(["strength schedule limit"] * 90)
    for index in range(17):
        service.append_message(
            conversation.conversation_id,
            role="user",
            content=(
                f"Turn {index}: I want strength and only have 35 minutes today. "
                f"{long_context}"
            ),
        )

    detail_before_close = service.get_conversation_detail(conversation.conversation_id)
    closed = service.close_conversation(conversation.conversation_id)
    keys = {item.memory_key for item in closed.candidate_memory_items}

    assert detail_before_close.conversation.rolling_summary.startswith("LLM rolling summary")
    assert detail_before_close.conversation.rolling_summary_until_message_id is not None
    assert closed.episode.summary.startswith("LLM episode summary")
    assert closed.episode.topics == ["strength", "schedule"]
    assert closed.episode.importance == 0.72
    assert {"primary_training_goal", "available_training_time_today"} <= keys
    session.close()


def test_working_memory_window_summarizes_oldest_messages_by_token_budget(tmp_path: Path):
    """Oversized working memory should persist a summary cursor and keep only a small live window."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    conversation = service.create_conversation("usr_test", title="Long chat")
    long_tail = " ".join(["readiness recovery strength"] * 160)
    saved_messages = []
    for index in range(6):
        saved_messages.append(
            service.append_message(
                conversation.conversation_id,
                role="user",
                content=f"Turn {index}: My goal is build strength. I slept 5 hours. {long_tail}",
            )
        )

    detail = service.get_conversation_detail(conversation.conversation_id)
    pre_close_items = service.list_memory_items("usr_test", status="candidate")
    context = service.get_memory_context(
        "usr_test",
        conversation_id=conversation.conversation_id,
        query="What should I do next?",
    )
    working_memory = context.sections["working_memory"]

    assert len(detail.messages) == 6
    assert detail.conversation.rolling_summary
    assert detail.conversation.rolling_summary_until_message_id is not None
    assert estimate_tokens(working_memory) <= 1800
    assert "Turn 0:" not in working_memory
    assert "Turn 5:" in working_memory
    assert {"primary_training_goal", "sleep_duration_hours"} <= {
        item.memory_key for item in pre_close_items
    }
    assert any(item.source_episode_id is None for item in pre_close_items)
    assert saved_messages[0].message_id not in context.memory_ids
    assert saved_messages[-1].message_id in context.memory_ids

    closed = service.close_conversation(conversation.conversation_id)
    post_close_items = service.list_memory_items("usr_test", status="candidate")
    pre_close_ids = {item.memory_id for item in pre_close_items}
    linked_pre_close_items = [
        item for item in post_close_items if item.memory_id in pre_close_ids
    ]

    assert pre_close_ids <= {item.memory_id for item in closed.candidate_memory_items}
    assert all(item.source_episode_id == closed.episode.episode_id for item in linked_pre_close_items)
    session.close()


def test_rolling_summary_self_compression_drops_oldest_content(tmp_path: Path):
    """Oversized rolling summaries should be capped by dropping oldest lines."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    conversation = service.create_conversation("usr_test", title="Rolling summary cap")
    row = session.get(Conversation, conversation.conversation_id)
    row.rolling_summary = "\n".join(
        f"old summary line {index}: " + "old context " * 30
        for index in range(180)
    )
    session.add(row)
    session.commit()

    long_tail = " ".join(["newest fallback marker recovery strength"] * 900)
    service.append_message(
        conversation.conversation_id,
        role="user",
        content=f"Newest fallback content marker. {long_tail}",
    )

    detail = service.get_conversation_detail(conversation.conversation_id)
    summary = detail.conversation.rolling_summary
    target_tokens = int(ROLLING_SUMMARY_TOKEN_BUDGET * ROLLING_SUMMARY_TARGET_RATIO)

    assert estimate_tokens(summary) <= target_tokens
    assert "old summary line 0" not in summary
    assert "Newest fallback content marker" in summary
    assert detail.conversation.rolling_summary_until_message_id is not None
    session.close()


def test_rolling_summary_self_compression_prefers_llm_rewrite(tmp_path: Path):
    """When available, LLM compression should rewrite oversized rolling summaries."""

    class RollingSummaryCompressionProvider(FakeMemoryLLMProvider):
        def __init__(self):
            self.compression_requests = []

        def generate_memory_rolling_summary(self, request):
            return "\n".join(
                f"oversized llm summary line {index}: " + "too much context " * 30
                for index in range(180)
            )

        def generate_memory_rolling_summary_compression(self, request):
            self.compression_requests.append(request)
            return "LLM compressed rolling summary: user wants strength and reported limited sleep."

    session, service = _make_service(tmp_path)
    provider = RollingSummaryCompressionProvider()
    service.llm_provider = provider
    service.create_user("usr_test")
    conversation = service.create_conversation("usr_test", title="LLM summary compression")
    long_tail = " ".join(["readiness recovery strength"] * 900)

    service.append_message(
        conversation.conversation_id,
        role="user",
        content=f"My goal is build strength. I slept 5 hours. {long_tail}",
    )

    detail = service.get_conversation_detail(conversation.conversation_id)
    summary = detail.conversation.rolling_summary
    target_tokens = int(ROLLING_SUMMARY_TOKEN_BUDGET * ROLLING_SUMMARY_TARGET_RATIO)

    assert provider.compression_requests
    assert provider.compression_requests[0].target_tokens == target_tokens
    assert summary == "LLM compressed rolling summary: user wants strength and reported limited sleep."
    assert estimate_tokens(summary) <= target_tokens
    assert detail.conversation.rolling_summary_until_message_id is not None
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


def test_llm_key_proposal_detector_marks_duplicates(tmp_path: Path):
    """The LLM key detector can reject duplicate/equivalent custom keys."""

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
    service = MemoryService(session=session, settings=settings, llm_provider=FakeMemoryLLMProvider())
    service.create_user("usr_test")

    response = service.update_profile(
        "usr_test",
        {"favorite_lifts": ["squat"]},
        extension_mode=True,
    )
    proposal = response.key_proposals[0]

    assert proposal.duplicate_decision == "duplicate"
    assert proposal.review_status == "rejected"
    assert proposal.nearest_existing_keys == ["exercise_preferences"]
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


def test_llm_plan_router_and_completed_plan_episode_writer(tmp_path: Path):
    """Plan routing and completed-plan episodic summary should use LLM nodes when available."""

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
    service = MemoryService(session=session, settings=settings, llm_provider=FakeMemoryLLMProvider())
    service.create_user("usr_test")
    service.patch_plan_progress(
        "plan_alpha",
        PlanProgressPatchRequest(
            user_id="usr_test",
            planned_sessions=[{"session_id": "s1", "title": "Upper body"}],
            event_type="plan_created",
        ),
    )

    context = service.get_memory_context(
        "usr_test",
        query="What does creatine do?",
        include_debug=True,
    )
    service.patch_plan_progress(
        "plan_alpha",
        PlanProgressPatchRequest(
            user_id="usr_test",
            status="completed",
            event_type="plan_completed",
            feedback="Good adherence overall.",
        ),
    )
    episode = session.exec(
        select(Episode).where(Episode.source_type == "plan_completion")
    ).one()

    assert context.plan_context_level == "summary"
    assert context.debug["plan_context_router_used"] == "llm"
    assert context.debug["plan_context_confidence"] == 0.9
    assert episode.summary.startswith("LLM completed-plan summary")
    assert json.loads(episode.topics_json) == ["training_plan", "plan_completion", "adherence"]
    session.close()


def test_oversized_memory_context_sections_warn_without_trimming(tmp_path: Path):
    """Oversized memory sections should warn the user without mutating context text."""

    session, service = _make_service(tmp_path)
    service.create_user("usr_test")
    long_preference = " ".join(["prefers slow tempo strength training"] * 1200)
    service.update_profile("usr_test", {"exercise_preferences": long_preference})

    context = service.get_memory_context(
        "usr_test",
        query="How should I train today?",
        include_debug=True,
    )
    profile_warning = next(
        warning
        for warning in context.budget_warnings
        if warning["type"] == "section_budget_exceeded" and warning["section"] == "profile"
    )

    assert long_preference in context.sections["profile"]
    assert profile_warning["token_estimate"] > profile_warning["token_budget"]
    assert "was not compacted" in profile_warning["message"]
    assert context.debug["budget_warnings"] == context.budget_warnings
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
    listed_sessions = service.list_training_sessions("usr_test")

    assert imported.imported is True
    assert imported.sessions_imported == 1
    assert duplicate.skipped_duplicate is True
    assert listed_sessions.session_count == 1
    assert listed_sessions.sessions[0].sport_type == "strength"
    assert listed_sessions.sessions[0].notes == "Felt solid."
    assert listed_sessions.sessions[0].exercises[0]["name"] == "Squat"
    context = service.get_memory_context("usr_test", query="Review my recent training.")
    assert "training_log_summary" not in context.sections
    session.close()


def test_xunji_import_labels_sport_type_drops_overlapping_less_detailed_session_and_filters(tmp_path: Path):
    """Xunji normalization should classify sessions, drop overlapping duplicates, and support filters."""

    session, service = _make_service(tmp_path)
    service = MemoryService(session=session, settings=service.settings, llm_provider=FakeMemoryLLMProvider())
    service.create_user("usr_test")
    base = datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc)
    less_detailed_start = int(base.timestamp() * 1000)
    less_detailed_end = int((base + timedelta(minutes=60)).timestamp() * 1000)
    detailed_start = int((base + timedelta(minutes=10)).timestamp() * 1000)
    detailed_end = int((base + timedelta(minutes=80)).timestamp() * 1000)
    running_start = int((base + timedelta(hours=4)).timestamp() * 1000)
    running_end = int((base + timedelta(hours=4, minutes=35)).timestamp() * 1000)

    imported = service.import_xunji(
        TrainingLogImportRequest(
            user_id="usr_test",
            external_id="xunji_overlap_batch",
            payload={
                "success": True,
                "res": {
                    "trains": [
                        {
                            "localid": "short",
                            "datestr": "2026-06-15",
                            "title": "Chest",
                            "start": less_detailed_start,
                            "end": less_detailed_end,
                            "movements": [{"name": "Bench Press", "sets": [{"weight": 80, "reps": 5}]}],
                        },
                        {
                            "localid": "detailed",
                            "datestr": "2026-06-15",
                            "title": "Strength detailed",
                            "start": detailed_start,
                            "end": detailed_end,
                            "movements": [
                                {
                                    "name": "Bench Press",
                                    "sets": [
                                        {"weight": 80, "reps": 5, "time": 180},
                                        {"weight": 85, "reps": 5, "time": 180},
                                    ],
                                },
                                {"name": "Squat", "sets": [{"weight": 120, "reps": 3}]},
                            ],
                            "notes": "More complete import.",
                        },
                        {
                            "localid": "run",
                            "datestr": "2026-06-15",
                            "title": "Running",
                            "start": running_start,
                            "end": running_end,
                            "movements": [{"name": "Running", "sets": [{"metrics": {"distance": 5000}}]}],
                        },
                    ]
                },
            },
        )
    )

    assert imported.sessions_imported == 2
    assert len(imported.dropped_sessions) == 1
    assert {item.title for item in imported.sessions} == {"Strength detailed", "Running"}
    assert {item.sport_type for item in imported.sessions} == {"strength", "running"}
    strength_only = service.list_training_sessions(
        "usr_test",
        start_date=date(2026, 6, 15),
        end_date=date(2026, 6, 15),
        sport_type="strength",
    )
    assert strength_only.session_count == 1
    assert strength_only.sessions[0].title == "Strength detailed"
    assert strength_only.sessions[0].duration_seconds == 70 * 60
    assert "time" not in strength_only.sessions[0].exercises[0]["sets"][0]
    assert "time" in strength_only.sessions[0].raw_payload["movements"][0]["sets"][0]
    context = service.get_memory_context("usr_test", query="Review my recent training.")
    assert "training_log_summary" not in context.sections
    session.close()


def test_xunji_sync_reads_api_client_and_imports_each_day(tmp_path: Path, monkeypatch):
    """The Xunji sync path should fetch day payloads and import them through the normal pipeline."""

    class FakeXunjiClient:
        def __init__(self, settings):
            self.settings = settings

        def read_day(self, datestr, include_full_data=True, force_refresh=False):
            return {
                "success": True,
                "res": {
                    "trains": [
                        {
                            "localid": f"run_{datestr}",
                            "datestr": datestr,
                            "title": "Running",
                            "start": f"{datestr}T06:00:00+00:00",
                            "end": f"{datestr}T06:30:00+00:00",
                            "movements": [{"name": "Running", "sets": [{"metrics": {"distance": 5000}}]}],
                        }
                    ]
                },
            }

    monkeypatch.setattr("app.memory.service.XunjiClient", FakeXunjiClient)
    session, service = _make_service(tmp_path)
    service = MemoryService(session=session, settings=service.settings, llm_provider=FakeMemoryLLMProvider())
    service.create_user("usr_test")

    response = service.sync_xunji_training_logs(
        TrainingLogXunjiSyncRequest(
            user_id="usr_test",
            start_date=date(2026, 6, 14),
            end_date=date(2026, 6, 15),
            include_full_data=True,
        )
    )

    listed = service.list_training_sessions("usr_test", sport_type="running")
    assert response.days_requested == 2
    assert len(response.imports) == 2
    assert response.errors == []
    assert listed.session_count == 2
    assert {item.title for item in listed.sessions} == {"Running"}
    session.close()
