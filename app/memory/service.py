"""Service layer for the User Profile and Memory block."""

from collections import Counter
from datetime import date, datetime, timedelta, timezone
import difflib
import hashlib
import json
import re
import time
from typing import Any
from uuid import uuid4

from sqlmodel import Session, select

from app.core.config import Settings
from app.core.errors import NotFoundError
from app.core.llm_provider import (
    CompletedPlanEpisodeRequest,
    MemoryEpisodeSummaryRequest,
    MemoryExtractionRequest,
    MemoryKeyProposalRequest,
    MemoryRollingSummaryCompressionRequest,
    MemoryRollingSummaryRequest,
    OpenAIProvider,
    TrainingSportTypeRequest,
)
from app.memory.catalogs import (
    DAILY_STATUS_KEYS,
    PROFILE_FIELD_KEYS,
    RECENT_STATUS_KEYS,
    default_valid_until,
    normalize_key,
)
from app.integrations.xunji import XunjiClient
from app.memory.graph import run_langgraph_or_nodes
from app.memory.nodes import (
    MEMORY_SECTION_TOKEN_BUDGETS,
    PlanContextRouter,
    build_memory_budget_warnings,
    estimate_tokens,
    render_sections,
)
from app.memory.schemas import (
    ConversationCloseResponse,
    ConversationDetailOut,
    ConversationMessageOut,
    ConversationOut,
    EpisodeOut,
    EpisodeSearchResponse,
    ExpiringRecentStatusResponse,
    KeyProposalOut,
    MemoryDeleteResponse,
    MemoryContextOut,
    MemoryItemApproveRequest,
    MemoryItemOut,
    MemoryItemReviewResponse,
    PlanProgressOut,
    PlanProgressPatchRequest,
    PlanActivationRequest,
    PlanActivationResponse,
    ProfileFieldOut,
    ProfilePatchResponse,
    ProfileResponse,
    RecentStatusOut,
    RecentStatusExpiryReminder,
    RecentStatusPatchResponse,
    TrainingLogImportRequest,
    TrainingLogImportResponse,
    TrainingLogXunjiSyncRequest,
    TrainingLogXunjiSyncResponse,
    TrainingSessionListResponse,
    TrainingSessionOut,
    UserOut,
)
from app.storage.models import (
    Conversation,
    ConversationMessage,
    Episode,
    MemoryItem,
    MemoryKeyProposal,
    MemoryTrace,
    ProfileField,
    RecentStatus,
    TrainingLogImport,
    TrainingPlanProgress,
    TrainingSession,
    UserAccount,
    utc_now,
)


WORKING_MEMORY_TOKEN_BUDGET = MEMORY_SECTION_TOKEN_BUDGETS["working_memory"]
WORKING_MEMORY_TARGET_RATIO = 0.3
ROLLING_SUMMARY_TOKEN_BUDGET = MEMORY_SECTION_TOKEN_BUDGETS["rolling_summary"]
ROLLING_SUMMARY_TARGET_RATIO = 0.5
CONVERSATION_INACTIVE_CLOSE_AFTER = timedelta(hours=1)
INACTIVE_CLOSURE_REASON = "inactive_timeout"
TRAINING_SPORT_TYPES = [
    "strength",
    "running",
    "other aerobic exercise",
    "other anaerobic exercise",
    "other Mixed Aerobic-Anaerobic Exercise",
]
OVERLAP_DROP_THRESHOLD = timedelta(minutes=20)


class MemoryService:
    """Stable service interface for user profile, memory, plans, and logs."""

    def __init__(
        self,
        session: Session,
        settings: Settings,
        llm_provider: OpenAIProvider | None = None,
    ):
        self.session = session
        self.settings = settings
        self.llm_provider = llm_provider or OpenAIProvider(settings)

    def create_user(
        self,
        user_id: str | None = None,
        display_name: str | None = None,
        timezone_name: str = "UTC",
    ) -> UserOut:
        """Create or update a local user account."""

        resolved_user_id = user_id or f"usr_{uuid4().hex[:12]}"
        user = self.session.get(UserAccount, resolved_user_id)
        now = utc_now()
        if user is None:
            user = UserAccount(
                user_id=resolved_user_id,
                display_name=display_name,
                timezone=timezone_name or "UTC",
                created_at=now,
                updated_at=now,
            )
        else:
            if display_name is not None:
                user.display_name = display_name
            if timezone_name:
                user.timezone = timezone_name
            user.updated_at = now
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return _user_out(user)

    def list_users(self) -> list[UserOut]:
        """Return local users for frontend selection and debugging."""

        users = self.session.exec(select(UserAccount)).all()
        users = sorted(users, key=lambda item: item.user_id)
        return [_user_out(user) for user in users]

    def delete_user(self, user_id: str) -> MemoryDeleteResponse:
        """Delete one local user and all rows owned by the memory block."""

        user = self.session.get(UserAccount, user_id)
        if user is None:
            raise NotFoundError(f"User not found: {user_id}")

        counts: dict[str, int] = {}
        self._delete_query_rows(
            "conversation_messages",
            select(ConversationMessage).where(ConversationMessage.user_id == user_id),
            counts,
        )
        self._delete_query_rows("episodes", select(Episode).where(Episode.user_id == user_id), counts)
        self._delete_query_rows("memory_items", select(MemoryItem).where(MemoryItem.user_id == user_id), counts)
        self._delete_query_rows("profile_fields", select(ProfileField).where(ProfileField.user_id == user_id), counts)
        self._delete_query_rows("recent_status", select(RecentStatus).where(RecentStatus.user_id == user_id), counts)
        self._delete_query_rows(
            "memory_key_proposals",
            select(MemoryKeyProposal).where(MemoryKeyProposal.user_id == user_id),
            counts,
        )
        self._delete_query_rows(
            "training_sessions",
            select(TrainingSession).where(TrainingSession.user_id == user_id),
            counts,
        )
        self._delete_query_rows(
            "training_log_imports",
            select(TrainingLogImport).where(TrainingLogImport.user_id == user_id),
            counts,
        )
        self._delete_query_rows(
            "training_plan_progress",
            select(TrainingPlanProgress).where(TrainingPlanProgress.user_id == user_id),
            counts,
        )
        self._delete_query_rows("memory_traces", select(MemoryTrace).where(MemoryTrace.user_id == user_id), counts)
        self._delete_query_rows("conversations", select(Conversation).where(Conversation.user_id == user_id), counts)
        self.session.delete(user)
        counts["user_accounts"] = 1
        self.session.commit()
        return MemoryDeleteResponse(
            deleted_id=user_id,
            detail="Deleted user and user-owned memory rows.",
            deleted_counts=counts,
        )

    def list_conversations(self, user_id: str | None = None) -> list[ConversationOut]:
        """Return memory conversations for selection and debugging."""

        self._close_inactive_conversations(user_id=user_id)
        statement = select(Conversation)
        if user_id:
            statement = statement.where(Conversation.user_id == user_id)
        rows = self.session.exec(statement).all()
        rows = sorted(rows, key=lambda item: item.updated_at, reverse=True)
        return [_conversation_out(conversation) for conversation in rows]

    def create_conversation(
        self,
        user_id: str,
        title: str | None = None,
        topic: str | None = None,
        conversation_id: str | None = None,
    ) -> ConversationOut:
        """Start a conversation for working memory."""

        self._ensure_user(user_id)
        now = utc_now()
        resolved_conversation_id = (conversation_id or "").strip() or f"conv_{uuid4().hex[:12]}"
        existing = self.session.get(Conversation, resolved_conversation_id)
        if existing is not None:
            if existing.user_id != user_id:
                raise ValueError("Conversation ID belongs to another user.")
            if self._conversation_is_inactive(existing, now):
                self._close_conversation(existing, INACTIVE_CLOSURE_REASON, extract_candidates=True)
                self.session.refresh(existing)
            if existing.status == "closed":
                self._reopen_conversation(existing)
                self.session.refresh(existing)
            if title is not None:
                existing.title = title
            if topic is not None:
                existing.topic = topic
            existing.updated_at = now
            self.session.add(existing)
            self.session.commit()
            self.session.refresh(existing)
            return _conversation_out(existing)

        conversation = Conversation(
            conversation_id=resolved_conversation_id,
            user_id=user_id,
            title=title,
            topic=topic,
            created_at=now,
            updated_at=now,
        )
        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(conversation)
        return _conversation_out(conversation)

    def ensure_conversation(
        self,
        user_id: str,
        conversation_id: str,
        title: str | None = None,
        topic: str | None = None,
    ) -> ConversationOut:
        """Return an existing conversation or create one with the supplied ID."""

        resolved_conversation_id = conversation_id.strip()
        if not resolved_conversation_id:
            raise ValueError("Conversation ID cannot be blank.")
        conversation = self.session.get(Conversation, resolved_conversation_id)
        if conversation is not None:
            if conversation.user_id != user_id:
                raise ValueError("Conversation ID belongs to another user.")
            now = utc_now()
            if self._conversation_is_inactive(conversation, now):
                self._close_conversation(conversation, INACTIVE_CLOSURE_REASON, extract_candidates=True)
                self.session.refresh(conversation)
            if conversation.status == "closed":
                self._reopen_conversation(conversation)
                self.session.refresh(conversation)
            return _conversation_out(conversation)
        return self.create_conversation(
            user_id=user_id,
            title=title,
            topic=topic,
            conversation_id=resolved_conversation_id,
        )

    def delete_conversation(self, conversation_id: str) -> MemoryDeleteResponse:
        """Delete one memory conversation plus its messages, episode, traces, and memory items."""

        conversation = self._get_conversation(conversation_id)
        counts: dict[str, int] = {}
        self._delete_query_rows(
            "conversation_messages",
            select(ConversationMessage).where(ConversationMessage.conversation_id == conversation_id),
            counts,
        )
        episodes = self.session.exec(select(Episode).where(Episode.conversation_id == conversation_id)).all()
        episode_ids = [episode.episode_id for episode in episodes]
        memory_items = self.session.exec(
            select(MemoryItem).where(MemoryItem.source_conversation_id == conversation_id)
        ).all()
        if episode_ids:
            episode_memory_items = self.session.exec(
                select(MemoryItem).where(MemoryItem.source_episode_id.in_(episode_ids))
            ).all()
            memory_items_by_id = {item.memory_id: item for item in memory_items}
            memory_items_by_id.update({item.memory_id: item for item in episode_memory_items})
            memory_items = list(memory_items_by_id.values())
        self._delete_rows("memory_items", memory_items, counts)
        self._delete_rows("episodes", episodes, counts)
        self._delete_query_rows(
            "memory_traces",
            select(MemoryTrace).where(MemoryTrace.conversation_id == conversation_id),
            counts,
        )
        self.session.delete(conversation)
        counts["conversations"] = 1
        self.session.commit()
        return MemoryDeleteResponse(
            deleted_id=conversation_id,
            detail="Deleted conversation, messages, episode, traces, and linked memory items.",
            deleted_counts=counts,
        )

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        source_block: str = "manual",
        qa_trace_id: str | None = None,
    ) -> ConversationMessageOut:
        """Append one message and update deterministic working-memory summary if needed."""

        conversation = self._get_conversation(conversation_id)
        now = utc_now()
        if self._conversation_is_inactive(conversation, now):
            self._close_conversation(conversation, INACTIVE_CLOSURE_REASON, extract_candidates=True)
            self.session.refresh(conversation)
        if conversation.status == "closed":
            self._reopen_conversation(conversation)
            self.session.refresh(conversation)
        message = ConversationMessage(
            message_id=f"msg_{uuid4().hex[:12]}",
            conversation_id=conversation_id,
            user_id=conversation.user_id,
            role=role.strip().lower(),
            content=content.strip(),
            source_block=source_block,
            qa_trace_id=qa_trace_id,
        )
        conversation.updated_at = utc_now()
        self.session.add(message)
        self.session.add(conversation)
        self.session.commit()
        self._update_rolling_summary_if_needed(conversation_id)
        self.session.refresh(message)
        return _message_out(message)

    def get_conversation_detail(self, conversation_id: str) -> ConversationDetailOut:
        """Return one conversation with all saved messages in order."""

        conversation = self._get_conversation(conversation_id)
        if self._conversation_is_inactive(conversation):
            self._close_conversation(
                conversation,
                INACTIVE_CLOSURE_REASON,
                extract_candidates=True,
            )
            conversation = self._get_conversation(conversation_id)
        messages = self._conversation_messages(conversation_id)
        return ConversationDetailOut(
            conversation=_conversation_out(conversation),
            messages=[_message_out(message) for message in messages],
        )

    def get_conversation_context(
        self,
        conversation_id: str,
        query: str = "",
        task_type: str = "qa",
        include_debug: bool = False,
    ) -> MemoryContextOut:
        """Build memory context scoped to one conversation."""

        conversation = self._get_conversation(conversation_id)
        if self._conversation_is_inactive(conversation):
            self._close_conversation(conversation, INACTIVE_CLOSURE_REASON, extract_candidates=True)
            conversation = self._get_conversation(conversation_id)
        return self.get_memory_context(
            user_id=conversation.user_id,
            query=query,
            conversation_id=conversation_id,
            task_type=task_type,
            include_debug=include_debug,
        )

    def close_conversation(
        self,
        conversation_id: str,
        closure_reason: str = "user_closed",
        extract_candidates: bool = True,
        program_context_id: str | None = None,
    ) -> ConversationCloseResponse:
        """Close a conversation idempotently and create one episode."""

        conversation = self._get_conversation(conversation_id)
        return self._close_conversation(
            conversation,
            closure_reason=closure_reason,
            extract_candidates=extract_candidates,
            program_context_id=program_context_id,
        )

    def _close_conversation(
        self,
        conversation: Conversation,
        closure_reason: str,
        extract_candidates: bool,
        program_context_id: str | None = None,
    ) -> ConversationCloseResponse:
        """Close one conversation and create or refresh its episode and candidates."""

        if conversation.status != "closed":
            conversation.status = "closed"
            conversation.closed_at = utc_now()
            conversation.updated_at = conversation.closed_at
            conversation.closure_reason = closure_reason
            self.session.add(conversation)
            self.session.commit()
            self.session.refresh(conversation)

        messages = self._conversation_messages(conversation.conversation_id)
        existing_episode = self.session.exec(
            select(Episode).where(Episode.conversation_id == conversation.conversation_id)
        ).first()
        episode = self._upsert_episode_from_conversation(conversation, messages, existing_episode)
        if program_context_id and episode.program_context_id != program_context_id:
            episode.program_context_id = program_context_id
            self.session.add(episode)
            self.session.commit()
            self.session.refresh(episode)
        candidates = []
        if extract_candidates:
            candidates = self._extract_candidates_for_episode(conversation, episode)
        return ConversationCloseResponse(
            conversation=_conversation_out(conversation),
            episode=_episode_out(episode),
            candidate_memory_items=[_memory_item_out(item) for item in candidates],
        )

    def _reopen_conversation(self, conversation: Conversation) -> None:
        """Reopen a closed conversation before a new turn is appended."""

        if conversation.status != "closed":
            return
        conversation.status = "open"
        conversation.closed_at = None
        conversation.closure_reason = None
        conversation.updated_at = utc_now()
        self.session.add(conversation)
        self.session.commit()

    def _close_inactive_conversations(self, user_id: str | None = None) -> None:
        """Lazily close open conversations inactive for more than one hour."""

        now = utc_now()
        statement = select(Conversation).where(Conversation.status == "open")
        if user_id:
            statement = statement.where(Conversation.user_id == user_id)
        conversations = self.session.exec(statement).all()
        for conversation in conversations:
            if self._conversation_is_inactive(conversation, now):
                self._close_conversation(
                    conversation,
                    closure_reason=INACTIVE_CLOSURE_REASON,
                    extract_candidates=True,
                )

    def _conversation_is_inactive(
        self,
        conversation: Conversation,
        now: datetime | None = None,
    ) -> bool:
        if conversation.status != "open":
            return False
        active_now = _ensure_aware(now or utc_now())
        return _ensure_aware(conversation.updated_at) < active_now - CONVERSATION_INACTIVE_CLOSE_AFTER

    def search_episodes(
        self,
        user_id: str,
        query: str = "",
        top_k: int = 5,
    ) -> EpisodeSearchResponse:
        """Search episodic memories with Chroma when available, then SQLite fallback."""

        self._ensure_user(user_id)
        semantic_error = None
        semantic_results: list[Episode] = []
        if query.strip():
            try:
                semantic_results = self._semantic_episode_search(user_id, query, top_k)
            except Exception as exc:
                semantic_error = str(exc)
        if semantic_results:
            return EpisodeSearchResponse(
                episodes=[_episode_out(item) for item in semantic_results],
                semantic_error=semantic_error,
            )
        fallback = self._keyword_episode_search(user_id, query, top_k)
        return EpisodeSearchResponse(
            episodes=[_episode_out(item) for item in fallback],
            semantic_error=semantic_error,
        )

    def update_profile(
        self,
        user_id: str,
        fields: dict[str, Any],
        source: str = "manual",
        extension_mode: bool = False,
    ) -> ProfilePatchResponse:
        """Apply manual profile edits, creating key proposals for unknown keys."""

        self._ensure_user(user_id)
        proposals = []
        for raw_key, value in fields.items():
            field_key = normalize_key(raw_key)
            if not self._is_key_allowed(user_id, "ProfileField", field_key):
                if extension_mode:
                    proposals.append(
                        self._create_key_proposal(
                            user_id=user_id,
                            target_type="ProfileField",
                            proposed_key=field_key,
                            example_value=value,
                            reason="Manual profile edit used a key outside the approved v1 catalog.",
                        )
                    )
                    continue
                raise ValueError(f"Unknown profile field key: {field_key}")
            self._upsert_profile_field(user_id, field_key, value, source=source)
        self.session.commit()
        return ProfilePatchResponse(
            user_id=user_id,
            fields=self.list_profile(user_id).fields,
            key_proposals=[_key_proposal_out(item) for item in proposals],
        )

    def list_profile(self, user_id: str) -> ProfileResponse:
        """Return approved profile fields for one user."""

        self._ensure_user(user_id)
        rows = self._latest_profile_rows(user_id)
        return ProfileResponse(user_id=user_id, fields=[_profile_field_out(item) for item in rows])

    def delete_profile_field(self, profile_field_id: str) -> MemoryDeleteResponse:
        """Delete one profile entry by its row ID."""

        field = self.session.get(ProfileField, profile_field_id)
        if field is None:
            raise NotFoundError(f"Profile field not found: {profile_field_id}")
        self.session.delete(field)
        self.session.commit()
        return MemoryDeleteResponse(
            deleted_id=profile_field_id,
            detail="Deleted profile field.",
            deleted_counts={"profile_fields": 1},
        )

    def list_profile_keys(self, user_id: str | None = None) -> list[str]:
        """Return approved profile keys, including user-approved custom keys."""

        return self._allowed_keys("ProfileField", user_id)

    def update_recent_status(
        self,
        user_id: str,
        statuses: dict[str, Any],
        observed_at: datetime | None = None,
        source: str = "manual",
        extension_mode: bool = False,
    ) -> RecentStatusPatchResponse:
        """Apply manual recent-status edits with editable default expirations."""

        self._ensure_user(user_id)
        observed = _ensure_aware(observed_at or utc_now())
        stored = []
        proposals = []
        for raw_key, value in statuses.items():
            status_key = normalize_key(raw_key)
            if not self._is_key_allowed(user_id, "RecentStatus", status_key):
                if extension_mode:
                    proposals.append(
                        self._create_key_proposal(
                            user_id=user_id,
                            target_type="RecentStatus",
                            proposed_key=status_key,
                            example_value=value,
                            reason="Manual recent-status edit used a key outside the approved v1 catalog.",
                        )
                    )
                    continue
                raise ValueError(f"Unknown recent status key: {status_key}")
            stored.append(
                self._store_recent_status(
                    user_id=user_id,
                    status_key=status_key,
                    value=value,
                    observed_at=observed,
                    valid_until=default_valid_until(status_key, observed),
                    source=source,
                )
            )
        self.session.commit()
        return RecentStatusPatchResponse(
            user_id=user_id,
            statuses=[_recent_status_out(item) for item in stored],
            key_proposals=[_key_proposal_out(item) for item in proposals],
        )

    def list_memory_items(
        self,
        user_id: str | None = None,
        status: str | None = None,
        category: str | None = None,
    ) -> list[MemoryItemOut]:
        """List memory items for review or debugging."""

        statement = select(MemoryItem)
        if user_id:
            statement = statement.where(MemoryItem.user_id == user_id)
        if status:
            statement = statement.where(MemoryItem.status == status)
        if category:
            statement = statement.where(MemoryItem.category == category)
        rows = self.session.exec(statement).all()
        rows = sorted(rows, key=lambda item: item.created_at, reverse=True)
        return [_memory_item_out(item) for item in rows]

    def approve_memory_item(
        self,
        memory_id: str,
        request: MemoryItemApproveRequest,
    ) -> MemoryItemReviewResponse:
        """Approve a candidate and write it to profile or recent-status storage."""

        item = self._get_memory_item(memory_id)
        target_type = _normalize_target_type(request.target_type or item.target_type)
        memory_key = normalize_key(request.memory_key or item.memory_key or "")
        value = request.value if request.value is not None else _json_loads(item.value_json)
        if value in ({}, None):
            value = item.text

        profile_field = None
        recent_status = None
        if target_type == "ProfileField":
            if not self._is_key_allowed(item.user_id, "ProfileField", memory_key):
                raise ValueError(f"Unknown profile field key: {memory_key}")
            profile_field = self._upsert_profile_field(
                item.user_id,
                memory_key,
                value,
                source="memory_review",
                source_memory_id=item.memory_id,
            )
        elif target_type == "RecentStatus":
            if not self._is_key_allowed(item.user_id, "RecentStatus", memory_key):
                raise ValueError(f"Unknown recent status key: {memory_key}")
            observed_at = item.created_at
            valid_until = request.valid_until or item.expires_at or default_valid_until(memory_key, observed_at)
            recent_status = self._store_recent_status(
                user_id=item.user_id,
                status_key=memory_key,
                value=value,
                observed_at=observed_at,
                valid_until=valid_until,
                source="memory_review",
                source_memory_id=item.memory_id,
                source_conversation_id=item.source_conversation_id,
            )
        else:
            raise ValueError("Approved memory item needs target_type ProfileField or RecentStatus.")

        now = utc_now()
        item.status = "approved"
        item.target_type = target_type
        item.memory_key = memory_key
        item.value_json = _json_dumps(value)
        item.review_note = request.review_note
        item.updated_at = now
        item.reviewed_at = now
        self.session.add(item)
        self.session.commit()
        return MemoryItemReviewResponse(
            memory_item=_memory_item_out(item),
            profile_field=_profile_field_out(profile_field) if profile_field else None,
            recent_status=_recent_status_out(recent_status) if recent_status else None,
        )

    def reject_memory_item(self, memory_id: str, review_note: str | None = None) -> MemoryItemReviewResponse:
        """Reject a candidate memory item and keep it for audit."""

        item = self._get_memory_item(memory_id)
        now = utc_now()
        item.status = "rejected"
        item.review_note = review_note
        item.updated_at = now
        item.reviewed_at = now
        self.session.add(item)
        self.session.commit()
        return MemoryItemReviewResponse(memory_item=_memory_item_out(item))

    def expire_memory_item(self, memory_id: str, reason: str | None = None) -> MemoryItemReviewResponse:
        """Expire a memory item and any linked recent status."""

        item = self._get_memory_item(memory_id)
        now = utc_now()
        item.status = "expired"
        item.review_note = reason
        item.expires_at = now
        item.updated_at = now
        self.session.add(item)
        linked_status = self.session.exec(
            select(RecentStatus).where(RecentStatus.source_memory_id == memory_id)
        ).first()
        if linked_status:
            linked_status.review_status = "expired"
            linked_status.expired_at = now
            linked_status.updated_at = now
            self.session.add(linked_status)
        self.session.commit()
        return MemoryItemReviewResponse(
            memory_item=_memory_item_out(item),
            recent_status=_recent_status_out(linked_status) if linked_status else None,
        )

    def list_key_proposals(
        self,
        user_id: str | None = None,
        review_status: str | None = None,
    ) -> list[KeyProposalOut]:
        """List proposed memory keys."""

        statement = select(MemoryKeyProposal)
        if user_id:
            statement = statement.where(MemoryKeyProposal.user_id == user_id)
        if review_status:
            statement = statement.where(MemoryKeyProposal.review_status == review_status)
        rows = self.session.exec(statement).all()
        rows = sorted(rows, key=lambda item: item.created_at, reverse=True)
        return [_key_proposal_out(item) for item in rows]

    def approve_key_proposal(
        self,
        proposal_id: str,
        proposed_key: str | None = None,
        target_type: str | None = None,
    ) -> KeyProposalOut:
        """Approve a proposed key so future writes can use it."""

        proposal = self._get_key_proposal(proposal_id)
        if proposed_key:
            proposal.proposed_key = normalize_key(proposed_key)
        if target_type:
            proposal.target_type = _normalize_target_type(target_type)
        proposal.review_status = "approved"
        proposal.reviewed_at = utc_now()
        self.session.add(proposal)
        self.session.commit()
        return _key_proposal_out(proposal)

    def reject_key_proposal(self, proposal_id: str) -> KeyProposalOut:
        """Reject a proposed key and keep it in audit history."""

        proposal = self._get_key_proposal(proposal_id)
        proposal.review_status = "rejected"
        proposal.reviewed_at = utc_now()
        self.session.add(proposal)
        self.session.commit()
        return _key_proposal_out(proposal)

    def list_expiring_recent_status(
        self,
        user_id: str,
        within_hours: int = 2,
        now: datetime | None = None,
    ) -> ExpiringRecentStatusResponse:
        """Return approved statuses whose valid_until is close."""

        self._ensure_user(user_id)
        active_now = _ensure_aware(now or utc_now())
        due_before = active_now + timedelta(hours=within_hours)
        statuses = self._recent_status_rows(user_id, include_expired=False)
        expiring = [
            item
            for item in statuses
            if active_now <= _ensure_aware(item.valid_until) <= due_before
        ]
        for item in expiring:
            item.expiration_reminder_status = "due"
            self.session.add(item)
        self.session.commit()
        reminders = [_recent_status_expiry_reminder(item, active_now) for item in expiring]
        return ExpiringRecentStatusResponse(
            statuses=[reminder.status for reminder in reminders],
            reminders=reminders,
        )

    def list_recent_status(self, user_id: str, include_expired: bool = False) -> list[RecentStatusOut]:
        """Return recent status rows for API, MCP, or debug views."""

        self._ensure_user(user_id)
        rows = (
            self._recent_status_rows(user_id, include_expired=True)
            if include_expired
            else self._latest_recent_status_rows(user_id)
        )
        rows = sorted(rows, key=lambda item: item.observed_at, reverse=True)
        return [_recent_status_out(item) for item in rows]

    def delete_recent_status(self, status_id: str) -> MemoryDeleteResponse:
        """Delete one recent-status entry by its row ID."""

        status = self.session.get(RecentStatus, status_id)
        if status is None:
            raise NotFoundError(f"Recent status not found: {status_id}")
        self.session.delete(status)
        self.session.commit()
        return MemoryDeleteResponse(
            deleted_id=status_id,
            detail="Deleted recent status.",
            deleted_counts={"recent_status": 1},
        )

    def list_recent_status_keys(self, user_id: str | None = None) -> list[str]:
        """Return approved recent-status keys, including user-approved custom keys."""

        return self._allowed_keys("RecentStatus", user_id)

    def update_recent_status_valid_until(self, status_id: str, valid_until: datetime) -> RecentStatusOut:
        """Update the editable expiration timestamp for one recent status."""

        status = self.session.get(RecentStatus, status_id)
        if status is None:
            raise NotFoundError(f"Recent status not found: {status_id}")
        status.valid_until = _ensure_aware(valid_until)
        status.review_status = "approved"
        status.expiration_reminder_status = "renewed"
        status.expired_at = None
        status.updated_at = utc_now()
        self.session.add(status)
        self.session.commit()
        return _recent_status_out(status)

    def expire_recent_status(self, status_id: str, reason: str | None = None) -> RecentStatusOut:
        """Expire one recent status without deleting it from audit storage."""

        status = self.session.get(RecentStatus, status_id)
        if status is None:
            raise NotFoundError(f"Recent status not found: {status_id}")
        now = utc_now()
        status.review_status = "expired"
        status.expired_at = now
        status.updated_at = now
        status.expiration_reminder_status = "expired_by_user"
        self.session.add(status)
        self.session.commit()
        return _recent_status_out(status)

    def get_active_plan(self, user_id: str) -> PlanProgressOut | None:
        """Return the latest active plan progress for a user."""

        self._ensure_user(user_id)
        rows = self.session.exec(
            select(TrainingPlanProgress).where(
                TrainingPlanProgress.user_id == user_id,
                TrainingPlanProgress.status == "active",
            )
        ).all()
        if not rows:
            return None
        row = sorted(rows, key=lambda item: item.updated_at, reverse=True)[0]
        return _plan_progress_out(row)

    def patch_plan_progress(self, plan_id: str, request: PlanProgressPatchRequest) -> PlanProgressOut:
        """Create or update a plan_progress item and append a progress event."""

        self._ensure_user(request.user_id)
        plan = self.session.get(TrainingPlanProgress, plan_id)
        now = utc_now()
        if plan is None:
            plan = TrainingPlanProgress(
                plan_id=plan_id,
                user_id=request.user_id,
                version_id=request.version_id,
                version=request.version or 1,
                status=request.status or "active",
                current_week=request.current_week,
                created_at=now,
                updated_at=now,
            )
        else:
            if plan.user_id != request.user_id:
                raise ValueError("Plan ID belongs to another user.")
            if request.version is not None:
                plan.version = request.version
            if request.version_id is not None:
                plan.version_id = request.version_id
            if request.status is not None:
                plan.status = request.status
            if request.current_week is not None:
                plan.current_week = request.current_week

        if request.planned_sessions is not None:
            plan.planned_sessions_json = _json_dumps(request.planned_sessions)

        progress = _json_loads(plan.plan_progress_json, default={})
        if not isinstance(progress, dict):
            progress = {}
        if request.progress_patch:
            progress.update(request.progress_patch)
        if request.session_id:
            sessions = progress.setdefault("sessions", {})
            session_progress = dict(sessions.get(request.session_id, {}))
            if request.session_status:
                session_progress["status"] = request.session_status
            if request.completed_at:
                session_progress["completed_at"] = request.completed_at.isoformat()
            session_progress["updated_at"] = now.isoformat()
            sessions[request.session_id] = session_progress
        plan.plan_progress_json = _json_dumps(progress)

        events = _json_loads(plan.progress_events_json, default=[])
        if not isinstance(events, list):
            events = []
        event = {
            "event_id": f"pevt_{uuid4().hex[:12]}",
            "event_type": request.event_type,
            "session_id": request.session_id,
            "session_status": request.session_status,
            "created_at": now.isoformat(),
        }
        if request.completed_at:
            event["completed_at"] = request.completed_at.isoformat()
        events.append(event)
        plan.progress_events_json = _json_dumps(events)

        if request.feedback:
            feedback = _json_loads(plan.feedback_json, default=[])
            if not isinstance(feedback, list):
                feedback = []
            feedback.append(
                {
                    "feedback_id": f"pfbk_{uuid4().hex[:12]}",
                    "text": request.feedback,
                    "created_at": now.isoformat(),
                }
            )
            plan.feedback_json = _json_dumps(feedback)

        if plan.status == "completed" and plan.completed_at is None:
            plan.completed_at = now
        plan.updated_at = now
        self.session.add(plan)
        self.session.commit()
        if plan.status == "completed":
            self._write_completed_plan_episode(plan)
        return _plan_progress_out(plan)

    def activate_plan(self, request: PlanActivationRequest) -> PlanActivationResponse:
        """Atomically replace the user's active plan with an approved version."""

        self._ensure_user(request.user_id)
        now = utc_now()
        active_rows = self.session.exec(
            select(TrainingPlanProgress).where(
                TrainingPlanProgress.user_id == request.user_id,
                TrainingPlanProgress.status == "active",
            )
        ).all()
        deactivated_ids: list[str] = []
        for row in active_rows:
            if row.plan_id == request.program_id:
                continue
            row.status = "superseded"
            row.updated_at = now
            deactivated_ids.append(row.plan_id)
            self.session.add(row)

        plan = self.session.get(TrainingPlanProgress, request.program_id)
        existing_progress = _json_loads(plan.plan_progress_json, default={}) if plan else {}
        existing_sessions = existing_progress.get("sessions", {}) if isinstance(existing_progress, dict) else {}
        existing_planned = _json_loads(plan.planned_sessions_json, default=[]) if plan else []
        progress_by_slot = {
            (item.get("week_number"), item.get("day_number")): existing_sessions.get(str(item.get("session_id")), {})
            for item in existing_planned
            if isinstance(item, dict)
        }
        next_sessions: dict[str, dict[str, Any]] = {}
        for session in request.planned_sessions:
            session_id = str(session.get("session_id", ""))
            if not session_id:
                raise ValueError("Every activated session must have a session_id.")
            previous = existing_sessions.get(session_id, {}) if isinstance(existing_sessions, dict) else {}
            if not previous:
                previous = progress_by_slot.get((session.get("week_number"), session.get("day_number")), {})
            next_sessions[session_id] = {
                "status": previous.get("status", "pending"),
                **({"completed_at": previous["completed_at"]} if previous.get("completed_at") else {}),
                "updated_at": now.isoformat(),
            }
        if plan is None:
            plan = TrainingPlanProgress(plan_id=request.program_id, user_id=request.user_id, created_at=now)
        elif plan.user_id != request.user_id:
            raise ValueError("Plan ID belongs to another user.")
        plan.version_id = request.version_id
        plan.version = request.version_number
        plan.status = "active"
        plan.current_week = 1
        plan.planned_sessions_json = _json_dumps(request.planned_sessions)
        plan.plan_snapshot_json = _json_dumps(request.plan_snapshot)
        plan.plan_progress_json = _json_dumps({"sessions": next_sessions})
        events = _json_loads(plan.progress_events_json, default=[])
        events.append(
            {
                "event_id": f"pevt_{uuid4().hex[:12]}",
                "event_type": "plan_activated",
                "version_id": request.version_id,
                "created_at": now.isoformat(),
            }
        )
        plan.progress_events_json = _json_dumps(events)
        plan.updated_at = now
        self.session.add(plan)
        self.session.commit()
        self.session.refresh(plan)
        return PlanActivationResponse(
            activated_plan=_plan_progress_out(plan),
            deactivated_plan_ids=deactivated_ids,
        )

    def import_xunji(self, request: TrainingLogImportRequest) -> TrainingLogImportResponse:
        """Import a raw Xunji payload idempotently by external ID or checksum."""

        self._ensure_user(request.user_id)
        payload_json = _json_dumps(request.payload)
        checksum = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        duplicate = self._find_duplicate_import(
            request.user_id,
            request.provider,
            request.external_id,
            checksum,
        )
        if duplicate is not None:
            return TrainingLogImportResponse(
                import_id=duplicate.import_id,
                imported=False,
                skipped_duplicate=True,
                message="Import skipped because this external ID or payload checksum already exists.",
            )
        import_row = TrainingLogImport(
            import_id=f"imp_{uuid4().hex[:12]}",
            user_id=request.user_id,
            provider=request.provider,
            external_id=request.external_id,
            checksum=checksum,
            raw_payload_json=payload_json,
        )
        self.session.add(import_row)
        self.session.commit()
        sessions, dropped_sessions = self._normalize_training_sessions(import_row, request.payload)
        self.session.commit()
        return TrainingLogImportResponse(
            import_id=import_row.import_id,
            imported=True,
            skipped_duplicate=False,
            sessions_imported=len(sessions),
            sessions=[_training_session_out(item) for item in sessions],
            dropped_sessions=dropped_sessions,
        )

    def sync_xunji_training_logs(self, request: TrainingLogXunjiSyncRequest) -> TrainingLogXunjiSyncResponse:
        """Read Xunji logs for a date range and import each day idempotently."""

        self._ensure_user(request.user_id)
        if request.end_date < request.start_date:
            raise ValueError("end_date must be on or after start_date.")
        client = XunjiClient(self.settings)
        imports: list[TrainingLogImportResponse] = []
        errors: list[dict[str, str]] = []
        current = request.start_date
        while current <= request.end_date:
            datestr = current.isoformat()
            try:
                payload = client.read_day(
                    datestr=datestr,
                    include_full_data=request.include_full_data,
                    force_refresh=request.force_refresh,
                )
                imports.append(
                    self.import_xunji(
                        TrainingLogImportRequest(
                            user_id=request.user_id,
                            external_id=f"xunji:{datestr}:{'full' if request.include_full_data else 'light'}",
                            payload=payload,
                            provider="xunji",
                        )
                    )
                )
            except Exception as exc:
                errors.append({"datestr": datestr, "error": str(exc)})
            current += timedelta(days=1)
        return TrainingLogXunjiSyncResponse(
            user_id=request.user_id,
            start_date=request.start_date,
            end_date=request.end_date,
            days_requested=(request.end_date - request.start_date).days + 1,
            imports=imports,
            errors=errors,
        )

    def list_training_sessions(
        self,
        user_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
        sport_type: str | None = None,
        limit: int = 200,
    ) -> TrainingSessionListResponse:
        """Return normalized training sessions filtered by date range and sport type."""

        self._ensure_user(user_id)
        rows = self.session.exec(
            select(TrainingSession).where(TrainingSession.user_id == user_id)
        ).all()
        filtered = []
        for row in rows:
            session_day = _ensure_aware(row.session_date).date()
            if start_date and session_day < start_date:
                continue
            if end_date and session_day > end_date:
                continue
            if sport_type and sport_type != "any" and row.sport_type != sport_type:
                continue
            filtered.append(row)
        filtered = sorted(filtered, key=lambda item: item.session_date, reverse=True)
        safe_limit = max(1, min(1000, limit))
        return TrainingSessionListResponse(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            sport_type=sport_type,
            session_count=len(filtered),
            sessions=[_training_session_out(item) for item in filtered[:safe_limit]],
        )

    def get_memory_context(
        self,
        user_id: str,
        query: str = "",
        conversation_id: str | None = None,
        task_type: str = "qa",
        include_debug: bool = False,
    ) -> MemoryContextOut:
        """Retrieve memory context for Q&A or future blocks."""

        self._ensure_user(user_id)
        started = time.perf_counter()
        errors: list[str] = []
        active_plan = self.get_active_plan(user_id)
        conversation_summary = ""
        if conversation_id:
            try:
                conversation_summary = self._get_conversation(conversation_id).rolling_summary
            except Exception as exc:
                errors.append(f"Conversation summary retrieval failed: {exc}")
        router_state = run_langgraph_or_nodes(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "task_type": task_type,
                "latest_user_query": query,
                "rolling_summary": conversation_summary,
                "active_plan_progress": active_plan,
                "errors": errors,
                "timings": {},
            },
            [PlanContextRouter(self.llm_provider)],
        )
        plan_context_level = router_state.get("plan_context_level", "none")
        errors = router_state.get("errors", errors)
        sections, memory_ids = self._build_context_sections(
            user_id=user_id,
            query=query,
            conversation_id=conversation_id,
            plan_context_level=plan_context_level,
            active_plan=active_plan,
            errors=errors,
        )
        section_caps = dict(MEMORY_SECTION_TOKEN_BUDGETS)
        if plan_context_level == "none":
            section_caps["active_plan_progress"] = 0
            context_max_tokens = self.settings.memory_non_plan_max_tokens
        elif plan_context_level == "summary":
            section_caps["active_plan_progress"] = self.settings.memory_plan_summary_max_tokens
            context_max_tokens = (
                self.settings.memory_non_plan_max_tokens + self.settings.memory_plan_summary_max_tokens
            )
        else:
            section_caps["active_plan_progress"] = self.settings.memory_plan_full_max_tokens
            context_max_tokens = self.settings.memory_context_max_tokens
        budget_warnings = build_memory_budget_warnings(
            sections=sections,
            section_caps=section_caps,
            max_tokens=context_max_tokens,
        )
        context = MemoryContextOut(
            user_id=user_id,
            conversation_id=conversation_id,
            task_type=task_type,
            plan_context_level=plan_context_level,
            active_plan_exists=active_plan is not None,
            rendered_context=render_sections(sections),
            sections=sections,
            memory_ids=memory_ids,
            token_estimate=sum(estimate_tokens(text) for text in sections.values()),
            budget_warnings=budget_warnings,
        )
        trace = MemoryTrace(
            trace_id=f"mtr_{uuid4().hex[:12]}",
            user_id=user_id,
            conversation_id=conversation_id,
            task_type=task_type,
            query=query,
            plan_context_level=plan_context_level,
            retrieved_memory_ids_json=_json_dumps(memory_ids),
            budget_warnings_json=_json_dumps(budget_warnings),
            errors_json=_json_dumps(errors),
            total_latency_ms=int((time.perf_counter() - started) * 1000),
        )
        self.session.add(trace)
        self.session.commit()
        context.trace_id = trace.trace_id
        if include_debug:
            context.debug = {
                "memory_trace_id": trace.trace_id,
                "memory_ids": memory_ids,
                "errors": errors,
                "plan_context_router_used": router_state.get("plan_context_router_used"),
                "plan_context_rationale": router_state.get("plan_context_rationale"),
                "plan_context_confidence": router_state.get("plan_context_confidence"),
                "memory_context_token_budget": context_max_tokens,
                "section_token_budgets": section_caps,
                "budget_warnings": budget_warnings,
                "section_token_estimates": {
                    key: estimate_tokens(value) for key, value in context.sections.items()
                },
            }
        return context

    def _build_context_sections(
        self,
        user_id: str,
        query: str,
        conversation_id: str | None,
        plan_context_level: str,
        active_plan: PlanProgressOut | None,
        errors: list[str],
    ) -> tuple[dict[str, str], list[str]]:
        """Assemble memory context sections."""

        sections = {
            "rolling_summary": "",
            "working_memory": "",
            "episodic_memories": "",
            "profile": "",
            "recent_status": "",
            "active_plan_progress": "",
        }
        memory_ids: list[str] = []
        if conversation_id:
            try:
                conversation = self._get_conversation(conversation_id)
                sections["rolling_summary"] = conversation.rolling_summary
                messages = self._live_working_memory_messages(
                    conversation,
                    self._conversation_messages(conversation_id),
                )
                sections["working_memory"] = _render_working_memory_messages(messages)
                memory_ids.extend(item.message_id for item in messages)
            except Exception as exc:
                errors.append(f"Working memory retrieval failed: {exc}")

        episodes = self.search_episodes(user_id, query=query, top_k=5).episodes
        sections["episodic_memories"] = "\n".join(
            f"[{item.episode_id} | {item.created_at.isoformat()} | importance={item.importance}] "
            f"{item.summary}"
            for item in episodes
        )
        memory_ids.extend(item.episode_id for item in episodes)

        profile = self.list_profile(user_id).fields
        sections["profile"] = "\n".join(
            f"[{item.profile_field_id} | updated={item.updated_at.isoformat()} | source={item.source}] "
            f"{item.field_key}: {item.value_text}"
            for item in profile
        )
        memory_ids.extend(item.profile_field_id for item in profile)

        recent_status = self._latest_recent_status_rows(user_id)
        recent_status = sorted(recent_status, key=lambda item: item.valid_until)
        sections["recent_status"] = "\n".join(
            f"[{item.status_id} | observed={item.observed_at.isoformat()} | "
            f"valid_until={item.valid_until.isoformat()}] {item.status_key}: {item.value_text}"
            for item in recent_status
        )
        memory_ids.extend(item.status_id for item in recent_status)

        if active_plan and plan_context_level != "none":
            memory_ids.append(active_plan.plan_id)
            if plan_context_level == "summary":
                sections["active_plan_progress"] = _render_plan_summary(active_plan)
            else:
                sections["active_plan_progress"] = _render_plan_full(active_plan)

        return sections, list(dict.fromkeys(memory_ids))

    def _delete_query_rows(self, label: str, statement: Any, counts: dict[str, int]) -> None:
        rows = self.session.exec(statement).all()
        self._delete_rows(label, rows, counts)

    def _delete_rows(self, label: str, rows: list[Any], counts: dict[str, int]) -> None:
        for row in rows:
            self.session.delete(row)
        counts[label] = len(rows)

    def _ensure_user(self, user_id: str) -> UserAccount:
        user = self.session.get(UserAccount, user_id)
        if user is None:
            user = UserAccount(user_id=user_id, timezone="UTC")
            self.session.add(user)
            self.session.commit()
        return user

    def _get_conversation(self, conversation_id: str) -> Conversation:
        conversation = self.session.get(Conversation, conversation_id)
        if conversation is None:
            raise NotFoundError(f"Conversation not found: {conversation_id}")
        return conversation

    def _conversation_messages(self, conversation_id: str) -> list[ConversationMessage]:
        rows = self.session.exec(
            select(ConversationMessage).where(ConversationMessage.conversation_id == conversation_id)
        ).all()
        return sorted(rows, key=lambda item: item.created_at)

    def _update_rolling_summary_if_needed(self, conversation_id: str) -> None:
        conversation = self._get_conversation(conversation_id)
        messages = self._conversation_messages(conversation_id)
        live_messages = self._live_working_memory_messages(conversation, messages)
        live_tokens = estimate_tokens(_render_working_memory_messages(live_messages))
        if live_tokens <= WORKING_MEMORY_TOKEN_BUDGET:
            return
        target_tokens = max(1, int(WORKING_MEMORY_TOKEN_BUDGET * WORKING_MEMORY_TARGET_RATIO))
        drop_count = len(live_messages)
        for index in range(len(live_messages)):
            remaining = live_messages[index + 1 :]
            if estimate_tokens(_render_working_memory_messages(remaining)) <= target_tokens:
                drop_count = index + 1
                break
        older = live_messages[:drop_count]
        if not older:
            return
        try:
            self._extract_candidates_for_messages(conversation, older, episode=None)
        except Exception:
            return
        try:
            conversation.rolling_summary = self.llm_provider.generate_memory_rolling_summary(
                MemoryRollingSummaryRequest(
                    existing_summary=conversation.rolling_summary,
                    messages=_message_payloads(older),
                )
            )
        except Exception:
            summary_lines = [
                conversation.rolling_summary.strip(),
                "Earlier conversation summary:",
                *_summarize_messages_deterministically(older),
            ]
            conversation.rolling_summary = "\n".join(line for line in summary_lines if line).strip()
        self._compress_rolling_summary_if_needed(conversation)
        conversation.rolling_summary_until_message_id = older[-1].message_id
        conversation.updated_at = utc_now()
        self.session.add(conversation)
        self.session.commit()

    def _compress_rolling_summary_if_needed(self, conversation: Conversation) -> None:
        """Keep the persisted rolling summary under its own section budget."""

        if estimate_tokens(conversation.rolling_summary) <= ROLLING_SUMMARY_TOKEN_BUDGET:
            return
        target_tokens = max(1, int(ROLLING_SUMMARY_TOKEN_BUDGET * ROLLING_SUMMARY_TARGET_RATIO))
        compressed_summary = ""
        if self.llm_provider and hasattr(self.llm_provider, "generate_memory_rolling_summary_compression"):
            try:
                compressed_summary = self.llm_provider.generate_memory_rolling_summary_compression(
                    MemoryRollingSummaryCompressionRequest(
                        rolling_summary=conversation.rolling_summary,
                        target_tokens=target_tokens,
                    )
                )
            except Exception:
                compressed_summary = ""
        if not compressed_summary:
            compressed_summary = conversation.rolling_summary
        if estimate_tokens(compressed_summary) > target_tokens:
            compressed_summary = _keep_newest_summary_lines(compressed_summary, target_tokens)
        conversation.rolling_summary = compressed_summary

    def _live_working_memory_messages(
        self,
        conversation: Conversation,
        messages: list[ConversationMessage],
    ) -> list[ConversationMessage]:
        """Return raw messages not yet covered by the rolling summary."""

        cursor = conversation.rolling_summary_until_message_id
        if cursor:
            for index, message in enumerate(messages):
                if message.message_id == cursor:
                    return messages[index + 1 :]
        if conversation.rolling_summary and len(messages) > 12:
            return messages[-12:]
        return messages

    def _create_episode_from_conversation(
        self,
        conversation: Conversation,
        messages: list[ConversationMessage],
    ) -> Episode:
        return self._upsert_episode_from_conversation(conversation, messages, existing_episode=None)

    def _upsert_episode_from_conversation(
        self,
        conversation: Conversation,
        messages: list[ConversationMessage],
        existing_episode: Episode | None,
    ) -> Episode:
        source_message_ids = [item.message_id for item in messages]
        if existing_episode is not None:
            existing_message_ids = _json_loads(existing_episode.source_message_ids_json, default=[])
            if existing_message_ids == source_message_ids:
                return existing_episode
        try:
            summary_result = self.llm_provider.generate_memory_episode_summary(
                MemoryEpisodeSummaryRequest(
                    title=conversation.title,
                    topic=conversation.topic,
                    rolling_summary=conversation.rolling_summary,
                    messages=_message_payloads(messages),
                )
            )
            summary = summary_result.summary
            topics = summary_result.topics
            importance = summary_result.importance
        except Exception:
            summary = _summarize_episode(conversation, messages)
            topics = _infer_topics(" ".join(item.content for item in messages))
            importance = _episode_importance(summary)
        if existing_episode is None:
            episode = Episode(
                episode_id=f"eps_{uuid4().hex[:12]}",
                user_id=conversation.user_id,
                conversation_id=conversation.conversation_id,
                source_type="conversation",
                title=conversation.title or conversation.topic,
                summary=summary,
                topics_json=_json_dumps(topics),
                importance=importance,
                source_message_ids_json=_json_dumps(source_message_ids),
                started_at=messages[0].created_at if messages else conversation.created_at,
                ended_at=messages[-1].created_at if messages else conversation.updated_at,
            )
        else:
            episode = existing_episode
            episode.title = conversation.title or conversation.topic
            episode.summary = summary
            episode.topics_json = _json_dumps(topics)
            episode.importance = importance
            episode.source_message_ids_json = _json_dumps(source_message_ids)
            episode.started_at = messages[0].created_at if messages else conversation.created_at
            episode.ended_at = messages[-1].created_at if messages else conversation.updated_at
        self.session.add(episode)
        self.session.commit()
        self._try_index_episode(episode)
        return episode

    def _extract_candidates_for_episode(
        self,
        conversation: Conversation,
        episode: Episode,
    ) -> list[MemoryItem]:
        return self._extract_candidates_for_messages(
            conversation,
            self._conversation_messages(conversation.conversation_id),
            episode=episode,
        )

    def _extract_candidates_for_messages(
        self,
        conversation: Conversation,
        messages: list[ConversationMessage],
        episode: Episode | None,
    ) -> list[MemoryItem]:
        """Extract reviewable candidates for messages that have not been processed yet."""

        user_messages = [message for message in messages if message.role in {"user", "human"}]
        message_ids = {message.message_id for message in user_messages}
        if not message_ids:
            return []
        existing_items = self.session.exec(
            select(MemoryItem).where(MemoryItem.source_conversation_id == conversation.conversation_id)
        ).all()
        existing_for_messages = []
        processed_message_ids: set[str] = set()
        changed = False
        for item in existing_items:
            item_message_ids = set(_json_loads(item.source_message_ids_json, default=[]))
            if not item_message_ids.intersection(message_ids):
                continue
            existing_for_messages.append(item)
            processed_message_ids.update(item_message_ids.intersection(message_ids))
            if episode is not None and item.source_episode_id is None:
                item.source_episode_id = episode.episode_id
                item.updated_at = utc_now()
                self.session.add(item)
                changed = True

        candidates = []
        for message in user_messages:
            if message.message_id in processed_message_ids:
                continue
            candidates.extend(self._extract_candidates_from_message(conversation, episode, message))
        for item in candidates:
            self.session.add(item)
        if candidates or changed:
            self.session.commit()
        for item in candidates:
            if item.target_type and item.memory_key and not self._is_key_allowed(
                item.user_id,
                item.target_type,
                item.memory_key,
            ):
                self._create_key_proposal(
                    user_id=item.user_id,
                    target_type=item.target_type,
                    proposed_key=item.memory_key,
                    example_value=_json_loads(item.value_json),
                    reason="LLM memory extraction proposed a useful key outside the approved catalog.",
                    source_memory_id=item.memory_id,
                )
        return existing_for_messages + candidates

    def _extract_candidates_with_llm(
        self,
        conversation: Conversation,
        episode: Episode | None,
        message: ConversationMessage,
    ) -> list[MemoryItem] | None:
        if not hasattr(self.llm_provider, "generate_memory_candidates"):
            return None
        result = self.llm_provider.generate_memory_candidates(
            MemoryExtractionRequest(
                message=_message_payload(message),
                profile_keys=self._allowed_keys("ProfileField", conversation.user_id),
                recent_status_keys=self._allowed_keys("RecentStatus", conversation.user_id),
            )
        )
        candidates = []
        for raw_item in result.items:
            target_type = _normalize_target_type(str(raw_item.get("target_type", "")))
            if target_type not in {"ProfileField", "RecentStatus"}:
                continue
            memory_key = normalize_key(str(raw_item.get("memory_key", "")))
            if not memory_key:
                continue
            value = raw_item.get("value")
            if value in ({}, None, ""):
                value = raw_item.get("text") or message.content
            category = str(raw_item.get("category") or "").strip().lower()
            if not category:
                category = "profile" if target_type == "ProfileField" else "recent_status"
            text = str(raw_item.get("text") or f"{memory_key}: {_stringify_value(value)}").strip()
            expires_at = (
                default_valid_until(memory_key, message.created_at)
                if target_type == "RecentStatus"
                else None
            )
            candidates.append(
                self._candidate_item(
                    conversation,
                    episode,
                    message,
                    category=category,
                    target_type=target_type,
                    key=memory_key,
                    value=value,
                    text=text,
                    confidence=_safe_confidence(raw_item.get("confidence"), default=0.65),
                    safety_relevant=bool(raw_item.get("safety_relevant", False)),
                    expires_at=expires_at,
                )
            )
        return candidates

    def _extract_candidates_from_message(
        self,
        conversation: Conversation,
        episode: Episode | None,
        message: ConversationMessage,
    ) -> list[MemoryItem]:
        try:
            llm_candidates = self._extract_candidates_with_llm(conversation, episode, message)
            if llm_candidates is not None:
                return _dedupe_candidates(llm_candidates)
        except Exception:
            pass

        text = message.content.strip()
        candidates: list[MemoryItem] = []

        def add_profile(key: str, value: Any, label: str, confidence: float = 0.65):
            candidates.append(
                self._candidate_item(
                    conversation,
                    episode,
                    message,
                    category="profile",
                    target_type="ProfileField",
                    key=key,
                    value=value,
                    text=label,
                    confidence=confidence,
                )
            )

        def add_status(key: str, value: Any, label: str, safety: bool = False, confidence: float = 0.65):
            candidates.append(
                self._candidate_item(
                    conversation,
                    episode,
                    message,
                    category="recent_status",
                    target_type="RecentStatus",
                    key=key,
                    value=value,
                    text=label,
                    confidence=confidence,
                    safety_relevant=safety,
                    expires_at=default_valid_until(key, message.created_at),
                )
            )

        goal = _first_match(text, [r"\bmy (?:main |primary )?goal is ([^.!\n]+)", r"\bi want to ([^.!\n]+)"])
        if goal:
            add_profile("primary_training_goal", goal, f"User training goal: {goal}")
        prefer = _first_match(text, [r"\bi prefer ([^.!\n]+)", r"\bi like ([^.!\n]+)"])
        if prefer:
            key = "food_preferences" if _looks_food_related(prefer) else "exercise_preferences"
            add_profile(key, prefer, f"User preference: {prefer}", confidence=0.55)
        dislike = _first_match(text, [r"\bi dislike ([^.!\n]+)", r"\bi hate ([^.!\n]+)", r"\bi do not like ([^.!\n]+)"])
        if dislike:
            key = "food_dislikes" if _looks_food_related(dislike) else "exercise_dislikes"
            add_profile(key, dislike, f"User dislike: {dislike}", confidence=0.55)
        allergy = _first_match(text, [r"\ballergic to ([^.!\n]+)", r"\ballergy to ([^.!\n]+)"])
        if allergy:
            add_profile("allergies_user_reported", allergy, f"User-reported allergy: {allergy}", confidence=0.8)

        for lift_name, key in [
            ("bench press", "bench_press_1rm_kg"),
            ("bench", "bench_press_1rm_kg"),
            ("squat", "squat_1rm_kg"),
            ("deadlift", "deadlift_1rm_kg"),
            ("overhead press", "overhead_press_1rm_kg"),
        ]:
            match = re.search(
                rf"\b{re.escape(lift_name)}\b.*?(?:1rm|max|pr).*?(\d+(?:\.\d+)?)\s*kg",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                value = float(match.group(1))
                add_profile(key, value, f"User reported {lift_name} 1RM: {value:g} kg", confidence=0.75)
                break

        height = _first_match(text, [r"\bheight (?:is )?(\d+(?:\.\d+)?)\s*cm", r"\bi am (\d+(?:\.\d+)?)\s*cm"])
        if height:
            add_profile("height_cm", float(height), f"User height: {height} cm", confidence=0.75)
        bodyweight = _first_match(
            text,
            [
                r"\bcurrent bodyweight (?:is )?(\d+(?:\.\d+)?)\s*kg",
                r"\bbodyweight (?:is )?(\d+(?:\.\d+)?)\s*kg",
                r"\bweigh (?:about )?(\d+(?:\.\d+)?)\s*kg",
            ],
        )
        if bodyweight:
            add_profile(
                "current_bodyweight_kg",
                float(bodyweight),
                f"User current bodyweight: {bodyweight} kg",
                confidence=0.7,
            )
        days = _first_match(text, [r"\btrain (\d+) days per week", r"\b(\d+) training days per week"])
        if days:
            add_profile(
                "preferred_training_days_per_week",
                int(days),
                f"User can train {days} days per week",
                confidence=0.7,
            )

        sleep_hours = _first_match(text, [r"\bslept (\d+(?:\.\d+)?) hours", r"\bsleep (?:was )?(\d+(?:\.\d+)?) hours"])
        if sleep_hours:
            add_status(
                "sleep_duration_hours",
                float(sleep_hours),
                f"User reported sleep duration: {sleep_hours} hours",
            )
        if re.search(r"\b(knee|shoulder|hip|back|elbow|ankle|wrist).{0,24}\bpain\b", text, flags=re.IGNORECASE) or re.search(
            r"\bpain\b", text, flags=re.IGNORECASE
        ):
            add_status("joint_pain", text, f"User reported pain: {text}", safety=True, confidence=0.7)
        if re.search(r"\binjur|flare", text, flags=re.IGNORECASE):
            add_status("injury_flare_up", text, f"User reported injury or flare-up: {text}", safety=True, confidence=0.75)
        if re.search(r"\bsick|ill|fever|cold symptoms|flu", text, flags=re.IGNORECASE):
            add_status("illness_symptoms", text, f"User reported illness symptoms: {text}", safety=True, confidence=0.75)
        if re.search(r"\bsore|soreness|doms", text, flags=re.IGNORECASE):
            add_status("muscle_soreness_general", text, f"User reported soreness: {text}", confidence=0.65)
        if re.search(r"\bstress|stressed", text, flags=re.IGNORECASE):
            add_status("stress_level", text, f"User reported stress: {text}", confidence=0.6)
        return _dedupe_candidates(candidates)

    def _candidate_item(
        self,
        conversation: Conversation,
        episode: Episode | None,
        message: ConversationMessage,
        category: str,
        target_type: str,
        key: str,
        value: Any,
        text: str,
        confidence: float,
        safety_relevant: bool = False,
        expires_at: datetime | None = None,
    ) -> MemoryItem:
        return MemoryItem(
            memory_id=f"mem_{uuid4().hex[:12]}",
            user_id=conversation.user_id,
            category=category,
            target_type=target_type,
            memory_key=key,
            text=text,
            value_json=_json_dumps(value),
            confidence=confidence,
            status="candidate",
            safety_relevant=safety_relevant,
            source_conversation_id=conversation.conversation_id,
            source_episode_id=episode.episode_id if episode is not None else None,
            source_message_ids_json=_json_dumps([message.message_id]),
            expires_at=expires_at,
        )

    def _upsert_profile_field(
        self,
        user_id: str,
        field_key: str,
        value: Any,
        source: str,
        source_memory_id: str | None = None,
    ) -> ProfileField:
        existing_rows = self.session.exec(
            select(ProfileField).where(ProfileField.user_id == user_id, ProfileField.field_key == field_key)
        ).all()
        existing = _latest_profile_row(existing_rows)
        superseded_rows = [row for row in existing_rows if row is not existing]
        for row in superseded_rows:
            row.review_status = "replaced"
            self.session.add(row)
        now = utc_now()
        stored_value = (
            _merge_recent_best_lifts(_json_loads(existing.value_json), value)
            if field_key == "recent_best_lifts_json" and existing is not None
            else _normalize_recent_best_lifts_value(value)
            if field_key == "recent_best_lifts_json"
            else value
        )
        if existing is None:
            existing = ProfileField(
                profile_field_id=f"pf_{uuid4().hex[:12]}",
                user_id=user_id,
                field_key=field_key,
                value_text=_stringify_value(stored_value),
                value_json=_json_dumps(stored_value),
                source=source,
                source_memory_id=source_memory_id,
                created_at=now,
                updated_at=now,
            )
        else:
            existing.value_text = _stringify_value(stored_value)
            existing.value_json = _json_dumps(stored_value)
            existing.source = source
            existing.source_memory_id = source_memory_id or existing.source_memory_id
            existing.review_status = "approved"
            existing.updated_at = now
        self.session.add(existing)
        for row in superseded_rows:
            row.replaced_by = existing.profile_field_id
            self.session.add(row)
        return existing

    def _store_recent_status(
        self,
        user_id: str,
        status_key: str,
        value: Any,
        observed_at: datetime,
        valid_until: datetime,
        source: str,
        source_memory_id: str | None = None,
        source_conversation_id: str | None = None,
    ) -> RecentStatus:
        now = utc_now()
        self._supersede_active_recent_status(user_id, status_key, now)
        row = RecentStatus(
            status_id=f"rs_{uuid4().hex[:12]}",
            user_id=user_id,
            status_key=status_key,
            value_text=_stringify_value(value),
            value_json=_json_dumps(value),
            observed_at=_ensure_aware(observed_at),
            valid_until=_ensure_aware(valid_until),
            review_status="approved",
            source=source,
            source_memory_id=source_memory_id,
            source_conversation_id=source_conversation_id,
        )
        self.session.add(row)
        return row

    def _latest_profile_rows(self, user_id: str) -> list[ProfileField]:
        rows = self.session.exec(
            select(ProfileField).where(
                ProfileField.user_id == user_id,
                ProfileField.review_status == "approved",
            )
        ).all()
        latest_by_key: dict[str, ProfileField] = {}
        for row in rows:
            current = latest_by_key.get(row.field_key)
            if current is None or _profile_sort_key(row) > _profile_sort_key(current):
                latest_by_key[row.field_key] = row
        return sorted(latest_by_key.values(), key=lambda item: item.field_key)

    def _latest_recent_status_rows(self, user_id: str) -> list[RecentStatus]:
        rows = self._recent_status_rows(user_id, include_expired=False)
        latest_by_key: dict[str, RecentStatus] = {}
        for row in rows:
            current = latest_by_key.get(row.status_key)
            if current is None or _recent_status_sort_key(row) > _recent_status_sort_key(current):
                latest_by_key[row.status_key] = row
        return list(latest_by_key.values())

    def _supersede_active_recent_status(
        self,
        user_id: str,
        status_key: str,
        superseded_at: datetime,
    ) -> None:
        rows = self.session.exec(
            select(RecentStatus).where(
                RecentStatus.user_id == user_id,
                RecentStatus.status_key == status_key,
                RecentStatus.review_status == "approved",
            )
        ).all()
        for row in rows:
            if _ensure_aware(row.valid_until) <= superseded_at:
                continue
            row.review_status = "expired"
            row.expired_at = superseded_at
            row.updated_at = superseded_at
            row.expiration_reminder_status = "superseded"
            self.session.add(row)

    def _recent_status_rows(self, user_id: str, include_expired: bool) -> list[RecentStatus]:
        rows = self.session.exec(
            select(RecentStatus).where(RecentStatus.user_id == user_id)
        ).all()
        now = utc_now()
        active = []
        for row in rows:
            is_expired = row.review_status == "expired" or _ensure_aware(row.valid_until) <= now
            if is_expired:
                if row.review_status != "expired":
                    row.review_status = "expired"
                    row.expired_at = _ensure_aware(row.valid_until)
                    row.updated_at = now
                    self.session.add(row)
                if include_expired:
                    active.append(row)
            elif row.review_status == "approved":
                active.append(row)
        self.session.commit()
        return active

    def _is_key_allowed(self, user_id: str, target_type: str, key: str) -> bool:
        normalized = normalize_key(key)
        if target_type == "ProfileField" and normalized in PROFILE_FIELD_KEYS:
            return True
        if target_type == "RecentStatus" and normalized in RECENT_STATUS_KEYS:
            return True
        proposals = self.session.exec(
            select(MemoryKeyProposal).where(
                MemoryKeyProposal.user_id == user_id,
                MemoryKeyProposal.target_type == target_type,
                MemoryKeyProposal.review_status == "approved",
            )
        ).all()
        return normalized in {proposal.proposed_key for proposal in proposals}

    def _allowed_keys(self, target_type: str, user_id: str | None = None) -> list[str]:
        base_keys = PROFILE_FIELD_KEYS if target_type == "ProfileField" else RECENT_STATUS_KEYS
        keys = set(base_keys)
        if user_id:
            proposals = self.session.exec(
                select(MemoryKeyProposal).where(
                    MemoryKeyProposal.user_id == user_id,
                    MemoryKeyProposal.target_type == target_type,
                    MemoryKeyProposal.review_status == "approved",
                )
            ).all()
            keys.update(proposal.proposed_key for proposal in proposals)
        return sorted(keys)

    def _create_key_proposal(
        self,
        user_id: str,
        target_type: str,
        proposed_key: str,
        example_value: Any,
        reason: str,
        source_memory_id: str | None = None,
    ) -> MemoryKeyProposal:
        normalized = normalize_key(proposed_key)
        existing = self.session.exec(
            select(MemoryKeyProposal).where(
                MemoryKeyProposal.user_id == user_id,
                MemoryKeyProposal.target_type == target_type,
                MemoryKeyProposal.proposed_key == normalized,
                MemoryKeyProposal.review_status == "pending",
            )
        ).first()
        if existing:
            return existing
        catalog = PROFILE_FIELD_KEYS if target_type == "ProfileField" else RECENT_STATUS_KEYS
        nearest = difflib.get_close_matches(normalized, sorted(catalog), n=5, cutoff=0.55)
        duplicate_decision = "not_duplicate"
        review_status = "pending"
        try:
            review = self.llm_provider.review_memory_key_proposal(
                MemoryKeyProposalRequest(
                    target_type=target_type,
                    proposed_key=normalized,
                    example_value=example_value,
                    reason=reason,
                    existing_keys=self._allowed_keys(target_type, user_id),
                )
            )
            if review.nearest_existing_keys:
                nearest = review.nearest_existing_keys[:5]
            if review.reason:
                reason = review.reason
            duplicate_decision = review.duplicate_decision
            if duplicate_decision in {"duplicate", "equivalent"}:
                review_status = "rejected"
        except Exception:
            pass
        proposal = MemoryKeyProposal(
            proposal_id=f"mkp_{uuid4().hex[:12]}",
            user_id=user_id,
            proposed_key=normalized,
            target_type=target_type,
            example_value_json=_json_dumps(example_value),
            reason=reason,
            nearest_existing_keys_json=_json_dumps(nearest),
            duplicate_decision=duplicate_decision,
            source_memory_id=source_memory_id,
            review_status=review_status,
        )
        self.session.add(proposal)
        self.session.commit()
        return proposal

    def _get_memory_item(self, memory_id: str) -> MemoryItem:
        item = self.session.get(MemoryItem, memory_id)
        if item is None:
            raise NotFoundError(f"Memory item not found: {memory_id}")
        return item

    def _get_key_proposal(self, proposal_id: str) -> MemoryKeyProposal:
        proposal = self.session.get(MemoryKeyProposal, proposal_id)
        if proposal is None:
            raise NotFoundError(f"Memory key proposal not found: {proposal_id}")
        return proposal

    def _semantic_episode_search(self, user_id: str, query: str, top_k: int) -> list[Episode]:
        embeddings = self.llm_provider.embed_texts([query])
        import chromadb

        client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
        collection = client.get_or_create_collection(self.settings.memory_chroma_collection)
        result = collection.query(
            query_embeddings=embeddings,
            n_results=top_k,
            where={"user_id": user_id},
        )
        ids = result.get("ids", [[]])[0]
        episodes = []
        for episode_id in ids:
            episode = self.session.get(Episode, episode_id)
            if episode is not None:
                episodes.append(episode)
        return episodes

    def _keyword_episode_search(self, user_id: str, query: str, top_k: int) -> list[Episode]:
        rows = self.session.exec(select(Episode).where(Episode.user_id == user_id)).all()
        if not query.strip():
            return sorted(rows, key=lambda item: item.created_at, reverse=True)[:top_k]
        terms = [term for term in re.split(r"\W+", query.lower()) if len(term) > 2]
        scored = []
        for episode in rows:
            haystack = f"{episode.summary} {episode.topics_json}".lower()
            score = sum(haystack.count(term) for term in terms)
            if score:
                scored.append((score, episode))
        scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return [episode for _, episode in scored[:top_k]]

    def _try_index_episode(self, episode: Episode) -> None:
        try:
            embeddings = self.llm_provider.embed_texts([episode.summary])
            import chromadb

            client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
            collection = client.get_or_create_collection(self.settings.memory_chroma_collection)
            collection.upsert(
                ids=[episode.episode_id],
                documents=[episode.summary],
                embeddings=embeddings,
                metadatas=[
                    {
                        "user_id": episode.user_id,
                        "conversation_id": episode.conversation_id or "",
                        "source_type": episode.source_type,
                    }
                ],
            )
            episode.embedding_id = episode.episode_id
            self.session.add(episode)
            self.session.commit()
        except Exception:
            return

    def _write_completed_plan_episode(self, plan: TrainingPlanProgress) -> None:
        existing = self.session.exec(
            select(Episode).where(
                Episode.user_id == plan.user_id,
                Episode.source_type == "plan_completion",
                Episode.title == plan.plan_id,
            )
        ).first()
        if existing:
            return
        try:
            summary_result = self.llm_provider.generate_completed_plan_episode(
                CompletedPlanEpisodeRequest(plan_payload=_plan_progress_out(plan).model_dump(mode="json"))
            )
            summary = summary_result.summary
            topics = summary_result.topics
            importance = summary_result.importance
        except Exception:
            summary = (
                f"Completed training plan {plan.plan_id} version {plan.version}. "
                f"Progress: {plan.plan_progress_json}. Feedback: {plan.feedback_json}."
            )
            topics = ["training_plan", "plan_completion"]
            importance = 0.75
        episode = Episode(
            episode_id=f"eps_{uuid4().hex[:12]}",
            user_id=plan.user_id,
            source_type="plan_completion",
            title=plan.plan_id,
            summary=summary,
            topics_json=_json_dumps(topics),
            importance=importance,
            started_at=plan.created_at,
            ended_at=plan.completed_at or plan.updated_at,
        )
        self.session.add(episode)
        self.session.commit()
        self._try_index_episode(episode)

    def _find_duplicate_import(
        self,
        user_id: str,
        provider: str,
        external_id: str | None,
        checksum: str,
    ) -> TrainingLogImport | None:
        rows = self.session.exec(
            select(TrainingLogImport).where(
                TrainingLogImport.user_id == user_id,
                TrainingLogImport.provider == provider,
            )
        ).all()
        for row in rows:
            if external_id and row.external_id == external_id:
                return row
            if row.checksum == checksum:
                return row
        return None

    def _normalize_training_sessions(
        self,
        import_row: TrainingLogImport,
        payload: Any,
    ) -> tuple[list[TrainingSession], list[dict[str, Any]]]:
        sessions_payload = _extract_training_session_payloads(payload)
        kept_payloads, dropped_sessions = _drop_overlapping_training_payloads(sessions_payload)
        rows = []
        for item in kept_payloads:
            if not isinstance(item, dict):
                continue
            external_id = str(item.get("id") or item.get("session_id") or item.get("external_id") or "")
            start_at = _parse_xunji_datetime(
                item.get("start")
                or item.get("start_time")
                or item.get("started_at")
                or item.get("date")
                or item.get("training_date")
            )
            end_at = _parse_xunji_datetime(
                item.get("end")
                or item.get("end_time")
                or item.get("ended_at")
            )
            session_date = _parse_xunji_datetime(
                item.get("date")
                or item.get("datestr")
                or item.get("training_date")
                or item.get("start_time")
                or item.get("started_at")
                or utc_now()
            )
            if start_at is not None:
                session_date = start_at
            duration_seconds = _session_duration_seconds(start_at, end_at, item)
            row = TrainingSession(
                session_id=f"ts_{uuid4().hex[:12]}",
                user_id=import_row.user_id,
                source_import_id=import_row.import_id,
                external_id=external_id or None,
                title=str(item.get("title") or item.get("name") or "").strip() or None,
                session_date=session_date,
                start_at=start_at,
                end_at=end_at,
                duration_seconds=duration_seconds,
                sport_type=self._classify_training_sport_type(item),
                exercises_json=_json_dumps(_strip_xunji_set_time(item.get("exercises") or item.get("movements") or [])),
                sets_summary_json=_json_dumps(_strip_xunji_set_time(item.get("sets_summary") or item.get("sets") or {})),
                rpe=_safe_float(item.get("rpe")),
                notes=item.get("notes") or item.get("comment"),
                raw_payload_json=_json_dumps(item),
            )
            self.session.add(row)
            rows.append(row)
        return rows, dropped_sessions

    def _classify_training_sport_type(self, item: dict[str, Any]) -> str:
        try:
            if hasattr(self.llm_provider, "classify_training_sport_type"):
                label = self.llm_provider.classify_training_sport_type(
                    TrainingSportTypeRequest(
                        session_payload=_compact_training_payload_for_label(item),
                        allowed_labels=TRAINING_SPORT_TYPES,
                    )
                )
                if label in TRAINING_SPORT_TYPES:
                    return label
        except Exception:
            pass
        return _fallback_training_sport_type(item)


def _extract_training_session_payloads(payload: Any) -> list[dict[str, Any]]:
    """Return candidate session dictionaries from raw Xunji or legacy payload shapes."""

    sessions_payload = payload
    if isinstance(payload, dict):
        res = payload.get("res")
        if isinstance(res, dict) and isinstance(res.get("trains"), list):
            sessions_payload = res["trains"]
        elif isinstance(res, list):
            sessions_payload = res
        elif "sessions" in payload:
            sessions_payload = payload.get("sessions")
        elif "trains" in payload:
            sessions_payload = payload.get("trains")
    if isinstance(sessions_payload, dict):
        sessions_payload = [sessions_payload]
    if not isinstance(sessions_payload, list):
        return []
    return [item for item in sessions_payload if isinstance(item, dict)]


def _drop_overlapping_training_payloads(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the more detailed session when two imported sessions overlap by over 20 minutes."""

    ordered = sorted(items, key=lambda item: (_training_start_or_date(item), -_training_detail_score(item)))
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for item in ordered:
        replacement_index = None
        drop_current = False
        for index, existing in enumerate(kept):
            if _training_overlap(existing, item) <= OVERLAP_DROP_THRESHOLD:
                continue
            if _training_detail_score(item) > _training_detail_score(existing):
                replacement_index = index
            else:
                drop_current = True
            break
        if replacement_index is not None:
            existing = kept[replacement_index]
            dropped.append(_dropped_training_session_info(existing, kept_reason="overlap_less_detailed"))
            kept[replacement_index] = item
        elif drop_current:
            dropped.append(_dropped_training_session_info(item, kept_reason="overlap_less_detailed"))
        else:
            kept.append(item)
    kept = sorted(kept, key=_training_start_or_date)
    return kept, dropped


def _training_overlap(left: dict[str, Any], right: dict[str, Any]) -> timedelta:
    left_start = _training_start(left)
    left_end = _training_end(left)
    right_start = _training_start(right)
    right_end = _training_end(right)
    if None in {left_start, left_end, right_start, right_end}:
        return timedelta(0)
    latest_start = max(left_start, right_start)
    earliest_end = min(left_end, right_end)
    return max(timedelta(0), earliest_end - latest_start)


def _training_start(item: dict[str, Any]) -> datetime | None:
    return _parse_xunji_datetime(item.get("start") or item.get("start_time") or item.get("started_at"))


def _training_end(item: dict[str, Any]) -> datetime | None:
    return _parse_xunji_datetime(item.get("end") or item.get("end_time") or item.get("ended_at"))


def _training_start_or_date(item: dict[str, Any]) -> datetime:
    return (
        _training_start(item)
        or _parse_xunji_datetime(item.get("date") or item.get("datestr") or item.get("training_date"))
        or utc_now()
    )


def _training_detail_score(item: dict[str, Any]) -> int:
    score = len(_json_dumps(item))
    movements = item.get("movements") or item.get("exercises") or []
    if isinstance(movements, list):
        score += len(movements) * 20
        for movement in movements:
            if isinstance(movement, dict):
                sets = movement.get("sets") or []
                if isinstance(sets, list):
                    score += len(sets) * 10
    for key in ("notes", "comment", "rpe", "title", "name"):
        if item.get(key):
            score += 25
    return score


def _dropped_training_session_info(item: dict[str, Any], kept_reason: str) -> dict[str, Any]:
    return {
        "external_id": str(item.get("id") or item.get("session_id") or item.get("external_id") or ""),
        "title": item.get("title") or item.get("name"),
        "start_at": (_training_start(item) or _training_start_or_date(item)).isoformat(),
        "end_at": _training_end(item).isoformat() if _training_end(item) else None,
        "reason": kept_reason,
    }


def _session_duration_seconds(
    start_at: datetime | None,
    end_at: datetime | None,
    item: dict[str, Any],
) -> int | None:
    if start_at and end_at and end_at >= start_at:
        return int((end_at - start_at).total_seconds())
    for key in ("duration_seconds", "duration_s", "duration", "workoutTime"):
        value = _safe_float(item.get(key))
        if value is not None:
            return int(value)
    return None


def _compact_training_payload_for_label(item: dict[str, Any]) -> dict[str, Any]:
    movements = item.get("movements") or item.get("exercises") or []
    movement_summaries = []
    if isinstance(movements, list):
        for movement in movements[:12]:
            if not isinstance(movement, dict):
                continue
            sets = movement.get("sets") if isinstance(movement.get("sets"), list) else []
            movement_summaries.append(
                {
                    "name": movement.get("name") or movement.get("title"),
                    "set_count": len(sets),
                    "metrics": movement.get("metrics"),
                }
            )
    return {
        "title": item.get("title") or item.get("name"),
        "type": item.get("type") or item.get("sport") or item.get("sport_type"),
        "start": item.get("start") or item.get("start_time") or item.get("started_at"),
        "end": item.get("end") or item.get("end_time") or item.get("ended_at"),
        "duration_seconds": _session_duration_seconds(_training_start(item), _training_end(item), item),
        "movements": movement_summaries,
        "notes": item.get("notes") or item.get("comment"),
    }


def _fallback_training_sport_type(item: dict[str, Any]) -> str:
    text = _json_dumps(_compact_training_payload_for_label(item)).lower()
    if any(term in text for term in ("running", "run", "跑步", "跑步机", "treadmill", "indoor running")):
        return "running"
    if any(term in text for term in ("hyrox", "crossfit", "conditioning", "interval", "metcon", "循环", "综合")):
        return "other Mixed Aerobic-Anaerobic Exercise"
    if any(term in text for term in ("cycling", "bike", "row", "elliptical", "swim", "骑行", "划船", "游泳")):
        return "other aerobic exercise"
    movements = item.get("movements") or item.get("exercises") or []
    if isinstance(movements, list) and movements:
        return "strength"
    if any(term in text for term in ("sprint", "jump", "throw", "冲刺", "跳", "投掷")):
        return "other anaerobic exercise"
    return "other Mixed Aerobic-Anaerobic Exercise"


def _user_out(user: UserAccount) -> UserOut:
    return UserOut(
        user_id=user.user_id,
        display_name=user.display_name,
        timezone=user.timezone,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


def _conversation_out(conversation: Conversation) -> ConversationOut:
    return ConversationOut(
        conversation_id=conversation.conversation_id,
        user_id=conversation.user_id,
        status=conversation.status,
        title=conversation.title,
        topic=conversation.topic,
        rolling_summary=conversation.rolling_summary,
        rolling_summary_until_message_id=conversation.rolling_summary_until_message_id,
        closure_reason=conversation.closure_reason,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        closed_at=conversation.closed_at,
    )


def _message_out(message: ConversationMessage) -> ConversationMessageOut:
    return ConversationMessageOut(
        message_id=message.message_id,
        conversation_id=message.conversation_id,
        user_id=message.user_id,
        role=message.role,
        content=message.content,
        source_block=message.source_block,
        qa_trace_id=message.qa_trace_id,
        created_at=message.created_at,
    )


def _episode_out(episode: Episode) -> EpisodeOut:
    return EpisodeOut(
        episode_id=episode.episode_id,
        user_id=episode.user_id,
        conversation_id=episode.conversation_id,
        program_context_id=episode.program_context_id,
        source_type=episode.source_type,
        title=episode.title,
        summary=episode.summary,
        topics=_json_loads(episode.topics_json, default=[]),
        importance=episode.importance,
        source_message_ids=_json_loads(episode.source_message_ids_json, default=[]),
        started_at=episode.started_at,
        ended_at=episode.ended_at,
        embedding_id=episode.embedding_id,
        created_at=episode.created_at,
    )


def _memory_item_out(item: MemoryItem) -> MemoryItemOut:
    return MemoryItemOut(
        memory_id=item.memory_id,
        user_id=item.user_id,
        category=item.category,
        target_type=item.target_type,
        memory_key=item.memory_key,
        text=item.text,
        value=_json_loads(item.value_json),
        confidence=item.confidence,
        status=item.status,
        safety_relevant=item.safety_relevant,
        source_conversation_id=item.source_conversation_id,
        source_episode_id=item.source_episode_id,
        source_message_ids=_json_loads(item.source_message_ids_json, default=[]),
        review_note=item.review_note,
        replacement_of=item.replacement_of,
        expires_at=item.expires_at,
        created_at=item.created_at,
        updated_at=item.updated_at,
        reviewed_at=item.reviewed_at,
    )


def _profile_field_out(item: ProfileField) -> ProfileFieldOut:
    return ProfileFieldOut(
        profile_field_id=item.profile_field_id,
        user_id=item.user_id,
        field_key=item.field_key,
        value=_json_loads(item.value_json),
        value_text=item.value_text,
        source=item.source,
        source_memory_id=item.source_memory_id,
        review_status=item.review_status,
        replaced_by=item.replaced_by,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


def _recent_status_out(item: RecentStatus) -> RecentStatusOut:
    return RecentStatusOut(
        status_id=item.status_id,
        user_id=item.user_id,
        status_key=item.status_key,
        value=_json_loads(item.value_json),
        value_text=item.value_text,
        severity=item.severity,
        impact=item.impact,
        observed_at=_ensure_aware(item.observed_at),
        valid_until=_ensure_aware(item.valid_until),
        review_status=item.review_status,
        source=item.source,
        source_memory_id=item.source_memory_id,
        source_conversation_id=item.source_conversation_id,
        expiration_reminder_status=item.expiration_reminder_status,
        created_at=_ensure_aware(item.created_at),
        updated_at=_ensure_aware(item.updated_at),
        expired_at=_ensure_aware(item.expired_at) if item.expired_at else None,
    )


def _recent_status_expiry_reminder(item: RecentStatus, now: datetime) -> RecentStatusExpiryReminder:
    status = _recent_status_out(item)
    suggested_valid_until = _suggested_recent_status_valid_until(item.status_key, now)
    if suggested_valid_until is None:
        prompt = (
            f"{item.status_key} expires at {status.valid_until.isoformat()}. "
            "Expire this status, or record a fresh recent-status value if it is still relevant."
        )
    else:
        prompt = (
            f"{item.status_key} expires at {status.valid_until.isoformat()}. "
            "Expire it if it is no longer true, or renew it with the suggested valid_until if it still applies."
        )
    return RecentStatusExpiryReminder(
        status_id=item.status_id,
        status_key=item.status_key,
        value=_json_loads(item.value_json),
        value_text=item.value_text,
        source=item.source,
        current_valid_until=status.valid_until,
        suggested_valid_until=suggested_valid_until,
        prompt=prompt,
        status=status,
    )


def _suggested_recent_status_valid_until(status_key: str, now: datetime) -> datetime | None:
    """Return a renewal suggestion when extending the same status is reasonable."""

    normalized_key = normalize_key(status_key)
    if normalized_key in DAILY_STATUS_KEYS:
        return None
    return default_valid_until(normalized_key, _ensure_aware(now))


def _key_proposal_out(item: MemoryKeyProposal) -> KeyProposalOut:
    return KeyProposalOut(
        proposal_id=item.proposal_id,
        user_id=item.user_id,
        proposed_key=item.proposed_key,
        target_type=item.target_type,
        example_value=_json_loads(item.example_value_json),
        reason=item.reason,
        nearest_existing_keys=_json_loads(item.nearest_existing_keys_json, default=[]),
        duplicate_decision=item.duplicate_decision,
        source_memory_id=item.source_memory_id,
        source_message_ids=_json_loads(item.source_message_ids_json, default=[]),
        review_status=item.review_status,
        created_at=item.created_at,
        reviewed_at=item.reviewed_at,
    )


def _plan_progress_out(item: TrainingPlanProgress) -> PlanProgressOut:
    plan_snapshot = _json_loads(item.plan_snapshot_json, default={})
    source_references = plan_snapshot.get("source_references") if isinstance(plan_snapshot, dict) else None
    if isinstance(source_references, dict) and "calculation_trace_ids" in source_references:
        plan_snapshot = dict(plan_snapshot)
        plan_snapshot["source_references"] = dict(source_references)
        plan_snapshot["source_references"].pop("calculation_trace_ids", None)
    return PlanProgressOut(
        plan_id=item.plan_id,
        user_id=item.user_id,
        version_id=item.version_id,
        version=item.version,
        status=item.status,
        current_week=item.current_week,
        planned_sessions=_json_loads(item.planned_sessions_json, default=[]),
        plan_snapshot=plan_snapshot,
        plan_progress=_json_loads(item.plan_progress_json, default={}),
        progress_events=_json_loads(item.progress_events_json, default=[]),
        feedback=_json_loads(item.feedback_json, default=[]),
        created_at=item.created_at,
        updated_at=item.updated_at,
        completed_at=item.completed_at,
    )


def _training_session_out(item: TrainingSession) -> TrainingSessionOut:
    return TrainingSessionOut(
        session_id=item.session_id,
        user_id=item.user_id,
        source_import_id=item.source_import_id,
        external_id=item.external_id,
        title=item.title,
        session_date=item.session_date,
        start_at=item.start_at,
        end_at=item.end_at,
        duration_seconds=item.duration_seconds,
        sport_type=item.sport_type,
        exercises=_strip_xunji_set_time(_json_loads(item.exercises_json, default=[])),
        sets_summary=_strip_xunji_set_time(_json_loads(item.sets_summary_json, default={})),
        rpe=item.rpe,
        notes=item.notes,
        raw_payload=_json_loads(item.raw_payload_json, default={}),
        created_at=item.created_at,
    )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _json_loads(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return {} if default is None else default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {} if default is None else default


def _normalize_recent_best_lifts_value(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            value = parsed
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, list):
        return value
    return [value]


def _merge_recent_best_lifts(existing_value: Any, new_value: Any) -> list[Any]:
    merged = list(_normalize_recent_best_lifts_value(existing_value))
    index_by_identity = {
        _recent_best_lift_identity(entry): index
        for index, entry in enumerate(merged)
    }
    for entry in _normalize_recent_best_lifts_value(new_value):
        identity = _recent_best_lift_identity(entry)
        existing_index = index_by_identity.get(identity)
        if existing_index is None:
            index_by_identity[identity] = len(merged)
            merged.append(entry)
        else:
            merged[existing_index] = entry
    return merged


def _recent_best_lift_identity(entry: Any) -> str:
    if isinstance(entry, dict):
        for key in ("exercise", "exercise_name", "lift", "movement", "name"):
            value = entry.get(key)
            if value not in (None, ""):
                return f"{key}:{normalize_key(str(value))}"
    return _json_dumps(entry)


def _strip_xunji_set_time(value: Any) -> Any:
    """Remove Xunji's meaningless set-level `time` field outside raw payloads."""

    if isinstance(value, dict):
        return {
            key: _strip_xunji_set_time(item)
            for key, item in value.items()
            if key != "time"
        }
    if isinstance(value, list):
        return [_strip_xunji_set_time(item) for item in value]
    return value


def _stringify_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return _json_dumps(value)
    return str(value)


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _normalize_target_type(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"profile", "profilefield", "profile_field"}:
        return "ProfileField"
    if normalized in {"recent_status", "recentstatus", "status"}:
        return "RecentStatus"
    if value in {"ProfileField", "RecentStatus"}:
        return value
    return value or ""


def _latest_profile_row(rows: list[ProfileField]) -> ProfileField | None:
    if not rows:
        return None
    return max(rows, key=_profile_sort_key)


def _profile_sort_key(row: ProfileField) -> tuple[datetime, datetime, str]:
    return (
        _ensure_aware(row.updated_at),
        _ensure_aware(row.created_at),
        row.profile_field_id,
    )


def _recent_status_sort_key(row: RecentStatus) -> tuple[datetime, datetime, str]:
    return (
        _ensure_aware(row.observed_at),
        _ensure_aware(row.updated_at),
        row.status_id,
    )


def _message_payload(message: ConversationMessage) -> dict[str, Any]:
    return {
        "message_id": message.message_id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at.isoformat(),
        "source_block": message.source_block,
        "qa_trace_id": message.qa_trace_id,
    }


def _message_payloads(messages: list[ConversationMessage]) -> list[dict[str, Any]]:
    return [_message_payload(message) for message in messages]


def _render_working_memory_messages(messages: list[ConversationMessage]) -> str:
    return "\n".join(
        f"[{item.message_id} | {item.created_at.isoformat()} | {item.role}] {item.content}"
        for item in messages
    )


def _safe_confidence(value: Any, default: float = 0.65) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _summarize_messages_deterministically(messages: list[ConversationMessage]) -> list[str]:
    lines = []
    for message in messages[-12:]:
        content = re.sub(r"\s+", " ", message.content).strip()
        if len(content) > 180:
            content = f"{content[:177]}..."
        lines.append(f"- {message.role}: {content}")
    return lines


def _keep_newest_summary_lines(text: str, target_tokens: int) -> str:
    """Drop oldest summary lines until the newest content fits the target."""

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return text[-target_tokens * 4 :].strip()

    selected: list[str] = []
    total = 0
    for line in reversed(lines):
        cost = estimate_tokens(line)
        if not selected and cost > target_tokens:
            return line[-target_tokens * 4 :].strip()
        if selected and total + cost > target_tokens:
            break
        selected.append(line)
        total += cost
    selected.reverse()
    return "\n".join(selected).strip()


def _summarize_episode(conversation: Conversation, messages: list[ConversationMessage]) -> str:
    lines = []
    if conversation.rolling_summary:
        lines.append(conversation.rolling_summary.strip())
    if conversation.title:
        lines.append(f"Conversation title: {conversation.title}.")
    if conversation.topic:
        lines.append(f"Topic: {conversation.topic}.")
    lines.extend(_summarize_messages_deterministically(messages))
    return "\n".join(lines).strip() or "Conversation closed without saved message content."


def _infer_topics(text: str) -> list[str]:
    normalized = text.lower()
    topic_terms = {
        "strength": ["squat", "deadlift", "bench", "strength", "powerlifting"],
        "nutrition": ["protein", "calorie", "food", "diet", "meal"],
        "recovery": ["sleep", "sore", "fatigue", "recovery", "stress"],
        "injury": ["pain", "injury", "rehab", "flare"],
        "running": ["run", "marathon", "pace", "vo2"],
        "plan": ["plan", "program", "session", "workout"],
    }
    topics = [topic for topic, terms in topic_terms.items() if any(term in normalized for term in terms)]
    return topics or ["general"]


def _episode_importance(summary: str) -> float:
    lower = summary.lower()
    if any(term in lower for term in ["pain", "injury", "illness", "allergy"]):
        return 0.85
    if any(term in lower for term in ["goal", "1rm", "plan", "program"]):
        return 0.7
    return 0.5


def _dedupe_candidates(candidates: list[MemoryItem]) -> list[MemoryItem]:
    seen = set()
    unique = []
    for item in candidates:
        key = (item.target_type, item.memory_key, item.value_json)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _first_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .,!?:;")
    return None


def _looks_food_related(value: str) -> bool:
    return bool(re.search(r"\b(food|meal|eat|diet|rice|meat|fish|egg|milk|vegetable|fruit)\b", value, re.IGNORECASE))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _ensure_aware(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        cleaned = value.strip().replace("Z", "+00:00")
        try:
            return _ensure_aware(datetime.fromisoformat(cleaned))
        except ValueError:
            try:
                return datetime.strptime(cleaned, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                return utc_now()
    return utc_now()


def _parse_xunji_datetime(value: Any) -> datetime | None:
    """Parse Xunji date/time values, including millisecond timestamps."""

    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _ensure_aware(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return _parse_datetime(value)


def _render_plan_summary(plan: PlanProgressOut) -> str:
    progress = plan.plan_progress.get("sessions", {}) if isinstance(plan.plan_progress, dict) else {}
    counts = Counter(
        item.get("status", "unknown")
        for item in progress.values()
        if isinstance(item, dict)
    )
    return (
        f"[{plan.plan_id}] version_id={plan.version_id} version={plan.version} "
        f"status={plan.status} current_week={plan.current_week}\n"
        f"Session status counts: {dict(counts)}"
    )


def _render_plan_full(plan: PlanProgressOut) -> str:
    payload = compact_active_plan_payload(
        {
            "plan_id": plan.plan_id,
            "version_id": plan.version_id,
            "version": plan.version,
            "status": plan.status,
            "current_week": plan.current_week,
            "planned_sessions": plan.planned_sessions,
            "plan_snapshot": plan.plan_snapshot,
            "plan_progress": plan.plan_progress,
            "completed_at": plan.completed_at.isoformat() if plan.completed_at else None,
        }
    )
    return f"[{plan.plan_id}]\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'), default=str)}"


def compact_active_plan_payload(plan: dict[str, Any]) -> dict[str, Any]:
    """Build the bounded active-plan read model used only inside MemoryContext."""

    exercise_schema = ["name", "mode", "volume", "volume_unit", "notes", "set_groups"]
    set_group_schema = [
        "count",
        "reps",
        "load_kg",
        "intensity_percent",
        "rpe",
        "rest_seconds",
    ]
    plan_snapshot = {
        key: value
        for key, value in plan.get("plan_snapshot", {}).items()
        if key
        not in {
            "source_references",
            "program_id",
            "version_id",
            "version_number",
            "user_id",
            "created_at",
        }
        and value not in (None, "", [], {})
    }
    return {
        "format": "compact_active_plan_v1",
        "schemas": {
            "exercise": exercise_schema,
            "set_group": set_group_schema,
        },
        "plan_id": plan.get("plan_id"),
        "version_id": plan.get("version_id"),
        "version": plan.get("version"),
        "status": plan.get("status"),
        "current_week": plan.get("current_week"),
        "planned_sessions": [_compact_planned_session(item) for item in plan.get("planned_sessions", [])],
        "plan_snapshot": plan_snapshot,
        "plan_progress": plan.get("plan_progress", {}),
        "completed_at": plan.get("completed_at"),
    }


def _compact_planned_session(session: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: session.get(key)
        for key in (
            "session_id",
            "planned_date",
            "week_number",
            "day_number",
            "session_type",
            "title",
            "goal",
            "duration_minutes",
        )
        if session.get(key) not in (None, "", [], {})
    }
    exercises = session.get("exercises", [])
    if exercises:
        compact["exercises"] = [_compact_prescribed_exercise(item) for item in exercises]
    return compact


def _compact_prescribed_exercise(exercise: dict[str, Any]) -> list[Any]:
    set_fields = (
        "reps",
        "load_kg",
        "intensity_percent",
        "rpe",
        "rest_seconds",
    )
    groups: list[list[Any]] = []
    for prescribed_set in exercise.get("sets", []):
        values = [prescribed_set.get(key) for key in set_fields]
        if groups and groups[-1][1:] == values:
            groups[-1][0] += 1
        else:
            groups.append([1, *values])
    return [
        exercise.get("name", ""),
        exercise.get("mode", "resistance"),
        exercise.get("volume"),
        exercise.get("volume_unit"),
        exercise.get("notes", ""),
        groups,
    ]


def _short_json(value: Any, max_chars: int = 400) -> str:
    text = _json_dumps(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."
