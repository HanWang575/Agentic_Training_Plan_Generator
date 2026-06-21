"""Pydantic schemas for the User Profile and Memory block."""

from datetime import date, datetime
from typing import Any, TypedDict

from pydantic import BaseModel, Field


class UserCreateRequest(BaseModel):
    """Input shape for creating or updating a local user."""

    user_id: str | None = None
    display_name: str | None = None
    timezone: str = "UTC"


class UserOut(BaseModel):
    """User account response."""

    user_id: str
    display_name: str | None = None
    timezone: str
    created_at: datetime
    updated_at: datetime


class MemoryDeleteResponse(BaseModel):
    """Response for explicit local memory deletion actions."""

    deleted_id: str
    detail: str
    deleted_counts: dict[str, int] = Field(default_factory=dict)


class ConversationCreateRequest(BaseModel):
    """Input shape for starting a memory-backed conversation."""

    user_id: str
    conversation_id: str | None = None
    title: str | None = None
    topic: str | None = None


class ConversationOut(BaseModel):
    """Conversation response."""

    conversation_id: str
    user_id: str
    status: str
    title: str | None = None
    topic: str | None = None
    rolling_summary: str = ""
    rolling_summary_until_message_id: str | None = None
    closure_reason: str | None = None
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None


class MessageAppendRequest(BaseModel):
    """Input shape for appending one conversation turn."""

    role: str
    content: str = Field(min_length=1)
    source_block: str = "manual"
    qa_trace_id: str | None = None


class ConversationMessageOut(BaseModel):
    """Saved conversation message response."""

    message_id: str
    conversation_id: str
    user_id: str
    role: str
    content: str
    source_block: str
    qa_trace_id: str | None = None
    created_at: datetime


class ConversationDetailOut(BaseModel):
    """Conversation metadata with all saved messages."""

    conversation: ConversationOut
    messages: list[ConversationMessageOut] = Field(default_factory=list)


class CloseConversationRequest(BaseModel):
    """Input shape for closing a conversation."""

    closure_reason: str = "user_closed"
    extract_candidates: bool = True
    program_context_id: str | None = None


class EpisodeOut(BaseModel):
    """Episode response."""

    episode_id: str
    user_id: str
    conversation_id: str | None = None
    program_context_id: str | None = None
    source_type: str
    title: str | None = None
    summary: str
    topics: list[str] = Field(default_factory=list)
    importance: float
    source_message_ids: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    embedding_id: str | None = None
    created_at: datetime


class MemoryItemOut(BaseModel):
    """General memory item response."""

    memory_id: str
    user_id: str
    category: str
    target_type: str | None = None
    memory_key: str | None = None
    text: str
    value: Any = None
    confidence: float
    status: str
    safety_relevant: bool
    source_conversation_id: str | None = None
    source_episode_id: str | None = None
    source_message_ids: list[str] = Field(default_factory=list)
    review_note: str | None = None
    replacement_of: str | None = None
    expires_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    reviewed_at: datetime | None = None


class ConversationCloseResponse(BaseModel):
    """Response after closing a conversation."""

    conversation: ConversationOut
    episode: EpisodeOut | None = None
    candidate_memory_items: list[MemoryItemOut] = Field(default_factory=list)


class EpisodeSearchResponse(BaseModel):
    """Episode search response."""

    episodes: list[EpisodeOut] = Field(default_factory=list)
    semantic_error: str | None = None


class ProfileFieldOut(BaseModel):
    """Approved profile field response."""

    profile_field_id: str
    user_id: str
    field_key: str
    value: Any
    value_text: str
    source: str
    source_memory_id: str | None = None
    review_status: str
    replaced_by: str | None = None
    created_at: datetime
    updated_at: datetime


class KeyProposalOut(BaseModel):
    """Memory key proposal response."""

    proposal_id: str
    user_id: str
    proposed_key: str
    target_type: str
    example_value: Any = None
    reason: str
    nearest_existing_keys: list[str] = Field(default_factory=list)
    duplicate_decision: str
    source_memory_id: str | None = None
    source_message_ids: list[str] = Field(default_factory=list)
    review_status: str
    created_at: datetime
    reviewed_at: datetime | None = None


class ProfileResponse(BaseModel):
    """User profile response."""

    user_id: str
    fields: list[ProfileFieldOut] = Field(default_factory=list)


class ProfilePatchRequest(BaseModel):
    """Manual profile edit input."""

    fields: dict[str, Any]
    source: str = "manual"
    extension_mode: bool = False


class ProfilePatchResponse(ProfileResponse):
    """Manual profile edit response."""

    key_proposals: list[KeyProposalOut] = Field(default_factory=list)


class RecentStatusOut(BaseModel):
    """Recent status response."""

    status_id: str
    user_id: str
    status_key: str
    value: Any
    value_text: str
    severity: str | None = None
    impact: str | None = None
    observed_at: datetime
    valid_until: datetime
    review_status: str
    source: str
    source_memory_id: str | None = None
    source_conversation_id: str | None = None
    expiration_reminder_status: str
    created_at: datetime
    updated_at: datetime
    expired_at: datetime | None = None


class RecentStatusPatchRequest(BaseModel):
    """Manual recent-status edit input."""

    statuses: dict[str, Any]
    observed_at: datetime | None = None
    source: str = "manual"
    extension_mode: bool = False


class RecentStatusPatchResponse(BaseModel):
    """Manual recent-status edit response."""

    user_id: str
    statuses: list[RecentStatusOut] = Field(default_factory=list)
    key_proposals: list[KeyProposalOut] = Field(default_factory=list)


class RecentStatusValidUntilRequest(BaseModel):
    """Update a recent status expiration date."""

    valid_until: datetime


class RecentStatusExpireRequest(BaseModel):
    """Expire one recent status while keeping it for audit."""

    reason: str | None = None


class RecentStatusExpiryReminder(BaseModel):
    """User-facing reminder for a recent status close to expiration."""

    status_id: str
    status_key: str
    value: Any = None
    value_text: str
    source: str
    current_valid_until: datetime
    suggested_valid_until: datetime | None = None
    prompt: str
    status: RecentStatusOut


class ExpiringRecentStatusResponse(BaseModel):
    """Statuses that are close to expiry."""

    statuses: list[RecentStatusOut] = Field(default_factory=list)
    reminders: list[RecentStatusExpiryReminder] = Field(default_factory=list)


class MemoryItemApproveRequest(BaseModel):
    """Approve or edit a candidate memory item."""

    target_type: str | None = None
    memory_key: str | None = None
    value: Any = None
    valid_until: datetime | None = None
    review_note: str | None = None


class MemoryItemRejectRequest(BaseModel):
    """Reject a candidate memory item."""

    review_note: str | None = None


class MemoryItemExpireRequest(BaseModel):
    """Expire a memory item or linked recent status."""

    reason: str | None = None


class MemoryItemReviewResponse(BaseModel):
    """Response after approving, rejecting, or expiring a memory item."""

    memory_item: MemoryItemOut
    profile_field: ProfileFieldOut | None = None
    recent_status: RecentStatusOut | None = None


class KeyProposalReviewRequest(BaseModel):
    """Approve or reject a key proposal."""

    proposed_key: str | None = None
    target_type: str | None = None
    review_note: str | None = None


class PlanProgressPatchRequest(BaseModel):
    """Create or update active training-plan progress."""

    user_id: str
    version_id: str | None = None
    version: int | None = None
    status: str | None = None
    current_week: int | None = None
    planned_sessions: list[dict[str, Any]] | None = None
    progress_patch: dict[str, Any] | None = None
    event_type: str = "progress_update"
    session_id: str | None = None
    session_status: str | None = None
    completed_at: datetime | None = None
    feedback: str | None = None


class PlanProgressOut(BaseModel):
    """Training plan progress response."""

    plan_id: str
    user_id: str
    version_id: str | None = None
    version: int
    status: str
    current_week: int | None = None
    planned_sessions: list[dict[str, Any]] = Field(default_factory=list)
    plan_snapshot: dict[str, Any] = Field(default_factory=dict)
    plan_progress: dict[str, Any] = Field(default_factory=dict)
    progress_events: list[dict[str, Any]] = Field(default_factory=list)
    feedback: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class PlanActivationRequest(BaseModel):
    """Activate one complete approved program version."""

    user_id: str
    program_id: str
    version_id: str
    version_number: int
    planned_sessions: list[dict[str, Any]]
    plan_snapshot: dict[str, Any]


class PlanActivationResponse(BaseModel):
    """Result of the atomic active-plan replacement."""

    activated_plan: PlanProgressOut
    deactivated_plan_ids: list[str] = Field(default_factory=list)


class TrainingLogImportRequest(BaseModel):
    """Import raw training log payload from Xunji."""

    user_id: str
    external_id: str | None = None
    payload: Any
    provider: str = "xunji"


class TrainingLogXunjiSyncRequest(BaseModel):
    """Read Xunji training logs from the Open API and import them."""

    user_id: str
    start_date: date
    end_date: date
    include_full_data: bool = True
    force_refresh: bool = False


class TrainingSessionOut(BaseModel):
    """Normalized training session response."""

    session_id: str
    user_id: str
    source_import_id: str
    external_id: str | None = None
    title: str | None = None
    session_date: datetime
    start_at: datetime | None = None
    end_at: datetime | None = None
    duration_seconds: int | None = None
    sport_type: str
    exercises: list[dict[str, Any]] = Field(default_factory=list)
    sets_summary: dict[str, Any] = Field(default_factory=dict)
    rpe: float | None = None
    notes: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class TrainingLogImportResponse(BaseModel):
    """Import result response."""

    import_id: str | None = None
    imported: bool
    skipped_duplicate: bool = False
    sessions_imported: int = 0
    sessions: list[TrainingSessionOut] = Field(default_factory=list)
    dropped_sessions: list[dict[str, Any]] = Field(default_factory=list)
    message: str | None = None


class TrainingLogXunjiSyncResponse(BaseModel):
    """Import result for a Xunji date-range sync."""

    user_id: str
    start_date: date
    end_date: date
    days_requested: int
    imports: list[TrainingLogImportResponse] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)


class TrainingLogSummaryResponse(BaseModel):
    """Recent training log summary response."""

    user_id: str
    days: int
    session_count: int
    sport_counts: dict[str, int] = Field(default_factory=dict)
    recent_sessions: list[TrainingSessionOut] = Field(default_factory=list)


class TrainingSessionListResponse(BaseModel):
    """Filtered training-session list response."""

    user_id: str
    start_date: date | None = None
    end_date: date | None = None
    sport_type: str | None = None
    session_count: int
    sessions: list[TrainingSessionOut] = Field(default_factory=list)


class MemoryContextOut(BaseModel):
    """Memory context sent to other blocks."""

    user_id: str
    conversation_id: str | None = None
    task_type: str = "qa"
    plan_context_level: str = "none"
    active_plan_exists: bool = False
    rendered_context: str = ""
    sections: dict[str, str] = Field(default_factory=dict)
    memory_ids: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    budget_warnings: list[dict[str, Any]] = Field(default_factory=list)
    debug: dict[str, Any] | None = None
    trace_id: str | None = None


class MemoryWorkflowState(TypedDict, total=False):
    """LangGraph-style state for memory context assembly."""

    user_id: str
    conversation_id: str | None
    task_type: str
    topic_hint: str | None
    incoming_message: str
    latest_user_query: str
    conversation_status: str
    closure_reason: str
    conversation_messages: list[ConversationMessageOut]
    rolling_summary: str
    episode_summary: str
    candidate_memory_items: list[MemoryItemOut]
    validated_memory_items: list[MemoryItemOut]
    memory_key_proposals: list[KeyProposalOut]
    profile_fields: list[ProfileFieldOut]
    recent_status_items: list[RecentStatusOut]
    plan_context_level: str
    plan_context_router_used: str
    plan_context_rationale: str
    plan_context_confidence: float
    active_plan_progress: PlanProgressOut | None
    training_log_summary: TrainingLogSummaryResponse | None
    retrieved_episodes: list[EpisodeOut]
    memory_context: MemoryContextOut
    budget_warnings: list[dict[str, Any]]
    memory_debug_ids: list[str]
    errors: list[str]
    timings: dict[str, int]
