"""SQLModel tables for source metadata, chunks, traces, citations, and safety flags."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, String
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for database rows."""

    return datetime.now(timezone.utc)


class SourceDocument(SQLModel, table=True):
    """One uploaded or registered document.

    Important fields:
        source_id: stable public ID, e.g. `src_...`.
        approval_status: `pending_review`, `approved`, `rejected`, `parse_error`, or `index_error`.
        topic_tags_json: JSON list stored as text for beginner-readable SQLite.
    """

    __tablename__ = "source_documents"

    source_id: str = Field(primary_key=True, index=True)
    title: str
    source_type: str
    file_path: str
    source_url: Optional[str] = None
    topic_tags_json: str = "[]"
    approval_status: str = Field(default="pending_review", index=True)
    uploaded_at: datetime = Field(default_factory=utc_now)
    reviewed_at: Optional[datetime] = None
    review_note: Optional[str] = None
    parser_status: str = "pending"
    parser_progress: float = 0.0
    parser_progress_message: Optional[str] = None
    parser_error: Optional[str] = None
    parser_warnings_json: str = "[]"
    parser_options_json: str = "{}"


class KnowledgeChunk(SQLModel, table=True):
    """A searchable chunk of approved or pending document text."""

    __tablename__ = "knowledge_chunks"

    chunk_id: str = Field(primary_key=True, index=True)
    source_id: str = Field(index=True, foreign_key="source_documents.source_id")
    chunk_index: int = Field(index=True)
    text: str
    page_number: Optional[int] = Field(default=None, index=True)
    section_heading: Optional[str] = None
    content_type: str = Field(default="plain_text", index=True)
    content_metadata_json: str = "{}"
    topic_tags_json: str = "[]"
    token_count: int = 0
    embedding_id: Optional[str] = None
    approval_status: str = Field(default="pending_review", index=True)
    quality_warnings_json: str = "[]"


class AnswerTrace(SQLModel, table=True):
    """A replayable record of one Q&A graph run."""

    __tablename__ = "answer_traces"

    trace_id: str = Field(primary_key=True, index=True)
    question: str
    topic_hint: Optional[str] = None
    user_id: Optional[str] = Field(default=None, index=True)
    conversation_id: Optional[str] = Field(default=None, index=True)
    safety_mode: str = "normal"
    planner_used: Optional[str] = None
    clean_question: Optional[str] = None
    clarified_question: Optional[str] = None
    search_query: Optional[str] = None
    query_terms_json: str = "[]"
    planner_notes_json: str = "[]"
    keyword_chunk_ids_json: str = "[]"
    semantic_chunk_ids_json: str = "[]"
    merged_chunk_ids_json: str = "[]"
    final_chunk_ids_json: str = "[]"
    answer: str = ""
    memory_context_json: str = "{}"
    memory_debug_ids_json: str = "[]"
    created_at: datetime = Field(default_factory=utc_now)
    total_latency_ms: int = 0
    error: Optional[str] = None
    timings_json: str = "{}"


class Citation(SQLModel, table=True):
    """A citation link from an answer trace to one retrieved source chunk."""

    __tablename__ = "citations"

    citation_id: str = Field(primary_key=True)
    trace_id: str = Field(index=True, foreign_key="answer_traces.trace_id")
    source_id: str = Field(index=True)
    chunk_id: str = Field(index=True)
    answer_section: str = "Evidence used"
    citation_text: str = ""


class SafetyFlag(SQLModel, table=True):
    """A safety decision attached to an answer trace."""

    __tablename__ = "safety_flags"

    flag_id: str = Field(primary_key=True)
    trace_id: str = Field(index=True, foreign_key="answer_traces.trace_id")
    flag_type: str
    severity: str
    explanation: str


class UserAccount(SQLModel, table=True):
    """A local-first user record for the personalization layer."""

    __tablename__ = "user_accounts"

    user_id: str = Field(primary_key=True, index=True)
    display_name: Optional[str] = None
    timezone: str = "UTC"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Conversation(SQLModel, table=True):
    """Thread-level working memory for one user conversation."""

    __tablename__ = "conversations"

    conversation_id: str = Field(primary_key=True, index=True)
    user_id: str = Field(index=True, foreign_key="user_accounts.user_id")
    status: str = Field(default="open", index=True)
    title: Optional[str] = None
    topic: Optional[str] = None
    rolling_summary: str = ""
    rolling_summary_until_message_id: Optional[str] = Field(default=None, index=True)
    closure_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    closed_at: Optional[datetime] = None


class ConversationMessage(SQLModel, table=True):
    """One raw conversation turn saved for working memory and later review."""

    __tablename__ = "conversation_messages"

    message_id: str = Field(primary_key=True, index=True)
    conversation_id: str = Field(index=True, foreign_key="conversations.conversation_id")
    user_id: str = Field(index=True, foreign_key="user_accounts.user_id")
    role: str = Field(index=True)
    content: str
    source_block: str = Field(default="manual", index=True)
    qa_trace_id: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utc_now)


class Episode(SQLModel, table=True):
    """A permanent episodic summary created from a closed conversation or plan."""

    __tablename__ = "episodes"

    episode_id: str = Field(primary_key=True, index=True)
    user_id: str = Field(index=True, foreign_key="user_accounts.user_id")
    conversation_id: Optional[str] = Field(default=None, index=True)
    source_type: str = Field(default="conversation", index=True)
    title: Optional[str] = None
    summary: str
    topics_json: str = "[]"
    importance: float = 0.5
    source_message_ids_json: str = "[]"
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    embedding_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class MemoryItem(SQLModel, table=True):
    """General memory record, usually a candidate awaiting review."""

    __tablename__ = "memory_items"

    memory_id: str = Field(primary_key=True, index=True)
    user_id: str = Field(index=True, foreign_key="user_accounts.user_id")
    category: str = Field(index=True)
    target_type: Optional[str] = Field(default=None, index=True)
    memory_key: Optional[str] = Field(default=None, index=True)
    text: str
    value_json: str = "{}"
    confidence: float = 0.5
    status: str = Field(default="candidate", index=True)
    safety_relevant: bool = False
    source_conversation_id: Optional[str] = Field(default=None, index=True)
    source_episode_id: Optional[str] = Field(default=None, index=True)
    source_message_ids_json: str = "[]"
    review_note: Optional[str] = None
    replacement_of: Optional[str] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    reviewed_at: Optional[datetime] = None


class ProfileField(SQLModel, table=True):
    """A clean current profile value that other blocks can consume directly."""

    __tablename__ = "profile_fields"

    profile_field_id: str = Field(primary_key=True, index=True)
    user_id: str = Field(index=True, foreign_key="user_accounts.user_id")
    field_key: str = Field(index=True)
    value_text: str
    value_json: str = "{}"
    source: str = "manual"
    source_memory_id: Optional[str] = Field(default=None, index=True)
    review_status: str = Field(default="approved", index=True)
    replaced_by: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class RecentStatus(SQLModel, table=True):
    """A short-lived readiness, health, schedule, or adherence status."""

    __tablename__ = "recent_status"

    status_id: str = Field(primary_key=True, index=True)
    user_id: str = Field(index=True, foreign_key="user_accounts.user_id")
    status_key: str = Field(index=True)
    value_text: str
    value_json: str = "{}"
    severity: Optional[str] = None
    impact: Optional[str] = None
    observed_at: datetime = Field(default_factory=utc_now)
    valid_until: datetime = Field(index=True)
    review_status: str = Field(default="approved", index=True)
    source: str = "manual"
    source_memory_id: Optional[str] = Field(default=None, index=True)
    source_conversation_id: Optional[str] = Field(default=None, index=True)
    expiration_reminder_status: str = "not_due"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    expired_at: Optional[datetime] = None


class MemoryKeyProposal(SQLModel, table=True):
    """A proposed extension to the profile or recent-status key catalog."""

    __tablename__ = "memory_key_proposals"

    proposal_id: str = Field(primary_key=True, index=True)
    user_id: str = Field(index=True, foreign_key="user_accounts.user_id")
    proposed_key: str = Field(index=True)
    target_type: str = Field(index=True)
    example_value_json: str = "{}"
    reason: str = ""
    nearest_existing_keys_json: str = "[]"
    duplicate_decision: str = "not_duplicate"
    source_memory_id: Optional[str] = Field(default=None, index=True)
    source_message_ids_json: str = "[]"
    review_status: str = Field(default="pending", index=True)
    created_at: datetime = Field(default_factory=utc_now)
    reviewed_at: Optional[datetime] = None


class TrainingPlanProgress(SQLModel, table=True):
    """Current plan progress plus append-only event history in JSON fields."""

    __tablename__ = "training_plan_progress"

    plan_id: str = Field(primary_key=True, index=True)
    user_id: str = Field(index=True, foreign_key="user_accounts.user_id")
    version: int = 1
    status: str = Field(default="active", index=True)
    current_week: Optional[int] = None
    planned_sessions_json: str = "[]"
    plan_progress_json: str = "{}"
    progress_events_json: str = "[]"
    feedback_json: str = "[]"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = None


class TrainingLogImport(SQLModel, table=True):
    """Raw imported payload from Xunji or a future training-log provider."""

    __tablename__ = "training_log_imports"

    import_id: str = Field(primary_key=True, index=True)
    user_id: str = Field(index=True, foreign_key="user_accounts.user_id")
    provider: str = Field(default="xunji", index=True)
    external_id: Optional[str] = Field(default=None, index=True)
    checksum: str = Field(index=True)
    raw_payload_json: str
    import_status: str = "imported"
    imported_at: datetime = Field(default_factory=utc_now)


class TrainingSession(SQLModel, table=True):
    """A normalized training session derived from an imported log payload."""

    __tablename__ = "training_sessions"

    session_id: str = Field(primary_key=True, index=True)
    user_id: str = Field(index=True, foreign_key="user_accounts.user_id")
    source_import_id: str = Field(index=True, foreign_key="training_log_imports.import_id")
    external_id: Optional[str] = Field(default=None, index=True)
    session_date: datetime = Field(index=True)
    sport_type: str = Field(default="unknown", index=True)
    exercises_json: str = "[]"
    sets_summary_json: str = "{}"
    rpe: Optional[float] = None
    notes: Optional[str] = None
    raw_payload_json: str = "{}"
    created_at: datetime = Field(default_factory=utc_now)


class MemoryTrace(SQLModel, table=True):
    """Debug trace for memory retrieval and budget warnings."""

    __tablename__ = "memory_traces"

    trace_id: str = Field(primary_key=True, index=True)
    user_id: str = Field(index=True, foreign_key="user_accounts.user_id")
    conversation_id: Optional[str] = Field(default=None, index=True)
    task_type: str = Field(default="qa", index=True)
    query: str = ""
    plan_context_level: str = "none"
    retrieved_memory_ids_json: str = "[]"
    budget_warnings_json: str = Field(
        default="{}",
        sa_column=Column("compaction_report_json", String, default="{}"),
    )
    errors_json: str = "[]"
    created_at: datetime = Field(default_factory=utc_now)
    total_latency_ms: int = 0
