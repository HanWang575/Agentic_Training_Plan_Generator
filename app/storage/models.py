"""SQLModel tables for source metadata, chunks, traces, citations, and safety flags."""

from datetime import datetime, timezone
from typing import Optional

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
