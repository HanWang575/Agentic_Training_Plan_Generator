"""Typed request, response, and graph-state objects for Q&A."""

from typing import Any, TypedDict

from pydantic import BaseModel, Field

from app.knowledge.schemas import RetrievedChunk


class AskQuestionRequest(BaseModel):
    """Input shape for `POST /api/qa/ask`."""

    question: str = Field(min_length=1)
    topic_hint: str | None = None
    debug: bool = False


class CitationOut(BaseModel):
    """Output citation shape shown to users and future function blocks."""

    source_id: str
    chunk_id: str
    title: str
    page_or_section: str | None = None


class AskQuestionResponse(BaseModel):
    """Output shape for `POST /api/qa/ask`."""

    answer: str
    source_ids: list[str]
    citations: list[CitationOut]
    safety_note: str | None = None
    insufficient_evidence: bool = False
    trace_id: str
    debug: dict[str, Any] | None = None


class AnswerTraceOut(BaseModel):
    """Output shape for `GET /api/qa/traces/{trace_id}`."""

    trace_id: str
    question: str
    topic_hint: str | None
    safety_mode: str
    planner_used: str | None = None
    clean_question: str | None = None
    clarified_question: str | None = None
    search_query: str | None = None
    query_terms: list[str] = Field(default_factory=list)
    planner_notes: list[str] = Field(default_factory=list)
    keyword_chunk_ids: list[str]
    semantic_chunk_ids: list[str]
    merged_chunk_ids: list[str]
    final_chunk_ids: list[str]
    answer: str
    total_latency_ms: int
    error: str | None
    timings: dict[str, int]
    citations: list[CitationOut]
    safety_flags: list[dict[str, str]]


class QAState(TypedDict, total=False):
    """LangGraph-style state shared by all Q&A nodes."""

    question: str
    topic_hint: str | None
    clean_question: str
    clarified_question: str
    safety_flags: list[dict[str, str]]
    safety_mode: str
    search_query: str
    query_terms: list[str]
    planner_notes: list[str]
    planner_used: str
    filters: dict[str, Any]
    keyword_results: list[RetrievedChunk]
    semantic_results: list[RetrievedChunk]
    semantic_error: str | None
    merged_results: list[RetrievedChunk]
    context_results: list[RetrievedChunk]
    evidence_context: str
    draft_answer: str
    final_answer: str
    context_citations: list[CitationOut]
    citations: list[CitationOut]
    trace_id: str
    errors: list[str]
    timings: dict[str, int]
    insufficient_evidence: bool
