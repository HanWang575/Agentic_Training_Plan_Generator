"""Typed service, API, graph, and plan contracts for the Program Generator."""

from datetime import date, datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class SetPrescriptionOut(BaseModel):
    set_id: str
    set_number: int
    reps: int | None = None
    load_kg: float | None = None
    intensity_percent: float | None = None
    rpe: float | None = None
    rest_seconds: int
    calculation_trace_ids: list[str] = Field(default_factory=list)


class ExercisePrescriptionOut(BaseModel):
    exercise_id: str
    name: str
    order: int
    mode: str = "resistance"
    notes: str = ""
    volume: float | None = None
    volume_unit: str | None = None
    calculation_trace_ids: list[str] = Field(default_factory=list)
    sets: list[SetPrescriptionOut] = Field(default_factory=list)


class ProgramSessionOut(BaseModel):
    session_id: str
    planned_date: datetime
    week_number: int
    day_number: int
    session_type: str
    goal: str
    duration_minutes: int
    duration_calculation_trace_id: str | None = None
    exercises: list[ExercisePrescriptionOut] = Field(default_factory=list)


class ProgramVersionResponse(BaseModel):
    program_id: str
    version_id: str
    version_number: int
    user_id: str
    title: str
    goal: str
    sport: str
    status: str
    reason: str
    screening: dict[str, Any] = Field(default_factory=dict)
    needs_analysis: dict[str, Any] = Field(default_factory=dict)
    periodization: dict[str, Any] = Field(default_factory=dict)
    planning_model: str
    rationale: str
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_references: dict[str, Any] = Field(default_factory=dict)
    sessions: list[ProgramSessionOut] = Field(default_factory=list)
    created_at: datetime


class ProgramLoadReferenceInput(BaseModel):
    exercise_name: str = Field(min_length=1)
    type: Literal["1rm", "recent_set"]
    weight_kg: float = Field(ge=0)
    reps: int | None = Field(default=None, ge=1)


class ProgramTurnRequest(BaseModel):
    user_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    load_references: list[ProgramLoadReferenceInput] = Field(default_factory=list)
    expected_context_revision: int | None = None


class ProgramTurnResponse(BaseModel):
    conversation_id: str
    context_id: str
    context_revision: int
    route: str
    status: str
    response: str
    pending_action: str | None = None
    draft_program_id: str | None = None
    draft_version_id: str | None = None
    trace_id: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ProgramContextResponse(BaseModel):
    context_id: str
    parent_context_id: str | None = None
    user_id: str
    conversation_id: str
    context_revision: int
    status: str
    active_task_type: str | None = None
    current_turn_route: str | None = None
    active_sop: str | None = None
    current_sop_step: str | None = None
    pending_action: str | None = None
    latest_program_id: str | None = None
    latest_version_id: str | None = None
    review_status: str | None = None
    approval_status: str | None = None
    sections: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ProgramReviewRequest(BaseModel):
    user_id: str = Field(min_length=1)
    decision: Literal["approve", "reject"]
    reason: str | None = None
    expected_version_number: int | None = None


class ProgramReviewResponse(BaseModel):
    program_id: str
    version_id: str
    review_status: str
    activation_status: str | None = None
    activated_plan_id: str | None = None
    deactivated_plan_ids: list[str] = Field(default_factory=list)
    message: str


class XunjiWriteRequest(BaseModel):
    user_id: str = Field(min_length=1)
    confirmed: bool


class XunjiWriteResponse(BaseModel):
    program_id: str
    version_id: str
    status: str
    external_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CalculatorRequest(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)
    source_references: list[str] = Field(default_factory=list)


class CalculationResult(BaseModel):
    tool_call_id: str
    tool_name: str
    value: float
    unit: str
    inputs: dict[str, Any]
    source_references: list[str] = Field(default_factory=list)
    formula: str
    rounding_policy: str


class ProgramTraceResponse(BaseModel):
    trace_id: str
    context_id: str
    version_id: str | None = None
    context_revision: int
    task_type: str
    sop_step: str | None = None
    action: str
    details: dict[str, Any] = Field(default_factory=dict)
    rag_query_count: int
    user_question_count: int
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    calculations: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    safety_flags: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime


class TrainingSummaryNoteOut(BaseModel):
    summary_id: str
    user_id: str
    week_start: datetime
    week_end: datetime
    timezone: str
    summary: str
    source_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class TrainingSummaryNotesResponse(BaseModel):
    user_id: str
    notes: list[TrainingSummaryNoteOut] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class WeeklySummaryRefreshRequest(BaseModel):
    user_id: str
    reference_date: date | None = None


class ProgramContextPatch(BaseModel):
    """One revision-checked update to a declared ProgramContext section."""

    expected_context_revision: int
    target_section: str
    operation: Literal["replace", "merge", "append"] = "replace"
    value: Any
    source_references: list[str] = Field(default_factory=list)
    trace_references: list[str] = Field(default_factory=list)
    supersession_reason: str | None = None


class ProgramGraphState(TypedDict, total=False):
    context: dict[str, Any]
    route_locked: bool
    route: str
    task_type: str
    route_rationale: str
    route_confidence: float
    rationale_question: str
    rationale_context: dict[str, Any]
    numeric_rationale_facts: dict[str, Any]
    rationale_answer: str
    current_step: str
    missing_inputs: list[str]
    evidence_sufficient: bool
    step_complete: bool
    sop_complete: bool
    has_raw_plan: bool
    has_calculated_plan: bool
    rag_query_count: int
    user_question_count: int
    max_rag_queries: int
    max_user_questions: int
    next_action: str
    errors: list[str]
    warnings: list[str]
