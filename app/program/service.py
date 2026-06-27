"""Readable orchestration service for program generation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import json
import re
import time as clock

from sqlmodel import Session, select

from app.core.config import Settings
from app.core.errors import ConfigurationError, NotFoundError
from app.core.llm_provider import (
    OpenAIProvider,
    ProgramInputExtractionRequest,
    ProgramPlanUpdateRequest,
    ProgramLiftMaxExtractionRequest,
    ProgramSOPStepDecisionRequest,
    StructuredOutputError,
)
from app.knowledge.service import KnowledgeService
from app.memory.nodes import render_sections
from app.memory.schemas import PlanActivationRequest
from app.memory.service import MemoryService, compact_active_plan_payload
from app.program.calculators import CalculatorService
from app.program.graph import run_program_generator_graph, run_program_rationale_graph
from app.program.nodes import ProgramContextPatchValidator
from app.program.schemas import (
    CalculationResult,
    CalculatorRequest,
    ExercisePrescriptionOut,
    ProgramContextResponse,
    ProgramContextPatch,
    ProgramGraphState,
    ProgramReviewRequest,
    ProgramReviewResponse,
    ProgramSessionOut,
    ProgramTraceResponse,
    ProgramTurnRequest,
    ProgramTurnResponse,
    ProgramVersionResponse,
    SetPrescriptionOut,
    TrainingSummaryNoteOut,
    TrainingSummaryNotesResponse,
    XunjiWriteRequest,
    XunjiWriteResponse,
)
from app.storage.models import (
    CalculationTrace,
    ExercisePrescription,
    ProgramCitation,
    ProgramContextRecord,
    ProgramRequest,
    ProgramSafetyFlag,
    ProgramSession,
    ProgramToolCall,
    ProgramTrace,
    ProgramVersion,
    SetPrescription,
    TrainingProgram,
    TrainingSummaryNote,
    utc_now,
)


NEW_PROGRAM_SOP = [
    "consultation_and_screening",
    "needs_analysis",
    "periodization",
    "planning_model",
    "programming_exercises_and_order",
    "programming_intensity_and_volume",
    "load_calculation",
    "volume_calculation",
    "rest_periods",
]

DETERMINISTIC_GENERATION_STEPS = {"load_calculation", "volume_calculation"}

PROGRAM_REQUIRED_INPUTS = ("goal", "days_per_week", "session_duration_minutes", "experience", "equipment", "screening")

PROGRAM_SECTION_DEFAULTS: dict[str, Any] = {
    "current_user_turn": {},
    "memory_context": {},
    "latest_four_week_summary_notes": [],
    "retrieved_evidence": {"current_step": None, "items": []},
    "user_clarification_answers": [],
    "sop_progress": {},
    "new_program_working_state": {},
    "draft_artifacts": {},
    "safety_flags_and_budget_warnings": [],
}

RED_FLAG_PATTERNS = {
    "chest_pain": r"\bchest pain\b",
    "fainting": r"\b(faint|fainted|fainting|passed out)\b",
    "severe_pain": r"\bsevere pain\b",
    "acute_injury": r"\b(acute injury|torn|fracture|broken bone)\b",
    "breathing_difficulty": r"\b(difficulty breathing|shortness of breath at rest)\b",
}


class ProgramService:
    """Public application service used by FastAPI, MCP, and Streamlit."""

    def __init__(
        self,
        session: Session,
        settings: Settings,
        memory_service: MemoryService | None = None,
        knowledge_service: KnowledgeService | None = None,
        llm_provider: OpenAIProvider | None = None,
    ):
        self.session = session
        self.settings = settings
        self.llm_provider = llm_provider or OpenAIProvider(settings)
        self.memory_service = memory_service or MemoryService(
            session=session,
            settings=settings,
            llm_provider=self.llm_provider,
        )
        self.knowledge_service = knowledge_service or KnowledgeService(
            session=session,
            settings=settings,
            llm_provider=self.llm_provider,
        )
        self.calculators = CalculatorService(settings)
        self.patch_validator = ProgramContextPatchValidator(set(PROGRAM_SECTION_DEFAULTS))

    def process_turn(self, conversation_id: str, request: ProgramTurnRequest) -> ProgramTurnResponse:
        """Process one user turn until output, clarification, or a saved draft."""

        if not conversation_id.strip() or not request.user_id.strip():
            raise ValueError("conversation_id and user_id are required.")
        started = clock.perf_counter()
        conversation = self.memory_service.ensure_conversation(
            user_id=request.user_id,
            conversation_id=conversation_id,
            title="Program Generator",
            topic="program_generator",
        )
        message = self.memory_service.append_message(
            conversation_id=conversation.conversation_id,
            role="user",
            content=request.message,
            source_block="program_generator",
        )
        context = self._load_or_create_context(request.user_id, conversation.conversation_id)
        if request.expected_context_revision is not None and context.context_revision != request.expected_context_revision:
            raise ValueError(
                f"ProgramContext revision conflict: expected {request.expected_context_revision}, "
                f"found {context.context_revision}."
            )
        sections = self._sections(context)
        sections["current_user_turn"] = {
            "message_id": message.message_id,
            "content": request.message,
            "created_at": message.created_at.isoformat(),
            "normalized_input": " ".join(request.message.split()),
        }
        memory_context = self.memory_service.get_memory_context(
            user_id=request.user_id,
            query=request.message,
            conversation_id=conversation.conversation_id,
            task_type="program_generator",
            include_debug=True,
        )
        sections["memory_context"] = _compact_memory_context_for_program_context(
            memory_context.model_dump(mode="json")
        )
        summary_response = self.list_weekly_summaries(request.user_id, limit=4)
        sections["latest_four_week_summary_notes"] = [item.model_dump(mode="json") for item in summary_response.notes]
        self._refresh_budget_warnings(sections)
        warnings = [
            item
            for item in sections.get("safety_flags_and_budget_warnings", [])
            if _is_budget_warning(item)
        ]
        if context.pending_action == "waiting_for_context_budget_decision":
            normalized = request.message.strip().lower()
            if normalized in {"stop", "close", "stop and close", "cancel"}:
                context.status = "closed"
                context.pending_action = None
                context.closed_at = utc_now()
                self._checkpoint(context, sections)
                response = "Program generation stopped and the conversation was closed. The ProgramContext remains available for audit."
                self._assistant_message(context, response)
                self.memory_service.close_conversation(
                    context.conversation_id,
                    closure_reason="program_context_budget_stopped",
                    program_context_id=context.context_id,
                )
                trace_id = self._trace(context, "context_budget_stop", {"warnings": warnings})
                return self._turn_response(context, response, trace_id, warnings=warnings)
            if normalized in {"continue", "continue generation", "proceed"}:
                progress = dict(sections.get("sop_progress", {}))
                pending_turn = progress.pop("context_budget_pending_turn", None)
                if pending_turn:
                    sections["current_user_turn"] = pending_turn
                progress["context_budget_override"] = True
                sections["sop_progress"] = progress
                context.pending_action = None
                context.status = "open"
        self._checkpoint(context, sections)

        initial_state = self._graph_state(context)
        routed = run_program_generator_graph(initial_state, self.llm_provider)
        route = routed.get("route", "new_program")
        task_type = "program_generation"
        context.current_turn_route = route
        context.active_task_type = task_type

        if route == "explain_plan_rationale":
            return self._rationale_response(context, started)

        budget_exceeded = any(item.startswith("ProgramContext") for item in warnings)
        budget_override = bool(sections.get("sop_progress", {}).get("context_budget_override"))
        if budget_exceeded and not budget_override:
            progress = dict(sections.get("sop_progress", {}))
            progress.setdefault("context_budget_pending_turn", sections.get("current_user_turn", {}))
            sections["sop_progress"] = progress
            context.status = "waiting_for_user"
            context.pending_action = "waiting_for_context_budget_decision"
            self._checkpoint(context, sections)
            response = (
                "The ProgramContext exceeds its configured token budget. Review the warnings, then reply "
                "'continue' to proceed without removing protected details or 'stop' to close the conversation."
            )
            trace_id = self._trace(context, "context_budget_warning", {"warnings": warnings})
            self._assistant_message(context, response)
            return self._turn_response(context, response, trace_id, warnings=warnings)

        effective_message = str(sections.get("current_user_turn", {}).get("content", request.message))

        if route == "new_program":
            self._reset_new_program_sections(sections)
            context.active_sop = "new_program_design"
            context.current_sop_step = NEW_PROGRAM_SOP[0]
        else:
            answers = list(sections.get("user_clarification_answers", []))
            answers.append(sections["current_user_turn"])
            sections["user_clarification_answers"] = answers
            context.pending_action = None

        request_row = ProgramRequest(
            request_id=f"preq_{uuid4().hex[:12]}",
            context_id=context.context_id,
            conversation_id=context.conversation_id,
            request_type=route,
            raw_request=effective_message,
        )
        self.session.add(request_row)

        working_key = self._working_key(task_type)
        working = dict(sections.get(working_key, {}))
        collected = dict(working.get("collected_inputs", {}))
        extracted_from_user = self._extract_inputs(effective_message, profile_text="", recent_status_text="")
        extracted_from_context = self._extract_inputs(
            "",
            profile_text=memory_context.sections.get("profile", ""),
            recent_status_text=memory_context.sections.get("recent_status", ""),
        )
        _merge_context_defaults(collected, extracted_from_context)
        collected.update(extracted_from_user)
        pending_load_names = list(working.get("pending_load_clarifications", []))
        if pending_load_names:
            load_references = dict(collected.get("exercise_load_references", {}))
            load_references.update(_extract_exercise_load_references(effective_message, pending_load_names))
            load_references.update(
                _normalize_structured_load_references(
                    [item.model_dump(mode="json") for item in request.load_references],
                    pending_load_names,
                )
            )
            if load_references:
                collected["exercise_load_references"] = load_references
        if route == "new_program":
            working["task_request"] = effective_message
        working["collected_inputs"] = collected
        if working.get("raw_plan"):
            working["pending_load_clarifications"] = self._missing_weighted_load_references(
                working["raw_plan"],
                collected,
                memory_context.sections.get("profile", ""),
            )
        sections[working_key] = working
        request_row.goal = str(collected.get("goal", "")) or None
        request_row.constraints_json = _dumps(
            {key: collected.get(key) for key in ("days_per_week", "session_duration_minutes", "equipment", "screening")}
        )

        red_flags = self._red_flags(effective_message, memory_context.sections.get("recent_status", ""))
        if red_flags:
            return self._blocked_safety_response(context, sections, red_flags, started)

        if route == "new_program" and not working.get("required_inputs_prompted"):
            working["required_inputs_prompted"] = True
            sections[working_key] = working
            first_message_inputs = self._extract_inputs(effective_message, profile_text="", recent_status_text="")
            missing_required = [
                key for key in PROGRAM_REQUIRED_INPUTS if first_message_inputs.get(key) in (None, "", [])
            ]
            if missing_required:
                self._checkpoint(context, sections)
                return self._ask_for_missing(context, missing_required, started)

        self._checkpoint(context, sections)
        max_actions = self.settings.program_max_sop_steps_per_run * (
            self.settings.program_max_rag_queries + self.settings.program_max_user_questions + 3
        )
        for _ in range(max_actions):
            state = run_program_generator_graph(self._graph_state(context, route_locked=True), self.llm_provider)
            action = state.get("next_action", "ask_user")
            if action == "ask_user":
                return self._ask_for_missing(context, state.get("missing_inputs", []), started)
            if action == "query_rag":
                self._query_evidence(context, effective_message)
                continue
            if action == "write_step_output":
                try:
                    self._write_sop_output(context, effective_message)
                except StructuredOutputError as exc:
                    return self._structured_output_error_response(context, exc, started)
                continue
            if action == "run_calculation":
                self._calculate_working_plan(context)
                continue
            if action == "finalize_output":
                return self._finalize(context, started)
        raise RuntimeError("Program workflow exceeded its bounded SOP action limit.")

    def get_context(self, context_id: str) -> ProgramContextResponse:
        return self._context_out(self._get_context(context_id))

    def list_context_ids(self) -> list[str]:
        contexts = self.session.exec(select(ProgramContextRecord)).all()
        return [
            context.context_id
            for context in sorted(contexts, key=lambda item: item.updated_at, reverse=True)
        ]

    def get_version(self, program_id: str, version_id: str) -> ProgramVersionResponse:
        program = self.session.get(TrainingProgram, program_id)
        version = self.session.get(ProgramVersion, version_id)
        if not program or not version or version.program_id != program_id:
            raise NotFoundError(f"Program version not found: {program_id}/{version_id}")
        sessions = self.session.exec(
            select(ProgramSession).where(ProgramSession.version_id == version_id)
        ).all()
        session_outputs = [self._session_out(item) for item in sorted(sessions, key=lambda row: row.planned_date)]
        return ProgramVersionResponse(
            program_id=program.program_id,
            version_id=version.version_id,
            version_number=version.version_number,
            user_id=program.user_id,
            title=program.title,
            goal=program.goal,
            sport=program.sport,
            status=version.status,
            reason=version.reason,
            screening=_loads(version.screening_json, {}),
            needs_analysis=_loads(version.needs_analysis_json, {}),
            periodization=_loads(version.periodization_json, {}),
            planning_model=version.planning_model,
            rationale=version.rationale,
            assumptions=_coerce_string_list(_loads(version.assumptions_json, [])),
            warnings=_coerce_string_list(_loads(version.warnings_json, [])),
            source_references=_loads(version.source_references_json, {}),
            sessions=session_outputs,
            created_at=version.created_at,
        )

    def review_program(
        self,
        program_id: str,
        version_id: str,
        request: ProgramReviewRequest,
    ) -> ProgramReviewResponse:
        version_output = self.get_version(program_id, version_id)
        if version_output.user_id != request.user_id:
            raise ValueError("Program version belongs to another user.")
        if request.expected_version_number is not None and version_output.version_number != request.expected_version_number:
            raise ValueError("Program version changed before review.")
        version = self.session.get(ProgramVersion, version_id)
        program = self.session.get(TrainingProgram, program_id)
        context = self._context_for_version(version_id)
        if request.decision == "reject":
            version.status = "rejected"
            version.reviewed_at = utc_now()
            program.generation_status = "rejected"
            if context:
                context.review_status = "rejected"
                context.approval_status = "rejected"
                context.status = "open"
                context.pending_action = None
                self._checkpoint(context, self._sections(context))
            self.session.add(version)
            self.session.add(program)
            self.session.commit()
            result = ProgramReviewResponse(
                program_id=program_id,
                version_id=version_id,
                review_status="rejected",
                message="Program version rejected. Send requested changes as a new Program Generator turn.",
            )
            self._record_review_message(context, f"Rejected program version {version_id}. {request.reason or ''}".strip())
            self._assistant_message_if_context(context, result.message)
            return result

        plan_snapshot = version_output.model_dump(mode="json")
        plan_snapshot.pop("sessions", None)
        try:
            activation = self.memory_service.activate_plan(
                PlanActivationRequest(
                    user_id=request.user_id,
                    program_id=program_id,
                    version_id=version_id,
                    version_number=version_output.version_number,
                    planned_sessions=[item.model_dump(mode="json") for item in version_output.sessions],
                    plan_snapshot=plan_snapshot,
                )
            )
        except Exception as exc:
            self.session.rollback()
            version = self.session.get(ProgramVersion, version_id)
            program = self.session.get(TrainingProgram, program_id)
            context = self._context_for_version(version_id)
            version.status = "approved"
            version.reviewed_at = utc_now()
            program.generation_status = "activation_failed"
            if context:
                context.review_status = "approved"
                context.approval_status = "activation_failed"
                context.status = "waiting_for_review"
                context.pending_action = "waiting_for_review"
                self._checkpoint(context, self._sections(context))
            self.session.add(version)
            self.session.add(program)
            self.session.commit()
            result = ProgramReviewResponse(
                program_id=program_id,
                version_id=version_id,
                review_status="approved",
                activation_status="activation_failed",
                message=f"Program approved, but activation failed and can be retried: {exc}",
            )
            self._record_review_message(context, f"Approved program version {version_id}.")
            self._assistant_message_if_context(context, result.message)
            return result
        version.status = "approved"
        version.reviewed_at = utc_now()
        program.generation_status = "active"
        program.updated_at = utc_now()
        if context:
            context.review_status = "approved"
            context.approval_status = "activated"
            context.status = "open"
            context.pending_action = None
            self._checkpoint(context, self._sections(context))
        self.session.add(version)
        self.session.add(program)
        self.session.commit()
        result = ProgramReviewResponse(
            program_id=program_id,
            version_id=version_id,
            review_status="approved",
            activation_status="activated",
            activated_plan_id=activation.activated_plan.plan_id,
            deactivated_plan_ids=activation.deactivated_plan_ids,
            message="Program approved and activated. Older active plans were deactivated.",
        )
        self._record_review_message(context, f"Approved and activated program version {version_id}.")
        self._assistant_message_if_context(context, result.message)
        return result

    def write_to_xunji(
        self,
        program_id: str,
        version_id: str,
        request: XunjiWriteRequest,
    ) -> XunjiWriteResponse:
        version = self.get_version(program_id, version_id)
        if version.user_id != request.user_id:
            raise ValueError("Program version belongs to another user.")
        if not request.confirmed:
            raise ValueError("Separate Xunji confirmation is required.")
        if version.status != "approved":
            raise ValueError("Only an approved program version can be written to Xunji.")
        if not self.settings.program_xunji_write_enabled:
            return XunjiWriteResponse(
                program_id=program_id,
                version_id=version_id,
                status="disabled",
                errors=["PROGRAM_XUNJI_WRITE_ENABLED is false."],
            )
        return XunjiWriteResponse(
            program_id=program_id,
            version_id=version_id,
            status="unavailable",
            errors=["The Xunji write endpoint/schema is not specified; no data was sent."],
        )

    def run_calculator(self, tool_name: str, request: CalculatorRequest) -> CalculationResult:
        return self.calculators.run(tool_name, request.inputs, request.source_references)

    def list_weekly_summaries(self, user_id: str, limit: int = 4) -> TrainingSummaryNotesResponse:
        rows = self.session.exec(
            select(TrainingSummaryNote).where(TrainingSummaryNote.user_id == user_id)
        ).all()
        rows = sorted(rows, key=lambda item: item.week_end, reverse=True)[: max(1, min(limit, 52))]
        warnings = [] if len(rows) >= min(limit, 4) else ["Fewer than four completed weekly summaries are available."]
        return TrainingSummaryNotesResponse(
            user_id=user_id,
            notes=[self._summary_out(item) for item in rows],
            warnings=warnings,
        )

    def refresh_weekly_summaries(self, user_id: str, reference_date: date | None = None) -> TrainingSummaryNotesResponse:
        """Create missing notes for the four completed natural weeks before a date."""

        user = self.memory_service.create_user(user_id)
        try:
            user_timezone = ZoneInfo(user.timezone)
        except ZoneInfoNotFoundError:
            user_timezone = timezone.utc
        reference = reference_date or datetime.now(user_timezone).date()
        current_monday = reference - timedelta(days=reference.weekday())
        for offset in range(1, 5):
            week_start_date = current_monday - timedelta(weeks=offset)
            week_end_date = week_start_date + timedelta(days=6)
            week_start = datetime.combine(week_start_date, time.min, tzinfo=user_timezone).astimezone(timezone.utc)
            week_end = datetime.combine(week_end_date, time.max, tzinfo=user_timezone).astimezone(timezone.utc)
            existing = self.session.exec(
                select(TrainingSummaryNote).where(
                    TrainingSummaryNote.user_id == user_id,
                    TrainingSummaryNote.week_start == week_start,
                )
            ).first()
            if existing:
                continue
            logs = self.memory_service.list_training_sessions(
                user_id=user_id,
                start_date=week_start_date,
                end_date=week_end_date,
                limit=200,
            )
            source_ids = [item.session_id for item in logs.sessions]
            sport_counts: dict[str, int] = {}
            total_minutes = 0
            total_sets = 0
            exercise_count = 0
            rpe_values: list[float] = []
            recovery_notes: list[str] = []
            for item in logs.sessions:
                sport_counts[item.sport_type] = sport_counts.get(item.sport_type, 0) + 1
                total_minutes += int((item.duration_seconds or 0) / 60)
                exercise_count += len(item.exercises)
                total_sets += _count_logged_sets(item.exercises, item.sets_summary)
                if item.rpe is not None:
                    rpe_values.append(float(item.rpe))
                if item.notes and re.search(r"\b(pain|fatigue|sore|sleep|recovery|ill|injur)\w*\b", item.notes, re.IGNORECASE):
                    recovery_notes.append(item.notes[:120])
            active_plan = self.memory_service.get_active_plan(user_id)
            planned_count, completed_count, plan_changes = _weekly_plan_signals(
                active_plan,
                week_start_date,
                week_end_date,
            )
            adherence = (
                f"{completed_count}/{planned_count} planned sessions marked complete"
                if planned_count
                else f"{logs.session_count} logged sessions (no dated active-plan baseline)"
            )
            average_rpe = round(sum(rpe_values) / len(rpe_values), 1) if rpe_values else None
            summary = (
                f"Adherence: {adherence}. Emphasis: {sport_counts or {'none': 0}}. "
                f"Workload: {logs.session_count} sessions, {exercise_count} exercises, about {total_sets} sets and "
                f"{total_minutes} minutes; average session RPE {average_rpe if average_rpe is not None else 'not reported'}. "
                f"Recovery/safety: {recovery_notes or ['no flagged note text']}. "
                f"Performance: use source session IDs for exercise-level results. Plan changes: {plan_changes or ['none recorded']}."
            )
            self.session.add(
                TrainingSummaryNote(
                    summary_id=f"wsum_{uuid4().hex[:12]}",
                    user_id=user_id,
                    week_start=week_start,
                    week_end=week_end,
                    timezone=user.timezone,
                    summary=summary,
                    source_ids_json=_dumps(source_ids),
                )
            )
        self.session.commit()
        return self.list_weekly_summaries(user_id, limit=4)

    def get_trace(self, trace_id: str) -> ProgramTraceResponse:
        trace = self.session.get(ProgramTrace, trace_id)
        if not trace:
            raise NotFoundError(f"Program trace not found: {trace_id}")
        tool_calls = self.session.exec(select(ProgramToolCall).where(ProgramToolCall.trace_id == trace_id)).all()
        calculations = self.session.exec(
            select(CalculationTrace).where(CalculationTrace.trace_id == trace_id)
        ).all()
        citations = self.session.exec(select(ProgramCitation).where(ProgramCitation.trace_id == trace_id)).all()
        flags = self.session.exec(select(ProgramSafetyFlag).where(ProgramSafetyFlag.trace_id == trace_id)).all()
        return ProgramTraceResponse(
            trace_id=trace.trace_id,
            context_id=trace.context_id,
            version_id=trace.version_id,
            context_revision=trace.context_revision,
            task_type=trace.task_type,
            sop_step=trace.sop_step,
            action=trace.action,
            details=_loads(trace.details_json, {}),
            rag_query_count=trace.rag_query_count,
            user_question_count=trace.user_question_count,
            warnings=_loads(trace.warnings_json, []),
            errors=_loads(trace.errors_json, []),
            tool_calls=[
                {"tool_call_id": item.tool_call_id, "tool_name": item.tool_name, "inputs": _loads(item.inputs_json, {}),
                 "output": _loads(item.output_json, {}), "status": item.status}
                for item in tool_calls
            ],
            calculations=[
                {"calculation_trace_id": item.calculation_trace_id, "tool_name": item.tool_name,
                 "inputs": _loads(item.inputs_json, {}), "result": _loads(item.result_json, {}),
                 "formula": item.formula, "unit": item.unit, "rounding_policy": item.rounding_policy}
                for item in calculations
            ],
            citations=[
                {"citation_id": item.citation_id, "source_id": item.source_id, "chunk_id": item.chunk_id,
                 "title": item.title, "page_or_section": item.page_or_section}
                for item in citations
            ],
            safety_flags=[
                {"safety_flag_id": item.safety_flag_id, "flag_type": item.flag_type,
                 "severity": item.severity, "explanation": item.explanation,
                 "blocks_planning": item.blocks_planning}
                for item in flags
            ],
            created_at=trace.created_at,
        )

    def _graph_state(self, context: ProgramContextRecord, route_locked: bool = False) -> ProgramGraphState:
        sections = self._sections(context)
        progress = sections.get("sop_progress", {})
        step = str(context.current_sop_step or "consultation_and_screening")
        counters = progress.get("counters", {}).get(step, {})
        return {
            "context": {**self._context_out(context).model_dump(mode="json"), "sections": sections},
            "route_locked": route_locked,
            "rag_query_count": int(counters.get("rag_queries", 0)),
            "user_question_count": int(counters.get("user_questions", 0)),
            "max_rag_queries": self.settings.program_max_rag_queries,
            "max_user_questions": self.settings.program_max_user_questions,
            "warnings": list(sections.get("safety_flags_and_budget_warnings", [])),
            "errors": [],
        }

    def _load_or_create_context(self, user_id: str, conversation_id: str) -> ProgramContextRecord:
        rows = self.session.exec(
            select(ProgramContextRecord).where(
                ProgramContextRecord.user_id == user_id,
                ProgramContextRecord.conversation_id == conversation_id,
                ProgramContextRecord.status != "closed",
            )
        ).all()
        if rows:
            return sorted(rows, key=lambda item: item.updated_at, reverse=True)[0]
        context = ProgramContextRecord(
            context_id=f"pctx_{uuid4().hex[:12]}",
            user_id=user_id,
            conversation_id=conversation_id,
            sections_json=_dumps(PROGRAM_SECTION_DEFAULTS),
        )
        self.session.add(context)
        self.session.commit()
        self.session.refresh(context)
        return context

    def _get_context(self, context_id: str) -> ProgramContextRecord:
        context = self.session.get(ProgramContextRecord, context_id)
        if not context:
            raise NotFoundError(f"ProgramContext not found: {context_id}")
        return context

    def _sections(self, context: ProgramContextRecord) -> dict[str, Any]:
        sections = _loads(context.sections_json, {})
        normalized = {
            key: sections.get(key, default.copy() if isinstance(default, (dict, list)) else default)
            for key, default in PROGRAM_SECTION_DEFAULTS.items()
        }
        memory_context = normalized.get("memory_context")
        if isinstance(memory_context, dict):
            normalized["memory_context"] = _compact_memory_context_for_program_context(memory_context)
        working = normalized.get("new_program_working_state")
        if isinstance(working, dict):
            normalized["new_program_working_state"] = _compact_calculated_plan_working_state(working)
        original_retrieved_evidence = normalized.get("retrieved_evidence")
        retrieved_evidence, sop_progress = _normalize_retrieved_evidence_section(
            original_retrieved_evidence,
            context.current_sop_step,
            normalized.get("sop_progress", {}),
        )
        if isinstance(original_retrieved_evidence, list):
            latest_items = self._latest_step_evidence_items(context.context_id, context.current_sop_step)
            if latest_items is not None:
                current_step = str(context.current_sop_step or sop_progress.get("current_step") or "")
                retrieved_evidence = {"current_step": current_step or None, "items": latest_items}
                if current_step:
                    sop_progress.setdefault("evidence_refs_by_step", {})[current_step] = _evidence_refs(latest_items)
        normalized["retrieved_evidence"] = retrieved_evidence
        normalized["sop_progress"] = sop_progress
        self._refresh_budget_warnings(normalized)
        return normalized

    def _latest_calculation_result_ids(self, context_id: str, fallback: Any) -> list[str]:
        trace = self.session.exec(
            select(ProgramTrace).where(
                ProgramTrace.context_id == context_id,
                ProgramTrace.action == "run_calculation",
            )
        ).all()
        latest = max(trace, key=lambda item: item.created_at, default=None)
        if latest is None:
            return list(dict.fromkeys(fallback if isinstance(fallback, list) else []))
        rows = self.session.exec(
            select(CalculationTrace).where(CalculationTrace.trace_id == latest.trace_id)
        ).all()
        return [item.calculation_trace_id for item in sorted(rows, key=lambda item: item.created_at)]

    def _latest_step_evidence_items(self, context_id: str, current_step: str | None) -> list[dict[str, Any]] | None:
        if not current_step:
            return None
        traces = self.session.exec(
            select(ProgramTrace).where(
                ProgramTrace.context_id == context_id,
                ProgramTrace.action == "query_rag",
                ProgramTrace.sop_step == current_step,
            )
        ).all()
        latest = max(traces, key=lambda item: item.created_at, default=None)
        if latest is None:
            return None
        calls = self.session.exec(
            select(ProgramToolCall).where(
                ProgramToolCall.trace_id == latest.trace_id,
                ProgramToolCall.tool_name == "search_knowledge",
            )
        ).all()
        if not calls:
            return None
        output = _loads(calls[-1].output_json, {})
        results = output.get("results", [])
        if not isinstance(results, list):
            return None
        return _dedupe_evidence_items(results)

    def _checkpoint(self, context: ProgramContextRecord, sections: dict[str, Any]) -> None:
        previous = _loads(context.sections_json, {})
        for section_name, value in sections.items():
            if previous.get(section_name) == value:
                continue
            self.patch_validator.validate(
                ProgramContextPatch(
                    expected_context_revision=context.context_revision,
                    target_section=section_name,
                    operation="replace",
                    value=value,
                ),
                current_revision=context.context_revision,
            )
        context.sections_json = _dumps(sections)
        context.context_revision += 1
        context.updated_at = utc_now()
        self.session.add(context)
        self.session.commit()
        self.session.refresh(context)

    def _reset_new_program_sections(self, sections: dict[str, Any]) -> None:
        budget_override = bool(sections.get("sop_progress", {}).get("context_budget_override"))
        for key in (
            "retrieved_evidence",
            "sop_progress",
            "new_program_working_state",
            "draft_artifacts",
        ):
            sections[key] = {"current_step": None, "items": []} if key == "retrieved_evidence" else {}
        if budget_override:
            sections["sop_progress"] = {"context_budget_override": True}

    def _context_budget_warnings(self, sections: dict[str, Any]) -> list[str]:
        caps = {
            "current_user_turn": 1000,
            "memory_context": 34000,
            "latest_four_week_summary_notes": 2000,
            "retrieved_evidence": 6000,
            "user_clarification_answers": 2000,
            "sop_progress": 2000,
            "new_program_working_state": 10000,
            "draft_artifacts": 5000,
            "safety_flags_and_budget_warnings": 1000,
        }
        estimates = {}
        for key, value in sections.items():
            if not value:
                continue
            if key == "memory_context" and isinstance(value, dict):
                estimates[key] = int(
                    value.get("token_estimate", max(1, len(_render_program_memory_context(value)) // 4))
                )
            else:
                estimates[key] = max(1, len(_dumps(value)) // 4)
        warnings = [
            f"ProgramContext section '{key}' is about {value} tokens, above its {caps[key]}-token cap."
            for key, value in estimates.items()
            if key in caps and value > caps[key]
        ]
        total = sum(estimates.values())
        if total > self.settings.program_max_context_tokens:
            warnings.append(
                f"ProgramContext is about {total} tokens, above its "
                f"{self.settings.program_max_context_tokens}-token limit."
            )
        return warnings

    def _refresh_budget_warnings(self, sections: dict[str, Any]) -> None:
        retained_warnings = [
            str(item)
            for item in sections.get("safety_flags_and_budget_warnings", [])
            if not _is_budget_warning(item)
        ]
        memory_warnings = []
        memory_context = sections.get("memory_context")
        if isinstance(memory_context, dict):
            memory_warnings = [
                str(item.get("message", item)) if isinstance(item, dict) else str(item)
                for item in memory_context.get("budget_warnings", [])
            ]
        sections["safety_flags_and_budget_warnings"] = list(
            dict.fromkeys(retained_warnings + memory_warnings)
        )
        context_warnings = self._context_budget_warnings(sections)
        sections["safety_flags_and_budget_warnings"] = list(
            dict.fromkeys(retained_warnings + memory_warnings + context_warnings)
        )

    def _working_key(self, task_type: str | None = None) -> str:
        return "new_program_working_state"

    def _extract_inputs(
        self,
        message: str,
        profile_text: str,
        recent_status_text: str = "",
    ) -> dict[str, Any]:
        extracted: dict[str, Any] = {}
        if self.llm_provider and hasattr(self.llm_provider, "extract_program_inputs"):
            try:
                llm_inputs = self.llm_provider.extract_program_inputs(
                    ProgramInputExtractionRequest(
                        user_message=message,
                        recent_status_text=recent_status_text,
                        profile_text=profile_text,
                    )
                )
                if isinstance(llm_inputs, dict):
                    extracted.update(llm_inputs)
            except (ConfigurationError, StructuredOutputError, ValueError, TypeError):
                extracted = {}
        if not extracted:
            extracted.update(self._extract_inputs_deterministic("", profile_text))
            extracted.update(self._extract_inputs_deterministic("", recent_status_text))
            extracted.update(self._extract_inputs_deterministic(message, ""))
        extracted.update(self._extract_lift_maxes(message))
        return extracted

    def _extract_inputs_deterministic(self, message: str, profile_text: str) -> dict[str, Any]:
        combined = f"{profile_text}\n{message}"
        lowered = combined.lower()
        result: dict[str, Any] = {}
        goals = {
            "strength": ("strength", "powerlifting"),
            "hypertrophy": ("muscle", "hypertrophy", "bodybuilding"),
            "endurance": ("endurance", "running", "run", "hyrox"),
            "general fitness": ("general fitness", "health", "fitness"),
        }
        for goal, terms in goals.items():
            if any(term in lowered for term in terms):
                result["goal"] = goal
                break
        days = re.search(r"(?:days_per_week|days per week|available_training_days|train)\D{0,12}([1-7])", lowered)
        if not days:
            days = re.search(r"\b([1-7])\s*(?:days?|times?)\s*(?:per|a)?\s*week", lowered)
        if days:
            result["days_per_week"] = int(days.group(1))
        duration = re.search(r"(?:session_duration_minutes|session duration|session|workout)\D{0,12}(\d{2,3})\s*(?:minutes?|mins?)", lowered)
        if not duration:
            duration = re.search(r"typical_session_duration_minutes\D{0,4}(\d{2,3})", lowered)
        if duration:
            result["session_duration_minutes"] = min(180, max(20, int(duration.group(1))))
        for level in ("beginner", "intermediate", "advanced"):
            if level in lowered:
                result["experience"] = level
                break
        equipment_terms = [term for term in ("full gym", "barbell", "dumbbell", "machine", "bodyweight", "home gym") if term in lowered]
        if equipment_terms:
            result["equipment"] = ", ".join(dict.fromkeys(equipment_terms))
        if re.search(r"\b(no injuries|no injury|no medical issues|healthy|cleared to train)\b", lowered):
            result["screening"] = "clear"
        elif re.search(r"\b(injury|pain|illness|medical|medication)\b", lowered):
            result["screening"] = "limited"
        named_patterns = {
            "goal": r"goal\s*[:=]\s*([^,;\n]+)",
            "experience": r"experience\s*[:=]\s*([^,;\n]+)",
            "equipment": r"equipment\s*[:=]\s*([^;\n]+)",
            "screening": r"screening\s*[:=]\s*([^,;\n]+)",
        }
        for key, pattern in named_patterns.items():
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()
        for key, pattern in {
            "days_per_week": r"days(?:_per_week| per week)?\s*[:=]\s*([1-7])",
            "session_duration_minutes": r"(?:duration|session_duration_minutes)\s*[:=]\s*(\d{2,3})",
        }.items():
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                result[key] = int(match.group(1))
        return result

    def _extract_lift_maxes(self, message: str) -> dict[str, float]:
        if not message.strip():
            return {}
        if self.llm_provider and hasattr(self.llm_provider, "extract_program_lift_maxes"):
            try:
                extracted = self.llm_provider.extract_program_lift_maxes(
                    ProgramLiftMaxExtractionRequest(user_message=message)
                )
                cleaned = _sanitize_lift_maxes(extracted)
                if cleaned:
                    return cleaned
            except (ConfigurationError, StructuredOutputError, ValueError, TypeError):
                pass
        return _extract_lift_maxes(message)

    def _red_flags(self, message: str, recent_status: str) -> list[dict[str, str]]:
        text = f"{message}\n{recent_status}".lower()
        return [
            {"flag_type": name, "severity": "high", "explanation": f"Reported {name.replace('_', ' ')}."}
            for name, pattern in RED_FLAG_PATTERNS.items()
            if re.search(pattern, text)
        ]

    def _blocked_safety_response(
        self,
        context: ProgramContextRecord,
        sections: dict[str, Any],
        flags: list[dict[str, str]],
        started: float,
    ) -> ProgramTurnResponse:
        sections["safety_flags_and_budget_warnings"] = list(
            dict.fromkeys(sections.get("safety_flags_and_budget_warnings", []) + [item["explanation"] for item in flags])
        )
        context.status = "waiting_for_safety_clearance"
        context.pending_action = "safety_clearance"
        self._checkpoint(context, sections)
        trace_id = self._trace(context, "safety_block", {"flags": flags}, started=started)
        for flag in flags:
            self.session.add(
                ProgramSafetyFlag(
                    safety_flag_id=f"psafe_{uuid4().hex[:12]}",
                    trace_id=trace_id,
                    blocks_planning=True,
                    **flag,
                )
            )
        self.session.commit()
        response = (
            "I cannot safely generate training while these red-flag symptoms are unresolved. "
            "Please seek appropriate medical evaluation or clearance before continuing."
        )
        self._assistant_message(context, response)
        return self._turn_response(context, response, trace_id, warnings=[item["explanation"] for item in flags])

    def _structured_output_error_response(
        self,
        context: ProgramContextRecord,
        exc: StructuredOutputError,
        started: float,
    ) -> ProgramTurnResponse:
        context.status = "open"
        context.pending_action = None
        self.session.add(context)
        self.session.commit()
        response = (
            "The chat model returned malformed JSON while drafting the program, so I did not save a new draft. "
            "I recorded the failure in the program trace. Please retry the same request; if it repeats, narrow "
            "the requested change for one turn."
        )
        trace_id = self._trace(
            context,
            "structured_output_error",
            {
                key: value
                for key, value in {
                    "error": str(exc),
                    "raw_excerpt": getattr(exc, "raw_excerpt", None),
                    "repair_excerpt": getattr(exc, "repair_excerpt", None),
                }.items()
                if value
            },
            errors=[str(exc)],
            started=started,
        )
        self._assistant_message(context, response)
        return self._turn_response(context, response, trace_id, warnings=[str(exc)])

    def _ask_for_missing(
        self,
        context: ProgramContextRecord,
        missing: list[str],
        started: float,
    ) -> ProgramTurnResponse:
        sections = self._sections(context)
        step = str(context.current_sop_step or "consultation_and_screening")
        progress = dict(sections.get("sop_progress", {}))
        counters = progress.setdefault("counters", {}).setdefault(step, {})
        counters["user_questions"] = int(counters.get("user_questions", 0)) + 1
        progress["pending_missing_inputs"] = missing
        sections["sop_progress"] = progress
        context.pending_action = "waiting_for_user"
        context.status = "waiting_for_user"
        self._checkpoint(context, sections)
        labels = {
            "goal": "primary training goal",
            "days_per_week": "available training days per week",
            "session_duration_minutes": "minutes available per session",
            "experience": "experience level (beginner, intermediate, or advanced)",
            "equipment": "available equipment",
            "screening": "current injuries, pain, medical limitations, or confirmation that none apply",
        }
        standard_missing = [item for item in missing if not item.startswith("load_reference:")]
        load_missing = [item.split(":", 1)[1] for item in missing if item.startswith("load_reference:")]
        requested_details = [labels[item] for item in standard_missing]
        if load_missing:
            requested_details.append(
                "load references for each weighted exercise listed in the template below"
            )
        question = "Before I design this safely, please provide: " + "; ".join(requested_details) + "."
        if load_missing:
            question += (
                "\n\nChoose `1rm` when you know a current one-rep max, or `recent_set` when you know a recent "
                "weighted set. Keep one object for each exercise and fill `weight_kg` plus `reps`; for `1rm`, "
                "use `reps: 1`.\n\n```json\n"
                + json.dumps(_load_reference_template(load_missing), indent=2)
                + "\n```"
            )
        trace_id = self._trace(context, "ask_user", {"missing_inputs": missing}, started=started)
        self._assistant_message(context, question)
        return self._turn_response(context, question, trace_id)

    def _query_evidence(self, context: ProgramContextRecord, message: str) -> None:
        sections = self._sections(context)
        step = str(context.current_sop_step or "consultation_and_screening")
        progress = dict(sections.get("sop_progress", {}))
        counters = progress.setdefault("counters", {}).setdefault(step, {})
        attempt = int(counters.get("rag_queries", 0)) + 1
        working = sections.get(self._working_key(str(context.active_task_type)), {})
        inputs = working.get("collected_inputs", {})
        query = (
            f"Evidence for SOP step '{step}' in an evidence-based {inputs.get('goal', 'general fitness')} program for "
            f"{inputs.get('experience', 'unknown')} trainee, attempt {attempt}: {message}"
        )
        started = clock.perf_counter()
        keyword, semantic, semantic_error = self.knowledge_service.search(
            query=query,
            keyword_top_k=self.settings.program_evidence_context_top_k,
            semantic_top_k=self.settings.program_evidence_context_top_k,
        )
        combined: dict[str, Any] = {}
        for item in [*keyword, *semantic]:
            existing = combined.get(item.chunk_id)
            if existing is None or item.score > existing.score:
                combined[item.chunk_id] = item
        ranked = sorted(combined.values(), key=lambda item: item.score, reverse=True)
        evidence = [asdict(item) for item in ranked[: self.settings.program_evidence_context_top_k]]
        counters["rag_queries"] = attempt
        progress.setdefault("evidence_sufficient_by_step", {})[step] = bool(evidence)
        progress.setdefault("evidence_refs_by_step", {})[step] = _evidence_refs(evidence)
        sections["sop_progress"] = progress
        sections["retrieved_evidence"] = {"current_step": step, "items": evidence}
        if not evidence and attempt >= self.settings.program_max_rag_queries:
            warning = (
                f"No sufficient approved evidence was retrieved for SOP step '{step}' after the query limit; "
                "assumptions are disclosed."
            )
            sections["safety_flags_and_budget_warnings"] = list(
                dict.fromkeys(sections["safety_flags_and_budget_warnings"] + [warning])
            )
        self._checkpoint(context, sections)
        trace_id = self._trace(
            context,
            "query_rag",
            {"query": query, "result_count": len(evidence), "semantic_error": semantic_error},
            latency_ms=int((clock.perf_counter() - started) * 1000),
        )
        self.session.add(
            ProgramToolCall(
                tool_call_id=f"ptool_{uuid4().hex[:12]}",
                trace_id=trace_id,
                tool_name="search_knowledge",
                inputs_json=_dumps({"query": query, "top_k": self.settings.program_evidence_context_top_k}),
                output_json=_dumps({"results": evidence, "semantic_error": semantic_error}),
            )
        )
        for item in evidence:
            self.session.add(
                ProgramCitation(
                    citation_id=f"pcit_{uuid4().hex[:12]}",
                    trace_id=trace_id,
                    source_id=item["source_id"],
                    chunk_id=item["chunk_id"],
                    title=item["title"],
                    page_or_section=str(item.get("page_number") or item.get("section_heading") or ""),
                )
            )
        self.session.commit()

    def _write_sop_output(self, context: ProgramContextRecord, message: str) -> None:
        self._write_generation_sop_output(context, message)

    def _write_generation_sop_output(self, context: ProgramContextRecord, message: str) -> None:
        sections = self._sections(context)
        working_key = self._working_key(str(context.active_task_type))
        working = dict(sections.get(working_key, {}))
        collected_inputs = dict(working.get("collected_inputs", {}))
        conservative_defaults = {
            "goal": "general fitness",
            "days_per_week": 2,
            "session_duration_minutes": 45,
            "experience": "beginner",
            "equipment": "bodyweight",
            "screening": "unknown_use_conservative_limits",
        }
        assumed = []
        for key, value in conservative_defaults.items():
            if collected_inputs.get(key) in (None, "", []):
                collected_inputs[key] = value
                assumed.append(f"{key}={value}")
        if assumed:
            sections["safety_flags_and_budget_warnings"].append(
                "Clarification limit reached; conservative assumptions used: " + ", ".join(assumed)
            )
        working["collected_inputs"] = collected_inputs
        task_request = str(working.get("task_request") or message)
        raw_plan = working.get("raw_plan")
        if not isinstance(raw_plan, dict) or not raw_plan:
            raw_plan = self._empty_raw_plan(task_request, collected_inputs)
            working["raw_plan"] = raw_plan

        sop = NEW_PROGRAM_SOP
        step = str(context.current_sop_step or sop[0])
        progress = dict(sections.get("sop_progress", {}))
        progress.setdefault("completed_steps", [])
        progress.setdefault("decisions", {})
        progress["active_sop"] = context.active_sop
        progress["support"] = {
            "evidence_chunk_ids": [item.get("chunk_id") for item in _all_evidence_refs(progress)],
            "memory_ids": sections.get("memory_context", {}).get("memory_ids", []),
        }
        started = clock.perf_counter()

        if step in DETERMINISTIC_GENERATION_STEPS:
            if step == "load_calculation":
                working["pending_load_clarifications"] = self._missing_weighted_load_references(
                    raw_plan,
                    collected_inputs,
                    sections.get("memory_context", {}).get("sections", {}).get("profile", ""),
                )
                if working["pending_load_clarifications"]:
                    sections[working_key] = working
                    sections["sop_progress"] = progress
                    self._checkpoint(context, sections)
                    return
                trace_id = self._trace(context, "run_calculation", {"title": raw_plan.get("title")})
                working["calculation_artifacts"] = self._materialize_plan(
                    raw_plan,
                    collected_inputs,
                    sections.get("memory_context", {}),
                    trace_id,
                    periodization=_periodization_from_sop_progress(progress, raw_plan),
                    intensity_volume_policy=_intensity_volume_policy_from_sop_progress(progress),
                )
                working.pop("calculated_plan", None)
                output = self._load_calculation_step_output(working["calculation_artifacts"])
            else:
                artifacts = working.get("calculation_artifacts")
                if not isinstance(artifacts, dict):
                    trace_id = self._trace(context, "run_calculation", {"title": raw_plan.get("title")})
                    artifacts = self._materialize_plan(
                        raw_plan,
                        collected_inputs,
                        sections.get("memory_context", {}),
                        trace_id,
                        periodization=_periodization_from_sop_progress(progress, raw_plan),
                        intensity_volume_policy=_intensity_volume_policy_from_sop_progress(progress),
                    )
                    working["calculation_artifacts"] = artifacts
                output = self._volume_calculation_step_output(artifacts)
        else:
            decision_request = ProgramSOPStepDecisionRequest(
                task_type=str(context.active_task_type),
                sop_step=step,
                user_message=task_request,
                collected_inputs=collected_inputs,
                memory_context=_render_program_memory_context_for_sop(sections.get("memory_context", {})),
                weekly_summaries=sections.get("latest_four_week_summary_notes", []),
                evidence=_current_evidence_items(sections),
                previous_decisions=dict(progress.get("decisions", {})),
                raw_plan=raw_plan,
            )
            output = self.llm_provider.generate_program_sop_step_decision(decision_request)
            updated_plan = self.llm_provider.update_program_plan(
                ProgramPlanUpdateRequest(
                    sop_step=step,
                    decision=output,
                    raw_plan=raw_plan,
                    collected_inputs=collected_inputs,
                    previous_decisions=dict(progress.get("decisions", {})),
                )
            )
            self._validate_raw_plan_structure(updated_plan, require_sessions=step in {
                "programming_exercises_and_order",
                "programming_intensity_and_volume",
                "rest_periods",
            })
            raw_plan = _strip_raw_plan_sop_metadata(updated_plan)
            working["raw_plan"] = raw_plan
            working["pending_load_clarifications"] = self._missing_weighted_load_references(
                raw_plan,
                collected_inputs,
                sections.get("memory_context", {}).get("sections", {}).get("profile", ""),
            )
            progress["rationale"] = raw_plan.get("rationale", "")

        if step == "rest_periods" and "calculation_artifacts" not in working:
            trace_id = self._trace(context, "run_calculation", {"title": raw_plan.get("title")})
            working["calculation_artifacts"] = self._materialize_plan(
                raw_plan,
                collected_inputs,
                sections.get("memory_context", {}),
                trace_id,
                periodization=_periodization_from_sop_progress(progress, raw_plan),
                intensity_volume_policy=_intensity_volume_policy_from_sop_progress(progress),
            )
            working.pop("calculated_plan", None)
        if step == "rest_periods":
            output = self._rest_periods_step_output(working.get("calculation_artifacts", {}))

        if step not in progress["completed_steps"]:
            progress["completed_steps"].append(step)
        progress["decisions"][step] = output
        step_index = sop.index(step)
        if step_index == len(sop) - 1:
            progress["sop_complete"] = True
            progress["current_step"] = step
        else:
            context.current_sop_step = sop[step_index + 1]
            progress["current_step"] = context.current_sop_step
        sections[working_key] = working
        sections["sop_progress"] = progress
        self._checkpoint(context, sections)
        self._trace(
            context,
            "write_step_output",
            {"step": step, "output": output},
            sop_step=step,
            latency_ms=int((clock.perf_counter() - started) * 1000),
        )

    def _empty_raw_plan(self, task_request: str, inputs: dict[str, Any]) -> dict[str, Any]:
        days = int(max(1, min(7, inputs.get("days_per_week", 3))))
        duration = int(max(20, min(180, inputs.get("session_duration_minutes", 60))))
        return {
            "schema": "raw_training_plan_v1",
            "title": "",
            "goal": inputs.get("goal", ""),
            "sport": "strength",
            "rationale": "",
            "task_request": task_request,
            "templates": {
                "session": {
                    "name": "",
                    "goal": "",
                    "duration_minutes": duration,
                    "exercises": [],
                },
                "exercise": {
                    "name": "",
                    "mode": "resistance",
                    "load_type": "weighted",
                    "sets": 0,
                    "reps": 0,
                    "intensity_percent": None,
                    "rpe": None,
                    "rest_seconds": None,
                    "notes": "",
                    "sets_template": [],
                },
                "set": {
                    "set_number": 1,
                    "reps": 0,
                    "load_kg": None,
                    "intensity_percent": None,
                    "rpe": None,
                    "rest_seconds": None,
                },
            },
            "sessions": [],
            "assumptions": [],
            "warnings": [],
        }

    def _validate_raw_plan_structure(self, raw_plan: dict[str, Any], require_sessions: bool = False) -> None:
        if not isinstance(raw_plan, dict):
            raise ValueError("PlanUpdater returned a non-object raw_plan.")
        templates = raw_plan.get("templates")
        if not isinstance(templates, dict):
            raise ValueError("PlanUpdater removed raw_plan.templates.")
        for name in ("session", "exercise", "set"):
            if not isinstance(templates.get(name), dict):
                raise ValueError(f"PlanUpdater removed raw_plan.templates.{name}.")
        sessions = raw_plan.get("sessions", [])
        if sessions is None:
            raw_plan["sessions"] = []
            sessions = []
        if not isinstance(sessions, list):
            raise ValueError("PlanUpdater returned raw_plan.sessions as a non-list.")
        if require_sessions and not sessions:
            raise ValueError("PlanUpdater did not create any session templates.")
        for session in sessions:
            if not isinstance(session.get("exercises", []), list):
                raise ValueError("PlanUpdater returned a session with non-list exercises.")
            for exercise in session.get("exercises", []):
                load_type = str(exercise.get("load_type") or "").strip().lower()
                if not load_type:
                    load_type = "bodyweight" if _is_bodyweight_exercise(exercise) else "weighted"
                    exercise["load_type"] = load_type
                if load_type not in {"weighted", "bodyweight"}:
                    raise ValueError(
                        f"Exercise '{exercise.get('name', '')}' has invalid load_type '{load_type}'."
                    )

    def _load_calculation_step_output(self, artifacts: dict[str, Any]) -> dict[str, Any]:
        by_exercise: dict[str, dict[str, Any]] = {}
        bodyweight: set[str] = set()
        for session in artifacts.get("calculated_sessions", []):
            for exercise in session.get("exercises", []):
                first_set = next(iter(exercise.get("sets", [])), {})
                name = str(exercise.get("name", ""))
                if first_set.get("load_kg") is None:
                    bodyweight.add(name)
                    continue
                entry = by_exercise.setdefault(
                    name,
                    {"exercise": name, "min_load_kg": None, "max_load_kg": None, "example_reps": first_set.get("reps")},
                )
                load = float(first_set.get("load_kg"))
                entry["min_load_kg"] = load if entry["min_load_kg"] is None else min(entry["min_load_kg"], load)
                entry["max_load_kg"] = load if entry["max_load_kg"] is None else max(entry["max_load_kg"], load)
        return {
            "weighted_load_summary": list(by_exercise.values()),
            "bodyweight_or_unloaded_exercises": sorted(bodyweight),
        }

    def _volume_calculation_step_output(self, artifacts: dict[str, Any]) -> dict[str, Any]:
        exercise_totals: dict[str, dict[str, Any]] = {}
        for session in artifacts.get("calculated_sessions", []):
            for exercise in session.get("exercises", []):
                name = str(exercise.get("name", ""))
                unit = str(exercise.get("volume_unit", ""))
                key = f"{name}:{unit}"
                entry = exercise_totals.setdefault(
                    key,
                    {"exercise": name, "volume_unit": unit, "total_volume": 0.0},
                )
                entry["total_volume"] += float(exercise.get("volume") or 0)
        return {
            "exercise_volume_summary": list(exercise_totals.values()),
            "weekly_volume_load_kg": artifacts.get("calculated_periodization", {}).get("weekly_volume_load_kg", {}),
        }

    def _rest_periods_step_output(self, artifacts: dict[str, Any]) -> list[dict[str, Any]]:
        rests: dict[str, dict[str, Any]] = {}
        for session in artifacts.get("calculated_sessions", []):
            for exercise in session.get("exercises", []):
                first_set = next(iter(exercise.get("sets", [])), {})
                name = str(exercise.get("name", ""))
                rests[name] = {"exercise": name, "rest_seconds": first_set.get("rest_seconds")}
        return list(rests.values())

    def _validate_raw_plan(self, raw_plan: dict[str, Any]) -> None:
        if not raw_plan.get("title") or not raw_plan.get("sessions"):
            raise ValueError("Program generator returned a plan without title or sessions.")
        if not isinstance(raw_plan["sessions"], list):
            raise ValueError("Program sessions must be a list.")
        for session in raw_plan["sessions"]:
            for exercise in session.get("exercises", []):
                load_type = str(exercise.get("load_type") or "").strip().lower()
                if not load_type:
                    load_type = "bodyweight" if _is_bodyweight_exercise(exercise) else "weighted"
                    exercise["load_type"] = load_type
                if load_type not in {"weighted", "bodyweight"}:
                    raise ValueError(
                        f"Exercise '{exercise.get('name', '')}' has invalid load_type '{load_type}'."
                    )

    def _missing_weighted_load_references(
        self,
        raw_plan: dict[str, Any],
        inputs: dict[str, Any],
        profile: str,
    ) -> list[str]:
        missing: list[str] = []
        seen: set[str] = set()
        references = inputs.get("exercise_load_references", {})
        for session in raw_plan.get("sessions", []):
            for exercise in session.get("exercises", []):
                name = str(exercise.get("name", "")).strip()
                key = _normalize_exercise_name(name)
                if not name or key in seen or _is_bodyweight_exercise(exercise):
                    continue
                seen.add(key)
                one_rm, _ = _one_rm_for_exercise(name, profile, inputs)
                if one_rm or references.get(key):
                    continue
                missing.append(name)
        return missing

    def _training_max_for_exercise(
        self,
        exercise_name: str,
        profile: str,
        inputs: dict[str, Any],
        trace_id: str,
        cache: dict[str, tuple[float | None, str]],
    ) -> tuple[float | None, str]:
        key = _normalize_exercise_name(exercise_name)
        if key in cache:
            return cache[key]
        one_rm, source = _one_rm_for_exercise(exercise_name, profile, inputs)
        if one_rm is not None:
            cache[key] = (one_rm, source)
            return cache[key]
        reference = inputs.get("exercise_load_references", {}).get(key)
        if not reference:
            cache[key] = (None, "missing_load_reference")
            return cache[key]
        weight = float(reference["weight_kg"])
        reps = int(reference.get("reps", 1))
        if reps == 1:
            cache[key] = (weight, f"user_reported_1rm:{key}")
            return cache[key]
        estimate_trace = self._calculation(
            trace_id,
            "estimate_repetition_max",
            {"weight": weight, "reps": reps, "method": "epley", "unit": "kg"},
            [f"user_reported_recent_performance:{key}"],
        )
        estimated = self._calculation_value(estimate_trace)
        cache[key] = (estimated, estimate_trace)
        return cache[key]

    def _calculate_working_plan(self, context: ProgramContextRecord) -> None:
        sections = self._sections(context)
        working_key = self._working_key(str(context.active_task_type))
        working = dict(sections.get(working_key, {}))
        raw = working["raw_plan"]
        inputs = working.get("collected_inputs", {})
        trace_id = self._trace(context, "run_calculation", {"title": raw.get("title")})
        artifacts = self._materialize_plan(
            raw,
            inputs,
            sections.get("memory_context", {}),
            trace_id,
            periodization=_periodization_from_sop_progress(sections.get("sop_progress", {}), raw),
            intensity_volume_policy=_intensity_volume_policy_from_sop_progress(sections.get("sop_progress", {})),
        )
        working["calculation_artifacts"] = artifacts
        working.pop("calculated_plan", None)
        sections[working_key] = working
        self._checkpoint(context, sections)

    def _materialize_plan(
        self,
        raw: dict[str, Any],
        inputs: dict[str, Any],
        memory_context: dict[str, Any],
        trace_id: str,
        periodization: dict[str, Any] | None = None,
        intensity_volume_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        periodization = periodization if isinstance(periodization, dict) else raw.get("periodization", {})
        weeks = int(max(1, min(52, _coerce_float(periodization.get("weeks"), 4))))
        raw_sessions = list(raw.get("sessions", []))
        days_source = len(raw_sessions) if raw_sessions else inputs.get("days_per_week", 3)
        days = int(max(1, min(7, days_source)))
        templates = raw_sessions[:days]
        if not templates:
            raise ValueError("At least one session template is required.")
        while len(templates) < days:
            templates.append(templates[len(templates) % len(raw["sessions"])])
        start_date = _next_monday(datetime.now(timezone.utc))
        profile = str(memory_context.get("sections", {}).get("profile", ""))
        intensity_volume_policy = intensity_volume_policy if isinstance(intensity_volume_policy, dict) else {}
        deload_weeks = _deload_weeks_from_periodization(periodization, weeks)
        exercise_policies = intensity_volume_policy.get("exercise_policies", {})
        if not isinstance(exercise_policies, dict):
            exercise_policies = {}
        session_exercise_policies = intensity_volume_policy.get("session_exercise_policies", {})
        if not isinstance(session_exercise_policies, dict):
            session_exercise_policies = {}
        frequency_trace = self._calculation(
            trace_id,
            "calculate_weekly_frequency",
            {"sessions": days, "weeks": 1},
            ["user_availability"],
        )
        weekly_workload: dict[str, float] = {str(week): 0.0 for week in range(1, weeks + 1)}
        sessions: list[dict[str, Any]] = []
        training_max_cache: dict[str, tuple[float | None, str]] = {}
        slot_progression_kg_cache: dict[str, float] = {}
        slot_set_count_cache: dict[str, int] = {}
        for week in range(1, weeks + 1):
            for day_index, template in enumerate(templates, start=1):
                session_id = f"psess_{uuid4().hex[:12]}"
                duration = self._calculation(
                    trace_id,
                    "calculate_expression",
                    {"expression": str(int(template.get("duration_minutes", inputs.get("session_duration_minutes", 60)))), "unit": "min"},
                    ["user_session_duration", "llm_session_template"],
                )
                exercises = []
                for exercise_index, exercise in enumerate(template.get("exercises", []), start=1):
                    exercise_id = f"pex_{uuid4().hex[:12]}"
                    mode = "resistance"
                    exercise_name = str(exercise.get("name", f"Exercise {exercise_index}"))
                    exercise_policy = _policy_for_exercise_slot(
                        day_index,
                        exercise_index,
                        exercise_name,
                        session_exercise_policies,
                        exercise_policies,
                    )
                    slot_policy_key = _exercise_slot_policy_key(day_index, exercise_index, exercise_name)
                    is_deload_week = week in deload_weeks
                    policy_sets = _policy_week_int(exercise_policy, "sets_per_week", week, None)
                    sets_count = int(max(0, min(20, policy_sets if policy_sets is not None else exercise.get("sets", 3))))
                    set_source_refs = ["programming_decision"]
                    if is_deload_week and policy_sets == 0:
                        previous_sets = slot_set_count_cache.get(slot_policy_key)
                        if previous_sets:
                            sets_count = previous_sets
                            set_source_refs.append("deload_zero_sets_replaced_with_previous_nonzero")
                    if sets_count == 0:
                        continue
                    slot_set_count_cache[slot_policy_key] = sets_count
                    set_count_trace = self._calculation(
                        trace_id, "calculate_expression", {"expression": str(sets_count), "unit": "sets"}, set_source_refs
                    )
                    sets = []
                    exercise_load = None
                    policy_reps = _policy_week_int(exercise_policy, "reps_per_set", week, None)
                    exercise_reps = int(max(1, min(100, policy_reps if policy_reps is not None else exercise.get("reps", 8))))
                    is_bodyweight = _is_bodyweight_exercise(exercise)
                    one_rm, one_rm_source = self._training_max_for_exercise(
                        exercise_name,
                        profile,
                        inputs,
                        trace_id,
                        training_max_cache,
                    )
                    if not is_bodyweight and one_rm is None:
                        raise ValueError(
                            f"Weighted exercise '{exercise.get('name', '')}' has no 1RM or recent performance record."
                        )
                    reps = exercise_reps
                    raw_intensity_value = _coerce_float(exercise.get("intensity_percent"), None)
                    weekly_estimated_progression_kg = _policy_week_float(
                        exercise_policy,
                        "weekly_estimated_progression_kg",
                        week,
                        0,
                    )
                    deload_intensity_reduction = _policy_week_float(
                        exercise_policy,
                        "deload_session_intensity_reduction_percent",
                        week,
                        0,
                    )
                    deload_reps_reduction = _policy_week_float(
                        exercise_policy,
                        "deload_session_reps_reduction_percent",
                        week,
                        0,
                    )
                    if is_deload_week and deload_reps_reduction:
                        adjusted_reps = max(1, round(exercise_reps * (1 - max(0, min(100, deload_reps_reduction)) / 100)))
                        reps = int(adjusted_reps)
                    rep_source_refs = ["programming_decision"]
                    if is_deload_week and deload_reps_reduction:
                        rep_source_refs.extend(["periodization_deload_week", "sop_intensity_volume_policy"])
                    rep_trace = self._calculation(
                        trace_id, "calculate_expression", {"expression": str(reps), "unit": "reps"}, rep_source_refs
                    )
                    rest_trace = self._calculation(
                        trace_id,
                        "calculate_rest_interval",
                        {"goal": template.get("goal", raw.get("goal", "general fitness")), "compound": exercise_index <= 2},
                        ["session_goal"],
                    )
                    policy_rpe = _policy_week_float(exercise_policy, "target_rpe", week, None)
                    rpe = float(max(1, min(10, policy_rpe if policy_rpe is not None else exercise.get("rpe", 7))))
                    rpe_trace = self._calculation(
                        trace_id, "calculate_expression", {"expression": str(rpe), "unit": "RPE"}, ["programming_decision"]
                    )
                    load_kg = None
                    intensity = raw_intensity_value
                    trace_ids = [set_count_trace, rep_trace, rest_trace, rpe_trace]
                    if not is_bodyweight and one_rm:
                        cumulative_progression_kg = slot_progression_kg_cache.get(slot_policy_key, 0.0) + float(
                            weekly_estimated_progression_kg or 0
                        )
                        slot_progression_kg_cache[slot_policy_key] = cumulative_progression_kg
                        effective_one_rm_trace = self._calculation(
                            trace_id,
                            "calculate_expression",
                            {
                                "expression": f"{one_rm} + {cumulative_progression_kg}",
                                "unit": "kg",
                            },
                            [one_rm_source, "sop_weekly_estimated_progression_kg"],
                        )
                        effective_one_rm = self._calculation_value(effective_one_rm_trace)
                        intensity_from_reps_trace = self._calculation(
                            trace_id,
                            "calculate_reps_rpe_intensity",
                            {"reps": int(reps), "target_rpe": rpe},
                            ["set_reps", "target_rpe"],
                        )
                        load_percent = self._calculation_value(intensity_from_reps_trace)
                        trace_ids.extend([effective_one_rm_trace, intensity_from_reps_trace])
                        if is_deload_week and deload_intensity_reduction:
                            deload_progressed = self._calculation(
                                trace_id,
                                "calculate_progression_increment",
                                {
                                    "current_value": load_percent,
                                    "percentage": -max(0, min(100, deload_intensity_reduction)),
                                    "unit": "%1RM",
                                },
                                ["periodization_deload_week", "sop_intensity_volume_policy"],
                            )
                            deload_row = self.session.get(CalculationTrace, deload_progressed)
                            load_percent = float(_loads(deload_row.result_json, {}).get("value", load_percent))
                            trace_ids.append(deload_progressed)
                        load_percent = min(load_percent, 100)
                        percentage_trace = self._calculation(
                            trace_id,
                            "calculate_percentage_load",
                            {"one_rm": effective_one_rm, "percentage": load_percent, "unit": "kg"},
                            [effective_one_rm_trace, intensity_from_reps_trace],
                        )
                        percentage_row = self.session.get(CalculationTrace, percentage_trace)
                        unrounded = float(_loads(percentage_row.result_json, {}).get("value", 0))
                        rounded_trace = self._calculation(
                            trace_id,
                            "round_training_load",
                            {
                                "load": unrounded,
                                "increment": self._load_rounding_increment(str(exercise.get("name", "")), inputs),
                                "unit": "kg",
                            },
                            [percentage_trace],
                        )
                        rounded_row = self.session.get(CalculationTrace, rounded_trace)
                        load_kg = float(_loads(rounded_row.result_json, {}).get("value", 0))
                        exercise_load = load_kg
                        intensity = load_percent
                        trace_ids.extend([percentage_trace, rounded_trace])
                    rest_seconds = int(self._calculation_value(rest_trace))
                    for set_number in range(1, sets_count + 1):
                        sets.append(
                            {
                                "set_id": f"pset_{uuid4().hex[:12]}",
                                "set_number": set_number,
                                "reps": reps,
                                "load_kg": load_kg,
                                "intensity_percent": intensity,
                                "rpe": rpe,
                                "rest_seconds": rest_seconds,
                                "calculation_trace_ids": list(trace_ids),
                            }
                        )
                    if exercise_load is not None:
                        volume_trace = self._calculation(
                            trace_id,
                            "calculate_volume_load",
                            {"sets": sets_count, "reps": int(reps), "load": exercise_load, "unit": "kg"},
                            [set_count_trace, "set_repetition_and_load_traces"],
                        )
                        volume_unit = "kg"
                    else:
                        volume_trace = self._calculation(
                            trace_id,
                            "calculate_expression",
                            {"expression": f"{sets_count} * {int(reps)}", "unit": "reps"},
                            [set_count_trace, "set_repetition_traces"],
                        )
                        volume_unit = "reps"
                    volume_value = self._calculation_value(volume_trace)
                    if volume_unit == "kg":
                        weekly_workload[str(week)] += volume_value
                    exercises.append(
                        {
                            "exercise_id": exercise_id,
                            "name": exercise_name,
                            "order": exercise_index,
                            "mode": mode,
                            "notes": str(exercise.get("notes", "")),
                            "volume": volume_value,
                            "volume_unit": volume_unit,
                            "calculation_trace_ids": [set_count_trace, volume_trace],
                            "sets": sets,
                        }
                    )
                sessions.append(
                    {
                        "session_id": session_id,
                        "planned_date": (start_date + timedelta(weeks=week - 1, days=day_index - 1)).isoformat(),
                        "week_number": week,
                        "day_number": day_index,
                        "session_type": str(template.get("name", f"Session {day_index}")),
                        "goal": str(template.get("goal", raw.get("goal", "general fitness"))),
                        "duration_minutes": int(self._calculation_value(duration)),
                        "duration_calculation_trace_id": duration,
                        "exercises": exercises,
                    }
                )
        return {
            "schema": "calculated_plan_delta_v1",
            "calculation_run_trace_id": trace_id,
            "calculated_sessions": sessions,
            "calculated_periodization": {
                "weeks": weeks,
                "sessions_per_week": days,
                "deload_weeks": sorted(deload_weeks),
                "exercise_policy_count": len(exercise_policies),
                "session_exercise_policy_count": len(session_exercise_policies),
                "week_policies": {
                    str(week): {"deload": week in deload_weeks}
                    for week in range(1, weeks + 1)
                },
                "frequency_calculation_trace_id": frequency_trace,
                "weekly_volume_load_kg": weekly_workload,
            },
        }

    def _load_rounding_increment(self, exercise_name: str, inputs: dict[str, Any]) -> float:
        text = f"{exercise_name} {inputs.get('equipment', '')}".lower()
        if "dumbbell" in text:
            return self.settings.program_dumbbell_load_rounding_kg
        if "machine" in text:
            return self.settings.program_machine_load_rounding_kg
        if "bodyweight" in text:
            return self.settings.program_bodyweight_load_rounding_kg
        return self.settings.program_barbell_load_rounding_kg

    def _calculation(
        self,
        trace_id: str,
        tool_name: str,
        inputs: dict[str, Any],
        source_references: list[str],
    ) -> str:
        result = self.calculators.run(tool_name, inputs, source_references)
        calculation_id = f"calc_{uuid4().hex[:12]}"
        self.session.add(
            ProgramToolCall(
                tool_call_id=result.tool_call_id,
                trace_id=trace_id,
                tool_name=tool_name,
                inputs_json=_dumps(inputs),
                output_json=result.model_dump_json(),
            )
        )
        self.session.add(
            CalculationTrace(
                calculation_trace_id=calculation_id,
                trace_id=trace_id,
                tool_name=tool_name,
                inputs_json=_dumps(inputs),
                result_json=result.model_dump_json(),
                formula=result.formula,
                unit=result.unit,
                rounding_policy=result.rounding_policy,
                source_references_json=_dumps(source_references),
            )
        )
        self.session.commit()
        return calculation_id

    def _calculation_value(self, calculation_id: str) -> float:
        row = self.session.get(CalculationTrace, calculation_id)
        return float(_loads(row.result_json, {}).get("value", 0))

    def _finalize(self, context: ProgramContextRecord, started: float) -> ProgramTurnResponse:
        sections = self._sections(context)
        working = sections[self._working_key(str(context.active_task_type))]
        plan = _build_calculated_plan_from_sections(working, sections)
        program = TrainingProgram(
            program_id=f"prog_{uuid4().hex[:12]}",
            user_id=context.user_id,
            title=str(plan["title"]),
            goal=str(plan.get("goal", "general fitness")),
            sport=str(plan.get("sport", "general_fitness")),
        )
        self.session.add(program)
        self.session.flush()
        existing_versions = self.session.exec(
            select(ProgramVersion).where(ProgramVersion.program_id == program.program_id)
        ).all()
        version_number = max((item.version_number for item in existing_versions), default=0) + 1
        version = ProgramVersion(
            version_id=f"pver_{uuid4().hex[:12]}",
            program_id=program.program_id,
            version_number=version_number,
            status="draft",
            reason="new_program",
            screening_json=_dumps(plan.get("screening", {})),
            needs_analysis_json=_dumps(plan.get("needs_analysis", {})),
            periodization_json=_dumps(plan.get("periodization", {})),
            planning_model=str(plan.get("planning_model") or plan.get("periodization", {}).get("model", "parallel")),
            rationale=str(plan.get("rationale", "")),
            assumptions_json=_dumps(_coerce_string_list(plan.get("assumptions", []))),
            warnings_json=_dumps(
                _coerce_string_list(plan.get("warnings", []))
                + _coerce_string_list(sections.get("safety_flags_and_budget_warnings", []))
            ),
            source_references_json=_dumps(
                {
                    "memory_trace_id": sections.get("memory_context", {}).get("trace_id"),
                    "memory_ids": sections.get("memory_context", {}).get("memory_ids", []),
                    "evidence": [
                        {"source_id": item.get("source_id"), "chunk_id": item.get("chunk_id")}
                        for item in _all_evidence_refs(sections.get("sop_progress", {}))
                    ],
                }
            ),
        )
        self.session.add(version)
        for session_data in plan["sessions"]:
            self._store_session(version.version_id, session_data)
        program.latest_version_id = version.version_id
        program.generation_status = version.status
        program.updated_at = utc_now()
        context.latest_program_id = program.program_id
        context.latest_version_id = version.version_id
        context.review_status = "pending"
        context.approval_status = "pending"
        context.pending_action = "waiting_for_review"
        context.status = "waiting_for_review"
        sections["draft_artifacts"] = {
            "program_id": program.program_id,
            "version_id": version.version_id,
            "version_number": version.version_number,
        }
        self.session.add(program)
        self.session.add(context)
        self.session.commit()
        self._checkpoint(context, sections)
        trace_id = self._trace(
            context,
            "finalize_output",
            {"program_id": program.program_id, "version_id": version.version_id},
            version_id=version.version_id,
            started=started,
        )
        response = (
            f"Draft {program.title} version {version.version_number} is ready for review. "
            f"It contains {len(plan['sessions'])} planned sessions. Approve, reject, or request changes."
        )
        self._assistant_message(context, response)
        return self._turn_response(context, response, trace_id)

    def _store_session(self, version_id: str, data: dict[str, Any]) -> None:
        session = ProgramSession(
            session_id=data["session_id"],
            version_id=version_id,
            planned_date=datetime.fromisoformat(data["planned_date"]),
            week_number=int(data["week_number"]),
            day_number=int(data["day_number"]),
            session_type=data["session_type"],
            goal=data["goal"],
            duration_minutes=int(data["duration_minutes"]),
            duration_calculation_trace_id=data.get("duration_calculation_trace_id"),
            exercise_order_json=_dumps([item["exercise_id"] for item in data["exercises"]]),
        )
        self.session.add(session)
        for exercise_data in data["exercises"]:
            exercise = ExercisePrescription(
                exercise_prescription_id=exercise_data["exercise_id"],
                session_id=session.session_id,
                exercise_name=exercise_data["name"],
                exercise_order=int(exercise_data["order"]),
                mode=exercise_data.get("mode", "resistance"),
                notes=exercise_data.get("notes", ""),
                volume=exercise_data.get("volume"),
                volume_unit=exercise_data.get("volume_unit"),
                calculation_trace_ids_json=_dumps(exercise_data.get("calculation_trace_ids", [])),
            )
            self.session.add(exercise)
            for set_data in exercise_data["sets"]:
                self.session.add(
                    SetPrescription(
                        set_prescription_id=set_data["set_id"],
                        exercise_prescription_id=exercise.exercise_prescription_id,
                        set_number=int(set_data["set_number"]),
                        reps=set_data.get("reps"),
                        load_kg=set_data.get("load_kg"),
                        intensity_percent=set_data.get("intensity_percent"),
                        rpe=set_data.get("rpe"),
                        rest_seconds=int(set_data["rest_seconds"]),
                        calculation_trace_ids_json=_dumps(set_data.get("calculation_trace_ids", [])),
                    )
                )

    def _session_out(self, session: ProgramSession) -> ProgramSessionOut:
        exercises = self.session.exec(
            select(ExercisePrescription).where(ExercisePrescription.session_id == session.session_id)
        ).all()
        exercise_outputs = []
        for exercise in sorted(exercises, key=lambda item: item.exercise_order):
            sets = self.session.exec(
                select(SetPrescription).where(
                    SetPrescription.exercise_prescription_id == exercise.exercise_prescription_id
                )
            ).all()
            exercise_outputs.append(
                ExercisePrescriptionOut(
                    exercise_id=exercise.exercise_prescription_id,
                    name=exercise.exercise_name,
                    order=exercise.exercise_order,
                    mode=exercise.mode,
                    notes=exercise.notes,
                    volume=exercise.volume,
                    volume_unit=exercise.volume_unit,
                    calculation_trace_ids=_loads(exercise.calculation_trace_ids_json, []),
                    sets=[
                        SetPrescriptionOut(
                            set_id=item.set_prescription_id,
                            set_number=item.set_number,
                            reps=item.reps,
                            load_kg=item.load_kg,
                            intensity_percent=item.intensity_percent,
                            rpe=item.rpe,
                            rest_seconds=item.rest_seconds,
                            calculation_trace_ids=_loads(item.calculation_trace_ids_json, []),
                        )
                        for item in sorted(sets, key=lambda row: row.set_number)
                    ],
                )
            )
        return ProgramSessionOut(
            session_id=session.session_id,
            planned_date=session.planned_date,
            week_number=session.week_number,
            day_number=session.day_number,
            session_type=session.session_type,
            goal=session.goal,
            duration_minutes=session.duration_minutes,
            duration_calculation_trace_id=session.duration_calculation_trace_id,
            exercises=exercise_outputs,
        )

    def _rationale_response(self, context: ProgramContextRecord, started: float) -> ProgramTurnResponse:
        sections = self._sections(context)
        rationale = str(sections.get("sop_progress", {}).get("rationale", "")).strip()
        if not rationale and context.latest_program_id and context.latest_version_id:
            rationale = self.get_version(context.latest_program_id, context.latest_version_id).rationale
        question = str(sections.get("current_user_turn", {}).get("content", ""))
        working_memory = str(
            sections.get("memory_context", {}).get("sections", {}).get("working_memory", "")
        )
        recent_conversation = working_memory[-6000:]
        numeric_facts = self._numeric_rationale_facts(context, question, recent_conversation)
        rationale_context = {
            "context_status": context.status,
            "active_task_type": context.active_task_type,
            "active_sop": context.active_sop,
            "current_sop_step": context.current_sop_step,
            "pending_action": context.pending_action,
            "latest_program_id": context.latest_program_id,
            "latest_version_id": context.latest_version_id,
            "review_status": context.review_status,
            "approval_status": context.approval_status,
            "sop_progress": sections.get("sop_progress", {}),
            "retrieved_evidence": sections.get("retrieved_evidence", []),
            "safety_flags_and_budget_warnings": sections.get("safety_flags_and_budget_warnings", []),
            "recent_program_conversation": recent_conversation,
            "recorded_rationale": rationale,
        }
        result = run_program_rationale_graph(
            {
                "rationale_question": question,
                "rationale_context": rationale_context,
                "numeric_rationale_facts": numeric_facts,
            },
            self.llm_provider,
        )
        response = str(result["rationale_answer"])
        trace_id = self._trace(
            context,
            "explain_plan_rationale",
            {"read_only": True, "builder": "llm", "numeric_facts_used": bool(numeric_facts)},
            started=started,
        )
        self._assistant_message(context, response)
        return self._turn_response(context, response, trace_id)

    def _numeric_rationale_facts(
        self,
        context: ProgramContextRecord,
        question: str,
        recent_conversation: str = "",
    ) -> dict[str, Any]:
        """Select a visible load and return its persisted calculation facts."""

        if not context.latest_program_id or not context.latest_version_id:
            return {}
        lowered = question.lower()
        if not any(
            term in lowered
            for term in ("weight", "load", "number", "calculate", "calculation", "intensity", "percent", "%", " kg", "rep")
        ):
            return {}
        version = self.get_version(context.latest_program_id, context.latest_version_id)
        exercise_scores = {
            exercise.exercise_id: _exercise_question_match_score(exercise.name, question)
            for planned_session in version.sessions
            for exercise in planned_session.exercises
        }
        exercise_term = _best_exercise_term(version, exercise_scores)
        if not exercise_term and recent_conversation:
            recent_scores = {
                exercise.exercise_id: _exercise_question_match_score(exercise.name, recent_conversation)
                for planned_session in version.sessions
                for exercise in planned_session.exercises
            }
            exercise_term = _best_exercise_term(version, recent_scores)
            exercise_scores = recent_scores if exercise_term else exercise_scores
        matched_exercise_ids = {exercise_id for exercise_id, score in exercise_scores.items() if score > 0}
        mentioned_reps = [
            int(value)
            for value in re.findall(r"(\d+)\s*(?:self[-\s]*weight\s*)?(?:pull[-\s]*ups?|reps?|repetitions?)", lowered)
        ]
        mentioned_rpes = [
            float(value)
            for value in re.findall(r"\brpe\s*(\d+(?:\.\d+)?)\b", lowered)
        ]
        wants_bodyweight = bool(
            re.search(r"\b(body\s*weight|bodyweight|self[-\s]*weight|unweighted)\b", lowered)
        )
        candidates = []
        for planned_session in version.sessions:
            for exercise in planned_session.exercises:
                exercise_score = exercise_scores.get(exercise.exercise_id, 0)
                if matched_exercise_ids and exercise.exercise_id not in matched_exercise_ids:
                    continue
                bodyweight = _is_bodyweight_exercise({"name": exercise.name, "notes": exercise.notes})
                for prescribed_set in exercise.sets:
                    if not matched_exercise_ids and prescribed_set.load_kg is None:
                        continue
                    candidates.append((planned_session, exercise, prescribed_set, exercise_score, bodyweight))
        if not candidates:
            return {}
        mentioned_loads = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*kg", question, re.IGNORECASE)]
        mentioned_percentages = [
            float(value)
            for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", question, re.IGNORECASE)
        ]
        if mentioned_percentages:
            candidates_with_intensity = [item for item in candidates if item[2].intensity_percent is not None]
            chosen = min(
                candidates_with_intensity or candidates,
                key=lambda item: min(
                    abs(float(item[2].intensity_percent or 0) - value)
                    for value in mentioned_percentages
                ),
            )
        elif mentioned_loads:
            candidates_with_load = [item for item in candidates if item[2].load_kg is not None]
            chosen = min(
                candidates_with_load or candidates,
                key=lambda item: min(
                    abs(float(item[2].load_kg) - value) if item[2].load_kg is not None else 1_000_000
                    for value in mentioned_loads
                ),
            )
        elif mentioned_reps or mentioned_rpes or matched_exercise_ids:
            chosen = max(
                candidates,
                key=lambda item: _prescription_question_score(
                    item[2],
                    exercise_score=item[3],
                    is_bodyweight=item[4],
                    mentioned_reps=mentioned_reps,
                    mentioned_rpes=mentioned_rpes,
                    wants_bodyweight=wants_bodyweight,
                ),
            )
        else:
            chosen = max(candidates, key=lambda item: float(item[2].load_kg or 0))
        planned_session, exercise, prescribed_set, _, is_bodyweight = chosen
        traces = [
            self.session.get(CalculationTrace, trace_id)
            for trace_id in prescribed_set.calculation_trace_ids
        ]
        traces = [trace for trace in traces if trace is not None]
        progression = next((trace for trace in traces if trace.tool_name == "calculate_progression_increment"), None)
        percentage = next((trace for trace in traces if trace.tool_name == "calculate_percentage_load"), None)
        rounding = next((trace for trace in traces if trace.tool_name == "round_training_load"), None)
        facts: dict[str, Any] = {
            "selected_prescription": {
                "exercise": exercise.name,
                "week": planned_session.week_number,
                "day": planned_session.day_number,
                "session_id": planned_session.session_id,
                "set_id": prescribed_set.set_id,
                "load_kg": prescribed_set.load_kg,
                "reps": prescribed_set.reps,
                "intensity_percent": prescribed_set.intensity_percent,
                "rpe": prescribed_set.rpe,
                "rest_seconds": prescribed_set.rest_seconds,
                "notes": exercise.notes,
                "is_bodyweight": is_bodyweight,
            },
            "selection": {
                "exercise_term": exercise_term,
                "weights_mentioned_kg": mentioned_loads,
                "intensities_mentioned_percent": mentioned_percentages,
                "reps_mentioned": mentioned_reps,
                "rpes_mentioned": mentioned_rpes,
                "rule": (
                    "closest mentioned intensity"
                    if mentioned_percentages
                    else "closest mentioned load"
                    if mentioned_loads
                    else "best exercise/reps/RPE match"
                    if mentioned_reps or mentioned_rpes or matched_exercise_ids
                    else "heaviest matching loaded set"
                ),
            },
            "calculation_chain": [
                {
                    "calculation_id": trace.calculation_trace_id,
                    "tool_name": trace.tool_name,
                    "inputs": _loads(trace.inputs_json, {}),
                    "result": _loads(trace.result_json, {}),
                    "formula": trace.formula,
                    "unit": trace.unit,
                    "rounding_policy": trace.rounding_policy,
                    "source_references": _loads(trace.source_references_json, []),
                }
                for trace in traces
            ],
            "matched_prescriptions": _matched_prescription_facts(
                candidates,
                mentioned_reps=mentioned_reps,
                mentioned_rpes=mentioned_rpes,
                wants_bodyweight=wants_bodyweight,
            ),
            "reported_bodyweight_performance": _extract_bodyweight_performance_claims(question),
        }
        for name, trace in (
            ("progression", progression),
            ("percentage_load", percentage),
            ("rounding", rounding),
        ):
            if trace:
                facts[name] = {
                    "tool_name": trace.tool_name,
                    "inputs": _loads(trace.inputs_json, {}),
                    "result": _loads(trace.result_json, {}),
                    "formula": trace.formula,
                    "unit": trace.unit,
                    "rounding_policy": trace.rounding_policy,
                    "source_references": _loads(trace.source_references_json, []),
                }
        if percentage:
            percentage_inputs = _loads(percentage.inputs_json, {})
            reported_maxes = self._extract_lift_maxes(question)
            lift_key = _lift_key_for_exercise(exercise.name)
            reported = reported_maxes.get(lift_key) if lift_key else None
            used = float(percentage_inputs.get("one_rm", 0))
            facts["one_rm_comparison"] = {
                "lift_key": lift_key,
                "saved_plan_one_rm_kg": used,
                "reported_in_current_question_kg": reported,
                "source_conflict": bool(reported and abs(reported - used) > 0.01),
            }
        return facts

    def _trace(
        self,
        context: ProgramContextRecord,
        action: str,
        details: dict[str, Any],
        version_id: str | None = None,
        sop_step: str | None = None,
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
        latency_ms: int | None = None,
        started: float | None = None,
    ) -> str:
        sections = self._sections(context)
        progress = sections.get("sop_progress", {})
        step = str(context.current_sop_step or "")
        counters = progress.get("counters", {}).get(step, {})
        trace_id = f"ptr_{uuid4().hex[:12]}"
        self.session.add(
            ProgramTrace(
                trace_id=trace_id,
                context_id=context.context_id,
                version_id=version_id,
                context_revision=context.context_revision,
                task_type=str(context.active_task_type or context.current_turn_route or "program_generation"),
                sop_step=sop_step if sop_step is not None else context.current_sop_step,
                action=action,
                details_json=_dumps(details),
                rag_query_count=int(counters.get("rag_queries", 0)),
                user_question_count=int(counters.get("user_questions", 0)),
                warnings_json=_dumps(warnings or []),
                errors_json=_dumps(errors or []),
                latency_ms=latency_ms if latency_ms is not None else int((clock.perf_counter() - started) * 1000) if started else 0,
            )
        )
        self.session.commit()
        return trace_id

    def _assistant_message(self, context: ProgramContextRecord, content: str) -> None:
        self.memory_service.append_message(
            conversation_id=context.conversation_id,
            role="assistant",
            content=content,
            source_block="program_generator",
        )

    def _assistant_message_if_context(self, context: ProgramContextRecord | None, content: str) -> None:
        if context:
            self._assistant_message(context, content)

    def _record_review_message(self, context: ProgramContextRecord | None, content: str) -> None:
        if context:
            self.memory_service.append_message(
                conversation_id=context.conversation_id,
                role="user",
                content=content,
                source_block="program_generator",
            )

    def _turn_response(
        self,
        context: ProgramContextRecord,
        response: str,
        trace_id: str,
        warnings: list[str] | None = None,
    ) -> ProgramTurnResponse:
        sections = self._sections(context)
        draft = sections.get("draft_artifacts", {})
        return ProgramTurnResponse(
            conversation_id=context.conversation_id,
            context_id=context.context_id,
            context_revision=context.context_revision,
            route=str(context.current_turn_route),
            status=context.status,
            response=response,
            pending_action=context.pending_action,
            draft_program_id=draft.get("program_id"),
            draft_version_id=draft.get("version_id"),
            trace_id=trace_id,
            warnings=warnings or list(sections.get("safety_flags_and_budget_warnings", [])),
        )

    def _context_out(self, context: ProgramContextRecord) -> ProgramContextResponse:
        return ProgramContextResponse(
            context_id=context.context_id,
            parent_context_id=context.parent_context_id,
            user_id=context.user_id,
            conversation_id=context.conversation_id,
            context_revision=context.context_revision,
            status=context.status,
            active_task_type=context.active_task_type,
            current_turn_route=context.current_turn_route,
            active_sop=context.active_sop,
            current_sop_step=context.current_sop_step,
            pending_action=context.pending_action,
            latest_program_id=context.latest_program_id,
            latest_version_id=context.latest_version_id,
            review_status=context.review_status,
            approval_status=context.approval_status,
            sections=self._sections(context),
            created_at=context.created_at,
            updated_at=context.updated_at,
        )

    def _context_for_version(self, version_id: str) -> ProgramContextRecord | None:
        return self.session.exec(
            select(ProgramContextRecord).where(ProgramContextRecord.latest_version_id == version_id)
        ).first()

    def _summary_out(self, item: TrainingSummaryNote) -> TrainingSummaryNoteOut:
        return TrainingSummaryNoteOut(
            summary_id=item.summary_id,
            user_id=item.user_id,
            week_start=item.week_start,
            week_end=item.week_end,
            timezone=item.timezone,
            summary=item.summary,
            source_ids=_loads(item.source_ids_json, []),
            created_at=item.created_at,
        )


def _active_plan_payload(memory_context: dict[str, Any]) -> dict[str, Any] | None:
    text = str(memory_context.get("sections", {}).get("active_plan_progress", "")).strip()
    if not text:
        return None
    json_start = text.find("{")
    if json_start < 0:
        return None
    try:
        return json.loads(text[json_start:])
    except json.JSONDecodeError:
        return None


def _compact_memory_context_for_program_context(memory_context: dict[str, Any]) -> dict[str, Any]:
    """Store a compact ProgramContext copy without changing the Memory block contract."""

    compacted = dict(memory_context)
    compacted.pop("rendered_context", None)
    compacted.pop("debug", None)
    memory_sections = compacted.get("sections")
    if isinstance(memory_sections, dict):
        sections = dict(memory_sections)
        active_plan = sections.get("active_plan_progress")
        if isinstance(active_plan, str):
            sections["active_plan_progress"] = _strip_plan_calculation_trace_ids(active_plan)
        compacted["sections"] = sections
    return compacted


def _render_program_memory_context(memory_context: dict[str, Any]) -> str:
    rendered_context = memory_context.get("rendered_context")
    if isinstance(rendered_context, str) and rendered_context.strip():
        return _strip_active_plan_calculation_trace_ids(rendered_context)
    memory_sections = memory_context.get("sections")
    if not isinstance(memory_sections, dict):
        return ""
    return render_sections(
        {
            str(name): _stringify_memory_context_section(value)
            for name, value in memory_sections.items()
        }
    )


def _render_program_memory_context_for_sop(memory_context: dict[str, Any]) -> str:
    """Render only the MemoryContext sections needed by SOPStepExecutor."""

    memory_sections = memory_context.get("sections")
    if not isinstance(memory_sections, dict):
        return ""
    allowed_sections = (
        "rolling_summary",
        "working_memory",
        "profile",
        "recent_status",
        "active_plan_summary",
        "latest_four_week_summary_notes",
    )
    return render_sections(
        {
            name: _stringify_memory_context_section(memory_sections.get(name))
            for name in allowed_sections
            if memory_sections.get(name) not in (None, "", [], {})
        }
    )


def _stringify_memory_context_section(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value in (None, "", [], {}):
        return ""
    return _dumps(value)


def _compact_calculated_plan_working_state(working: dict[str, Any]) -> dict[str, Any]:
    compacted = dict(working)
    compacted.pop("step_outputs", None)
    if "calculated_plan" not in compacted or compacted.get("calculation_artifacts"):
        return compacted
    calculated_plan = compacted.get("calculated_plan")
    if not isinstance(calculated_plan, dict):
        return compacted
    compacted["calculation_artifacts"] = _calculation_artifacts_from_calculated_plan(
        calculated_plan,
        compacted.get("raw_plan", {}),
    )
    compacted.pop("calculated_plan", None)
    return compacted


def _calculation_artifacts_from_calculated_plan(
    calculated_plan: dict[str, Any],
    raw_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_periodization = (raw_plan or {}).get("periodization", {})
    calculated_periodization = calculated_plan.get("periodization", {})
    return {
        "schema": "calculated_plan_delta_v1",
        "calculated_sessions": calculated_plan.get("sessions", []),
        "calculated_periodization": {
            key: value
            for key, value in calculated_periodization.items()
            if not isinstance(raw_periodization, dict) or raw_periodization.get(key) != value
        },
    }


RAW_PLAN_SOP_METADATA_KEYS = {
    "screening",
    "needs_analysis",
    "periodization",
    "planning_model",
    "changed_elements",
    "unchanged_elements",
}


def _strip_raw_plan_sop_metadata(raw_plan: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(raw_plan or {})
    for key in RAW_PLAN_SOP_METADATA_KEYS:
        cleaned.pop(key, None)
    return cleaned


def _build_calculated_plan_from_sections(working: dict[str, Any], sections: dict[str, Any]) -> dict[str, Any]:
    if isinstance(working.get("calculated_plan"), dict):
        raw_plan = dict(working["calculated_plan"])
    else:
        raw_plan = _strip_raw_plan_sop_metadata(dict(working.get("raw_plan") or {}))
    artifacts = working.get("calculation_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Calculated plan is missing calculation artifacts.")
    calculated_sessions = artifacts.get("calculated_sessions")
    if not isinstance(calculated_sessions, list):
        raise ValueError("Calculation artifacts are missing calculated_sessions.")
    calculated_periodization = artifacts.get("calculated_periodization", {})
    if not isinstance(calculated_periodization, dict):
        raise ValueError("Calculation artifacts have invalid calculated_periodization.")
    metadata = _program_metadata_from_sop_progress(sections.get("sop_progress", {}), working, raw_plan)
    periodization = {
        **metadata["periodization"],
        **calculated_periodization,
    }
    return {
        **raw_plan,
        "screening": metadata["screening"],
        "needs_analysis": metadata["needs_analysis"],
        "periodization": periodization,
        "planning_model": metadata["planning_model"],
        "rationale": metadata["rationale"],
        "assumptions": metadata["assumptions"],
        "warnings": metadata["warnings"],
        "sessions": calculated_sessions,
    }


def _program_metadata_from_sop_progress(
    sop_progress: dict[str, Any],
    working: dict[str, Any],
    raw_plan: dict[str, Any],
) -> dict[str, Any]:
    legacy_raw = working.get("raw_plan", {}) if isinstance(working.get("raw_plan"), dict) else {}
    legacy_metadata = working.get("plan_metadata", {}) if isinstance(working.get("plan_metadata"), dict) else {}
    screening_decision = _sop_decision_payload(sop_progress, "consultation_and_screening")
    needs_decision = _sop_decision_payload(sop_progress, "needs_analysis")
    periodization_decision = _periodization_from_sop_progress(sop_progress, legacy_metadata or legacy_raw)
    planning_decision = _sop_decision_payload(sop_progress, "planning_model")
    planning_model = planning_decision.get("planning_model", planning_decision) if isinstance(planning_decision, dict) else {}
    if not isinstance(planning_model, dict):
        planning_model = {"model": str(planning_model)}
    if planning_model.get("model") and not periodization_decision.get("model"):
        periodization_decision["model"] = planning_model.get("model")
    screening = screening_decision.get("screening", screening_decision) if isinstance(screening_decision, dict) else {}
    needs_analysis = needs_decision.get("needs_analysis", needs_decision) if isinstance(needs_decision, dict) else {}
    return {
        "screening": (
            screening
            if isinstance(screening, dict) and screening
            else legacy_metadata.get("screening", legacy_raw.get("screening", {}))
        ),
        "needs_analysis": (
            needs_analysis
            if isinstance(needs_analysis, dict) and needs_analysis
            else legacy_metadata.get("needs_analysis", legacy_raw.get("needs_analysis", {}))
        ),
        "periodization": periodization_decision,
        "planning_model": str(
            planning_model.get("model")
            or legacy_metadata.get("planning_model")
            or periodization_decision.get("model")
            or "parallel"
        ),
        "rationale": str(raw_plan.get("rationale") or sop_progress.get("rationale") or ""),
        "assumptions": _coerce_string_list(raw_plan.get("assumptions", [])),
        "warnings": _coerce_string_list(raw_plan.get("warnings", [])),
    }


def _metadata_from_legacy_raw_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    periodization = raw_plan.get("periodization", {}) if isinstance(raw_plan.get("periodization"), dict) else {}
    planning_model = raw_plan.get("planning_model")
    if isinstance(planning_model, dict):
        planning_model = planning_model.get("model")
    return {
        "screening": raw_plan.get("screening", {}) if isinstance(raw_plan.get("screening"), dict) else {},
        "needs_analysis": raw_plan.get("needs_analysis", {}) if isinstance(raw_plan.get("needs_analysis"), dict) else {},
        "periodization": periodization,
        "planning_model": str(planning_model or periodization.get("model") or "parallel"),
    }


def _periodization_from_sop_progress(sop_progress: dict[str, Any], raw_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_plan = raw_plan if isinstance(raw_plan, dict) else {}
    decision = _sop_decision_payload(sop_progress, "periodization")
    periodization = decision.get("periodization", decision) if isinstance(decision, dict) else {}
    if not isinstance(periodization, dict) or not periodization:
        periodization = raw_plan.get("periodization", {}) if isinstance(raw_plan.get("periodization"), dict) else {}
    if not periodization:
        periodization = {"weeks": 4, "macrocycle": "four_week_block", "phases": []}
    return dict(periodization)


def _merge_context_defaults(collected: dict[str, Any], defaults: dict[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in collected or collected.get(key) in (None, "", []):
            collected[key] = value


def _intensity_volume_policy_from_sop_progress(sop_progress: dict[str, Any]) -> dict[str, Any]:
    decision = _sop_decision_payload(sop_progress, "programming_intensity_and_volume")
    payload = decision.get("programming_intensity_and_volume", decision) if isinstance(decision, dict) else {}
    policies: dict[str, dict[str, Any]] = {}
    session_policies: dict[str, dict[str, Any]] = {}
    for item in _iter_dicts(payload):
        name = _first_present(item, "lift", "exercise", "exercise_name", "name")
        if not name:
            continue
        policy = {
            "name": str(name),
            "weekly_estimated_progression_kg": _first_present(
                item,
                "weekly_estimated_progression_kg",
                "weekly_progression_kg",
            ),
            "deload_session_intensity_reduction_percent": _first_present(
                item,
                "deload_session_intensity_reduction_percent",
                "deload_session_weight_reduction_percent",
            ),
            "deload_session_reps_reduction_percent": item.get("deload_session_reps_reduction_percent"),
            "sets_per_week": item.get("sets_per_week"),
            "reps_per_set": item.get("reps_per_set"),
            "target_rpe": item.get("target_rpe"),
            "slot_role": _first_present(item, "slot_role", "role", "session_role"),
        }
        session_index = _coerce_float(
            _first_present(item, "session_index", "session_day", "day_index", "day_number", "day"),
            None,
        )
        exercise_order = _coerce_float(_first_present(item, "exercise_order", "exercise_index", "order"), None)
        if session_index is not None and exercise_order is not None:
            session_policies[_exercise_slot_policy_key(int(session_index), int(exercise_order), str(name))] = policy
        if any(_policy_value_present(value) for key, value in policy.items() if key != "name"):
            policies[_normalize_exercise_name(str(name))] = policy
    return {"exercise_policies": policies, "session_exercise_policies": session_policies}


def _policy_value_present(value: Any) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, list):
        return any(item not in (None, "") for item in value)
    return True


def _policy_week_float(policy: dict[str, Any], key: str, week: int, default: float | None = 0) -> float | None:
    return _coerce_float(_policy_week_value(policy, key, week, default), default)


def _policy_week_int(policy: dict[str, Any], key: str, week: int, default: int | None = 0) -> int | None:
    value = _policy_week_float(policy, key, week, None if default is None else float(default))
    if value is None:
        return default
    return int(round(value))


def _policy_week_value(policy: dict[str, Any], key: str, week: int, default: Any = None) -> Any:
    value = policy.get(key)
    if isinstance(value, list):
        index = week - 1
        if 0 <= index < len(value):
            return value[index]
        return default
    if value in (None, ""):
        return default
    return value


def _policy_has_weekly_values(policy: dict[str, Any], key: str) -> bool:
    return isinstance(policy.get(key), list)


def _exercise_slot_policy_key(day_index: int, exercise_index: int, exercise_name: str) -> str:
    return f"{int(day_index)}:{int(exercise_index)}:{_normalize_exercise_name(exercise_name)}"


def _policy_for_exercise_slot(
    day_index: int,
    exercise_index: int,
    exercise_name: str,
    session_exercise_policies: dict[str, Any],
    exercise_policies: dict[str, Any],
) -> dict[str, Any]:
    exact_slot = session_exercise_policies.get(_exercise_slot_policy_key(day_index, exercise_index, exercise_name))
    if isinstance(exact_slot, dict):
        return exact_slot
    normalized = _normalize_exercise_name(exercise_name)
    for key, value in session_exercise_policies.items():
        if not isinstance(value, dict):
            continue
        parts = str(key).split(":", 2)
        if len(parts) == 3 and parts[0] == str(day_index) and parts[2] == normalized:
            return value
    return _policy_for_exercise(exercise_name, exercise_policies)


def _deload_weeks_from_periodization(periodization: dict[str, Any], total_weeks: int) -> set[int]:
    weeks: set[int] = set()
    for key in (
        "deload_week_numbers",
        "deload_week_number",
    ):
        weeks.update(_parse_week_numbers(periodization.get(key), total_weeks))

    cursor = 1
    mesocycles = periodization.get("mesocycles", [])
    if isinstance(mesocycles, list):
        for mesocycle in mesocycles:
            if not isinstance(mesocycle, dict):
                continue
            duration = int(max(1, _coerce_float(_first_present(mesocycle, "weeks", "duration_weeks"), 1)))
            if _contains_deload_text(mesocycle):
                weeks.update(range(cursor, min(total_weeks, cursor + duration - 1) + 1))
            cursor += duration

    phases = periodization.get("phases", [])
    if isinstance(phases, list):
        for phase in phases:
            if isinstance(phase, dict) and _contains_deload_text(phase):
                weeks.update(_parse_week_numbers(phase.get("weeks"), total_weeks))
    return {week for week in weeks if 1 <= week <= total_weeks}


def _policy_for_exercise(exercise_name: str, exercise_policies: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_exercise_name(exercise_name)
    exact = exercise_policies.get(normalized)
    if isinstance(exact, dict):
        return exact
    for key, value in exercise_policies.items():
        if not isinstance(value, dict):
            continue
        key_text = str(key)
        if key_text and (key_text in normalized or normalized in key_text):
            return value
    return {}


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _coerce_float(value: Any, default: float | None = 0) -> float | None:
    if value in (None, ""):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:-|to)\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if range_match:
        return (float(range_match.group(1)) + float(range_match.group(2))) / 2
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        return float(match.group(0))
    return default


def _parse_week_numbers(value: Any, total_weeks: int) -> set[int]:
    weeks: set[int] = set()
    if value in (None, ""):
        return weeks
    if isinstance(value, (int, float)):
        week = int(value)
        return {week} if 1 <= week <= total_weeks else set()
    if isinstance(value, list):
        for item in value:
            weeks.update(_parse_week_numbers(item, total_weeks))
        return weeks
    if isinstance(value, dict):
        for item in value.values():
            weeks.update(_parse_week_numbers(item, total_weeks))
        return weeks
    text = str(value)
    for start, end in re.findall(r"(\d+)\s*(?:-|to)\s*(\d+)", text, flags=re.IGNORECASE):
        low, high = sorted((int(start), int(end)))
        weeks.update(range(low, high + 1))
    for item in re.findall(r"\d+", text):
        weeks.add(int(item))
    return {week for week in weeks if 1 <= week <= total_weeks}


def _contains_deload_text(value: Any) -> bool:
    if isinstance(value, dict):
        text = " ".join(str(child) for child in value.values())
    elif isinstance(value, list):
        text = " ".join(str(child) for child in value)
    else:
        text = str(value)
    lowered = text.lower()
    return any(term in lowered for term in ("deload", "resensitization", "resensitisation", "fatigue reduction"))


def _sop_decision_payload(sop_progress: dict[str, Any], step: str) -> dict[str, Any]:
    decisions = sop_progress.get("decisions", {}) if isinstance(sop_progress, dict) else {}
    output = decisions.get(step, {}) if isinstance(decisions, dict) else {}
    if not isinstance(output, dict):
        return {}
    decision = output.get("decision", output)
    return decision if isinstance(decision, dict) else {}


def _normalize_retrieved_evidence_section(
    retrieved_evidence: Any,
    current_step: str | None,
    sop_progress: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    progress = dict(sop_progress) if isinstance(sop_progress, dict) else {}
    raw_refs_by_step = progress.get("evidence_refs_by_step", {})
    if not isinstance(raw_refs_by_step, dict):
        raw_refs_by_step = {}
    refs_by_step = {
        str(step): list(refs) if isinstance(refs, list) else []
        for step, refs in raw_refs_by_step.items()
    }
    step = str(current_step or progress.get("current_step") or "")
    if isinstance(retrieved_evidence, list):
        current_items: list[dict[str, Any]] = []
        current_chunk_ids: set[str] = set()
        for item in retrieved_evidence:
            if not isinstance(item, dict):
                continue
            item_steps = item.get("sop_steps")
            if isinstance(item_steps, list):
                for item_step in item_steps:
                    refs_by_step.setdefault(str(item_step), [])
                    _append_unique_ref(refs_by_step[str(item_step)], item)
            if step and (not isinstance(item_steps, list) or step in item_steps):
                chunk_id = str(item.get("chunk_id") or "")
                if chunk_id not in current_chunk_ids:
                    current_items.append(_evidence_item_without_sop_steps(item))
                    current_chunk_ids.add(chunk_id)
        progress["evidence_refs_by_step"] = refs_by_step
        return {"current_step": step or None, "items": current_items}, progress
    if isinstance(retrieved_evidence, dict):
        packet_step = str(retrieved_evidence.get("current_step") or step or "")
        items = _dedupe_evidence_items(retrieved_evidence.get("items", []))
        if packet_step and items:
            refs_by_step.setdefault(packet_step, [])
            for item in items:
                _append_unique_ref(refs_by_step[packet_step], item)
            progress["evidence_refs_by_step"] = refs_by_step
        return {"current_step": packet_step or None, "items": items}, progress
    return {"current_step": step or None, "items": []}, progress


def _current_evidence_items(sections: dict[str, Any]) -> list[dict[str, Any]]:
    retrieved = sections.get("retrieved_evidence", {})
    if isinstance(retrieved, dict):
        return _dedupe_evidence_items(retrieved.get("items", []))
    if isinstance(retrieved, list):
        return _dedupe_evidence_items(retrieved)
    return []


def _coerce_string_list(value: Any) -> list[str]:
    items: list[str] = []

    def add(item: Any) -> None:
        if item in (None, ""):
            return
        if isinstance(item, str):
            text = item.strip()
            if text:
                items.append(text)
            return
        if isinstance(item, list):
            for child in item:
                add(child)
            return
        if isinstance(item, dict):
            text = _dumps(item)
        else:
            text = str(item)
        if text:
            items.append(text)

    add(value)
    return items


def _dedupe_evidence_items(items: Any) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    if not isinstance(items, list):
        return deduped
    for item in items:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "")
        if chunk_id in seen_chunk_ids:
            continue
        deduped.append(_evidence_item_without_sop_steps(item))
        seen_chunk_ids.add(chunk_id)
    return deduped


def _evidence_refs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for item in items:
        _append_unique_ref(refs, item)
    return refs


def _all_evidence_refs(sop_progress: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    refs_by_step = sop_progress.get("evidence_refs_by_step", {}) if isinstance(sop_progress, dict) else {}
    if not isinstance(refs_by_step, dict):
        return refs
    for step_refs in refs_by_step.values():
        if not isinstance(step_refs, list):
            continue
        for item in step_refs:
            if isinstance(item, dict):
                _append_unique_ref(refs, item)
    return refs


def _append_unique_ref(refs: list[dict[str, Any]], item: dict[str, Any]) -> None:
    ref = {"source_id": item.get("source_id"), "chunk_id": item.get("chunk_id")}
    if not ref["chunk_id"]:
        return
    if ref not in refs:
        refs.append(ref)


def _evidence_item_without_sop_steps(item: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(item)
    cleaned.pop("sop_steps", None)
    return cleaned


def _count_logged_sets(exercises: list[dict[str, Any]], sets_summary: dict[str, Any]) -> int:
    def count(value: Any) -> int:
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            for key in ("sets", "set_list", "items"):
                if key in value:
                    nested = value[key]
                    return int(nested) if isinstance(nested, (int, float)) else count(nested)
            return sum(count(item) for item in value.values())
        return 0

    exercise_sets = sum(count(item) for item in exercises)
    return exercise_sets or count(sets_summary)


def _weekly_plan_signals(active_plan: Any, week_start: date, week_end: date) -> tuple[int, int, list[str]]:
    if not active_plan:
        return 0, 0, []
    planned_ids: list[str] = []
    for session in active_plan.planned_sessions:
        raw_date = session.get("planned_date")
        if not raw_date:
            continue
        try:
            planned_date = datetime.fromisoformat(str(raw_date)).date()
        except ValueError:
            continue
        if week_start <= planned_date <= week_end:
            planned_ids.append(str(session.get("session_id", "")))
    progress = active_plan.plan_progress.get("sessions", {})
    completed = sum(
        1
        for session_id in planned_ids
        if progress.get(session_id, {}).get("status") in {"complete", "completed", "done"}
    )
    change_types: list[str] = []
    for event in active_plan.progress_events:
        raw_created_at = event.get("created_at")
        if not raw_created_at:
            continue
        try:
            event_date = datetime.fromisoformat(str(raw_created_at)).date()
        except ValueError:
            continue
        if week_start <= event_date <= week_end:
            change_types.append(str(event.get("event_type", "progress_update")))
    return len(planned_ids), completed, list(dict.fromkeys(change_types))


def _extract_lift_maxes(message: str) -> dict[str, float]:
    values: dict[str, float] = {}
    lift_aliases = (
        ("bench press", "bench_press_1rm_kg"),
        ("bench", "bench_press_1rm_kg"),
        ("squat", "squat_1rm_kg"),
        ("deadlift", "deadlift_1rm_kg"),
        ("overhead press", "overhead_press_1rm_kg"),
    )
    exact_keys: set[str] = set()
    for _, key in lift_aliases:
        named = re.search(rf"{re.escape(key)}\s*[:=]\s*(\d+(?:\.\d+)?)", message, re.IGNORECASE)
        if named:
            values[key] = float(named.group(1))
            exact_keys.add(key)

    sbd = re.search(
        r"\bSBD\b\D{0,12}(\d+(?:\.\d+)?)\s*kg?\D{0,12}(\d+(?:\.\d+)?)\s*kg?\D{0,12}(\d+(?:\.\d+)?)\s*kg?",
        message,
        re.IGNORECASE,
    )
    if sbd:
        for key, group_index in (
            ("squat_1rm_kg", 1),
            ("bench_press_1rm_kg", 2),
            ("deadlift_1rm_kg", 3),
        ):
            values.setdefault(key, float(sbd.group(group_index)))

    max_marker = r"(?:1\s*rm|one\s*rep\s*max|max|\bpr\b|personal\s+record)"
    for lift_name, key in lift_aliases:
        if key in exact_keys:
            continue
        for lift_match in re.finditer(rf"\b{re.escape(lift_name)}\b", message, re.IGNORECASE):
            segment = message[lift_match.start() : lift_match.start() + 100]
            segment = re.split(r"[\n.;]", segment, maxsplit=1)[0]
            direct = re.search(
                rf"\b{re.escape(lift_name)}\b\s*{max_marker}\s*[:=]?\s*(\d+(?:\.\d+)?)\s*kg",
                segment,
                re.IGNORECASE,
            )
            marked = re.search(
                rf"\b{re.escape(lift_name)}\b[^\n.;]{{0,50}}?{max_marker}[^\n.;]{{0,25}}?(\d+(?:\.\d+)?)\s*kg",
                segment,
                re.IGNORECASE,
            )
            match = direct or marked
            if match:
                values[key] = float(match.group(1))
                break

    for lift_name, key in lift_aliases:
        if key in exact_keys:
            continue
        natural = re.search(
            rf"{max_marker}[^\n.;]{{0,35}}?\b{re.escape(lift_name)}\b[^\n.;]{{0,35}}?(\d+(?:\.\d+)?)\s*kg",
            message,
            re.IGNORECASE,
        )
        if natural:
            values[key] = float(natural.group(1))
    return values


def _sanitize_lift_maxes(value: Any) -> dict[str, float]:
    allowed = {
        "squat_1rm_kg",
        "bench_press_1rm_kg",
        "deadlift_1rm_kg",
        "overhead_press_1rm_kg",
    }
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key, raw in value.items():
        if key not in allowed:
            continue
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            continue
        if 1 <= parsed <= 500:
            cleaned[key] = parsed
    return cleaned


def _lift_key_for_exercise(exercise_name: str) -> str | None:
    normalized = exercise_name.lower()
    for term, key in (
        ("bench", "bench_press_1rm_kg"),
        ("squat", "squat_1rm_kg"),
        ("deadlift", "deadlift_1rm_kg"),
        ("overhead", "overhead_press_1rm_kg"),
    ):
        if term in normalized:
            return key
    return None


def _best_exercise_term(version: ProgramVersionResponse, exercise_scores: dict[str, int]) -> str | None:
    best_name = None
    best_score = 0
    for planned_session in version.sessions:
        for exercise in planned_session.exercises:
            score = exercise_scores.get(exercise.exercise_id, 0)
            if score > best_score:
                best_name = exercise.name
                best_score = score
    return best_name.lower() if best_name and best_score > 0 else None


def _exercise_question_match_score(exercise_name: str, question: str) -> int:
    lowered = question.lower()
    score = 0
    for alias in _exercise_aliases(exercise_name):
        if alias and alias in lowered:
            score = max(score, 100 + len(alias))
    name_tokens = set(_match_tokens(exercise_name))
    question_tokens = set(_match_tokens(question))
    if name_tokens and name_tokens <= question_tokens:
        score = max(score, 60 + len(name_tokens) * 5)
    elif name_tokens:
        overlap = name_tokens & question_tokens
        if overlap and len(overlap) >= min(2, len(name_tokens)):
            score = max(score, 20 + len(overlap) * 5)
        distinctive_overlap = overlap & {"bench", "squat", "deadlift", "overhead", "pull"}
        if distinctive_overlap:
            score = max(score, 45 + len(distinctive_overlap) * 5)
    return score


def _exercise_aliases(exercise_name: str) -> list[str]:
    base = exercise_name.lower().strip()
    aliases = {base, base.replace("-", " "), base.replace("-", ""), _normalize_exercise_name(base).replace("_", " ")}
    if "pull" in base and "up" in base:
        aliases.update({"pull-up", "pull up", "pullup", "pull-ups", "pull ups", "pullups"})
        if "paused" in base:
            aliases.update({"paused pull-up", "paused pull up", "paused pullup", "paused self-weight pull-up"})
        if "weighted" in base:
            aliases.update({"weighted pull-up", "weighted pull up", "weighted pullup"})
    return sorted(aliases, key=len, reverse=True)


def _match_tokens(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[a-z0-9]+", text.lower().replace("-", " ")):
        if token in {"self", "weight", "body", "bodyweight", "the", "your", "plan", "sets", "set"}:
            continue
        if token == "ups":
            token = "up"
        elif token == "pullups":
            token = "pullup"
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]
        if token == "pullup":
            tokens.extend(["pull", "up"])
        else:
            tokens.append(token)
    return tokens


def _prescription_question_score(
    prescribed_set: SetPrescriptionOut,
    exercise_score: int,
    is_bodyweight: bool,
    mentioned_reps: list[int],
    mentioned_rpes: list[float],
    wants_bodyweight: bool,
) -> float:
    score = float(exercise_score)
    if mentioned_reps and prescribed_set.reps is not None:
        score += max(0.0, 30.0 - min(abs(prescribed_set.reps - reps) for reps in mentioned_reps) * 6.0)
    if mentioned_rpes and prescribed_set.rpe is not None:
        score += max(0.0, 20.0 - min(abs(float(prescribed_set.rpe) - rpe) for rpe in mentioned_rpes) * 8.0)
    if wants_bodyweight and is_bodyweight and prescribed_set.load_kg is None:
        score += 25.0
    if prescribed_set.load_kg is not None:
        score += 2.0
    return score


def _matched_prescription_facts(
    candidates: list[tuple[Any, Any, SetPrescriptionOut, int, bool]],
    mentioned_reps: list[int],
    mentioned_rpes: list[float],
    wants_bodyweight: bool,
) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda item: _prescription_question_score(
            item[2],
            exercise_score=item[3],
            is_bodyweight=item[4],
            mentioned_reps=mentioned_reps,
            mentioned_rpes=mentioned_rpes,
            wants_bodyweight=wants_bodyweight,
        ),
        reverse=True,
    )
    facts: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, float | None, float | None]] = set()
    mentioned_reps_set = set(mentioned_reps)
    for planned_session, exercise, prescribed_set, _, is_bodyweight in ranked:
        if mentioned_reps_set and prescribed_set.reps not in mentioned_reps_set:
            continue
        if not mentioned_reps_set and mentioned_rpes and prescribed_set.rpe is not None:
            if all(abs(float(prescribed_set.rpe) - rpe) > 0.01 for rpe in mentioned_rpes):
                continue
        key = (exercise.name, prescribed_set.reps, prescribed_set.rpe, prescribed_set.load_kg)
        if key in seen:
            continue
        seen.add(key)
        facts.append(
            {
                "exercise": exercise.name,
                "week": planned_session.week_number,
                "day": planned_session.day_number,
                "session_id": planned_session.session_id,
                "set_id": prescribed_set.set_id,
                "reps": prescribed_set.reps,
                "load_kg": prescribed_set.load_kg,
                "intensity_percent": prescribed_set.intensity_percent,
                "rpe": prescribed_set.rpe,
                "rest_seconds": prescribed_set.rest_seconds,
                "notes": exercise.notes,
                "is_bodyweight": is_bodyweight,
                "calculation_trace_ids": prescribed_set.calculation_trace_ids,
            }
        )
        if len(facts) >= 8:
            break
    return facts


def _extract_bodyweight_performance_claims(question: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for match in re.finditer(
        r"(?:more than|over|at least|>=|>)?\s*(\d+)\s*(?:self[-\s]*weight|body\s*weight|bodyweight)?\s*pull[-\s]*ups?",
        question,
        flags=re.IGNORECASE,
    ):
        claims.append({"exercise": "pull-up", "reps": int(match.group(1)), "text": match.group(0).strip()})
    return claims


def _strip_active_plan_calculation_trace_ids(rendered_context: str) -> str:
    marker = "## Active Plan Progress"
    section_start = rendered_context.find(marker)
    if section_start < 0:
        return rendered_context
    section_end = rendered_context.find("\n\n## ", section_start + len(marker))
    if section_end < 0:
        section_end = len(rendered_context)
    original_section = rendered_context[section_start:section_end]
    cleaned_section = _strip_plan_calculation_trace_ids(original_section)
    return rendered_context[:section_start] + cleaned_section + rendered_context[section_end:]


def _is_budget_warning(value: Any) -> bool:
    text = str(value)
    return text.startswith(("ProgramContext section ", "ProgramContext is about ", "Memory section ", "Memory context is about "))


def _strip_plan_calculation_trace_ids(plan_text: str) -> str:
    json_start = plan_text.find("{")
    if json_start < 0:
        return plan_text
    prefix = plan_text[:json_start]
    try:
        payload = json.loads(plan_text[json_start:])
    except (json.JSONDecodeError, TypeError):
        return plan_text
    if isinstance(payload, dict) and payload.get("format") != "compact_active_plan_v1":
        return prefix + json.dumps(
            compact_active_plan_payload(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    plan_snapshot = payload.get("plan_snapshot") if isinstance(payload, dict) else None
    source_references = plan_snapshot.get("source_references") if isinstance(plan_snapshot, dict) else None
    if not isinstance(source_references, dict) or "calculation_trace_ids" not in source_references:
        return plan_text
    source_references = dict(source_references)
    source_references.pop("calculation_trace_ids", None)
    plan_snapshot = dict(plan_snapshot)
    plan_snapshot["source_references"] = source_references
    payload = dict(payload)
    payload["plan_snapshot"] = plan_snapshot
    return prefix + _dumps(payload)


def _one_rm_for_exercise(
    exercise_name: str,
    profile: str,
    task_inputs: dict[str, Any] | None = None,
) -> tuple[float | None, str]:
    normalized = exercise_name.lower()
    keys = {
        "bench": "bench_press_1rm_kg",
        "squat": "squat_1rm_kg",
        "deadlift": "deadlift_1rm_kg",
        "overhead": "overhead_press_1rm_kg",
    }
    for term, key in keys.items():
        if term not in normalized:
            continue
        if task_inputs and task_inputs.get(key) not in (None, ""):
            return float(task_inputs[key]), "current_user_turn_1rm"
        match = re.search(rf"{re.escape(key)}:\s*([0-9]+(?:\.[0-9]+)?)", profile)
        if match:
            return float(match.group(1)), "profile_1rm"
    return None, "missing_1rm"


def _normalize_exercise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _is_bodyweight_exercise(exercise: dict[str, Any]) -> bool:
    load_type = str(exercise.get("load_type", "")).strip().lower()
    if load_type == "weighted":
        return False
    if load_type == "bodyweight":
        return True
    text = f"{exercise.get('name', '')} {exercise.get('notes', '')}".lower()
    if "weighted" in text or "external load" in text:
        return False
    return any(
        term in text
        for term in (
            "bodyweight",
            "body weight",
            "pull-up",
            "pull up",
            "pullup",
            "push-up",
            "push up",
            "pushup",
            "plank",
            "archer",
            "inverted row",
        )
    )


def _load_reference_template(exercise_names: list[str]) -> dict[str, Any]:
    return {
        "load_references": [
            {
                "exercise_name": exercise_name,
                "type": "recent_set",
                "weight_kg": None,
                "reps": None,
            }
            for exercise_name in exercise_names
        ]
    }


def _extract_exercise_load_references(
    message: str,
    exercise_names: list[str],
) -> dict[str, dict[str, Any]]:
    references = _extract_structured_load_references_from_message(message, exercise_names)
    for exercise_name in exercise_names:
        key = _normalize_exercise_name(exercise_name)
        if key in references:
            continue
        segment = _exercise_message_segment(message, exercise_name, exercise_names)
        if not segment:
            continue
        performance_match = re.search(
            r"(\d+(?:\.\d+)?)\s*kg\s*(?:x|for)\s*(\d+)\s*(?:reps?)?",
            segment,
            re.IGNORECASE,
        )
        if performance_match:
            references[key] = {
                "exercise_name": exercise_name,
                "weight_kg": float(performance_match.group(1)),
                "reps": int(performance_match.group(2)),
                "source": "current_user_turn_recent_performance",
            }
            continue
        one_rm_match = re.search(
            r"(?:\b1\s*rm\b|\bone\s*rep\s*max\b|\bpr\b|\bmax\b)\D{0,20}(\d+(?:\.\d+)?)\s*kg",
            segment,
            re.IGNORECASE,
        )
        if not one_rm_match:
            one_rm_match = re.search(
                r"(\d+(?:\.\d+)?)\s*kg\D{0,20}(?:\b1\s*rm\b|\bone\s*rep\s*max\b|\bpr\b|\bmax\b)",
                segment,
                re.IGNORECASE,
            )
        if one_rm_match:
            references[key] = {
                "exercise_name": exercise_name,
                "weight_kg": float(one_rm_match.group(1)),
                "reps": 1,
                "source": "current_user_turn_1rm",
            }
    return references


def _extract_structured_load_references_from_message(
    message: str,
    exercise_names: list[str],
) -> dict[str, dict[str, Any]]:
    candidates: list[Any] = []
    for block in re.findall(r"```(?:json)?\s*(.*?)```", message, flags=re.IGNORECASE | re.DOTALL):
        candidates.append(_loads(block.strip(), None))
    stripped = message.strip()
    candidates.append(_loads(stripped, None))
    if "{" in stripped and "}" in stripped:
        candidates.append(_loads(stripped[stripped.find("{") : stripped.rfind("}") + 1], None))
    for candidate in candidates:
        references = _normalize_structured_load_references(_load_reference_items(candidate), exercise_names)
        if references:
            return references
    return {}


def _load_reference_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        raw_items = payload.get("load_references", payload.get("references", payload.get("exercise_load_references")))
        if raw_items is None and {"exercise_name", "weight_kg"}.issubset(payload):
            raw_items = [payload]
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = None
    if not isinstance(raw_items, list):
        return []
    return [item for item in raw_items if isinstance(item, dict)]


def _normalize_structured_load_references(
    items: list[dict[str, Any]],
    exercise_names: list[str],
) -> dict[str, dict[str, Any]]:
    allowed = {_normalize_exercise_name(name): name for name in exercise_names}
    references: dict[str, dict[str, Any]] = {}
    for item in items:
        key = _normalize_exercise_name(str(item.get("exercise_name") or item.get("exercise") or item.get("name") or ""))
        if key not in allowed:
            continue
        try:
            weight = float(item.get("weight_kg"))
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        raw_type = str(item.get("type") or item.get("reference_type") or "").strip().lower().replace("-", "_").replace(" ", "_")
        try:
            reps = int(item.get("reps")) if item.get("reps") not in (None, "") else None
        except (TypeError, ValueError):
            reps = None
        if raw_type in {"1rm", "1_rm", "one_rm", "one_rep_max", "max", "pr"}:
            reps = 1
            source = "current_user_turn_1rm"
        elif raw_type in {"recent_set", "recent", "working_set", "set"}:
            if reps is None or reps < 1:
                continue
            source = "current_user_turn_recent_performance"
        else:
            if reps is None:
                continue
            source = "current_user_turn_1rm" if reps == 1 else "current_user_turn_recent_performance"
        references[key] = {
            "exercise_name": allowed[key],
            "weight_kg": weight,
            "reps": reps,
            "source": source,
        }
    return references


def _exercise_aliases(exercise_name: str) -> list[str]:
    aliases = [exercise_name]
    normalized = exercise_name.lower()
    if "overhead press" in normalized:
        aliases.append("OHP")
    if "back squat" in normalized:
        aliases.append("squat")
    return sorted(dict.fromkeys(aliases), key=len, reverse=True)


def _exercise_message_segment(message: str, exercise_name: str, exercise_names: list[str]) -> str:
    start = None
    end_of_name = None
    for alias in _exercise_aliases(exercise_name):
        match = re.search(re.escape(alias), message, re.IGNORECASE)
        if match and (start is None or match.start() < start):
            start = match.start()
            end_of_name = match.end()
    if start is None or end_of_name is None:
        return ""
    end = min(len(message), start + 180)
    for other_name in exercise_names:
        if _normalize_exercise_name(other_name) == _normalize_exercise_name(exercise_name):
            continue
        for alias in _exercise_aliases(other_name):
            match = re.search(re.escape(alias), message[end_of_name:], re.IGNORECASE)
            if match:
                end = min(end, end_of_name + match.start())
    return message[start:end]


def _next_monday(now: datetime) -> datetime:
    days = (7 - now.weekday()) % 7
    if days == 0:
        days = 7
    next_date = (now + timedelta(days=days)).date()
    return datetime.combine(next_date, time(hour=9), tzinfo=timezone.utc)


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default
