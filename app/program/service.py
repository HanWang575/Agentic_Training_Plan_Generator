"""Readable orchestration service for program generation and adaptation."""

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
from app.core.llm_provider import OpenAIProvider, ProgramPlanRequest
from app.knowledge.service import KnowledgeService
from app.memory.schemas import PlanActivationRequest
from app.memory.service import MemoryService
from app.program.calculators import CalculatorService
from app.program.graph import run_program_generator_graph
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
    ProgramAdaptation,
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
    "programming_decisions",
    "exercise_selection",
    "frequency",
    "exercise_order",
    "load_and_repetitions",
    "volume",
    "rest_periods",
    "final_review",
]

ADAPTATION_SOP = [
    "classify_trigger",
    "load_active_baseline",
    "protect_goals_and_constraints",
    "analyze_future_effects",
    "choose_smallest_coherent_change",
    "recalculate_affected_numbers",
    "create_reviewable_revision",
]

PROGRAM_SECTION_DEFAULTS: dict[str, Any] = {
    "current_user_turn": {},
    "memory_context": {},
    "latest_four_week_summary_notes": [],
    "retrieved_evidence": [],
    "user_clarification_answers": [],
    "sop_progress": {},
    "new_program_working_state": {},
    "revision_adaptation_working_state": {},
    "draft_artifacts": {},
    "source_and_trace_references": {},
    "calculation_results": [],
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
        sections["memory_context"] = memory_context.model_dump(mode="json")
        summary_response = self.list_weekly_summaries(request.user_id, limit=4)
        sections["latest_four_week_summary_notes"] = [item.model_dump(mode="json") for item in summary_response.notes]
        self._inherit_active_program_context(context, sections)
        warnings = [str(item.get("message", item)) for item in memory_context.budget_warnings]
        warnings.extend(self._context_budget_warnings(sections))
        sections["safety_flags_and_budget_warnings"] = list(
            dict.fromkeys(sections.get("safety_flags_and_budget_warnings", []) + warnings)
        )
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
                progress["context_budget_override"] = True
                sections["sop_progress"] = progress
                context.pending_action = None
        budget_exceeded = any(item.startswith("ProgramContext") for item in warnings)
        budget_override = bool(sections.get("sop_progress", {}).get("context_budget_override"))
        if budget_exceeded and not budget_override:
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
        self._checkpoint(context, sections)

        initial_state = self._graph_state(context)
        routed = run_program_generator_graph(initial_state)
        route = routed.get("route", "new_program")
        task_type = routed.get("task_type", "program_generation")
        context.current_turn_route = route
        context.active_task_type = task_type

        if route == "explain_plan_rationale":
            return self._rationale_response(context, started)

        if route == "new_program":
            self._reset_new_program_sections(sections)
            context.active_sop = "new_program_design"
            context.current_sop_step = NEW_PROGRAM_SOP[0]
        elif route == "revise_current_plan":
            self._start_adaptation_sop(sections)
            context.active_sop = "current_plan_revision_adaptation"
            context.current_sop_step = ADAPTATION_SOP[0]
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
            raw_request=request.message,
        )
        self.session.add(request_row)

        working_key = self._working_key(task_type)
        working = dict(sections.get(working_key, {}))
        collected = dict(working.get("collected_inputs", {}))
        collected.update(self._extract_inputs(request.message, memory_context.sections.get("profile", "")))
        if task_type == "program_adaptation":
            collected = {**self._inputs_from_active_plan(_active_plan_payload(sections.get("memory_context", {}))), **collected}
        if route in {"new_program", "revise_current_plan"}:
            working["task_request"] = request.message
        working["collected_inputs"] = collected
        sections[working_key] = working
        request_row.goal = str(collected.get("goal", "")) or None
        request_row.constraints_json = _dumps(
            {key: collected.get(key) for key in ("days_per_week", "session_duration_minutes", "equipment", "screening")}
        )

        red_flags = self._red_flags(request.message, memory_context.sections.get("recent_status", ""))
        if red_flags:
            return self._blocked_safety_response(context, sections, red_flags, started)

        self._checkpoint(context, sections)
        max_actions = self.settings.program_max_sop_steps_per_run * (
            self.settings.program_max_rag_queries + self.settings.program_max_user_questions + 3
        )
        for _ in range(max_actions):
            state = run_program_generator_graph(self._graph_state(context, route_locked=True))
            action = state.get("next_action", "ask_user")
            if action == "ask_user":
                return self._ask_for_missing(context, state.get("missing_inputs", []), started)
            if action == "query_rag":
                self._query_evidence(context, request.message)
                continue
            if action == "write_step_output":
                self._write_sop_output(context, request.message)
                continue
            if action == "run_calculation":
                self._calculate_working_plan(context)
                continue
            if action == "finalize_output":
                return self._finalize(context, started)
        raise RuntimeError("Program workflow exceeded its bounded SOP action limit.")

    def get_context(self, context_id: str) -> ProgramContextResponse:
        return self._context_out(self._get_context(context_id))

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
            assumptions=_loads(version.assumptions_json, []),
            warnings=_loads(version.warnings_json, []),
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
            adaptation = self._adaptation_for_version(version_id)
            if adaptation:
                adaptation.approval_status = "rejected"
                self.session.add(adaptation)
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
            adaptation = self._adaptation_for_version(version_id)
            if adaptation:
                adaptation.approval_status = "activation_failed"
                self.session.add(adaptation)
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
        adaptation = self._adaptation_for_version(version_id)
        if adaptation:
            adaptation.approval_status = "approved"
            self.session.add(adaptation)
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

    def _inherit_active_program_context(
        self,
        context: ProgramContextRecord,
        sections: dict[str, Any],
    ) -> None:
        """Link a new conversation context to the active version's audit context."""

        active_plan = _active_plan_payload(sections.get("memory_context", {}))
        if not active_plan:
            return
        snapshot = active_plan.get("plan_snapshot", {})
        program_id = str(snapshot.get("program_id") or active_plan.get("plan_id") or "")
        version_id = str(active_plan.get("version_id") or snapshot.get("version_id") or "")
        if not program_id or not version_id:
            return
        if not context.latest_program_id:
            context.latest_program_id = program_id
        if not context.latest_version_id:
            context.latest_version_id = version_id
        prior_contexts = self.session.exec(
            select(ProgramContextRecord).where(
                ProgramContextRecord.user_id == context.user_id,
                ProgramContextRecord.latest_version_id == version_id,
                ProgramContextRecord.context_id != context.context_id,
            )
        ).all()
        if not prior_contexts:
            return
        prior = sorted(prior_contexts, key=lambda item: item.updated_at, reverse=True)[0]
        context.parent_context_id = prior.context_id
        prior_sections = self._sections(prior)
        for key in (
            "retrieved_evidence",
            "sop_progress",
            "new_program_working_state",
            "revision_adaptation_working_state",
            "draft_artifacts",
            "source_and_trace_references",
            "calculation_results",
        ):
            if not sections.get(key) and prior_sections.get(key):
                sections[key] = prior_sections[key]

    def _inputs_from_active_plan(self, active_plan: dict[str, Any] | None) -> dict[str, Any]:
        if not active_plan:
            return {}
        snapshot = active_plan.get("plan_snapshot", {})
        sessions = active_plan.get("planned_sessions", [])
        needs = snapshot.get("needs_analysis", {})
        screening = snapshot.get("screening", {})
        first_session = sessions[0] if sessions else {}
        week_one_days = {
            item.get("day_number")
            for item in sessions
            if int(item.get("week_number", 1)) == 1 and item.get("day_number") is not None
        }
        values = {
            "goal": snapshot.get("goal"),
            "days_per_week": len(week_one_days) or None,
            "session_duration_minutes": first_session.get("duration_minutes"),
            "experience": needs.get("experience"),
            "equipment": needs.get("equipment"),
            "screening": screening.get("status"),
        }
        return {key: value for key, value in values.items() if value not in (None, "", [])}

    def _get_context(self, context_id: str) -> ProgramContextRecord:
        context = self.session.get(ProgramContextRecord, context_id)
        if not context:
            raise NotFoundError(f"ProgramContext not found: {context_id}")
        return context

    def _sections(self, context: ProgramContextRecord) -> dict[str, Any]:
        sections = _loads(context.sections_json, {})
        return {key: sections.get(key, default.copy() if isinstance(default, (dict, list)) else default)
                for key, default in PROGRAM_SECTION_DEFAULTS.items()}

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
        for key in (
            "retrieved_evidence",
            "sop_progress",
            "new_program_working_state",
            "revision_adaptation_working_state",
            "draft_artifacts",
            "source_and_trace_references",
            "calculation_results",
        ):
            sections[key] = [] if key in {"retrieved_evidence", "calculation_results"} else {}

    def _start_adaptation_sop(self, sections: dict[str, Any]) -> None:
        previous = dict(sections.get("sop_progress", {}))
        history = list(previous.get("sop_history", []))
        if previous.get("completed_steps"):
            history.append(
                {
                    "active_sop": previous.get("active_sop"),
                    "completed_steps": previous.get("completed_steps", []),
                    "rationale": previous.get("rationale", ""),
                }
            )
        sections["sop_progress"] = {"sop_history": history[-3:]}
        sections["revision_adaptation_working_state"] = {}

    def _context_budget_warnings(self, sections: dict[str, Any]) -> list[str]:
        caps = {
            "current_user_turn": 1000,
            "memory_context": 34000,
            "latest_four_week_summary_notes": 2000,
            "retrieved_evidence": 6000,
            "user_clarification_answers": 2000,
            "sop_progress": 2000,
            "new_program_working_state": 10000,
            "revision_adaptation_working_state": 10000,
            "draft_artifacts": 5000,
            "source_and_trace_references": 5000,
            "calculation_results": 3000,
            "safety_flags_and_budget_warnings": 1000,
        }
        estimates = {}
        for key, value in sections.items():
            if not value:
                continue
            if key == "memory_context" and isinstance(value, dict):
                estimates[key] = int(value.get("token_estimate", max(1, len(str(value.get("rendered_context", ""))) // 4)))
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

    def _working_key(self, task_type: str) -> str:
        return "revision_adaptation_working_state" if task_type == "program_adaptation" else "new_program_working_state"

    def _extract_inputs(self, message: str, profile_text: str) -> dict[str, Any]:
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
            "I cannot safely generate or adapt training while these red-flag symptoms are unresolved. "
            "Please seek appropriate medical evaluation or clearance before continuing."
        )
        self._assistant_message(context, response)
        return self._turn_response(context, response, trace_id, warnings=[item["explanation"] for item in flags])

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
        question = "Before I design this safely, please provide: " + "; ".join(labels[item] for item in missing) + "."
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
        evidence = [{**asdict(item), "sop_steps": [step]} for item in ranked[: self.settings.program_evidence_context_top_k]]
        counters["rag_queries"] = attempt
        progress.setdefault("evidence_sufficient_by_step", {})[step] = bool(evidence)
        sections["sop_progress"] = progress
        stored_evidence = list(sections.get("retrieved_evidence", []))
        by_chunk_id = {item.get("chunk_id"): item for item in stored_evidence}
        for item in evidence:
            existing = by_chunk_id.get(item.get("chunk_id"))
            if existing:
                existing["sop_steps"] = list(dict.fromkeys(existing.get("sop_steps", []) + [step]))
            else:
                stored_evidence.append(item)
                by_chunk_id[item.get("chunk_id")] = item
        sections["retrieved_evidence"] = stored_evidence
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
        sections = self._sections(context)
        working_key = self._working_key(str(context.active_task_type))
        working = dict(sections.get(working_key, {}))
        active_plan = _active_plan_payload(sections.get("memory_context", {}))
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
        started = clock.perf_counter()
        raw_plan = working.get("raw_plan")
        if not raw_plan:
            task_request = str(working.get("task_request") or message)
            request = ProgramPlanRequest(
                task_type=str(context.active_task_type),
                user_message=task_request,
                collected_inputs=collected_inputs,
                memory_context=str(sections.get("memory_context", {}).get("rendered_context", "")),
                weekly_summaries=sections.get("latest_four_week_summary_notes", []),
                evidence=sections.get("retrieved_evidence", []),
                active_plan=active_plan,
            )
            raw_plan = self.llm_provider.generate_program_plan(request)
            self._validate_raw_plan(raw_plan)
            working["raw_plan"] = raw_plan
            if context.active_task_type == "program_adaptation":
                working["revision_trigger"] = task_request
                working["trigger_classification"] = _classify_adaptation_trigger(task_request)
                active_snapshot = (active_plan or {}).get("plan_snapshot", {})
                active_goal = active_snapshot.get("goal")
                working["protected_goals"] = [active_goal] if active_goal else []
                working["changed_elements"] = raw_plan.get("changed_elements", [])
                working["unchanged_elements"] = raw_plan.get("unchanged_elements", [])
        sop = ADAPTATION_SOP if context.active_task_type == "program_adaptation" else NEW_PROGRAM_SOP
        progress = dict(sections.get("sop_progress", {}))
        progress.setdefault("completed_steps", [])
        progress.setdefault("decisions", {})
        progress["active_sop"] = context.active_sop
        progress["rationale"] = raw_plan.get("rationale", "")
        progress["support"] = {
            "evidence_chunk_ids": [item.get("chunk_id") for item in sections.get("retrieved_evidence", [])],
            "memory_ids": sections.get("memory_context", {}).get("memory_ids", []),
        }
        working.setdefault("step_outputs", {})
        step = str(context.current_sop_step or sop[0])
        output = self._sop_step_output(step, raw_plan, working.get("collected_inputs", {}), active_plan, working)
        working["step_outputs"][step] = output
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

    def _validate_raw_plan(self, raw_plan: dict[str, Any]) -> None:
        if not raw_plan.get("title") or not raw_plan.get("sessions"):
            raise ValueError("Program generator returned a plan without title or sessions.")
        if not isinstance(raw_plan["sessions"], list):
            raise ValueError("Program sessions must be a list.")

    def _sop_step_output(
        self,
        step: str,
        raw_plan: dict[str, Any],
        inputs: dict[str, Any],
        active_plan: dict[str, Any] | None,
        working: dict[str, Any],
    ) -> Any:
        """Select the recorded decision for one ordered SOP step."""

        sessions = raw_plan.get("sessions", [])
        exercises = [item for session in sessions for item in session.get("exercises", [])]
        outputs = {
            "consultation_and_screening": raw_plan.get("screening", {"status": inputs.get("screening")}),
            "needs_analysis": raw_plan.get("needs_analysis", inputs),
            "periodization": raw_plan.get("periodization", {}),
            "planning_model": raw_plan.get("periodization", {}).get("model", "parallel"),
            "programming_decisions": [{"name": item.get("name"), "goal": item.get("goal")} for item in sessions],
            "exercise_selection": [item.get("name") for item in exercises],
            "frequency": inputs.get("days_per_week"),
            "exercise_order": [[item.get("name") for item in session.get("exercises", [])] for session in sessions],
            "load_and_repetitions": [
                {"exercise": item.get("name"), "reps": item.get("reps"), "intensity_percent": item.get("intensity_percent")}
                for item in exercises
            ],
            "volume": [{"exercise": item.get("name"), "sets": item.get("sets"), "reps": item.get("reps")} for item in exercises],
            "rest_periods": [{"exercise": item.get("name"), "rest_seconds": item.get("rest_seconds")} for item in exercises],
            "final_review": {"assumptions": raw_plan.get("assumptions", []), "warnings": raw_plan.get("warnings", [])},
            "classify_trigger": working.get("trigger_classification", {}),
            "load_active_baseline": {"plan_id": (active_plan or {}).get("plan_id"), "version_id": (active_plan or {}).get("version_id")},
            "protect_goals_and_constraints": raw_plan.get("unchanged_elements", []),
            "analyze_future_effects": raw_plan.get("rationale", ""),
            "choose_smallest_coherent_change": raw_plan.get("changed_elements", []),
            "recalculate_affected_numbers": [item.get("name") for item in exercises],
            "create_reviewable_revision": {"title": raw_plan.get("title"), "warnings": raw_plan.get("warnings", [])},
        }
        return outputs.get(step, {})

    def _calculate_working_plan(self, context: ProgramContextRecord) -> None:
        sections = self._sections(context)
        working_key = self._working_key(str(context.active_task_type))
        working = dict(sections.get(working_key, {}))
        raw = working["raw_plan"]
        inputs = working.get("collected_inputs", {})
        trace_id = self._trace(context, "run_calculation", {"title": raw.get("title")})
        calculated = self._materialize_plan(raw, inputs, sections.get("memory_context", {}), trace_id)
        working["calculated_plan"] = calculated
        sections[working_key] = working
        calculation_ids = [item.calculation_trace_id for item in self.session.exec(
            select(CalculationTrace).where(CalculationTrace.trace_id == trace_id)
        ).all()]
        sections["calculation_results"] = list(
            dict.fromkeys(sections.get("calculation_results", []) + calculation_ids)
        )
        self._checkpoint(context, sections)

    def _materialize_plan(
        self,
        raw: dict[str, Any],
        inputs: dict[str, Any],
        memory_context: dict[str, Any],
        trace_id: str,
    ) -> dict[str, Any]:
        weeks = int(max(1, min(12, raw.get("periodization", {}).get("weeks", 4))))
        days = int(max(1, min(7, inputs.get("days_per_week", len(raw.get("sessions", [])) or 3))))
        templates = list(raw.get("sessions", []))[:days]
        if not templates:
            raise ValueError("At least one session template is required.")
        while len(templates) < days:
            templates.append(templates[len(templates) % len(raw["sessions"])])
        start_date = _next_monday(datetime.now(timezone.utc))
        profile = str(memory_context.get("sections", {}).get("profile", ""))
        frequency_trace = self._calculation(
            trace_id,
            "calculate_weekly_frequency",
            {"sessions": days, "weeks": 1},
            ["user_availability"],
        )
        weekly_workload: dict[str, float] = {str(week): 0.0 for week in range(1, weeks + 1)}
        sessions: list[dict[str, Any]] = []
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
                    mode = str(exercise.get("mode", "resistance"))
                    sets_count = int(max(1, min(10, exercise.get("sets", 3))))
                    set_count_trace = self._calculation(
                        trace_id, "calculate_expression", {"expression": str(sets_count), "unit": "sets"}, ["programming_decision"]
                    )
                    sets = []
                    exercise_load = None
                    is_aerobic = mode == "aerobic"
                    exercise_reps = None if is_aerobic else int(max(1, min(100, exercise.get("reps", 8))))
                    for set_number in range(1, sets_count + 1):
                        reps = exercise_reps
                        rep_trace = None if reps is None else self._calculation(
                            trace_id, "calculate_expression", {"expression": str(reps), "unit": "reps"}, ["programming_decision"]
                        )
                        raw_intensity = exercise.get("intensity_percent")
                        intensity = None if raw_intensity is None and is_aerobic else float(
                            max(1, min(100, raw_intensity if raw_intensity is not None else 70))
                        )
                        intensity_trace = None if intensity is None else self._calculation(
                            trace_id,
                            "calculate_expression",
                            {"expression": str(intensity), "unit": "%max" if is_aerobic else "%1RM"},
                            ["programming_decision"],
                        )
                        rest_trace = self._calculation(
                            trace_id,
                            "calculate_rest_interval",
                            {"goal": template.get("goal", raw.get("goal", "general fitness")), "compound": exercise_index <= 2},
                            ["session_goal"],
                        )
                        rpe = float(max(1, min(10, exercise.get("rpe", 7))))
                        rpe_trace = self._calculation(
                            trace_id, "calculate_expression", {"expression": str(rpe), "unit": "RPE"}, ["programming_decision"]
                        )
                        load_kg = None
                        trace_ids = [item for item in (set_count_trace, rep_trace, intensity_trace, rest_trace, rpe_trace) if item]
                        duration_seconds = None
                        distance_meters = None
                        pace_seconds_per_km = None
                        if is_aerobic:
                            proposed_duration = int(
                                exercise.get("duration_seconds")
                                or max(
                                    60,
                                    int(
                                        template.get("duration_minutes", 30)
                                        * 60
                                        / max(1, len(template.get("exercises", [])) * sets_count)
                                    ),
                                )
                            )
                            raw_pace = int(exercise.get("pace_seconds_per_km") or 360)
                            pace_increment = self.settings.program_pace_rounding_seconds_per_km
                            proposed_pace = int(round(raw_pace / pace_increment) * pace_increment)
                            duration_trace = self._calculation(
                                trace_id,
                                "calculate_expression",
                                {"expression": str(proposed_duration), "unit": "s"},
                                ["programming_decision", "session_duration"],
                            )
                            pace_trace = self._calculation(
                                trace_id,
                                "calculate_expression",
                                {"expression": str(proposed_pace), "unit": "s/km"},
                                ["programming_decision"],
                            )
                            distance_trace = self._calculation(
                                trace_id,
                                "calculate_distance_pace_time",
                                {"duration_seconds": proposed_duration, "pace_seconds_per_km": proposed_pace},
                                [duration_trace, pace_trace],
                            )
                            duration_seconds = int(self._calculation_value(duration_trace))
                            pace_seconds_per_km = int(self._calculation_value(pace_trace))
                            distance_meters = self._calculation_value(distance_trace)
                            trace_ids.extend([duration_trace, pace_trace, distance_trace])
                        one_rm = _one_rm_for_exercise(str(exercise.get("name", "")), profile)
                        if one_rm and not is_aerobic:
                            progressed = self._calculation(
                                trace_id,
                                "calculate_progression_increment",
                                {"current_value": intensity, "percentage": (week - 1) * 1.5, "unit": "%1RM"},
                                ["periodization_week"],
                            )
                            progressed_value = self.session.get(CalculationTrace, progressed)
                            progressed_percent = float(_loads(progressed_value.result_json, {}).get("value", intensity))
                            percentage_trace = self._calculation(
                                trace_id,
                                "calculate_percentage_load",
                                {"one_rm": one_rm, "percentage": min(progressed_percent, 95), "unit": "kg"},
                                ["profile_1rm", "program_intensity"],
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
                            intensity = min(progressed_percent, 95)
                            trace_ids.extend([progressed, percentage_trace, rounded_trace])
                        sets.append(
                            {
                                "set_id": f"pset_{uuid4().hex[:12]}",
                                "set_number": set_number,
                                "reps": reps,
                                "load_kg": load_kg,
                                "intensity_percent": intensity,
                                "rpe": rpe,
                                "rest_seconds": int(self._calculation_value(rest_trace)),
                                "duration_seconds": duration_seconds,
                                "distance_meters": distance_meters,
                                "pace_seconds_per_km": pace_seconds_per_km,
                                "calculation_trace_ids": trace_ids,
                            }
                        )
                    if is_aerobic:
                        distance_per_set = float(sets[0].get("distance_meters") or 0)
                        volume_trace = self._calculation(
                            trace_id,
                            "calculate_expression",
                            {"expression": f"{sets_count} * {distance_per_set}", "unit": "m"},
                            [set_count_trace, "set_distance_traces"],
                        )
                        volume_unit = "m"
                    elif exercise_load is not None:
                        volume_trace = self._calculation(
                            trace_id,
                            "calculate_volume_load",
                            {"sets": sets_count, "reps": int(exercise_reps), "load": exercise_load, "unit": "kg"},
                            [set_count_trace, "set_repetition_and_load_traces"],
                        )
                        volume_unit = "kg"
                    else:
                        volume_trace = self._calculation(
                            trace_id,
                            "calculate_expression",
                            {"expression": f"{sets_count} * {int(exercise_reps)}", "unit": "reps"},
                            [set_count_trace, "set_repetition_traces"],
                        )
                        volume_unit = "reps"
                    volume_value = self._calculation_value(volume_trace)
                    if volume_unit == "kg":
                        weekly_workload[str(week)] += volume_value
                    exercises.append(
                        {
                            "exercise_id": exercise_id,
                            "name": str(exercise.get("name", f"Exercise {exercise_index}")),
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
            **raw,
            "sessions": sessions,
            "periodization": {
                **raw.get("periodization", {}),
                "weeks": weeks,
                "sessions_per_week": days,
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
        plan = working["calculated_plan"]
        program_id = context.latest_program_id if context.active_task_type == "program_adaptation" else None
        program = self.session.get(TrainingProgram, program_id) if program_id else None
        if not program:
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
            status="draft_adaptation" if context.active_task_type == "program_adaptation" else "draft",
            reason="adaptation" if context.active_task_type == "program_adaptation" else "new_program",
            screening_json=_dumps(plan.get("screening", {})),
            needs_analysis_json=_dumps(plan.get("needs_analysis", {})),
            periodization_json=_dumps(plan.get("periodization", {})),
            planning_model=str(plan.get("periodization", {}).get("model", "parallel")),
            rationale=str(plan.get("rationale", "")),
            assumptions_json=_dumps(plan.get("assumptions", [])),
            warnings_json=_dumps(plan.get("warnings", []) + sections.get("safety_flags_and_budget_warnings", [])),
            source_references_json=_dumps(
                {
                    "memory_trace_id": sections.get("memory_context", {}).get("trace_id"),
                    "memory_ids": sections.get("memory_context", {}).get("memory_ids", []),
                    "evidence": [
                        {"source_id": item.get("source_id"), "chunk_id": item.get("chunk_id")}
                        for item in sections.get("retrieved_evidence", [])
                    ],
                    "calculation_trace_ids": sections.get("calculation_results", []),
                }
            ),
        )
        self.session.add(version)
        for session_data in plan["sessions"]:
            self._store_session(version.version_id, session_data)
        if context.active_task_type == "program_adaptation" and context.latest_version_id:
            self.session.add(
                ProgramAdaptation(
                    adaptation_id=f"padapt_{uuid4().hex[:12]}",
                    from_version_id=context.latest_version_id,
                    to_version_id=version.version_id,
                    trigger=str(working.get("revision_trigger", "user_feedback")),
                    user_feedback=str(sections.get("current_user_turn", {}).get("content", "")),
                    changed_json=_dumps(working.get("changed_elements", [])),
                    unchanged_json=_dumps(working.get("unchanged_elements", [])),
                    rationale=version.rationale,
                    evidence_references_json=_dumps(
                        [
                            {"source_id": item.get("source_id"), "chunk_id": item.get("chunk_id")}
                            for item in sections.get("retrieved_evidence", [])
                        ]
                    ),
                    calculation_trace_ids_json=_dumps(sections.get("calculation_results", [])),
                )
            )
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
        sections["source_and_trace_references"] = {
            "memory_trace_id": sections.get("memory_context", {}).get("trace_id"),
            "evidence_chunk_ids": [item.get("chunk_id") for item in sections.get("retrieved_evidence", [])],
            "calculation_trace_ids": sections.get("calculation_results", []),
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
                        duration_seconds=set_data.get("duration_seconds"),
                        distance_meters=set_data.get("distance_meters"),
                        pace_seconds_per_km=set_data.get("pace_seconds_per_km"),
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
                            duration_seconds=item.duration_seconds,
                            distance_meters=item.distance_meters,
                            pace_seconds_per_km=item.pace_seconds_per_km,
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
        response = rationale or "No recorded plan rationale is available in this ProgramContext yet."
        trace_id = self._trace(context, "explain_plan_rationale", {"read_only": True}, started=started)
        self._assistant_message(context, response)
        return self._turn_response(context, response, trace_id)

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
        return ProgramTurnResponse(
            conversation_id=context.conversation_id,
            context_id=context.context_id,
            context_revision=context.context_revision,
            route=str(context.current_turn_route),
            status=context.status,
            response=response,
            pending_action=context.pending_action,
            draft_program_id=context.latest_program_id,
            draft_version_id=context.latest_version_id,
            trace_id=trace_id,
            warnings=warnings or list(self._sections(context).get("safety_flags_and_budget_warnings", [])),
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

    def _adaptation_for_version(self, version_id: str) -> ProgramAdaptation | None:
        return self.session.exec(
            select(ProgramAdaptation).where(ProgramAdaptation.to_version_id == version_id)
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


def _classify_adaptation_trigger(message: str) -> dict[str, Any]:
    text = message.lower()
    if re.search(r"\b(meeting|schedule|calendar|travel|work conflict|overlap|busy)\b", text):
        category = "schedule_conflict"
    elif re.search(r"\b(fatigue|tired|sore|sleep|under.recover|recovery)\b", text):
        category = "recovery_limitation"
    elif re.search(r"\b(pain|injury|ill|sick)\b", text):
        category = "pain_illness_or_safety"
    elif re.search(r"\b(missed|cannot do|can't do|couldn't do)\b", text):
        category = "missed_session_reason_unclear"
    else:
        category = "general_revision"
    return {
        "category": category,
        "adjustment_priority": ["frequency", "volume", "intensity", "exercise_selection_or_order", "rest"],
    }


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


def _one_rm_for_exercise(exercise_name: str, profile: str) -> float | None:
    normalized = exercise_name.lower()
    keys = {
        "bench": "bench_press_1rm_kg",
        "squat": "squat_1rm_kg",
        "deadlift": "deadlift_1rm_kg",
        "overhead": "overhead_press_1rm_kg",
        "press": "overhead_press_1rm_kg",
    }
    for term, key in keys.items():
        if term not in normalized:
            continue
        match = re.search(rf"{re.escape(key)}:\s*([0-9]+(?:\.[0-9]+)?)", profile)
        if match:
            return float(match.group(1))
    return None


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
