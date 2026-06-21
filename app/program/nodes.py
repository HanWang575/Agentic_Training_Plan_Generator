"""Small typed nodes used by the Program Generator LangGraph."""

from app.program.schemas import ProgramContextPatch, ProgramGraphState


class TaskAndSOPRouter:
    """Classify the current turn using pending state before message wording."""

    def __call__(self, state: ProgramGraphState) -> ProgramGraphState:
        context = state.get("context", {})
        if state.get("route_locked"):
            return {
                "route": str(context.get("current_turn_route") or "new_program"),
                "task_type": str(context.get("active_task_type") or "program_generation"),
            }
        sections = context.get("sections", {})
        pending = context.get("pending_action")
        message = str(sections.get("current_user_turn", {}).get("content", "")).lower()
        if pending == "waiting_for_user":
            route = "clarification_answer"
        elif any(term in message for term in ("why", "rationale", "explain the plan", "explain this plan")):
            route = "explain_plan_rationale"
        elif any(
            term in message
            for term in ("revise", "adapt", "change my plan", "missed", "cannot do", "can't do", "couldn't do")
        ):
            route = "revise_current_plan"
        else:
            route = "new_program"
        task_type = "program_adaptation" if route == "revise_current_plan" else "program_generation"
        if route == "clarification_answer":
            task_type = str(context.get("active_task_type") or "program_generation")
        return {"route": route, "task_type": task_type}


class SOPStepReadinessAssessor:
    """Report missing raw inputs and the current step's evidence/output state."""

    REQUIRED_INPUTS = ("goal", "days_per_week", "session_duration_minutes", "experience", "equipment", "screening")

    def __call__(self, state: ProgramGraphState) -> ProgramGraphState:
        context = state.get("context", {})
        sections = context.get("sections", {})
        working_key = (
            "revision_adaptation_working_state"
            if state.get("task_type") == "program_adaptation"
            else "new_program_working_state"
        )
        working = sections.get(working_key, {})
        inputs = working.get("collected_inputs", {})
        step = str(context.get("current_sop_step") or "consultation_and_screening")
        progress = sections.get("sop_progress", {})
        completed_steps = progress.get("completed_steps", [])
        missing = [key for key in self.REQUIRED_INPUTS if inputs.get(key) in (None, "", [])]
        return {
            "missing_inputs": missing,
            "current_step": step,
            "evidence_sufficient": bool(progress.get("evidence_sufficient_by_step", {}).get(step)),
            "step_complete": step in completed_steps,
            "sop_complete": bool(progress.get("sop_complete")),
            "has_raw_plan": bool(working.get("raw_plan")),
            "has_calculated_plan": bool(working.get("calculated_plan")),
        }


class ReasoningActionRouter:
    """Choose one explicit next action for the current SOP step."""

    def __call__(self, state: ProgramGraphState) -> ProgramGraphState:
        if state.get("route") == "explain_plan_rationale":
            return {"next_action": "explain_plan_rationale"}
        if state.get("missing_inputs") and state.get("user_question_count", 0) < state.get("max_user_questions", 5):
            return {"next_action": "ask_user"}
        if not state.get("evidence_sufficient") and state.get("rag_query_count", 0) < state.get("max_rag_queries", 5):
            return {"next_action": "query_rag"}
        if not state.get("step_complete"):
            return {"next_action": "write_step_output"}
        if state.get("sop_complete") and not state.get("has_calculated_plan"):
            return {"next_action": "run_calculation"}
        if state.get("sop_complete"):
            return {"next_action": "finalize_output"}
        return {"next_action": "write_step_output"}


class SOPStepExecutor:
    """Name-bearing contract; the service performs the LLM-backed write."""

    def __call__(self, state: ProgramGraphState) -> ProgramGraphState:
        return {"next_action": "write_step_output"}


class ProgramContextPatchValidator:
    """Reject stale revisions, unknown sections, and invalid operations."""

    def __init__(self, allowed_sections: set[str]):
        self.allowed_sections = allowed_sections

    def validate(self, patch: ProgramContextPatch, current_revision: int) -> None:
        if patch.expected_context_revision != current_revision:
            raise ValueError(
                f"Stale ProgramContext patch: expected revision {patch.expected_context_revision}, "
                f"found {current_revision}."
            )
        if patch.target_section not in self.allowed_sections:
            raise ValueError(f"Unknown ProgramContext section: {patch.target_section}")
        if patch.operation == "append" and not isinstance(patch.value, list):
            raise ValueError("Append patches must contain a list value.")
        if patch.operation == "merge" and not isinstance(patch.value, dict):
            raise ValueError("Merge patches must contain an object value.")
