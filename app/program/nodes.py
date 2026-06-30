"""Small typed nodes used by the Program Generator LangGraph."""

from app.core.errors import ConfigurationError
from app.core.llm_provider import ProgramRationaleRequest, ProgramTaskRouteRequest
from app.program.schemas import ProgramContextPatch, ProgramGraphState


DETERMINISTIC_GENERATION_STEPS = {"load_calculation", "volume_calculation"}


class TaskAndSOPRouter:
    """Classify one user turn through the configured chat model."""

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    def __call__(self, state: ProgramGraphState) -> ProgramGraphState:
        context = state.get("context", {})
        if state.get("route_locked"):
            return {
                "route": str(context.get("current_turn_route") or "new_program"),
                "task_type": str(context.get("active_task_type") or "program_generation"),
            }
        sections = context.get("sections", {})
        if not self.llm_provider or not hasattr(self.llm_provider, "generate_program_task_route"):
            raise ConfigurationError("Program task routing requires a configured LLM provider.")
        result = self.llm_provider.generate_program_task_route(
            ProgramTaskRouteRequest(
                user_turn=str(sections.get("current_user_turn", {}).get("content", "")),
                pending_action=context.get("pending_action"),
                active_task_type=context.get("active_task_type"),
                active_sop=context.get("active_sop"),
                current_sop_step=context.get("current_sop_step"),
                latest_version_id=context.get("latest_version_id"),
                approval_status=context.get("approval_status"),
            )
        )
        route = result.route
        if route not in {
            "new_program",
            "explain_plan_rationale",
            "clarification_answer",
        }:
            raise ValueError(f"TaskAndSOPRouter returned invalid route: {route}")
        return {
            "route": route,
            "task_type": "program_generation",
            "route_rationale": result.rationale,
            "route_confidence": result.confidence,
        }


class RationaleAnswerBuilder:
    """Use the configured chat model to answer from stored rationale context."""

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    def __call__(self, state: ProgramGraphState) -> ProgramGraphState:
        if not self.llm_provider or not hasattr(self.llm_provider, "generate_program_rationale"):
            raise ConfigurationError("Program rationale answers require a configured LLM provider.")
        context = state.get("rationale_context", {})
        answer = self.llm_provider.generate_program_rationale(
            ProgramRationaleRequest(
                question=str(state.get("rationale_question", "")),
                sop_context=context,
                recorded_rationale=str(context.get("recorded_rationale", "")),
                numeric_facts=state.get("numeric_rationale_facts") or None,
            )
        )
        return {"rationale_answer": answer}


class SOPStepReadinessAssessor:
    """Report missing raw inputs and the current step's evidence/output state."""

    REQUIRED_INPUTS = (
        "goal",
        "age",
        "sex",
        "height_cm",
        "bodyweight_kg",
        "medical_history",
        "baseline_assessments",
        "lifestyle_work_demands",
        "days_per_week",
        "training_days",
        "session_duration_minutes",
        "program_duration_weeks",
        "experience",
        "equipment",
    )

    def __call__(self, state: ProgramGraphState) -> ProgramGraphState:
        context = state.get("context", {})
        sections = context.get("sections", {})
        working_key = "new_program_working_state"
        working = sections.get(working_key, {})
        inputs = working.get("collected_inputs", {})
        step = str(context.get("current_sop_step") or "consultation_and_screening")
        progress = sections.get("sop_progress", {})
        completed_steps = progress.get("completed_steps", [])
        missing = [key for key in self.REQUIRED_INPUTS if inputs.get(key) in (None, "", [])]
        if step == "load_calculation":
            if working.get("pending_bodyweight_clarification"):
                missing.append("bodyweight_kg")
            missing.extend(
                f"load_reference:{exercise_name}"
                for exercise_name in working.get("pending_load_clarifications", [])
            )
        evidence_sufficient = bool(progress.get("evidence_sufficient_by_step", {}).get(step))
        if step in DETERMINISTIC_GENERATION_STEPS:
            evidence_sufficient = True
        return {
            "missing_inputs": missing,
            "current_step": step,
            "evidence_sufficient": evidence_sufficient,
            "step_complete": step in completed_steps,
            "sop_complete": bool(progress.get("sop_complete")),
            "has_raw_plan": bool(working.get("raw_plan")),
            "has_calculated_plan": bool(working.get("calculation_artifacts")),
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
