"""LangGraph runner for program route, readiness, and action decisions."""

from app.program.nodes import RationaleAnswerBuilder, ReasoningActionRouter, SOPStepReadinessAssessor, TaskAndSOPRouter
from app.program.schemas import ProgramGraphState


def build_program_generator_graph(llm_provider):
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(ProgramGraphState)
    workflow.add_node("task_and_sop_router", TaskAndSOPRouter(llm_provider))
    workflow.add_node("sop_step_readiness_assessor", SOPStepReadinessAssessor())
    workflow.add_node("reasoning_action_router", ReasoningActionRouter())
    workflow.set_entry_point("task_and_sop_router")
    workflow.add_edge("task_and_sop_router", "sop_step_readiness_assessor")
    workflow.add_edge("sop_step_readiness_assessor", "reasoning_action_router")
    workflow.add_edge("reasoning_action_router", END)
    return workflow.compile()


def run_program_generator_graph(initial_state: ProgramGraphState, llm_provider) -> ProgramGraphState:
    """Use LangGraph when installed and the same nodes sequentially otherwise."""

    try:
        return build_program_generator_graph(llm_provider).invoke(initial_state)
    except ModuleNotFoundError:
        state = dict(initial_state)
        for node in (TaskAndSOPRouter(llm_provider), SOPStepReadinessAssessor(), ReasoningActionRouter()):
            state.update(node(state))
        return state


def build_program_rationale_graph(llm_provider):
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(ProgramGraphState)
    workflow.add_node("rationale_answer_builder", RationaleAnswerBuilder(llm_provider))
    workflow.set_entry_point("rationale_answer_builder")
    workflow.add_edge("rationale_answer_builder", END)
    return workflow.compile()


def run_program_rationale_graph(initial_state: ProgramGraphState, llm_provider) -> ProgramGraphState:
    """Run the grounded LLM rationale node through LangGraph when available."""

    try:
        return build_program_rationale_graph(llm_provider).invoke(initial_state)
    except ModuleNotFoundError:
        state = dict(initial_state)
        state.update(RationaleAnswerBuilder(llm_provider)(state))
        return state
