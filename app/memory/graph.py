"""LangGraph runner plus deterministic fallback for memory workflows."""

from collections.abc import Iterable

from app.memory.schemas import MemoryWorkflowState


def run_nodes(initial_state: MemoryWorkflowState, nodes: Iterable) -> MemoryWorkflowState:
    """Run memory workflow nodes in order."""

    state = dict(initial_state)
    for node in nodes:
        updates = node(state)
        state.update(updates)
    return state


def build_langgraph(nodes: list):
    """Build a LangGraph StateGraph from memory node callables."""

    from langgraph.graph import END, StateGraph

    workflow = StateGraph(MemoryWorkflowState)
    previous_name = None
    for index, node in enumerate(nodes):
        name = f"node_{index}_{node.__class__.__name__}"
        workflow.add_node(name, node)
        if previous_name is None:
            workflow.set_entry_point(name)
        else:
            workflow.add_edge(previous_name, name)
        previous_name = name
    if previous_name:
        workflow.add_edge(previous_name, END)
    return workflow.compile()


def run_langgraph_or_nodes(initial_state: MemoryWorkflowState, nodes: list) -> MemoryWorkflowState:
    """Run with LangGraph when installed, otherwise run nodes directly."""

    try:
        graph = build_langgraph(nodes)
    except ModuleNotFoundError:
        return run_nodes(initial_state, nodes)
    return graph.invoke(initial_state)

