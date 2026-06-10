"""LangGraph runner plus a deterministic fallback runner.

The service uses LangGraph when the package is installed. The deterministic
runner stays available so local syntax checks and lightweight tests can run
before optional dependencies are installed.
"""

from collections.abc import Iterable

from app.qa.schemas import QAState


def run_nodes(initial_state: QAState, nodes: Iterable) -> QAState:
    """Run Q&A nodes in order.

    Parameters:
        initial_state: QAState with raw question fields.
        nodes: iterable of callable nodes.
    Returns:
        QAState: merged state after every node runs.
    """

    state = dict(initial_state)
    for node in nodes:
        updates = node(state)
        state.update(updates)
    return state


def build_langgraph(nodes: list):
    """Build a LangGraph StateGraph from the same node callables.

    Returns:
        Compiled graph object when LangGraph is installed.
    """

    from langgraph.graph import END, StateGraph

    workflow = StateGraph(QAState)
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


def run_langgraph_or_nodes(initial_state: QAState, nodes: list) -> QAState:
    """Run the Q&A workflow with LangGraph when it is installed.

    Parameters:
        initial_state: QAState with raw question fields.
        nodes: ordered graph node callables.
    Returns:
        QAState: final graph state. Falls back to `run_nodes` only when the
        optional LangGraph package is not installed.
    """

    try:
        graph = build_langgraph(nodes)
    except ModuleNotFoundError:
        return run_nodes(initial_state, nodes)
    return graph.invoke(initial_state)
