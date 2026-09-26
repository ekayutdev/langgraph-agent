"""Plain-LangGraph plan -> execute -> review loop (no lgkit), kept as a reference.

The first draft is always rejected so the loop is exercised; MAX_ROUNDS stops
it even if the reviewer never approves.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

MAX_ROUNDS = 3


class AgentState(TypedDict, total=False):
    task: str
    plan: str
    result: str
    approved: bool
    attempts: int


def planner(state: AgentState) -> AgentState:
    return {"plan": f"Plan for: {state['task']}", "attempts": 0}


def executor(state: AgentState) -> AgentState:
    attempts = state.get("attempts", 0) + 1
    suffix = " (draft)" if attempts == 1 else ""
    return {"result": f"Executed: {state['plan']}{suffix}", "attempts": attempts}


def reviewer(state: AgentState) -> AgentState:
    return {"approved": not state["result"].endswith("(draft)")}


def route(state: AgentState) -> str:
    if state["approved"] or state["attempts"] >= MAX_ROUNDS:
        return "end"
    return "executor"


def build():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("reviewer", reviewer)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_conditional_edges("reviewer", route, {"executor": "executor", "end": END})
    return graph.compile()


def run(task: str) -> dict:
    return build().invoke({"task": task})


if __name__ == "__main__":
    print(run("Create API documentation"))
