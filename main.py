from typing import TypedDict

from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    task: str
    plan: str
    result: str
    approved: bool


def planner(state):
    return {
        "plan": f"Plan for: {state['task']}"
    }


def executor(state):
    return {
        "result": f"Executed: {state['plan']}"
    }


def reviewer(state):
    approved = len(state["result"]) > 10

    return {
        "approved": approved
    }


def route(state):
    if state["approved"]:
        return "end"

    return "executor"


graph = StateGraph(AgentState)

graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("reviewer", reviewer)

graph.set_entry_point("planner")

graph.add_edge("planner", "executor")
graph.add_edge("executor", "reviewer")

graph.add_conditional_edges(
    "reviewer",
    route,
    {
        "executor": "executor",
        "end": END
    }
)

app = graph.compile()

result = app.invoke(
    {
        "task": "Create API documentation"
    }
)

print(result)