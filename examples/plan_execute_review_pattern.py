"""plan_execute_review demo: plan once, write, review, revise until approved.

    uv run python examples/plan_execute_review_pattern.py            # scripted LLM, no API key
    LGKIT_REAL_LLM=1 OPENAI_API_KEY=sk-... uv run python examples/plan_execute_review_pattern.py
"""

import os

from lgkit.builder import build_graph
from lgkit.patterns import plan_execute_review
from lgkit.testing import ScriptedLLM, using_llm


def make_llm():
    if os.environ.get("LGKIT_REAL_LLM"):
        from lgkit.llm import build_llm

        return build_llm()
    return ScriptedLLM([
        "1. Say what the endpoint does. 2. Show a request. 3. Show a response.",
        "GET /health returns service status.",
        {"verdict": "revise", "feedback": "Add the request and response examples from the plan."},
        "GET /health returns service status.\nRequest: curl /health\nResponse: {\"status\": \"ok\"}",
        {"verdict": "approve", "feedback": ""},
    ])


def main() -> dict:
    per = plan_execute_review(
        "doc",
        planner="Break the task into the steps a good answer must cover.",
        executor="Write the answer following the plan. Address any reviewer feedback.",
        reviewer="Approve only if every step of the plan is covered.",
        max_rounds=3,
    )
    app = build_graph(per.to_workflow(name="doc"))
    with using_llm(make_llm()):
        state = app.invoke({"task": "Document the GET /health endpoint."})
    scratch = state["scratch"]
    summary = {
        "approved": scratch["doc__review"]["verdict"] == "approve" and not scratch.get("doc__exhausted"),
        "rounds": scratch.get("doc__round", 0),
        "result": scratch["doc__result"],
    }
    print(summary)
    return summary


if __name__ == "__main__":
    main()
