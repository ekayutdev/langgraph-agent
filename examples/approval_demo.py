"""approval_gate demo: an agent reviews a draft and escalates to you when unsure.

    uv run python examples/approval_demo.py                  # scripted LLM, no API key
    LGKIT_REAL_LLM=1 OPENAI_API_KEY=sk-... uv run python examples/approval_demo.py
"""

import os

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from lgkit.builder import build_graph
from lgkit.patterns import approval_gate
from lgkit.testing import ScriptedLLM, using_llm


def make_llm():
    if os.environ.get("LGKIT_REAL_LLM"):
        from lgkit.llm import build_llm

        return build_llm()
    return ScriptedLLM([{"choice": "approve", "confidence": 0.55, "reason": "Fine but terse."}])


def main() -> dict:
    gate = approval_gate(
        "publish_check",
        message="Publish this announcement?\n\n{scratch.draft}",
        choices=["approve", "revise"],
        approver="agent",
        criteria="Approve only if the announcement is complete, accurate and polite.",
        min_confidence=0.8,
    )
    app = build_graph(gate.to_workflow(), checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "demo"}}

    with using_llm(make_llm()):
        app.invoke({"task": "publish", "scratch": {"draft": "Release moves to Friday."}}, cfg)
        snap = app.get_state(cfg)
        if snap.interrupts:
            ask = snap.interrupts[0].value
            print(f"Agent escalated. Advice: {ask['advice']}  Error: {ask['advice_error']}")
            answer = input(f"{ask['message']}\nChoose {ask['choices']}: ").strip() or "approve"
            app.invoke(Command(resume={"choice": answer, "comment": "via demo"}), cfg)

    result = app.get_state(cfg).values["scratch"]["publish_check"]
    print(result)
    return result


if __name__ == "__main__":
    main()
