# langgraph-agent (`lgkit`)

Build LangGraph workflows from plain spec data, plus ready-made patterns.
`lgtools` (laggraph-hermes) uses this as its kernel.

## Install

```bash
uv sync                              # in this repo: core + dev tools
uv add --editable ../langgraph-agent # in another project (local path; not published to PyPI)
uv add langchain-openai              # only if you use the OpenAI provider
```

## Approval gate

```python
from langgraph.checkpoint.memory import InMemorySaver
from lgkit.builder import build_graph
from lgkit.patterns import approval_gate

gate = approval_gate(
    "publish_check",
    message="Publish?\n\n{scratch.draft}",
    choices=["approve", "revise"],
    approver="agent",              # "human" (default) | "agent"
    criteria="Approve only if complete and polite.",
    min_confidence=0.8,            # below this, or on any error, a human is asked
)
app = build_graph(gate.to_workflow(), checkpointer=InMemorySaver())
```

| Setting | Who decides |
|---|---|
| `approver="human"` | you, via `interrupt()` |
| `approver="human", advise=True` | you, with the agent's recommendation shown |
| `approver="agent"` | the agent; escalates to you on `escalate`, low confidence or failure |

Every mode writes `scratch[<id>] = {"choice", "comment", "by", "advice"}` and routes on `signal = choice`.

Run the offline demo: `uv run python examples/approval_demo.py`

## Plan → execute → review

```python
from lgkit.builder import build_graph
from lgkit.patterns import plan_execute_review

per = plan_execute_review(
    "doc",
    planner="Break the task into steps.",
    executor="Write the document following the plan.",   # or a NodeSpec + executor_result_key
    reviewer="Approve only if every step of the plan is covered.",
    max_rounds=3,                                         # most times the execute step runs
)
app = build_graph(per.to_workflow())
state = app.invoke({"task": "Document the GET /health endpoint."})
state["scratch"]["doc__result"]      # the work
state["scratch"]["doc__review"]      # {"verdict": "approve" | "revise", "feedback": ...}
state["scratch"].get("doc__exhausted")  # True when the rounds ran out — NOT an approval
```

The fragment has two named exits, `per.exits["approved"]` and `per.exits["exhausted"]`, each a `(node id, condition)` pair — wire the exhausted one into an `approval_gate` to let a human decide.

Its building blocks are ordinary node kinds you can use directly: `llm` (one LLM call; text or structured output; a structured field can be the routing signal) and `loop_limit` (`again` until `max_rounds`, then `exhausted`).

Run the offline demo: `uv run python examples/plan_execute_review_pattern.py`

## Tests

```bash
uv run pytest -q
```
