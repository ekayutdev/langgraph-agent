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

## Tests

```bash
uv run pytest -q
```
