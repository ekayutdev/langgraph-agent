"""llm node — one LLM call. The result goes to scratch; a structured result
can also emit the routing signal.

It has no tools and no inner loop: looping is the graph's job. On failure it
RAISES (after ``max_attempts``) — unlike the advisor there is no human waiting
to catch it, and a node that swallowed the failure would leave a stale or
missing signal for the router to act on.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphBubbleUp
from pydantic import BaseModel, Field, create_model

from lgkit.nodes._llm import resolve_llm
from lgkit.nodes._template import flat_state

_TYPES: dict[str, Any] = {
    "string": str,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def output_model(
    fields: list[dict[str, Any]], signal_field: str | None, signal_values: list[str]
) -> type[BaseModel]:
    """Build the structured-output schema. The signal field is constrained to
    ``signal_values`` so the model cannot answer something the graph has no
    edge for."""
    defs: dict[str, Any] = {}
    for f in fields:
        name = f["name"]
        if name == signal_field:
            typ: Any = Literal[tuple(signal_values)]
        else:
            typ = _TYPES.get(str(f.get("type") or "string"), str)
        desc = f.get("description")
        if f.get("required", True):
            defs[name] = (typ, Field(..., description=desc))
        else:
            defs[name] = (typ | None, Field(None, description=desc))
    return create_model("LlmOutput", **defs)


def _text(out: Any) -> str:
    content = out.content if isinstance(out, BaseMessage) else out
    return content if isinstance(content, str) else str(content)


def _structured(out: Any, schema: type[BaseModel]) -> dict[str, Any]:
    # A fake/scripted model returns a plain AIMessage (itself a pydantic
    # model) whose .content is the reply: parse that as JSON before the
    # BaseModel branch can misread it.
    if isinstance(out, BaseMessage):
        data = json.loads(_text(out))
    elif isinstance(out, BaseModel):
        data = out.model_dump()
    else:
        data = out
    return schema.model_validate(data).model_dump()


def _call(params, ctx, node_id: str, messages: list, schema: type[BaseModel] | None) -> Any:
    """One model call, returning the raw reply.

    A dry-run that can hand out per-visit scripted replies (``next_mock``) is
    asked directly: ``model_for()`` builds a fresh fake model on every call,
    which would replay the FIRST scripted reply each time a loop revisits this
    node.
    """
    dry = getattr(ctx, "dry_run", None) if ctx is not None else None
    if dry is not None and hasattr(dry, "next_mock"):
        raw = dry.next_mock(node_id)
        if raw is None:
            raise RuntimeError("no scripted reply left")
        return AIMessage(content=raw)
    model = resolve_llm(params, ctx, node_id)
    return (model if schema is None else model.with_structured_output(schema)).invoke(messages)


def run(state, params, prompt, ctx=None, resolved=None) -> dict[str, Any]:
    node_id = params.get("__node_id") or "llm"
    message = str(params.get("message") or "{task}").format_map(flat_state(state))
    fields = list(params.get("output_fields") or [])
    signal_field = params.get("signal_field")
    schema = (
        output_model(fields, signal_field, list(params.get("signal_values") or []))
        if fields
        else None
    )
    messages = ([SystemMessage(prompt)] if prompt else []) + [HumanMessage(message)]
    attempts = max(1, int(params.get("max_attempts", 2)))

    last_error = ""
    for _ in range(attempts):
        try:
            out = _call(params, ctx, node_id, messages, schema)
            result: Any = _text(out) if schema is None else _structured(out, schema)
            break
        except GraphBubbleUp:
            raise
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
    else:
        hint = ""
        if ctx is not None and getattr(ctx, "dry_run", None) is not None:
            hint = f" — dry-run: script this node's reply with mock_scripts['{node_id}']"
        raise RuntimeError(
            f"llm node '{node_id}' failed after {attempts} attempts: {last_error}{hint}"
        )

    delta: dict[str, Any] = {
        "scratch": {params.get("result_key") or node_id: result},
        "events": [{"node": node_id, "result": result, "tools": []}],
    }
    if signal_field:
        delta["signal"] = result[signal_field]
    return delta


__all__ = ["output_model", "run"]
